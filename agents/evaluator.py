# deep dive agent logic (3: z.ai (GLM-4.5-flash), openrouter (nemotron), google-gemini (2.5-flash-lite))
"""
Research Evaluator Module
A LangChain-based evaluator that judges search result relevance and determines whether additional 
searches are needed for a given subquestion / subtopic
"""

from typing import Dict, List, Optional
from typing_extensions import TypedDict, Annotated
from enum import Enum
import random
import logging
from operator import add

from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field

from tools.web_search import get_ws_provider

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelevanceLevel(str, Enum):
    """Relevance levels for search results"""
    HIGHLY_RELEVANT = "highly_relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    MINIMALLY_RELEVANT = "minimally_relevant"
    NOT_RELEVANT = "not_relevant"

class EvaluationResult(BaseModel):
    """Structured output for evaluation results"""
    relevance_level: RelevanceLevel = Field(
        description="How relevant the search results are to the subquestion"
    )
    relevance_score: float = Field(
        description="Numeric relevance score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    stop_search: bool = Field(
        description="Whether to stop searching (True) or continue (False)"
    )
    reasoning: str = Field(
        description="Explanation of the evaluation decision"
    )
    next_search_term: Optional[str] = Field(
        description="Suggested keyword/term for next search if stop_search is False",
        default=None
    )

# === State definition ===
class ResearchState(TypedDict):
    """
    State for the research evaluator graph
    
    Attributes:
        subquestion (str): The specific question being researched
        new_search_results (List{Document}): Fresh results from the latest search action
        accumulated context (List[str]): Summaries or raw text of what we have so far (Optional)
        iteration (int): Current search depth
        max_iterations (int): safety limit
        evaluation_history (List[EvaluationResult]): Track decision made
        final_answer_ready (bool): Flag to signal completion"""
    subquestion: str
    new_search_results: List[Document]  # page_content is the content snippet, metadata includes url and title
    iteration: int
    max_iterations: int
    evaluation_history: Annotated[List[EvaluationResult], add]
    current_query: str

# --- Evaluator module ---
class ResearchEvaluator:
    WEB_SEARCH_PROVIDERS = {"duckduckgo", "searchxng", "whoogle"}

    def __init__(self, 
                 host: str = "http://192.168.0.162:11434",
                 model_name: str = "qwen3:4b", 
                 temperature: float = 0.1,
                 max_search_iterations: int = 5,
                 relevance_threshold: float = 0.8,
                 embedding_model: str = "embeddinggemma:latest",
                 chroma_collection_name: str = "research",
                 chroma_domain: str = "192.168.0.162",
                 chroma_port: int = 9000):
        """
        Initialize the evaluator with Ollama models"""
        self.llm = ChatOllama(model=model_name, base_url=host, temperature=temperature)
        self.max_search_iterations = max_search_iterations
        self.relevance_threshold = relevance_threshold
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=host
        )
        print(f"Evaluator LLM: {model_name} initialized")
        # initialize the local Chroma client
        chroma_client = chromadb.HttpClient(host=chroma_domain, port=chroma_port)
        self.vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=self.embeddings,
            client=chroma_client
        )
        print(f"Vector store client created")

        # initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=EvaluationResult)

        # create evaluation prompt
        self.evaluator_chain = self._build_evaluator_chain()

    def _build_evaluator_chain(self):
        """Builds the LLM chain for judging relevance"""
        system_prompt = """You are an expert research evaluator. Your task is to assess whether the gathered information is sufficient to answer a specific sub-question.
        
        You have access to a vector database of research snippets. 
        1. Analyze the 'Current Sub-question'.
        2. Review the 'Retrieved Context' which represents the most relevant information gathered so far.
        3. Determine if this information comprehensively answers the sub-question.
        
        If the information is missing or incomplete, provide a specific 'next_search_term' to fill the gap.
        If the information is sufficient, set 'stop_search' to True.
        
        Output must strictly follow the JSON format."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            Current sub-question: {subquestion}
             
            Retrieved Context (from vector DB):
            {context}
            
            Previous Search Term: {current_query}
             
            {format_instructions}""")
        ])

        return prompt | self.llm | self.output_parser


    def web_search(self, state: ResearchState, max_results: int = 10) -> Dict:
        query = state["current_query"]
        iteration = state["iteration"]

        if not query:
            logger.warning(f"query cannot be empty or None: {query}") 

        logger.info(f"[Search] Iteration {iteration}: {query}")

        provider_name = random.choice(list(self.WEB_SEARCH_PROVIDERS))
        web_search_tool = get_ws_provider(provider_name=provider_name)
        results = web_search_tool.search(query=query, max_results=max_results)
        
        docs = []
        for result in results:
            content = result.get("content")
            if not content:
                continue
            docs.append(Document(page_content=content, 
                                 metadata={"source": result.get("source"), "url": result.get("url"), "title": result.get("title")}))
        return {
            "new_search_results": docs,
            "iteration": iteration + 1
        }
    
    def index_if_needed(self, state: ResearchState) -> Dict:
        docs = state.get("new_search_results", [])
        if docs:
            logger.info(f"[Index] Adding {len(docs)} docs to chroma")
            self.vector_store.add_documents(docs)
        return {}
    
    def evaluate(self, state: ResearchState) -> Dict:
        iteration = state["iteration"]
        subquestion = state["subquestion"]
        current_query = state["current_query"]

        # --- Interation 0: raw web results only
        if iteration == 1:
            context = "\n\n".join(d.page_content for d in state['new_search_results'])
            source = "RAW_WEB"
        else:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(subquestion)
            context = "\n\n".join(d.page_content for d in docs) or "No relevant info from local database"
            source = "CHROMA"

        logger.info(f"[Evaluate] Source={source}, Iteration={iteration}")

        result: EvaluationResult = self.evaluator_chain.invoke({
            "subquestion": subquestion,
            "iteration": iteration,
            "context": context,
            "current_query": current_query,
            "format_instructions": PydanticOutputParser(
                pydantic_object=EvaluationResult
            ).get_format_instructions()
        })

        return {
            "evaluation_history": [result],
            "current_query": result.next_search_term or current_query
        }
    
    def should_continue(self, state: ResearchState):
        latest = state["evaluation_history"][-1]

        if latest.stop_search:
            logger.info("[Stop] Evaluator says efficient")
            return "stop"
        
        if state["iteration"] >= state['max_iterations']:
            logger.info("[Stop] Max iterations reached")
            return "stop"
        
        return "continue"
    
    def build_graph(self):
        g = StateGraph(ResearchState)

        g.add_node("search", self.web_search)
        g.add_node("index", self.index_if_needed)
        g.add_node("evaluate", self.evaluate)

        g.set_entry_point("search")
        g.add_edge("search", "index")
        g.add_edge("index", "evaluate")
        g.add_conditional_edges(
            "evaluate",
            self.should_continue,
            {
                "continue": "search",
                "stop": END
            }
        )

        return g.compile()
    
# ----
# Example run 
# ----
if __name__ == "__main__":
    evaluator = ResearchEvaluator()
    app = evaluator.build_graph()

    state = {
        "subquestion": "What are the core benefits of using LangGraph?",
        "current_query": "LangGraph benefits",
        "new_search_results": [],
        "iteration": 1,
        "max_iterations": 2,
        "evaluation_history": []
    }

    for step in app.stream(state):
        for node, value in step.items():
            print(f"\n--- {node.upper()} ---")
            if node == "evaluate":
                res = value["evaluation_history"][-1]
                print(f"Decision: {'STOP' if res.stop_search else 'CONTINUE'}")
                print(f"Reason: {res.reasoning}")
