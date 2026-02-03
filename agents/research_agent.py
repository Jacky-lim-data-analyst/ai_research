"""LangGrapgh research agent

A comprehensive research agent using LangGraph for workflow orchestration.
This agent queries a local vector store, fetch webpage, analyzes results, 
and generates detailed research reports
"""

from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
import operator
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.search_tools import chroma_search, vector_store
from utils.text_io import string_to_text_file

def _structured_output_to_dict(response):
    if not response:
        print("Empty structured llm response")
        return {}
    
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        else:
            return response.dict()
    except Exception:
        print("Unexpected output format from structured llm")
        return {}

# ============================================================================
# State Definition
# ============================================================================
class ResearchState(TypedDict):
    """State for the research workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    search_results: list[dict]
    research_findings: str
    iterations: int
    max_iterations: int
    should_continue: bool
    final_report: str

# ============================================================================
# Node Functions
# ============================================================================

def initialize_research(state: ResearchState) -> dict:
    """Initialize research process and extract user query"""
    messages = state['messages']

    # extract user query from the last human message
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    return {
        "query": user_query,
        "iterations": 0,
        "max_iterations": state.get("max_iterations", 3),
        "should_continue": True,
        "search_results": [],
        "research_findings": "",
    }

def search_vector_store(state: ResearchState) -> dict:
    """Query the local vector store for relevant info"""
    query = state['query']
    current_findings = state.get('research_findings', '')

    # perform vector search
    search_result = chroma_search.invoke({"query": query})

    # extract urls and metadata from search results
    results = []
    retrieval_res = vector_store.similarity_search_with_score(query, k=5)

    for doc, score in retrieval_res:
        results.append({
            "url": doc.metadata.get("url", ""),
            "title": doc.metadata.get("title", "Unknown"),
            "snippet": doc.page_content,
            "score": round(float(score), 2)
        })

    # updated findings
    updated_findings = current_findings + f"\n\n## Search Results (Iteration {state['iterations'] + 1})\n{search_result}"

    return {
        "search_results": results,
        "research_findings": updated_findings,
        "iterations": state["iterations"] + 1,
        "messages": [AIMessage(content=f"Retrieved {len(results)} results from vector store")]
    }

class AnalyzeResults(BaseModel):
    key_findings: str = Field(description="Summary of what we had found so far")
    gaps: str = Field(description="What is still missing in the research")
    should_continue: bool = Field(description="Whether we should continue the research loop or not")
    reason: str = Field(description="why we should continue or stop?")

def analyze_results(state: ResearchState, llm: ChatOllama) -> dict:
    """Analyze search results and determine if more research is needed"""
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research analyst. Analyze the search results and determine:
1. What key information have we found?
2. What gaps still exist in answering the user's query?
3. Should we continue searching or do we have enough information?

Current query: {query}
Iteration: {iteration} of {max_iterations}

Respond with:
{{
    "key_findings": "summary of what we found",
    "gaps": "what's still missing",
    "should_continue": true/false,
    "reason": "why continue or stop"
}}"""), 
        ("user", "Research findings so far:\n{findings}")
    ])

    structured_llm = llm.with_structured_output(AnalyzeResults)
    chain = analysis_prompt | structured_llm

    response = chain.invoke({
        "query": state["query"],
        "iteration": state["iterations"],
        "max_iterations": state["max_iterations"],
        "findings": state["research_findings"]
    })

    # get the response
    response_dict = _structured_output_to_dict(response)
    should_continue = (
        state["iterations"] < state["max_iterations"] and
        response_dict.get("should_continue")
    )

    return {
        "should_continue": should_continue,
        "messages": [AIMessage(content=f"""Analysis:
                               key findings: {response_dict.get("key_findings", "")}
                                gaps: {response_dict.get("gaps", "")}""")]
    }

def generate_report(state: ResearchState, llm: ChatOllama) -> dict:
    """Generate the final comprehensive research report."""
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research report writer. Based on the research findings,
generate a comprehensive, well-structured report that answers the user's query.

The report should:
- Start with an executive summary
- Organize information into clear sections
- Include specific details and examples from the research
- Cite sources when appropriate
- End with key takeaways or conclusions

User Query: {query}
Current Date: {date}"""),
        ("user", "Research Findings:\n{findings}\n\nGenerate a comprehensive report.")
    ])

    chain = report_prompt | llm

    response = chain.invoke({
        "query": state["query"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "findings": state["research_findings"]
    })

    return {
        "final_report": response.content,
        "messages": [AIMessage(content="Generated final research report")]
    }

def refine_query(state: ResearchState, llm: ChatOllama) -> dict:
    """Refine the search query based on current findings to fill gaps"""
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", """Based on the current research findings and gaps, suggest a refined search query
that would help fill the missing information. Return ONLY the refined query text, nothing else.

Original Query: {original_query}"""),
        ("user", "Current Findings:\n{findings}\n\nWhat should we search for next?")
    ])

    chain = refinement_prompt | llm

    response = chain.invoke({
        "original_query": state["query"],
        "findings": state["research_findings"]
    })

    # update query for next iteration
    refined_query = response.content

    return {
        "query": refined_query.strip() if isinstance(refined_query, str) else refined_query,
        "messages": [AIMessage(content=f"Refined search query: {refined_query}")],
    }

# ============================================================================
# Routing Functions
# ============================================================================

def should_continue_research(state: ResearchState) -> Literal["continue", "generate_report"]:
    """Determine if we should continue researching or generate the final report"""
    if state["should_continue"] and state["iterations"] < state["max_iterations"]:
        return "continue"
    return "generate_report"

# ============================================================================
# Graph Construction
# ============================================================================

def create_research_graph(llm: ChatOllama, query_llm: ChatOllama):
    """Create the LangGraph research workflow"""
    # create the graph
    workflow = StateGraph(ResearchState)

    # add nodes
    workflow.add_node("initialize", initialize_research)
    workflow.add_node("search", search_vector_store)
    workflow.add_node("analyze", lambda state: analyze_results(state, llm))
    workflow.add_node("refine", lambda state: refine_query(state, query_llm))
    workflow.add_node("generate_report", lambda state: generate_report(state, llm))

    # define the flow
    workflow.set_entry_point("initialize")

    workflow.add_edge("initialize", "search")
    workflow.add_edge("search", "analyze")

    workflow.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "continue": "refine",
            "generate_report": "generate_report"
        }
    )

    workflow.add_edge("refine", "search")
    workflow.add_edge("generate_report", END)

    return workflow.compile()

# ============================================================================
# Main Execution Function
# ============================================================================

def run_research(query: str, llm: ChatOllama, query_llm: ChatOllama, max_iterations: int = 3) -> dict:
    """Run the research workflow for a given query
    Args:
        query: The research question or topic
        llm: The language model to use
        max_iterations: Maximum number of search iterations
        
    Returns:
        Dictionary containing the final report and all intermediate results"""
    # create the graph
    graph = create_research_graph(llm, query_llm)

    # initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "search_results": [],
        "research_findings": "",
        "iterations": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
        "final_report": "",
    }

    # run the graph 
    result = graph.invoke(initial_state)

    return result

if __name__ == "__main__":
    llm = ChatOllama(
        base_url="http://192.168.0.162:11434",
        model="qwen3:4b",
        temperature=0.0
    )

    query_llm = ChatOllama(
        base_url="http://192.168.0.162:11434",
        model="granite4:3b",
        temperature=0.3
    )

    result = run_research(
        query="real Purposes and significances of NASA Artemis project",
        llm=llm,
        query_llm=query_llm
    )

    print("\n" + "=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print(result["final_report"])
    print("\n" + "=" * 80)

    # save to text file
    output_path = "./data/reports/artemis_project.txt"
    string_to_text_file(result["final_report"], file_path=output_path)
    print(f"Report saved to {output_path}")
