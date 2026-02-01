"""Breaks down user query into subquestions and keywords pairs"""

from typing import List, Tuple
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
import os
from config.settings import AppSettings
from config.prompts import PLANNER_PROMPT
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# custom exceptions class

# model for individual sub-question/keyword pairs
class SubQuestionPair(BaseModel):
    """A single subquestion and its corresponding search keyword"""
    subquestion: str = Field(
        description="A specific sub-question that helps answer the main research question",
        min_length=10,
        max_length=400
    )
    keyword: str = Field(
        description="Web search-optimized keyword or phrase for the sub-question, maximum of 8 words",
        min_length=2,
        max_length=80
    )

# main model with exactly 3 pairs
class ResearchDecomposition(BaseModel):
    """Decomposition of a research question into exactly 3 sub-questions and keywords"""
    subquestion_pairs: List[SubQuestionPair] = Field(
        description="Exactly 3 pairs of sub-questions and keywords",
        min_length=3,
        max_length=3
    )

    @field_validator('subquestion_pairs')
    def validate_exactly_three_pairs(cls, v):
        """Ensure exactly 3 pairs are provided"""
        if len(v) != 3:
            raise ValueError(f'Must provide exactly 3 pairs, got {len(v)} instead')
        return v
        
    def get_pairs_as_tuples(self) -> List[Tuple[str, str]]:
        """Return pairs as list of tuples"""
        return [(pair.subquestion, pair.keyword) for pair in self.subquestion_pairs]
    
class QuestionDecomposer:
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.3):
        """Initialize a user query decomposer that generates fixed pairs
        of subquestions and search keywords"""
        self.llm = self._initialize_llm(
            model_name=model_name,
            temperature=temperature
        )

        # create output parser
        self.parser = PydanticOutputParser(pydantic_object=ResearchDecomposition)

        # create prompt with explicit 3-pair requirement
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", PLANNER_PROMPT),
            ("human", "Decompose this research question into exactly 3 sub-questions and keywords:\n\n{research_question}")
        ])

        # create the chain
        self.chain = self.prompt_template | self.llm | self.parser

    def _initialize_llm(self, model_name: str, temperature: float):
        """
        Initialize the LLM with optimal configuration for planning tasks

        Model configuration rationale:
        - model: gpt-oss - Best for complex reasoning
        - temperature: 0.3 - Low for consistency, slight creativity for diverse topics
        - max_tokens: 2048 - Sufficient for detailed plan with 3 subtopics with reasoning
        - include reasoning
        """
        settings = AppSettings()
        os.environ["GROQ_API_KEY"] = settings.groq_api_key.get_secret_value()
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=2048,
            reasoning_format='parsed',
            reasoning_effort='medium',
            max_retries=3
        )
    
    def decompose(self, research_question: str) -> ResearchDecomposition:
        """
        Decompose a research question into exactly sub-questions and keywords.
        
        Args:
            research_question: The main research question to decompose
            
        Returns:
            ResearchDecomposition object with pairs of subquestions and keywords
        """
        result = self.chain.invoke({
            "research_question": research_question,
            "format_instructions": self.parser.get_format_instructions()
        })

        return result
    
    def decompose_to_dict(self, research_question: str) -> dict:
        """Decompose and return as structured dictionary."""
        result = self.decompose(research_question)
        pairs = result.get_pairs_as_tuples()

        return {
            "research_question": research_question,
            "pairs": pairs,
            "pair_count": len(pairs)
        }
    
def main():
    # initialize decomposer
    decomposer = QuestionDecomposer()
    # sample question
    question = "What are the effects of social media on mental health?"

    # pass to LLM
    result = decomposer.decompose_to_dict(question)

    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
