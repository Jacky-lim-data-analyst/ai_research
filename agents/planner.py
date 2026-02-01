# Logic for breaking down topics (groq:gpt-oss:20b)
"""
Research Planner Agent
Breaks down user topics into 3 focused research subtopics with structured output
"""

from typing import List, Tuple
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
import os
from config.settings import AppSettings
import yaml
from pathlib import Path

class ResearchSubtopic(BaseModel):
    """Individual research subtopic with clear scope and expectations."""
    subtopic_id: int = Field(
        description="Unique identifier (1-3)",
        ge=1,
        le=3
    )
    research_question: str = Field(
        description="Specific, answerable research question to investigate"
    )
    expected_outcomes: List[str] = Field(
        description="2-4 concrete deliverables or insights expected from this research",
        min_length=2,
        max_length=4,
    )
    search_keywords: List[str] = Field(
        description="3-5 relevant keywords for web search",
        min_length=3,
        max_length=5
    )

class ResearchPlan(BaseModel):
    """Complete research plan with multiple subtopics"""
    research_objective: str = Field(
        description="Overall goal and purpose of the research"
    )
    subtopics: List[ResearchSubtopic] = Field(
        description="Exactly 3 research subtopics",
        min_length=3,
        max_length=3
    )
    synthesis_strategy: str = Field(
        description="How the 3 subtopics will be synthesized into a cohesive final report"
    )

class PlannerAgent:
    """Agent responsible for breaking down research topics into subtopics."""
    
    def __init__(self, config_path: str = "config/agents.yaml"):
        """
        Initailize the planner agent.

        Args:
            config_path: Path to YAML file containing system prompts
        """
        self.config_path = Path(config_path)
        self.system_prompt = self._load_system_prompt()

        self.llm = self._initialize_llm()

    def _load_system_prompt(self) -> str:
        """Load system prompt from YAML config"""
        with open(self.config_path, 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['planner']['system']
    
    def _initialize_llm(self) -> ChatGroq:
        """
        Initialize the LLM with optimal configuration for planning tasks

        Model configuration rationale:
        - model: gpt-oss - Best for complex reasoning
        - temperature: 0.3 - Low for consistency, slight creativity for diverse topics
        - max_tokens: 5000 - Sufficient for detailed plan with 3 subtopics with reasoning
        - include reasoning
        """
        settings = AppSettings()
        os.environ["GROQ_API_KEY"] = settings.groq_api_key.get_secret_value()
        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=4096,
            reasoning_format='parsed',
            reasoning_effort='medium'
            max_retries=3
        )
    
    def create_research_plan(self, user_topic: str) -> ResearchPlan:
        """Breaks down a user topic into a structured research plan.
        
        Args: 
            user_topic (str): Research topic provided by user
        
        Returns:
            ResearchPlan object with 3 subtopics and structured metadata"""
        # create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Research Topic: {topic}
             Create a comprehensive research plan by breaking this topic into exactly 3 distinct, complementary subtopics. 
             Each subtopic should be investigable through web research and together they provide complete coverage of the main topic.""")
        ])

        # use structured output with Pydantic model
        structured_llm = self.llm.with_structured_output(ResearchPlan)
        chain = prompt | structured_llm

        # generate the research plan
        return chain.invoke({"topic": user_topic})
    
    def display_plan(self, plan: ResearchPlan) -> None:
        """Print the research plan for debugging"""
        print("\n" + "-"*60)
        print(f'RESEARCH objective: {plan.research_objective}')
        print("-"*60)

        for subtopic in plan.subtopics:
            print(f"\n{'-'*60}")
            print(f"Research Question: {subtopic.research_question}")
            print(f"{'-'*60}")
            print(f"\nExpected Outcomes:")
            for outcome in subtopic.expected_outcomes:
                print(f"  • {outcome}")
            print(f"\nSearch Keywords: {', '.join(subtopic.search_keywords)}")

if __name__ == "__main__":
    settings = AppSettings()
    os.environ["GROQ_API_KEY"] = settings.groq_api_key.get_secret_value()

    model = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=1024,
        max_retries=3
    )

    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of Canada?")
    ]

    response = model.invoke(messages)
    print(response.content)
