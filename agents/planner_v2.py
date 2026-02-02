"""Breaks down user query into subquestions and keywords pairs"""

from typing import List, Tuple, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import PydanticOutputParser
import os
from config.settings import AppSettings
from config.prompts import PLANNER_PROMPT
import logging
from abc import ABC, abstractmethod

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model config for each provider
class LLMConfig(BaseModel):
    """Configuration for a specific LLM provider"""
    provider: Literal["groq", "openrouter", "zai"]
    model_name: str
    temperature: float = 0.3

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
    
# chain of responsibility pattern implementation
class LLMHandler(ABC):
    """Abstract base handler for LLM providers"""
    def __init__(self, config: LLMConfig, settings: AppSettings):
        self.config = config
        self.settings = settings
        self._next_handler: Optional['LLMHandler'] = None
        self._llm = None

    def set_next(self, handler: 'LLMHandler') -> 'LLMHandler':
        """Set the next handler in the chain"""
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def _initialize_llm(self):
        """Initialize the specific LLM provider"""
        pass

    def handle(self, prompt_template: ChatPromptTemplate, parser: PydanticOutputParser,
               research_question: str) -> Optional[ResearchDecomposition]:
        """Attempt process the request with this handler. 
        If it fails, pass to next handler in the chain"""
        try:
            logger.info(f"Attempting to use {self.config.provider} with model {self.config.model_name}")

            # initialize LLM if not already
            if self._llm is None:
                self._llm = self._initialize_llm()

            # create and execute the chain
            chain = prompt_template | self._llm | parser
            result = chain.invoke({
                "research_question": research_question,
                "format_instructions": parser.get_format_instructions()
            })

            logger.info(f"Successfully processed with {self.config.provider}")
            return result
        except Exception as e:
            logger.warning(f"Failed to process with {self.config.provider}: {str(e)}")

            # pass to next handler
            if self._next_handler:
                logging.info("Falling back to next handler...")
                return self._next_handler.handle(prompt_template, parser, research_question)
            else:
                logger.error("All handlers failed, no more fallbacks available")
                raise Exception(f"All LLM providers failed. Last error: {str(e)}")
            
class GroqHandler(LLMHandler):
    """Handler for Groq LLM provider"""
    def _initialize_llm(self):
        os.environ["GROQ_API_KEY"] = self.settings.groq_api_key.get_secret_value()
        return ChatGroq(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=2048,
            reasoning_format='parsed',
            reasoning_effort='medium',
            max_retries=3
        )
    
class OpenRouterHandler(LLMHandler):
    """Handler for OpenRouter LLM provider"""
    def _initialize_llm(self):
        return ChatOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=self.config.model_name,
            temperature=self.config.temperature,
            reasoning={"effort": "medium"},
            max_retries=3
        )
    
class ZhipuHandler(LLMHandler):
    """Handler for Zhipu AI LLM provider"""
    def _initialize_llm(self):
        # os.environ["ZHIPU_API_KEY"] = self.settings.zhipu_api_key.get_secret_value()
        # return ChatZhipuAI(
        #     model=self.config.model_name,
        #     temperature=self.config.temperature
        # )
        return ChatOpenAI(
            api_key=self.settings.zhipu_api_key,
            base_url="https://api.z.ai/api/paas/v4/",
            model=self.config.model_name,
            temperature=self.config.temperature,
            # reasoning={"effort": "medium"},
            max_retries=3
        )

class QuestionDecomposer:
    """Main decomposer using chain of responsibility pattern"""
    # default config for each provider
    DEFAULT_CONFIGS = [
        LLMConfig(provider="groq", model_name="openai/gpt-oss-20b", temperature=0.3),
        LLMConfig(provider="openrouter", model_name="nvidia/nemotron-3-nano-30b-a3b:free", temperature=0.3),
        LLMConfig(provider="zai", model_name="glm-4.7-flash", temperature=0.3)
    ]
    def __init__(self, configs: Optional[List[LLMConfig]] = None):
        """Initialize the question decomposer with a chain of LLM providers.
        
        Args:
            configs: List of LLMConfig objects. If None, uses DEFAULT_CONFIGS.
                    The first config is primary, others are fallbacks in order."""
        self.settings = AppSettings()
        self.configs = configs or self.DEFAULT_CONFIGS

        # create output parser
        self.parser = PydanticOutputParser(pydantic_object=ResearchDecomposition)

        # create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", PLANNER_PROMPT),
            ("human", "Decompose this research question into exactly 3 sub-questions and keywords:\n\n{research_question}")
        ])

        # build the chain of responsibility
        self.handler_chain = self._build_handler_chain()

    def _build_handler_chain(self) -> LLMHandler:
        """Build the chain of responsibility from configs"""
        if not self.configs:
            raise ValueError("At least one LLM config must be provided")
        
        # create handlers
        handlers = []
        for config in self.configs:
            if config.provider == "groq":
                handlers.append(GroqHandler(config, self.settings))
            elif config.provider == "openrouter":
                handlers.append(OpenRouterHandler(config, self.settings))
            elif config.provider == "zai":
                handlers.append(ZhipuHandler(config, self.settings))
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
        # link handlers together
        for i in range(len(handlers) - 1):
            handlers[i].set_next(handlers[i + 1])

        return handlers[0]
    
    def decompose(self, research_question: str) -> Optional[ResearchDecomposition]:
        """
        Decompose a research question into exactly sub-questions and keywords.
        
        Args:
            research_question: The main research question to decompose
            
        Returns:
            ResearchDecomposition object with pairs of subquestions and keywords
        Raises:
            Exception: if all providers in chain failed
        """
        result = self.handler_chain.handle(
            self.prompt_template,
            self.parser,
            research_question
        )

        return result
    
    def decompose_to_dict(self, research_question: str) -> dict:
        """Decompose and return as structured dictionary."""
        result = self.decompose(research_question)
        if result is None:
            return {}
        
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

    # pass to LLM chains
    result = decomposer.decompose_to_dict(question)

    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
