# translategemma
"""
Translation Agent
Translates technical research reports from English to Chinese with quality assurance.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# from langchain.chat_models import BaseChatModel
import yaml
from pathlib import Path

from config.settings import AppSettings

class TranslatedReport(BaseModel):
    """Completed translated technical report with quality metadata"""
    title: str = Field(
        description="Chinese tranlation of the title"
    )
    body: str = Field(
        description="Chinese translation of the report in English"
    )
    summary: str = Field(
        description="One-sentence report summarization in Chinese"
    )
    quality_score: int = Field(
        description="Self-assesed translation quality score (1-10)",
        ge=1,
        le=10
    )
    translator_notes: str = Field(
        description="Overall notes about translation challenges and decisions"
    )

class TranslatorAgent:
    """Agent responsible for translating technical reports from English to Chinese"""
    def __init__(self, config_path: str = "config/prompts.yaml"):
        """Initialize the translator llm
        Args:
            config_path: Path to YAML file containing system prompts
        """
        self.config_path = Path(config_path)
        self.system_prompt = self._load_system_prompt()
        self.llm = self._initialize_llm()

    def _load_system_prompt(self) -> str:
        """Load system prompt from YAML configuration"""
        with open(self.config_path, 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['translator']['system']
    
    def _initialize_llm(self) -> ChatOllama:
        """Initialize the LLM with optimal configuration for translation tasks.
        
        Model configuration rationale:
        - model: translategemma - excellent multilingual capabilities
        - temperature: 0.1 - Very low for consistency and accuracy
        - top_p: 0.95 - High nucleus sampling for natural Chinese expression
        """
        return ChatOllama(
            model="translategemma:4b",
            temperature=0.1,
            top_p=0.95
        )

    def translate_report(
        self, 
        english_report: str,
        report_title: str,
        domain: str = "technical"
    ) -> TranslatedReport:
        """
        Translate an English technical report to Chinese
        Args:
            english_report: The full English report text
            report_title: Title of report
            domain: Domain/field of the report (technical/scientific/business)
        Returns:
            TranslatedReport object with structured translation and metadata
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """English Report Title: {title}
             Domain: {domain}
             
             English Report Content:
             {content}
             
             Translate this technical report to Chinese.""")
        ])

        # structured output
        structured_llm = self.llm.with_structured_output(TranslatedReport)
        chain = prompt | structured_llm

        translated_report = chain.invoke({
            "title": report_title,
            "domain": domain,
            "content": english_report
        })

    def display_example(self, translation: TranslatedReport) -> None:
        """Print the translation for debugging/logging"""
        print("\n" + "="*80)
        print(f"TRANSLATION REPORT")
        print("="*80)
        print(f"中文标题: {translation.title}")
        print(f"Quality Score: {translation.quality_score}/10")

        print(f"\n{'─'*80}")
        print(f"概括:{translation.summary}")
        print(f"{'─'*80}")

        print(f"\n{'─'*80}")
        print("TRANSLATED TEXT")
        print(f"{'─'*80}")
        print(translation.body)

        print(f"\n{'='*80}")
        print("TRANSLATOR NOTES")
        print(f"{'='*80}")
        print(translation.translator_notes)
        print("="*80 + "\n")


if __name__ == "__main__":
    settings = AppSettings()
    port = settings.ollama_port

    model = ChatOllama(
        model="qwen3:4b",
        validate_model_on_init=True,
        base_url=f"http://192.168.0.155:{port}",
        temperature=0.5
    )

    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of Canada?")
    ]
    response = model.invoke(messages)
    # print(response)
    print(response.content)