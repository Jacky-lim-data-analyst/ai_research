"""
Translation Agent
Translates technical research reports from English to Chinese with quality assurance.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config.prompts import TRANSLATOR_AGENT_PROMPT

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

class TranslatorAgent:
    """Agent responsible for translating technical reports from English to Chinese"""
    def __init__(self, 
                 base_url: str = "http://192.168.0.162:11434",
                 model_name: str = "translategemma:4b",
                 temperature: float = 0.1,
                 top_p: float = 0.95):
        """Initialize the translator llm
        Args:
        - model_name: translategemma - excellent multilingual capabilities
        - temperature: 0.1 - Very low for consistency and accuracy
        - top_p: 0.95 - High nucleus sampling for natural Chinese expression
        """
        # self.config_path = Path(config_path)
        self.system_prompt = TRANSLATOR_AGENT_PROMPT
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p
        )

    def translate_report(
        self, 
        english_report: str,
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
            ("human", """
             English Report Content:
             {content}
             
             Translate this technical report to Chinese.""")
        ])

        # structured output
        structured_llm = self.llm.with_structured_output(TranslatedReport)
        chain = prompt | structured_llm

        return chain.invoke({
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

def main():
    """Main entry point for execution"""
    print("Initializing Translator Agent...")
    agent = TranslatorAgent()
    
    sample_report = """
    Title: Impact of Large Language Models on Software Engineering
    Abstract: This report explores how generative AI is transforming the 
    Software Development Life Cycle (SDLC). We analyze productivity gains 
    in coding, testing, and documentation phases. Our findings suggest 
    a 40% increase in initial draft speed for boilerplate code.
    """
    
    try:
        print("Starting translation (this may take a moment)...")
        result = agent.translate_report(sample_report)
        agent.display_example(result)
    except Exception as e:
        print(f"An error occurred during translation: {e}")

if __name__ == "__main__":
    main()
