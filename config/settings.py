# settings with environment variables
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class AppSettings(BaseSettings):
    """Basic application settings loaded from environment variables."""

    groq_api_key: SecretStr
    openrouter_api_key: SecretStr
    gemini_api_key: SecretStr
    zhipu_api_key: SecretStr
    tavily_api_key: SecretStr

    # ollama
    ollama_port: int = 11434

    # other locally deployed search engine
    searxng_host: str = "localhost"
    searxng_port: int = 8888

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

if __name__ == "__main__":
    settings = AppSettings()
    print(f"searxng host: {settings.searxng_host}")
    print(f"searxng port: {settings.searxng_port}")
    print(f"google json api key: {settings.google_json_api_key}")
    api_key = settings.google_json_api_key.get_secret_value()
    print(api_key)
