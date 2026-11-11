# SecretStr prevents accidental exposure of sensitive configuration values.
# Import Path to configure the knowledge document storage directory.
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SupportIQ"
    app_env: str = "development"
    debug: bool = True

    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./supportiq.db"

    # Directory used to store uploaded knowledge documents.
    knowledge_upload_dir: Path = Path("data/knowledge_documents")

    # Maximum permitted document upload size: 10 MB.
    knowledge_max_upload_size: int = 10 * 1024 * 1024

    # OpenAI LLM configuration.
    # The API key is loaded securely from the .env file.
    openai_api_key: SecretStr | None = None

    # Model used for generating customer-support responses.
    llm_model: str = "gpt-4o-mini"

    # Lower temperature produces more consistent answers.
    llm_temperature: float = 0.2

    # Maximum number of tokens in a generated response.
    llm_max_tokens: int = 800

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()