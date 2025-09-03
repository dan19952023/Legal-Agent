from pydantic_settings import BaseSettings
from pydantic import Field
import os
import logging

class Settings(BaseSettings):
    llm_api_key: str = Field(alias="LLM_API_KEY", default="super-secret")
    llm_base_url: str = Field(alias="LLM_BASE_URL", default="https://api.openai.com/v1")
    llm_model_id: str = Field(alias="LLM_MODEL_ID", default="gpt-4o-mini")

    # app state
    app_env: str = Field(alias="APP_ENV", default="development")

    # Server
    host: str = Field(alias="HOST", default="0.0.0.0")
    port: int = Field(alias="PORT", default=80)

    persistent_storage_path: str = Field(alias="PERSISTENT_STORAGE_PATH", default="/storage")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

logger = logging.getLogger(__name__)

logger.info(f"Persistent storage path: {settings.persistent_storage_path}")
os.makedirs(settings.persistent_storage_path, exist_ok=True)