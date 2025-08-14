from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
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
    port: int = Field(alias="PORT", default=8080)

    persistent_storage_path: str = Field(alias="PERSISTENT_STORAGE_PATH", default="/storage")

    @field_validator('llm_api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v == "super-secret":
            logging.warning("Using default API key. Please set LLM_API_KEY environment variable.")
        return v
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

logger = logging.getLogger(__name__)

# Validate critical settings
if settings.llm_api_key == "super-secret":
    logger.error("LLM_API_KEY not set. Please set this environment variable.")
    logger.error("Example: export LLM_API_KEY='your-api-key-here'")

logger.info(f"Persistent storage path: {settings.persistent_storage_path}")
os.makedirs(settings.persistent_storage_path, exist_ok=True)