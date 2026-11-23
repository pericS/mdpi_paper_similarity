"""Application settings, read from environment variables or a local .env file."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DATABASE_URL_SCHEME = "postgresql+psycopg://"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """All configurable values. Missing or invalid values stop the application at startup."""

    model_config = SettingsConfigDict(
        env_prefix="PAPER_SIMILARITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Validation errors would otherwise print the database password.
        hide_input_in_errors=True,
    )

    database_url: SecretStr
    log_level: LogLevel = LogLevel.INFO
    embedding_cache_dir: Path = Path("~/.cache/paper-similarity")

    @field_validator("database_url")
    @classmethod
    def require_psycopg_driver(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith(DATABASE_URL_SCHEME):
            raise ValueError(f"must start with {DATABASE_URL_SCHEME}")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the settings, reading the environment only once per process."""
    return Settings()
