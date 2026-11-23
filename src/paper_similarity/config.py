"""Application settings, read from environment variables or a local .env file."""

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """All configurable values. Invalid values stop the application at startup."""

    model_config = SettingsConfigDict(
        env_prefix="PAPER_SIMILARITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: LogLevel = LogLevel.INFO


@lru_cache
def get_settings() -> Settings:
    """Return the settings, reading the environment only once per process."""
    return Settings()
