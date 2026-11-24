"""Application factory."""

from importlib.metadata import version

from fastapi import FastAPI

from paper_similarity.api import health
from paper_similarity.config import Settings, get_settings
from paper_similarity.logging_config import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Tests can pass their own settings."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="Paper Similarity Service",
        description="Finds indexed arXiv papers similar to a submitted manuscript.",
        version=version("paper-similarity"),
    )
    app.include_router(health.router)
    return app
