"""Application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI

from paper_similarity.api import health, papers
from paper_similarity.config import Settings, get_settings
from paper_similarity.db.session import create_db_engine, create_session_factory
from paper_similarity.embeddings.base import Embedder
from paper_similarity.embeddings.fastembed_embedder import FastEmbedEmbedder
from paper_similarity.errors import register_error_handlers
from paper_similarity.logging_config import configure_logging


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    """Build the application. Tests can pass their own settings and embedder."""
    app_settings = settings or get_settings()
    configure_logging(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs before the first request. A missing model stops the startup; the
        # database is not contacted yet, so a short outage does not.
        engine = create_db_engine(app_settings)
        app.state.session_factory = create_session_factory(engine)
        app.state.embedder = embedder or FastEmbedEmbedder(
            app_settings.embedding_cache_dir, max_text_tokens=app_settings.max_text_tokens
        )
        yield
        engine.dispose()

    app = FastAPI(
        title="Paper Similarity Service",
        description="Finds indexed arXiv papers similar to a submitted manuscript.",
        version=version("paper-similarity"),
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(papers.router)
    return app
