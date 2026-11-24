"""Command line interface."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from paper_similarity.arxiv.client import ArxivClient, create_http_client
from paper_similarity.arxiv.errors import ArxivError
from paper_similarity.config import Settings, get_settings
from paper_similarity.db.locks import INGESTION_LOCK_KEY, LockNotAvailableError, advisory_lock
from paper_similarity.db.session import create_db_engine, create_session_factory
from paper_similarity.embeddings.base import EmbeddingError
from paper_similarity.embeddings.fastembed_embedder import FastEmbedEmbedder, download_model
from paper_similarity.logging_config import configure_logging
from paper_similarity.services.ingestion import IngestionReport, IngestionService

logger = logging.getLogger(__name__)

# arXiv asks to refine larger queries; see its API manual.
MAX_RESULTS_LIMIT = 2000

app = typer.Typer(
    help="Scientific paper similarity service.",
    no_args_is_help=True,
    # The decorated traceback can print local variables, such as the database URL.
    pretty_exceptions_enable=False,
)


@app.command()
def ingest(
    query: Annotated[
        str,
        typer.Option(help='arXiv search query, e.g. "cat:cs.CL AND abs:retrieval".'),
    ],
    max_results: Annotated[
        int,
        typer.Option(min=1, max=MAX_RESULTS_LIMIT, help="Number of papers to read, newest first."),
    ] = 100,
) -> None:
    """Add arXiv papers to the index. Running it again only processes changes."""
    if not query.strip():
        raise typer.BadParameter("must not be blank", param_hint="--query")
    settings = _load_settings()
    engine = create_db_engine(settings)
    total = IngestionReport()
    try:
        with (
            _errors_as_messages(),
            advisory_lock(engine, INGESTION_LOCK_KEY),
            create_http_client() as http_client,
        ):
            service = IngestionService(
                source=ArxivClient(http_client),
                embedder=FastEmbedEmbedder(
                    settings.embedding_cache_dir, max_text_tokens=settings.max_text_tokens
                ),
                session_factory=create_session_factory(engine),
            )
            for report in service.ingest(query, max_results):
                total += report
                typer.echo(f"Saved a page. So far: {_describe(total)}")
    finally:
        engine.dispose()
        # Also after a failure: pages saved before it stay saved.
        typer.echo(f"Result: {_describe(total)}")

    if total.usable == 0:
        _fail("No usable papers were found. Check the query: arXiv ignores unknown field names.")


@app.command(name="download-model")
def download_model_command() -> None:
    """Download the embedding model, so that later commands work offline."""
    settings = _load_settings()
    try:
        path = download_model(settings.embedding_cache_dir.expanduser())
    except Exception as error:
        logger.debug("Download failed", exc_info=True)
        _fail(f"Could not download the embedding model: {error}")
    typer.echo(f"The model is available in {path}")


def _load_settings() -> Settings:
    try:
        settings = get_settings()
    except ValidationError as error:
        _fail(f"Invalid configuration: {error}")
    configure_logging(settings)
    return settings


@contextmanager
def _errors_as_messages() -> Iterator[None]:
    """Turn expected problems into a short message and exit code 1."""
    try:
        yield
    except LockNotAvailableError:
        _fail("Another ingestion is already running.")
    except ArxivError as error:
        _fail(f"arXiv: {error}")
    except EmbeddingError as error:
        logger.debug("Embedding failed", exc_info=True)
        _fail(f"Embedding: {error}")
    except SQLAlchemyError as error:
        _fail(_describe_database_error(error))


def _describe_database_error(error: SQLAlchemyError) -> str:
    """Describe the problem without the full message, which may contain stored data."""
    kind = type(error).__name__
    if not isinstance(error, DBAPIError):
        return f"Database: {kind}"
    sqlstate = getattr(error.orig, "sqlstate", None)
    if sqlstate == "42P01":
        return "Database: tables are missing. Run: alembic upgrade head"
    if sqlstate is None:
        return f"Database: {kind}. Is it running, and is the database URL correct?"
    return f"Database: {kind} (SQLSTATE {sqlstate})"


def _describe(report: IngestionReport) -> str:
    return (
        f"{report.created} created, {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.skipped} skipped"
    )


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)
