"""Fixtures that run tests against a real PostgreSQL with pgvector, in a container."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, make_url, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.community.postgres import PostgresContainer

from paper_similarity.config import Settings
from paper_similarity.db.session import create_db_engine
from paper_similarity.main import create_app
from tests.fakes import FakeEmbedder

PROJECT_ROOT = Path(__file__).parents[2]
# Keep in sync with compose.yaml.
DATABASE_IMAGE = "pgvector/pgvector:0.8.7-pg18-trixie"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in this folder, so `-m "not integration"` skips them all."""
    for item in items:
        if item.path.is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.integration)


def make_engine(url: str) -> Engine:
    return create_db_engine(Settings(_env_file=None, database_url=SecretStr(url)))


def run_alembic(engine: Engine, action: str, revision: str) -> None:
    config = Config(PROJECT_ROOT / "alembic.ini")
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        getattr(command, action)(config, revision)


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with PostgresContainer(DATABASE_IMAGE, driver="psycopg") as container:
        yield container.get_connection_url()


@pytest.fixture(scope="session")
def engine(database_url: str) -> Iterator[Engine]:
    """A database with all migrations applied, shared by all tests."""
    engine = make_engine(database_url)
    run_alembic(engine, "upgrade", "head")
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session whose changes are rolled back after the test, even if it commits."""
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()


@pytest.fixture
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    """Like `session`, for code that opens and commits its own sessions."""
    with engine.connect() as connection:
        transaction = connection.begin()
        yield sessionmaker(
            connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        transaction.rollback()


@pytest.fixture
def committed_database(engine: Engine) -> Iterator[Engine]:
    """For tests that really commit. The tables are emptied afterwards."""
    yield engine
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE arxiv_papers, submitted_papers"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def api(
    database_url: str, committed_database: Engine, embedder: FakeEmbedder
) -> Iterator[TestClient]:
    """The complete application with a real database and a fake embedder."""
    settings = Settings(_env_file=None, database_url=SecretStr(database_url))
    with TestClient(create_app(settings, embedder)) as client:
        yield client


@pytest.fixture
def empty_database_engine(database_url: str) -> Iterator[Engine]:
    """A separate database without tables, for testing the migrations themselves."""
    admin = make_engine(database_url).execution_options(isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text("CREATE DATABASE migration_test"))
    url = make_url(database_url).set(database="migration_test")
    engine = make_engine(url.render_as_string(hide_password=False))
    yield engine
    engine.dispose()
    with admin.connect() as connection:
        connection.execute(text("DROP DATABASE migration_test"))
    admin.dispose()
