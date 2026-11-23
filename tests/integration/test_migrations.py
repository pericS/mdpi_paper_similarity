from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect

from paper_similarity.db.models import Base
from tests.integration.conftest import run_alembic

PAPER_TABLES = {"arxiv_papers", "submitted_papers"}


def table_names(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_migrations_can_be_applied_reverted_and_applied_again(
    empty_database_engine: Engine,
) -> None:
    run_alembic(empty_database_engine, "upgrade", "head")
    assert table_names(empty_database_engine) >= PAPER_TABLES

    run_alembic(empty_database_engine, "downgrade", "base")
    assert not table_names(empty_database_engine) & PAPER_TABLES

    run_alembic(empty_database_engine, "upgrade", "head")
    assert table_names(empty_database_engine) >= PAPER_TABLES


def test_models_match_the_migrated_database(engine: Engine) -> None:
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert differences == []
