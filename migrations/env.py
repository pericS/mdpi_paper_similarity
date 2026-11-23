"""Alembic entry point: connects to the database and runs migrations."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

from paper_similarity.config import get_settings
from paper_similarity.db.models import Base
from paper_similarity.db.session import create_db_engine


def run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


def main() -> None:
    if context.is_offline_mode():
        raise RuntimeError("Offline (--sql) migrations are not supported")

    # Tests pass a connection to their own database.
    connection: Connection | None = context.config.attributes.get("connection")
    if connection is not None:
        run_migrations(connection)
        return

    if context.config.config_file_name is not None:
        fileConfig(context.config.config_file_name, disable_existing_loggers=False)
    engine = create_db_engine(get_settings())
    with engine.connect() as new_connection:
        run_migrations(new_connection)
    engine.dispose()


main()
