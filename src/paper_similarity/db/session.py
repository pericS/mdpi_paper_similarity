"""Creation of the database engine and the session factory."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from paper_similarity.config import Settings

CONNECT_TIMEOUT_SECONDS = 5
# How long to wait for a free connection of the pool.
POOL_TIMEOUT_SECONDS = 5


def create_db_engine(settings: Settings, *, statement_timeout_seconds: int = 0) -> Engine:
    """Create the connection pool. No connection is opened until the first query.

    A statement timeout of 0 means no limit, for migrations and ingestion.
    """
    return create_engine(
        settings.database_url.get_secret_value(),
        # Replaces connections that died while idle, e.g. after a database restart.
        pool_pre_ping=True,
        pool_timeout=POOL_TIMEOUT_SECONDS,
        connect_args={
            "connect_timeout": CONNECT_TIMEOUT_SECONDS,
            "options": f"-c statement_timeout={statement_timeout_seconds}s",
        },
        # Keeps query parameters, such as manuscript text, out of error messages.
        hide_parameters=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a factory for short-lived sessions, one per request or ingestion batch."""
    # Without expire_on_commit=False, reading an object after commit runs another query.
    return sessionmaker(engine, expire_on_commit=False)
