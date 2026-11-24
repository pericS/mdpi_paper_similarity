"""Locks that work across processes."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, func, select

# Any fixed number works, but it must not be used for another purpose.
INGESTION_LOCK_KEY = 7_203_451


class LockNotAvailableError(Exception):
    """Another process holds the lock."""


@contextmanager
def advisory_lock(engine: Engine, key: int) -> Iterator[None]:
    """Hold a PostgreSQL advisory lock, or fail immediately if it is already taken.

    The lock belongs to a separate connection. If the process dies, PostgreSQL closes
    that connection and releases the lock, so it cannot be left behind.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        # Closing a pooled connection keeps it open for reuse, and the lock with it.
        # A detached connection is really closed.
        connection.detach()
        if not connection.scalar(select(func.pg_try_advisory_lock(key))):
            raise LockNotAvailableError
        yield
