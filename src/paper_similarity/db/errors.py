"""Describing database errors without revealing stored data."""

from sqlalchemy.exc import DBAPIError, SQLAlchemyError


def sqlstate(error: SQLAlchemyError) -> str | None:
    """Return PostgreSQL's five-character error code, if there is one."""
    if not isinstance(error, DBAPIError):
        return None
    code: str | None = getattr(error.orig, "sqlstate", None)
    return code


def describe_safely(error: SQLAlchemyError) -> str:
    """Describe an error by type, code and constraint only.

    The message itself can contain a whole row: PostgreSQL adds "Failing row contains ...".
    """
    description = type(error).__name__
    if code := sqlstate(error):
        description += f" (SQLSTATE {code})"
    diagnostics = getattr(getattr(error, "orig", None), "diag", None)
    if constraint := getattr(diagnostics, "constraint_name", None):
        description += f" (constraint {constraint})"
    return description
