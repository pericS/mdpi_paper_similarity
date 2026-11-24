"""Logging setup, shared by the API and the command line."""

import logging
import traceback
from contextvars import ContextVar

from sqlalchemy.exc import SQLAlchemyError

from paper_similarity.config import Settings
from paper_similarity.db.errors import describe_safely

HANDLER_NAME = "paper-similarity"
NO_REQUEST = "-"

# A context variable has a separate value for each request, also in worker threads.
request_id: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST)


class _RequestIdFilter(logging.Filter):
    """Adds the current request ID to every log line."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.value)
    for existing in [handler for handler in root.handlers if handler.name == HANDLER_NAME]:
        root.removeHandler(existing)  # otherwise a second call would log everything twice
    handler = logging.StreamHandler()
    handler.name = HANDLER_NAME
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)

    # Uvicorn's messages get the same format.
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    # RequestContextMiddleware logs requests, with the request ID and the duration.
    logging.getLogger("uvicorn.access").disabled = True
    # It logs every request to arXiv with its full URL at INFO.
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def describe_exception(error: BaseException) -> str:
    """Describe an exception and where it happened, without its message.

    Messages can contain submitted text. The stack contains only our source code.
    """
    summary = type(error).__name__
    if isinstance(error, SQLAlchemyError):
        summary = describe_safely(error)
    if error.__cause__ is not None:
        summary += f" caused by {type(error.__cause__).__name__}"
    stack = "".join(traceback.format_tb(error.__traceback__))
    return f"{summary}\n{stack}".rstrip()
