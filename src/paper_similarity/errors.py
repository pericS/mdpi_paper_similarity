"""Translates application errors into HTTP responses."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from paper_similarity.embeddings.base import EmbeddingError, UnembeddableTextError
from paper_similarity.services.similarity import PaperNotFoundError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    # Handlers are chosen by the most specific error class.
    app.add_exception_handler(RequestValidationError, _handle_invalid_request)
    app.add_exception_handler(PaperNotFoundError, _handle_paper_not_found)
    app.add_exception_handler(UnembeddableTextError, _handle_unembeddable_text)
    app.add_exception_handler(EmbeddingError, _handle_embedding_error)
    for error_class in (OperationalError, InterfaceError, PoolTimeoutError):
        app.add_exception_handler(error_class, _handle_database_unavailable)


def _handle_invalid_request(request: Request, error: Exception) -> JSONResponse:
    """Like FastAPI's own handler, but without repeating the input.

    The input can be a whole manuscript, or contain characters that cannot be sent back.
    """
    if not isinstance(error, RequestValidationError):
        raise error
    details = [
        {
            "type": problem["type"],
            "loc": [_printable(part) for part in problem["loc"]],
            "msg": problem["msg"],
        }
        for problem in error.errors()
    ]
    return JSONResponse({"detail": details}, status_code=422)


def _printable(part: str | int) -> str | int:
    """Replace characters that cannot be encoded. Field names come from the client."""
    if isinstance(part, int):
        return part
    return part.encode("utf-8", errors="replace").decode()


def _handle_paper_not_found(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": "There is no submitted paper with this ID"}, status_code=404)


def _handle_unembeddable_text(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=422)


def _handle_embedding_error(request: Request, error: Exception) -> JSONResponse:
    logger.error("Embedding failed", exc_info=error)
    return JSONResponse({"detail": "The embedding model is unavailable"}, status_code=503)


def _handle_database_unavailable(request: Request, error: Exception) -> JSONResponse:
    # Only the type: the full message can contain stored data.
    logger.error("Database unavailable: %s", type(error).__name__)
    return JSONResponse({"detail": "The database is unavailable"}, status_code=503)
