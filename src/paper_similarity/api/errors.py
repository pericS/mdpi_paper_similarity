"""One error format for every response, and the translation of application errors."""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.exceptions import HTTPException

from paper_similarity.embeddings.base import (
    EmbeddingError,
    TextTooLongError,
    UnembeddableTextError,
)
from paper_similarity.logging_config import describe_exception, request_id
from paper_similarity.services.similarity import PaperNotFoundError

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    field: str = Field(examples=["body.text"])
    message: str = Field(examples=["Field required"])


class ErrorBody(BaseModel):
    code: str = Field(description="Stable identifier for programs.", examples=["not_found"])
    message: str = Field(description="Explanation for people.")
    details: list[ErrorDetail] = Field(description="Invalid fields. Empty for other errors.")
    request_id: str = Field(description="Find the matching log lines with this ID.")


class ErrorResponse(BaseModel):
    error: ErrorBody


def documented_errors(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """Describe error responses for the generated API documentation."""
    return {
        status: {"model": ErrorResponse, "description": HTTPStatus(status).phrase}
        for status in statuses
    }


def error_response(
    status: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=code, message=message, details=details or [], request_id=request_id.get()
        )
    )
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def internal_error_response(error: Exception) -> JSONResponse:
    logger.error("Unexpected error: %s", describe_exception(error))
    return error_response(500, "internal_error", "An unexpected error occurred")


def register_error_handlers(app: FastAPI) -> None:
    # The handler of the most specific error class is used.
    app.add_exception_handler(RequestValidationError, _handle_invalid_request)
    app.add_exception_handler(HTTPException, _handle_http_exception)
    app.add_exception_handler(PaperNotFoundError, _handle_paper_not_found)
    app.add_exception_handler(TextTooLongError, _handle_text_too_long)
    app.add_exception_handler(UnembeddableTextError, _handle_unembeddable_text)
    app.add_exception_handler(EmbeddingError, _handle_embedding_error)
    for error_class in (OperationalError, InterfaceError, PoolTimeoutError):
        app.add_exception_handler(error_class, _handle_database_unavailable)
    # Everything else: see RequestContextMiddleware.


def _handle_invalid_request(request: Request, error: Exception) -> JSONResponse:
    """List what is wrong, without repeating the input.

    The input can be a whole manuscript, or contain characters that cannot be sent back.
    """
    if not isinstance(error, RequestValidationError):
        raise error
    details = [
        ErrorDetail(field=_field_name(problem["loc"]), message=problem["msg"])
        for problem in error.errors()
    ]
    return error_response(422, "invalid_request", "The request is invalid", details)


def _field_name(location: tuple[str | int, ...]) -> str:
    """Join ("body", "text") to "body.text". Names may come from the client."""
    name = ".".join(str(part) for part in location)
    return name.encode("utf-8", errors="replace").decode()


def _handle_http_exception(request: Request, error: Exception) -> JSONResponse:
    """For example an unknown path (404) or a wrong method (405)."""
    if not isinstance(error, HTTPException):
        raise error
    status = HTTPStatus(error.status_code)
    return error_response(
        status,
        code=status.name.lower(),
        message=str(error.detail),
        headers=dict(error.headers) if error.headers else None,
    )


def _handle_paper_not_found(request: Request, error: Exception) -> JSONResponse:
    return error_response(404, "paper_not_found", "There is no submitted paper with this ID")


def _handle_text_too_long(request: Request, error: Exception) -> JSONResponse:
    return error_response(422, "text_too_long", str(error))


def _handle_unembeddable_text(request: Request, error: Exception) -> JSONResponse:
    return error_response(422, "unusable_text", str(error))


def _handle_embedding_error(request: Request, error: Exception) -> JSONResponse:
    logger.error("Embedding failed: %s", describe_exception(error))
    return error_response(503, "embedding_unavailable", "The embedding model is unavailable")


def _handle_database_unavailable(request: Request, error: Exception) -> JSONResponse:
    # Without the stack: during an outage this is logged for every request.
    logger.error("Database unavailable: %s", type(error).__name__)
    return error_response(503, "database_unavailable", "The database is unavailable")
