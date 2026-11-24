"""Code that runs around every request."""

import logging
import re
import time
from urllib.parse import quote
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from paper_similarity.api.errors import error_response, internal_error_response
from paper_similarity.logging_config import request_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# Other values could be used to write misleading lines into the log.
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

# The longest valid text needs at most 1.2 MB: every character written as two \uXXXX.
MAX_BODY_BYTES = 2 * 1024 * 1024


class RequestContextMiddleware:
    """Gives each request an ID, logs it, and handles unexpected errors.

    Unexpected errors are handled here because Starlette's own handler passes them on
    to Uvicorn, which logs the complete message, including stored data.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        current_id = _incoming_request_id(scope) or uuid4().hex
        token = request_id.set(current_id)
        started = time.perf_counter()
        status = 0  # 0 means that no response was started

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (REQUEST_ID_HEADER.lower().encode(), current_id.encode()),
                ]
            await send(message)

        try:
            await self._app(scope, receive, send_with_id)
        except Exception as error:
            response = internal_error_response(error)
            if status == 0:
                await response(scope, receive, send_with_id)
        finally:
            milliseconds = (time.perf_counter() - started) * 1000
            # Health checks are frequent and would hide other requests.
            level = logging.DEBUG if scope["path"].startswith("/health/") else logging.INFO
            # quote() makes line breaks in the path harmless.
            logger.log(
                level,
                "%s %s %d %.0f ms",
                scope["method"],
                quote(scope["path"]),
                status,
                milliseconds,
            )
            request_id.reset(token)


def _incoming_request_id(scope: Scope) -> str | None:
    for name, value in scope["headers"]:
        if name == REQUEST_ID_HEADER.lower().encode():
            text = value.decode("latin-1")
            return text if VALID_REQUEST_ID.fullmatch(text) else None
    return None


class BodySizeLimitMiddleware:
    """Rejects large bodies before they are parsed.

    The Content-Length header is only a first check, because it can be missing or wrong.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        if _declared_length(scope) > self._max_bytes:
            await self._reject(scope, receive, send)
            return

        # One buffer, because keeping every message costs far more than its bytes:
        # a client can send the body in a million pieces of one byte.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break  # the client disconnected; the application is told below
            body += message.get("body", b"")
            if len(body) > self._max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                message = {"type": "http.request", "body": bytes(body), "more_body": False}
                break

        first: Message | None = message

        async def receive_collected() -> Message:
            nonlocal first
            if first is None:
                return await receive()
            collected, first = first, None
            return collected

        await self._app(scope, receive_collected, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = error_response(
            413, "request_too_large", f"The request body is larger than {self._max_bytes} bytes"
        )
        await response(scope, receive, send)


def _declared_length(scope: Scope) -> int:
    for name, value in scope["headers"]:
        if name == b"content-length" and value.isdigit():
            return int(value)
    return 0
