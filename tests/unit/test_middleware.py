"""The body limit alone, with exactly controlled ASGI messages and without a server."""

import asyncio
import json
import time

from starlette.types import Message, Receive, Scope, Send

from paper_similarity.api.middleware import BodySizeLimitMiddleware

DISCONNECT: Message = {"type": "http.disconnect"}


def chunk(body: bytes, *, more: bool = True) -> Message:
    return {"type": "http.request", "body": body, "more_body": more}


class Exchange:
    """Sends prepared messages through the middleware and records what comes out."""

    def __init__(self, *incoming: Message, max_bytes: int = 10) -> None:
        self._incoming = list(reversed(incoming))
        self.received: list[Message] = []  # by the application
        self.sent: list[Message] = []  # to the client
        scope: Scope = {"type": "http", "headers": []}
        middleware = BodySizeLimitMiddleware(self._app, max_bytes)
        asyncio.run(middleware(scope, self._receive, self._send))

    async def _receive(self) -> Message:
        return self._incoming.pop()

    async def _send(self, message: Message) -> None:
        self.sent.append(message)

    async def _app(self, scope: Scope, receive: Receive, send: Send) -> None:
        message = await receive()
        self.received.append(message)
        while message.get("more_body", False):
            message = await receive()
            self.received.append(message)

    @property
    def body(self) -> bytes:
        return b"".join(message["body"] for message in self.received)


def test_body_in_several_chunks_is_passed_on_unchanged() -> None:
    exchange = Exchange(chunk(b"abc"), chunk(b""), chunk(b"def"), chunk(b"ghij", more=False))

    assert exchange.body == b"abcdefghij"  # exactly at the limit
    assert len(exchange.received) == 1
    assert exchange.sent == []


def test_empty_body_is_passed_on() -> None:
    assert Exchange(chunk(b"", more=False)).body == b""


def test_one_byte_too_many_is_rejected() -> None:
    exchange = Exchange(chunk(b"0123456789"), chunk(b"x", more=False))

    assert exchange.received == []
    start, content = exchange.sent
    assert start["status"] == 413
    assert json.loads(content["body"])["error"]["code"] == "request_too_large"


def test_disconnect_while_sending_is_passed_on() -> None:
    exchange = Exchange(chunk(b"abc"), DISCONNECT)

    assert exchange.received == [DISCONNECT]


def test_disconnect_after_the_body_can_still_be_seen() -> None:
    async def scenario() -> list[Message]:
        incoming = [DISCONNECT, chunk(b"abc", more=False)]
        received: list[Message] = []

        async def receive() -> Message:
            return incoming.pop()

        async def send(message: Message) -> None:
            raise AssertionError("nothing should be sent")

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            received.extend([await receive(), await receive()])

        await BodySizeLimitMiddleware(app)({"type": "http", "headers": []}, receive, send)
        return received

    body, disconnect = asyncio.run(scenario())

    assert body["body"] == b"abc"
    assert disconnect == DISCONNECT


def test_many_tiny_chunks_do_not_block_the_event_loop() -> None:
    count = 300_000
    messages = [chunk(b"x")] * (count - 1) + [chunk(b"x", more=False)]

    started = time.perf_counter()
    exchange = Exchange(*messages, max_bytes=count)

    assert exchange.body == b"x" * count
    # Takes about 0.05 s. Removing chunks from the front of a list took 4 s.
    assert time.perf_counter() - started < 2
