"""Error format, logging, limits and health checks of the complete application."""

import io
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from paper_similarity.api.middleware import MAX_BODY_BYTES
from paper_similarity.api.papers import MAX_TEXT_CHARACTERS
from paper_similarity.config import Settings
from paper_similarity.db.session import create_db_engine
from paper_similarity.logging_config import HANDLER_NAME
from paper_similarity.main import create_app
from tests.fakes import FakeEmbedder
from tests.integration.conftest import run_alembic

PAPERS = "/api/v1/papers"
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def log(api: TestClient) -> io.StringIO:
    """Everything the application writes to its log."""
    output = io.StringIO()
    handler = next(h for h in logging.getLogger().handlers if h.name == HANDLER_NAME)
    assert isinstance(handler, logging.StreamHandler)
    handler.setStream(output)
    return output


@pytest.fixture
def api_without_database(embedder: FakeEmbedder) -> Iterator[TestClient]:
    url = SecretStr("postgresql+psycopg://u:top-secret@127.0.0.1:1/x")
    with TestClient(create_app(Settings(_env_file=None, database_url=url), embedder)) as api:
        yield api


# Error format


def test_every_kind_of_error_has_the_same_format(api: TestClient) -> None:
    responses = {
        "not_found": api.get("/unknown"),
        "method_not_allowed": api.get(PAPERS),
        "bad_request": api.post(PAPERS, content=b'{"text": "\xff"}', headers=JSON),
        "invalid_request": api.post(PAPERS, json={}),
        "request_too_large": api.post(PAPERS, content=b"x" * (MAX_BODY_BYTES + 1), headers=JSON),
        "paper_not_found": api.get("/api/v1/posts/01a20e77-0000-7000-8000-000000000000/similar"),
    }

    for code, response in responses.items():
        error = response.json()["error"]
        assert error.keys() == {"code", "message", "details", "request_id"}
        assert error["code"] == code
        assert error["request_id"] == response.headers["X-Request-ID"]
    assert [response.status_code for response in responses.values()] == [
        404, 405, 400, 422, 413, 404
    ]  # fmt: skip
    assert responses["method_not_allowed"].headers["Allow"] == "POST"


def test_error_format_is_documented(api: TestClient) -> None:
    operation = api.get("/openapi.json").json()["paths"][PAPERS]["post"]

    for status in ("413", "422", "503"):
        schema = operation["responses"][status]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/ErrorResponse")


def test_unexpected_error_reveals_nothing(
    api: TestClient, embedder: FakeEmbedder, log: io.StringIO
) -> None:
    embedder.error = RuntimeError("Confidential text in a message")

    response = api.post(PAPERS, json={"text": "Confidential text"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "Confidential" not in response.text
    assert "Confidential" not in log.getvalue()
    # Enough to find the problem:
    assert "RuntimeError" in log.getvalue()
    assert "submission.py" in log.getvalue()
    assert response.headers["X-Request-ID"] in log.getvalue()


def test_failed_database_rule_does_not_log_the_manuscript(
    api: TestClient, committed_database: Engine, log: io.StringIO
) -> None:
    rule = "ALTER TABLE submitted_papers ADD CONSTRAINT always_fails CHECK (text = 'other')"
    with committed_database.begin() as connection:
        connection.execute(text(rule))
    try:
        response = api.post(PAPERS, json={"text": "Confidential text"})
    finally:
        with committed_database.begin() as connection:
            connection.execute(text("ALTER TABLE submitted_papers DROP CONSTRAINT always_fails"))

    assert response.status_code == 500
    assert "Confidential" not in response.text + log.getvalue()
    assert "IntegrityError (SQLSTATE 23514) (constraint always_fails)" in log.getvalue()


# Request ID and request log


def test_each_request_gets_its_own_id(api: TestClient) -> None:
    ids = {api.get("/health/live").headers["X-Request-ID"] for _ in range(3)}

    assert len(ids) == 3


def test_id_from_the_client_is_used(api: TestClient) -> None:
    response = api.get("/unknown", headers={"X-Request-ID": "client-42"})

    assert response.headers["X-Request-ID"] == "client-42"
    assert response.json()["error"]["request_id"] == "client-42"


@pytest.mark.parametrize("value", ["", "a b", "x" * 65, "id;DROP", "čđ".encode()])
def test_unsafe_id_from_the_client_is_replaced(api: TestClient, value: str | bytes) -> None:
    sent = value.encode() if isinstance(value, str) else value

    response = api.get("/unknown", headers={b"X-Request-ID": sent})

    assert response.status_code == 404
    assert len(response.headers["X-Request-ID"]) == 32


def test_request_and_its_work_are_logged_with_the_same_id(
    api: TestClient, log: io.StringIO
) -> None:
    api.post(PAPERS, json={"text": "Some text"}, headers={"X-Request-ID": "client-42"})

    # The first line is written in a worker thread.
    stored, request = [line for line in log.getvalue().splitlines() if "[client-42]" in line]
    assert "Stored submitted paper" in stored
    assert "POST /api/v1/papers 201" in request


def test_path_cannot_add_lines_to_the_log(api: TestClient, log: io.StringIO) -> None:
    api.get("/a%0A2026-01-01 ERROR forged")

    assert len(log.getvalue().splitlines()) == 1


def test_health_checks_do_not_fill_the_log(api: TestClient, log: io.StringIO) -> None:
    api.get("/health/live")
    api.get("/health/ready")

    assert log.getvalue() == ""


# Limits


def test_body_without_declared_length_is_limited_too(api: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(MAX_BODY_BYTES // 100_000 + 1):
            yield b"x" * 100_000

    response = api.post(PAPERS, content=chunks(), headers=JSON)

    assert response.status_code == 413


def test_longest_text_fits_also_in_its_longest_encoding(api: TestClient) -> None:
    # json.dumps writes an emoji as two escape sequences: twelve bytes for one character.
    body = json.dumps({"text": "\N{GRINNING FACE}" * MAX_TEXT_CHARACTERS})
    assert len(body) > 12 * MAX_TEXT_CHARACTERS

    assert api.post(PAPERS, content=body, headers=JSON).status_code == 201


def test_long_query_is_stopped(database_url: str) -> None:
    settings = Settings(_env_file=None, database_url=SecretStr(database_url))
    engine = create_db_engine(settings, statement_timeout_seconds=1)

    with engine.connect() as connection, pytest.raises(OperationalError, match="timeout"):
        connection.execute(text("SELECT pg_sleep(3)"))
    engine.dispose()


# Health


def test_ready_when_the_database_can_be_used(api: TestClient) -> None:
    response = api.get("/health/ready")

    assert (response.status_code, response.json()) == (200, {"status": "ready"})


def test_alive_but_not_ready_without_database(api_without_database: TestClient) -> None:
    live = api_without_database.get("/health/live")
    ready = api_without_database.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "not_ready"
    assert "top-secret" not in ready.text


def test_not_ready_before_migrations(empty_database_engine: Engine, embedder: FakeEmbedder) -> None:
    url = SecretStr(empty_database_engine.url.render_as_string(hide_password=False))
    with TestClient(create_app(Settings(_env_file=None, database_url=url), embedder)) as api:
        response = api.get("/health/ready")

    assert response.status_code == 503


@pytest.mark.parametrize("table", ["arxiv_papers", "submitted_papers"])
def test_not_ready_if_one_table_is_missing(
    empty_database_engine: Engine, embedder: FakeEmbedder, table: str
) -> None:
    run_alembic(empty_database_engine, "upgrade", "head")
    with empty_database_engine.begin() as connection:
        connection.execute(text(f"DROP TABLE {table}"))

    url = SecretStr(empty_database_engine.url.render_as_string(hide_password=False))
    with TestClient(create_app(Settings(_env_file=None, database_url=url), embedder)) as api:
        response = api.get("/health/ready")

    assert response.status_code == 503
