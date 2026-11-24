from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from paper_similarity.api.papers import MAX_TEXT_CHARACTERS, MAX_TITLE_CHARACTERS
from paper_similarity.config import Settings
from paper_similarity.db.models import ArxivPaper, SubmittedPaper
from paper_similarity.embeddings.base import EmbeddingError, TextTooLongError
from paper_similarity.main import create_app
from tests.fakes import FakeEmbedder

URL = "/api/v1/papers"


def stored_papers(engine: Engine) -> list[SubmittedPaper]:
    with Session(engine) as session:
        return list(session.scalars(select(SubmittedPaper)))


def test_paper_is_stored_with_its_vector(
    api: TestClient, committed_database: Engine, embedder: FakeEmbedder
) -> None:
    response = api.post(URL, json={"title": " A title ", "text": " Some\n text "})

    assert response.status_code == 201
    assert response.json().keys() == {"id", "status"}
    assert response.json()["status"] == "created"
    (paper,) = stored_papers(committed_database)
    assert paper.id == UUID(response.json()["id"])
    assert (paper.title, paper.text) == ("A title", "Some\n text")
    assert paper.embedding == pytest.approx(embedder.embed(["A title\n\nSome text"])[0])
    assert paper.embedding_model == embedder.model_id


@pytest.mark.parametrize("title", [None, "", "   "])
def test_title_is_optional(
    api: TestClient, committed_database: Engine, embedder: FakeEmbedder, title: str | None
) -> None:
    response = api.post(URL, json={"title": title, "text": "Some text"})

    assert response.status_code == 201
    assert stored_papers(committed_database)[0].title is None
    assert embedder.embedded_texts == ["Some text"]


def test_submitted_paper_does_not_become_part_of_the_index(
    api: TestClient, committed_database: Engine
) -> None:
    api.post(URL, json={"text": "Some text"})

    with Session(committed_database) as session:
        assert session.scalar(select(func.count()).select_from(ArxivPaper)) == 0


def test_same_text_creates_another_paper(api: TestClient, committed_database: Engine) -> None:
    ids = {api.post(URL, json={"text": "Same"}).json()["id"] for _ in range(2)}

    assert len(ids) == 2
    assert len(stored_papers(committed_database)) == 2


def test_text_in_other_scripts_is_stored_unchanged(
    api: TestClient, committed_database: Engine
) -> None:
    text = "Čačak, 蛋白质折叠, ∑ → ∞, 😀, '; DROP TABLE submitted_papers; --"

    assert api.post(URL, json={"text": text}).status_code == 201
    assert stored_papers(committed_database)[0].text == text


def test_longest_allowed_values_are_accepted(api: TestClient) -> None:
    body = {"title": "t" * MAX_TITLE_CHARACTERS, "text": "t" * MAX_TEXT_CHARACTERS}

    assert api.post(URL, json=body).status_code == 201


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "text"),
        ({"title": "Only a title"}, "text"),
        ({"text": None}, "text"),
        ({"text": 5}, "text"),
        ({"text": ["a"]}, "text"),
        ({"text": ""}, "text"),
        ({"text": " \n\t"}, "text"),
        ({"text": "\x1c"}, "text"),  # whitespace only for Python, not for Pydantic
        ({"text": "\x1c", "title": "A title"}, "text"),
        ({"text": "null \x00 character"}, "text"),
        ({"text": "t" * (MAX_TEXT_CHARACTERS + 1)}, "text"),
        ({"text": "Text", "title": 5}, "title"),
        ({"text": "Text", "title": "t" * (MAX_TITLE_CHARACTERS + 1)}, "title"),
        ({"text": "Text", "title": "null \x00 character"}, "title"),
        ({"text": "Text", "titel": "Misspelled"}, "titel"),
        ([], "body"),
    ],
)
def test_invalid_request_is_rejected_and_names_the_field(
    api: TestClient, committed_database: Engine, body: object, field: str
) -> None:
    response = api.post(URL, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"].split(".")[-1] == field
    assert stored_papers(committed_database) == []


@pytest.mark.parametrize(
    "body",
    [
        r'{"text": "half of an emoji \ud83d"}',
        r'{"text": "Text", "title": "\ud83d"}',
        r'{"text": "Text", "\ud83d": "in a field name"}',
    ],
)
def test_incomplete_unicode_character_is_rejected(api: TestClient, body: str) -> None:
    response = api.post(URL, content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_rejected_text_is_not_repeated_in_the_response(api: TestClient) -> None:
    response = api.post(URL, json={"title": 5, "text": "Confidential" * 10_000})

    assert response.status_code == 422
    assert "Confidential" not in response.text


def test_broken_json_is_rejected(api: TestClient) -> None:
    response = api.post(URL, content="{", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_text_with_too_many_tokens_is_rejected_with_the_limit(
    api: TestClient, committed_database: Engine, embedder: FakeEmbedder
) -> None:
    embedder.error = TextTooLongError("The text has 9 tokens, the limit is 8")

    response = api.post(URL, json={"text": "Some text"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "text_too_long"
    assert "limit is 8" in response.json()["error"]["message"]
    assert stored_papers(committed_database) == []


def test_embedding_failure_stores_nothing_and_hides_details(
    api: TestClient, committed_database: Engine, embedder: FakeEmbedder
) -> None:
    embedder.error = EmbeddingError("internal path /secret/model.onnx")

    response = api.post(URL, json={"text": "Some text"})

    assert response.status_code == 503
    assert "secret" not in response.text
    assert stored_papers(committed_database) == []

    embedder.error = None
    assert api.post(URL, json={"text": "Some text"}).status_code == 201


def test_unavailable_database_is_reported(embedder: FakeEmbedder) -> None:
    settings = Settings(
        _env_file=None, database_url=SecretStr("postgresql+psycopg://u:top-secret@127.0.0.1:1/x")
    )
    # The application starts without a database; only the request fails.
    with TestClient(create_app(settings, embedder)) as api:
        response = api.post(URL, json={"text": "Some text"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"


def test_manuscript_is_not_logged(api: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        api.post(URL, json={"title": "Secret title", "text": "Confidential text"})

    assert "Stored submitted paper" in caplog.text
    assert "Secret" not in caplog.text
    assert "Confidential" not in caplog.text
