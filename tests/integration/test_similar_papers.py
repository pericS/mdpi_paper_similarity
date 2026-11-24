from collections.abc import Sequence
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from pydantic import SecretStr
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from paper_similarity.config import Settings
from paper_similarity.db.models import EMBEDDING_DIMENSIONS, SubmittedPaper
from paper_similarity.db.session import create_session_factory
from paper_similarity.main import create_app
from paper_similarity.repositories.arxiv_papers import ArxivPaperRepository, EmbeddedPaper
from paper_similarity.services.ingestion import IngestionService
from tests.fakes import FakeEmbedder, FakePaperSource, make_page, make_record

MODEL = "fake-model"


def vector(*first_values: float) -> list[float]:
    """A vector with these first values and zeros after them."""
    return [*first_values, *[0.0] * (EMBEDDING_DIMENSIONS - len(first_values))]


def index(
    engine: Engine, papers: dict[str, Sequence[float]], model: str = MODEL, title: str = "A title"
) -> None:
    with Session(engine) as session, session.begin():
        ArxivPaperRepository(session).upsert(
            [
                EmbeddedPaper(make_record(arxiv_id, title=title), "0" * 64, list(values), model)
                for arxiv_id, values in papers.items()
            ]
        )


def submit(engine: Engine, values: list[float], model: str = MODEL) -> UUID:
    paper = SubmittedPaper(text="Some text", embedding=values, embedding_model=model)
    with Session(engine) as session, session.begin():
        session.add(paper)
        session.flush()
        return paper.id


def get_similar(api: TestClient, paper_id: object, query: str = "") -> Response:
    return api.get(f"/api/v1/posts/{paper_id}/similar{query}")


def matched_ids(response: Response) -> list[str]:
    assert response.status_code == 200
    return [match["arxiv_id"] for match in response.json()["matches"]]


def test_results_are_ordered_by_cosine_similarity(
    api: TestClient, committed_database: Engine
) -> None:
    index(
        committed_database,
        {
            "2401.00001": vector(0, 1),  # perpendicular
            "2401.00002": vector(-1, 0),  # opposite
            "2401.00003": vector(1, 0),  # same direction
            "2401.00004": vector(1, 1),  # 45 degrees
        },
    )
    paper_id = submit(committed_database, vector(1, 0))

    response = get_similar(api, paper_id)

    assert matched_ids(response) == ["2401.00003", "2401.00004", "2401.00001", "2401.00002"]
    scores = [match["score"] for match in response.json()["matches"]]
    assert scores == pytest.approx([1, 1 / np.sqrt(2), 0, -1], abs=1e-6)


def test_length_of_a_vector_does_not_matter(api: TestClient, committed_database: Engine) -> None:
    index(committed_database, {"2401.00001": vector(0.1, 0), "2401.00002": vector(9, 9)})
    paper_id = submit(committed_database, vector(5, 0))

    response = get_similar(api, paper_id)

    assert matched_ids(response) == ["2401.00001", "2401.00002"]
    assert response.json()["matches"][0]["score"] == 1


def test_response_contains_what_is_needed_to_show_a_paper(
    api: TestClient, committed_database: Engine
) -> None:
    index(committed_database, {"2401.00001": vector(1)})
    paper_id = submit(committed_database, vector(1))

    assert get_similar(api, paper_id).json() == {
        "paper_id": str(paper_id),
        "embedding_model": MODEL,
        "matches": [
            {
                "arxiv_id": "2401.00001",
                "title": "A title",
                "authors": ["First Author", "Second Author"],
                "primary_category": "cs.CL",
                "published_at": "2024-01-01T00:00:00Z",
                "url": "https://arxiv.org/abs/2401.00001",
                "score": 1.0,
            }
        ],
    }


def test_equal_scores_are_always_ordered_by_id(api: TestClient, committed_database: Engine) -> None:
    ids = ["2401.00003", "hep-th/9901001", "2401.00001", "2401.00002"]
    for arxiv_id in ids:
        index(committed_database, {arxiv_id: vector(1)})
    # An update moves the row to another place in the table.
    index(committed_database, {"2401.00001": vector(1)}, title="Changed")
    paper_id = submit(committed_database, vector(1))

    results = [matched_ids(get_similar(api, paper_id)) for _ in range(3)]

    assert results == [sorted(ids)] * 3
    assert matched_ids(get_similar(api, paper_id, "?limit=2")) == sorted(ids)[:2]


@pytest.mark.parametrize(("query", "count"), [("", 10), ("?limit=1", 1), ("?limit=100", 12)])
def test_number_of_results_can_be_limited(
    api: TestClient, committed_database: Engine, query: str, count: int
) -> None:
    index(committed_database, {f"2401.{n:05}": vector(1, n) for n in range(12)})
    paper_id = submit(committed_database, vector(1))

    ids = matched_ids(get_similar(api, paper_id, query))

    # The best ones are returned, not just any.
    assert ids == [f"2401.{n:05}" for n in range(count)]


@pytest.mark.parametrize("limit", ["0", "-1", "101", "ten", "1.5", ""])
def test_invalid_limit_is_rejected(api: TestClient, committed_database: Engine, limit: str) -> None:
    paper_id = submit(committed_database, vector(1))

    response = get_similar(api, paper_id, f"?limit={limit}")

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "limit"]


def test_unknown_id_is_not_found(api: TestClient, committed_database: Engine) -> None:
    index(committed_database, {"2401.00001": vector(1)})

    response = get_similar(api, uuid4())

    assert response.status_code == 404
    assert response.json() == {"detail": "There is no submitted paper with this ID"}


@pytest.mark.parametrize("paper_id", ["paper-123", "1", "2401.00001", "0193"])
def test_malformed_id_is_rejected(api: TestClient, paper_id: str) -> None:
    response = get_similar(api, paper_id)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["path", "paper_id"]


def test_empty_index_gives_no_matches(api: TestClient, committed_database: Engine) -> None:
    paper_id = submit(committed_database, vector(1))

    assert matched_ids(get_similar(api, paper_id)) == []


def test_vectors_of_another_model_are_not_compared(
    api: TestClient, committed_database: Engine
) -> None:
    index(committed_database, {"2401.00001": vector(1)}, model="other-model")
    index(committed_database, {"2401.00002": vector(0, 1)})
    paper_id = submit(committed_database, vector(1))

    assert matched_ids(get_similar(api, paper_id)) == ["2401.00002"]


def test_other_submitted_papers_are_never_returned(
    api: TestClient, committed_database: Engine
) -> None:
    submit(committed_database, vector(1))
    paper_id = submit(committed_database, vector(1))

    assert matched_ids(get_similar(api, paper_id)) == []


def test_unavailable_database_is_reported(embedder: FakeEmbedder) -> None:
    settings = Settings(
        _env_file=None, database_url=SecretStr("postgresql+psycopg://u:top-secret@127.0.0.1:1/x")
    )
    with TestClient(create_app(settings, embedder)) as api:
        response = get_similar(api, uuid4())

    assert response.status_code == 503


def test_complete_workflow_also_works_after_a_restart(
    database_url: str, committed_database: Engine, embedder: FakeEmbedder
) -> None:
    """Ingest, submit and search, then search again with a new application."""
    page = make_page(
        make_record("2401.00001", title="Soil erosion", abstract="Vineyards and rain"),
        make_record("2401.00002", title="Neural networks", abstract="Searching deep models"),
        make_record("2401.00003", title="Neural retrieval", abstract="Searching scientific papers"),
    )
    service = IngestionService(
        FakePaperSource(page), embedder, create_session_factory(committed_database)
    )
    list(service.ingest("any", 10))
    settings = Settings(_env_file=None, database_url=SecretStr(database_url))

    with TestClient(create_app(settings, embedder)) as api:
        body = {"title": "Neural retrieval", "text": "We are searching scientific papers"}
        paper_id = api.post("/api/v1/papers", json=body).json()["id"]
        first = get_similar(api, paper_id)
    embedder.embedded_texts.clear()
    with TestClient(create_app(settings, embedder)) as api:
        second = get_similar(api, paper_id)

    assert matched_ids(first) == ["2401.00003", "2401.00002", "2401.00001"]
    assert second.json() == first.json()
    assert embedder.embedded_texts == []  # the stored vector was used
