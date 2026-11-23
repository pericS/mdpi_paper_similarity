from fastapi.testclient import TestClient

from paper_similarity.config import Settings
from paper_similarity.main import create_app


def test_live_returns_ok() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_documents_the_health_endpoint() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health/live" in response.json()["paths"]
