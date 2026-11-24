import asyncio

import httpx2
from anyio.to_thread import current_default_thread_limiter
from fastapi.testclient import TestClient

from paper_similarity.config import Settings
from paper_similarity.main import create_app


def test_live_returns_ok(settings: Settings) -> None:
    client = TestClient(create_app(settings))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_live_answers_while_all_worker_threads_are_busy(settings: Settings) -> None:
    async def scenario() -> int:
        # Sync endpoints, such as submissions, run in these threads.
        workers = current_default_thread_limiter()
        workers.total_tokens = 1
        workers.acquire_on_behalf_of_nowait("a long submission")

        transport = httpx2.ASGITransport(app=create_app(settings))
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await asyncio.wait_for(client.get("/health/live"), timeout=2)
        return response.status_code

    assert asyncio.run(scenario()) == 200


def test_openapi_documents_the_health_endpoint(settings: Settings) -> None:
    client = TestClient(create_app(settings))

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health/live" in response.json()["paths"]
