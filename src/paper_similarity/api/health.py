"""Health endpoints."""

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import JSONResponse

from paper_similarity.api.dependencies import SessionFactoryDependency
from paper_similarity.api.errors import documented_errors, error_response
from paper_similarity.db.errors import describe_safely
from paper_similarity.repositories.arxiv_papers import ArxivPaperRepository
from paper_similarity.repositories.submitted_papers import SubmittedPaperRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"


@router.get("/live", summary="Check that the process is running")
async def live() -> LiveResponse:
    """Report liveness only. External dependencies are deliberately not checked."""
    # Async, so it does not wait for a worker thread when all are busy with submissions.
    return LiveResponse()


@router.get(
    "/ready",
    summary="Check that requests can be served",
    response_model=ReadyResponse,
    responses=documented_errors(503),
)
def ready(session_factory: SessionFactoryDependency) -> ReadyResponse | JSONResponse:
    """Check that the database can be reached and contains both tables.

    The embedding model is not checked: without it, the application does not start.
    """
    try:
        with session_factory() as session:
            ArxivPaperRepository(session).check_available()
            SubmittedPaperRepository(session).check_available()
    except SQLAlchemyError as error:
        logger.warning("Not ready: %s", describe_safely(error))
        return error_response(503, "not_ready", "The database is not ready")
    return ReadyResponse()
