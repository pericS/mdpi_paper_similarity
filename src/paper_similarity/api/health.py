"""Health endpoints."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]


@router.get("/live", summary="Check that the process is running")
def live() -> HealthResponse:
    """Report liveness only. External dependencies are deliberately not checked."""
    return HealthResponse(status="ok")
