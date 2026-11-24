"""Retrieving similar papers."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field

from paper_similarity.api.dependencies import SimilarityServiceDependency
from paper_similarity.api.errors import documented_errors
from paper_similarity.repositories.arxiv_papers import SimilarPaper

DEFAULT_LIMIT = 10
MAX_LIMIT = 100
SCORE_DECIMALS = 6

# The task specifies "posts" here, although the paper is created at /api/v1/papers.
router = APIRouter(prefix="/api/v1/posts", tags=["papers"])


class Match(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    primary_category: str
    published_at: datetime
    url: str
    score: float = Field(
        description="Cosine similarity, from -1 to 1; higher is more similar. "
        "Useful for ordering, but not a percentage or a probability."
    )

    @classmethod
    def from_paper(cls, paper: SimilarPaper) -> Match:
        return cls(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            primary_category=paper.primary_category,
            published_at=paper.published_at,
            url=paper.url,
            score=round(1 - paper.distance, SCORE_DECIMALS),
        )


class SimilarPapersResponse(BaseModel):
    paper_id: UUID
    embedding_model: str = Field(description="Only papers embedded with this model are compared.")
    matches: list[Match] = Field(description="Most similar first. Empty if there are none.")


@router.get(
    "/{paper_id}/similar",
    summary="Retrieve similar papers",
    responses=documented_errors(404, 422, 503),
)
def get_similar_papers(
    service: SimilarityServiceDependency,
    paper_id: Annotated[UUID, Path(description="The ID returned when the paper was submitted.")],
    limit: Annotated[
        int, Query(ge=1, le=MAX_LIMIT, description="Maximum number of results.")
    ] = DEFAULT_LIMIT,
) -> SimilarPapersResponse:
    """Compare a submitted paper with all indexed arXiv papers."""
    result = service.find_similar(paper_id, limit)
    return SimilarPapersResponse(
        paper_id=paper_id,
        embedding_model=result.embedding_model,
        matches=[Match.from_paper(paper) for paper in result.papers],
    )
