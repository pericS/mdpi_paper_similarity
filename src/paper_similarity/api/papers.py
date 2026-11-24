"""Submitting papers."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, status
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from paper_similarity.api.dependencies import SubmissionServiceDependency

MAX_TITLE_CHARACTERS = 500
MAX_TEXT_CHARACTERS = 100_000

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])


def _without_null_characters(value: str) -> str:
    # PostgreSQL cannot store them.
    if "\x00" in value:
        raise ValueError("must not contain null characters")
    return value


def _not_blank(value: str) -> str:
    if not value.split():
        raise ValueError("must not be blank")
    return value


def _blank_to_none(value: str | None) -> str | None:
    return value if value and value.split() else None


Title = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=MAX_TITLE_CHARACTERS),
    AfterValidator(_without_null_characters),
]
Text = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=MAX_TEXT_CHARACTERS),
    AfterValidator(_without_null_characters),
    AfterValidator(_not_blank),
]


class SubmitPaperRequest(BaseModel):
    model_config = ConfigDict(
        # Reports misspelled field names instead of ignoring them.
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "title": "Dense retrieval for scientific literature",
                    "text": "We study how neural embeddings can find related scientific papers.",
                }
            ]
        },
    )

    title: Annotated[Title | None, AfterValidator(_blank_to_none)] = Field(
        default=None, description="Optional. Embedded together with the text."
    )
    text: Text = Field(description="The abstract or manuscript text.")


class SubmitPaperResponse(BaseModel):
    id: UUID = Field(description="Use this ID to retrieve similar papers.")
    status: Literal["created"] = "created"


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Submit a paper",
    responses={503: {"description": "The embedding model or the database is unavailable"}},
)
def submit_paper(
    request: SubmitPaperRequest, service: SubmissionServiceDependency
) -> SubmitPaperResponse:
    """Embed and store a paper or manuscript, to be compared with the indexed arXiv papers.

    Every valid request creates a new paper, also when the same text was sent before.
    """
    return SubmitPaperResponse(id=service.submit(request.title, request.text))
