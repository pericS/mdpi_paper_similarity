"""Reads and writes submitted manuscripts."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_similarity.db.models import SubmittedPaper


@dataclass(frozen=True)
class StoredVector:
    embedding: list[float]
    embedding_model: str


class SubmittedPaperRepository:
    """Does not commit: the caller decides when a transaction ends."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def check_available(self) -> None:
        """Raise an error if the table cannot be read."""
        self._session.execute(select(SubmittedPaper.id).limit(1))

    def add(self, paper: SubmittedPaper) -> None:
        self._session.add(paper)

    def find_vector(self, paper_id: UUID) -> StoredVector | None:
        """Return the vector of a paper, without loading its text."""
        row = self._session.execute(
            select(SubmittedPaper.embedding, SubmittedPaper.embedding_model).where(
                SubmittedPaper.id == paper_id
            )
        ).one_or_none()
        return None if row is None else StoredVector(row.embedding, row.embedding_model)
