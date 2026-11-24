"""Finds indexed papers that are similar to a submitted one."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from paper_similarity.repositories.arxiv_papers import ArxivPaperRepository, SimilarPaper
from paper_similarity.repositories.submitted_papers import SubmittedPaperRepository


class PaperNotFoundError(Exception):
    """There is no submitted paper with this ID."""


@dataclass(frozen=True)
class SimilarityResult:
    embedding_model: str
    papers: list[SimilarPaper]


class SimilarityService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_similar(self, paper_id: UUID, limit: int) -> SimilarityResult:
        """Return the most similar arXiv papers, most similar first."""
        with self._session_factory() as session:
            stored = SubmittedPaperRepository(session).find_vector(paper_id)
            if stored is None:
                raise PaperNotFoundError
            papers = ArxivPaperRepository(session).find_similar(
                stored.embedding, embedding_model=stored.embedding_model, limit=limit
            )
        return SimilarityResult(stored.embedding_model, papers)
