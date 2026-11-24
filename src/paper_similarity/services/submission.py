"""Accepts manuscripts that will be compared with the index."""

import logging
from uuid import UUID, uuid7

from sqlalchemy.orm import Session, sessionmaker

from paper_similarity.db.models import SubmittedPaper
from paper_similarity.embeddings.base import Embedder
from paper_similarity.embeddings.text import build_embedding_text
from paper_similarity.repositories.submitted_papers import SubmittedPaperRepository

logger = logging.getLogger(__name__)


class SubmissionService:
    def __init__(self, embedder: Embedder, session_factory: sessionmaker[Session]) -> None:
        self._embedder = embedder
        self._session_factory = session_factory

    def submit(self, title: str | None, text: str) -> UUID:
        """Embed and store a manuscript. Returns its ID."""
        # Embedding comes first: no database connection is held during it, and a
        # failure leaves nothing behind.
        (embedding,) = self._embedder.embed([build_embedding_text(title, text)])
        paper = SubmittedPaper(
            id=uuid7(),
            title=title,
            text=text,
            embedding=embedding,
            embedding_model=self._embedder.model_id,
        )
        with self._session_factory.begin() as session:
            SubmittedPaperRepository(session).add(paper)
        # The manuscript may be confidential, so neither title nor text is logged.
        logger.info("Stored submitted paper %s (%d characters)", paper.id, len(text))
        return paper.id
