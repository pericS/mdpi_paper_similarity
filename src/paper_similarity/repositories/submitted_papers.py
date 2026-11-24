"""Reads and writes submitted manuscripts."""

from sqlalchemy.orm import Session

from paper_similarity.db.models import SubmittedPaper


class SubmittedPaperRepository:
    """Does not commit: the caller decides when a transaction ends."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, paper: SubmittedPaper) -> None:
        self._session.add(paper)
