"""Reads and writes the index of arXiv papers."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from paper_similarity.arxiv.feed import ArxivRecord
from paper_similarity.db.models import ArxivPaper


@dataclass(frozen=True)
class StoredPaper:
    """What is already stored about a paper, without its vector."""

    metadata: dict[str, Any]
    content_hash: str
    embedding_model: str

    @property
    def version(self) -> int:
        version: int = self.metadata["version"]
        return version


@dataclass(frozen=True)
class EmbeddedPaper:
    """A paper that is ready to be stored."""

    record: ArxivRecord
    content_hash: str
    embedding: list[float]
    embedding_model: str


@dataclass(frozen=True)
class SimilarPaper:
    """A search result: what is needed to show a paper, without its vector."""

    arxiv_id: str
    title: str
    authors: list[str]
    primary_category: str
    url: str
    published_at: datetime
    distance: float  # cosine distance: 0 same direction, 1 perpendicular, 2 opposite


def metadata_values(record: ArxivRecord) -> dict[str, Any]:
    """Return the columns that come directly from arXiv."""
    return {
        "version": record.version,
        "title": record.title,
        "abstract": record.abstract,
        "authors": list(record.authors),
        "categories": list(record.categories),
        "primary_category": record.primary_category,
        "url": record.url,
        "published_at": record.published_at,
        "revised_at": record.revised_at,
    }


METADATA_COLUMNS = (
    ArxivPaper.version,
    ArxivPaper.title,
    ArxivPaper.abstract,
    ArxivPaper.authors,
    ArxivPaper.categories,
    ArxivPaper.primary_category,
    ArxivPaper.url,
    ArxivPaper.published_at,
    ArxivPaper.revised_at,
)


class ArxivPaperRepository:
    """Does not commit: the caller decides when a transaction ends."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def check_available(self) -> None:
        """Raise an error if the table cannot be read."""
        self._session.execute(select(ArxivPaper.id).limit(1))

    def find_stored(self, arxiv_ids: Sequence[str]) -> dict[str, StoredPaper]:
        """Return the papers that already exist, by arXiv ID."""
        rows = self._session.execute(
            select(
                ArxivPaper.arxiv_id,
                ArxivPaper.content_hash,
                ArxivPaper.embedding_model,
                *METADATA_COLUMNS,
            ).where(ArxivPaper.arxiv_id.in_(arxiv_ids))
        ).mappings()
        return {
            row["arxiv_id"]: StoredPaper(
                metadata={column.key: row[column.key] for column in METADATA_COLUMNS},
                content_hash=row["content_hash"],
                embedding_model=row["embedding_model"],
            )
            for row in rows
        }

    def find_similar(
        self, embedding: Sequence[float], *, embedding_model: str, limit: int
    ) -> list[SimilarPaper]:
        """Return the closest papers, closest first.

        Compares with every paper (there is no approximate index), so none is missed.
        """
        distance = ArxivPaper.embedding.cosine_distance(embedding).label("distance")
        rows = self._session.execute(
            select(
                ArxivPaper.arxiv_id,
                ArxivPaper.title,
                ArxivPaper.authors,
                ArxivPaper.primary_category,
                ArxivPaper.url,
                ArxivPaper.published_at,
                distance,
            )
            # Vectors of different models are not comparable.
            .where(ArxivPaper.embedding_model == embedding_model)
            # The ID makes the order the same every time, also for equal distances.
            .order_by(distance, ArxivPaper.arxiv_id)
            .limit(limit)
        ).mappings()
        return [SimilarPaper(**row) for row in rows]

    def upsert(self, papers: Sequence[EmbeddedPaper]) -> None:
        """Insert papers, or update those that exist. Never replaces a newer version.

        Each arXiv ID may appear only once in `papers`.
        """
        if not papers:
            return
        statement = insert(ArxivPaper).values(
            [
                {
                    "arxiv_id": paper.record.arxiv_id,
                    **metadata_values(paper.record),
                    "content_hash": paper.content_hash,
                    "embedding": paper.embedding,
                    "embedding_model": paper.embedding_model,
                }
                for paper in papers
            ]
        )
        new = statement.excluded  # the values that could not be inserted
        changed = {
            column.key: new[column.key]
            for column in (
                *METADATA_COLUMNS,
                ArxivPaper.content_hash,
                ArxivPaper.embedding,
                ArxivPaper.embedding_model,
            )
        }
        self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[ArxivPaper.arxiv_id],
                # The model's onupdate does not apply to upserts.
                set_={**changed, "updated_at": func.now()},
                where=ArxivPaper.version <= new.version,
            )
        )

    def update_metadata(
        self, record: ArxivRecord, *, content_hash: str, embedding_model: str
    ) -> None:
        """Update a paper whose vector is still correct.

        The conditions prevent new metadata from being combined with another vector,
        if the paper changed after it was read.
        """
        self._session.execute(
            update(ArxivPaper)
            .where(
                ArxivPaper.arxiv_id == record.arxiv_id,
                ArxivPaper.version <= record.version,
                ArxivPaper.content_hash == content_hash,
                ArxivPaper.embedding_model == embedding_model,
            )
            .values(**metadata_values(record))
        )
