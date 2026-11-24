"""Adds arXiv papers to the index."""

import hashlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from paper_similarity.arxiv.feed import ArxivPage, ArxivRecord
from paper_similarity.embeddings.base import Embedder, UnembeddableTextError
from paper_similarity.embeddings.text import build_embedding_text
from paper_similarity.repositories.arxiv_papers import (
    ArxivPaperRepository,
    EmbeddedPaper,
    StoredPaper,
    metadata_values,
)

logger = logging.getLogger(__name__)


class PaperSource(Protocol):
    def iter_pages(self, query: str, max_results: int) -> Iterator[ArxivPage]: ...


@dataclass(frozen=True)
class IngestionReport:
    """How many papers were handled, and how. Reports can be added together."""

    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0

    def __add__(self, other: IngestionReport) -> IngestionReport:
        return IngestionReport(
            created=self.created + other.created,
            updated=self.updated + other.updated,
            unchanged=self.unchanged + other.unchanged,
            skipped=self.skipped + other.skipped,
        )

    @property
    def usable(self) -> int:
        return self.created + self.updated + self.unchanged


@dataclass(frozen=True)
class _Candidate:
    """A paper from arXiv, together with what is already stored about it."""

    record: ArxivRecord
    stored: StoredPaper | None
    text: str
    content_hash: str


class IngestionService:
    def __init__(
        self, source: PaperSource, embedder: Embedder, session_factory: sessionmaker[Session]
    ) -> None:
        self._source = source
        self._embedder = embedder
        self._session_factory = session_factory

    def ingest(self, query: str, max_results: int) -> Iterator[IngestionReport]:
        """Save one page at a time and yield its report after it was committed."""
        for page in self._source.iter_pages(query, max_results):
            yield self._ingest_page(page)

    def _ingest_page(self, page: ArxivPage) -> IngestionReport:
        to_embed, to_update, unchanged = self._compare_with_stored(page)

        # Embedding is slow, so no database connection is held during it.
        embedded = self._embed(to_embed)
        unembeddable = len(to_embed) - len(embedded)

        with self._session_factory.begin() as session:
            repository = ArxivPaperRepository(session)
            repository.upsert([paper for _, paper in embedded])
            for candidate in to_update:
                repository.update_metadata(
                    candidate.record,
                    content_hash=candidate.content_hash,
                    embedding_model=self._embedder.model_id,
                )

        created = sum(candidate.stored is None for candidate, _ in embedded)
        return IngestionReport(
            created=created,
            updated=len(embedded) - created + len(to_update),
            unchanged=unchanged,
            skipped=page.skipped + unembeddable,
        )

    def _compare_with_stored(
        self, page: ArxivPage
    ) -> tuple[list[_Candidate], list[_Candidate], int]:
        """Split a page into: needs a vector, needs new metadata only, unchanged (count)."""
        records = _newest_versions(page.records)
        with self._session_factory() as session:
            stored = ArxivPaperRepository(session).find_stored(
                [record.arxiv_id for record in records]
            )

        to_embed, to_update = [], []
        for record in records:
            text = build_embedding_text(record.title, record.abstract)
            candidate = _Candidate(
                record=record,
                stored=stored.get(record.arxiv_id),
                text=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
            if self._needs_embedding(candidate):
                to_embed.append(candidate)
            elif _needs_metadata_update(candidate):
                to_update.append(candidate)
        # Repeated entries of the same paper also count as unchanged.
        unchanged = len(page.records) - len(to_embed) - len(to_update)
        return to_embed, to_update, unchanged

    def _needs_embedding(self, candidate: _Candidate) -> bool:
        stored = candidate.stored
        if stored is None:
            return True
        if candidate.record.version < stored.version:
            return False
        return (
            stored.content_hash != candidate.content_hash
            or stored.embedding_model != self._embedder.model_id
        )

    def _embed(self, candidates: Sequence[_Candidate]) -> list[tuple[_Candidate, EmbeddedPaper]]:
        if not candidates:
            return []
        try:
            vectors: list[list[float] | None] = [
                *self._embedder.embed([candidate.text for candidate in candidates])
            ]
        except UnembeddableTextError:
            # Very rare: find the paper responsible and save the others.
            vectors = [self._embed_or_skip(candidate) for candidate in candidates]

        return [
            (
                candidate,
                EmbeddedPaper(
                    record=candidate.record,
                    content_hash=candidate.content_hash,
                    embedding=vector,
                    embedding_model=self._embedder.model_id,
                ),
            )
            for candidate, vector in zip(candidates, vectors, strict=True)
            if vector is not None
        ]

    def _embed_or_skip(self, candidate: _Candidate) -> list[float] | None:
        try:
            return self._embedder.embed([candidate.text])[0]
        except UnembeddableTextError as error:
            logger.warning("Skipping arXiv paper %s: %s", candidate.record.arxiv_id, error)
            return None


def _newest_versions(records: Sequence[ArxivRecord]) -> list[ArxivRecord]:
    """Keep one record per paper. An upsert cannot change the same row twice."""
    newest: dict[str, ArxivRecord] = {}
    for record in records:
        current = newest.get(record.arxiv_id)
        if current is None or record.version > current.version:
            newest[record.arxiv_id] = record
    return list(newest.values())


def _needs_metadata_update(candidate: _Candidate) -> bool:
    stored = candidate.stored
    return (
        stored is not None
        and candidate.record.version >= stored.version
        and metadata_values(candidate.record) != stored.metadata
    )
