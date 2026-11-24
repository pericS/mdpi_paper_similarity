"""Simple replacements for expensive or external parts of the application."""

import hashlib
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import numpy as np

from paper_similarity.arxiv.feed import ArxivPage, ArxivRecord
from paper_similarity.db.models import EMBEDDING_DIMENSIONS
from paper_similarity.embeddings.base import EmbeddingError, UnembeddableTextError


class FakeEmbedder:
    """Counts words into vector positions: texts sharing words get similar vectors.

    Fast, needs no model and gives the same result on every run.
    """

    def __init__(self, model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self.embedded_texts: list[str] = []
        self.error: EmbeddingError | None = None  # raised by embed(), if set

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        self.embedded_texts.extend(texts)
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        words = text.lower().split()
        if not words:
            raise UnembeddableTextError("The text contains nothing to embed")
        vector = np.zeros(EMBEDDING_DIMENSIONS)
        for word in words:
            # hash() differs between Python runs; SHA-256 does not.
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4]) % EMBEDDING_DIMENSIONS] += 1
        result: list[float] = (vector / np.linalg.norm(vector)).tolist()
        return result


def make_record(
    arxiv_id: str = "2401.00001",
    *,
    version: int = 1,
    title: str = "A title",
    abstract: str = "An abstract",
    categories: tuple[str, ...] = ("cs.CL",),
) -> ArxivRecord:
    return ArxivRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=("First Author", "Second Author"),
        categories=categories,
        primary_category=categories[0],
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        revised_at=datetime(2024, 1, version, tzinfo=UTC),
    )


class FakePaperSource:
    """Returns prepared pages. An exception in the list is raised instead."""

    def __init__(self, *pages: ArxivPage | Exception) -> None:
        self._pages = pages

    def iter_pages(self, query: str, max_results: int) -> Iterator[ArxivPage]:
        for page in self._pages:
            if isinstance(page, Exception):
                raise page
            yield page


def make_page(*records: ArxivRecord, skipped: int = 0) -> ArxivPage:
    return ArxivPage(total_results=len(records) + skipped, records=tuple(records), skipped=skipped)
