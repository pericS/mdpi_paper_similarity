"""Simple replacements for expensive or external parts of the application."""

import hashlib
from collections.abc import Sequence

import numpy as np

from paper_similarity.db.models import EMBEDDING_DIMENSIONS
from paper_similarity.embeddings.base import EmbeddingError, UnembeddableTextError


class FakeEmbedder:
    """Counts words into vector positions: texts sharing words get similar vectors.

    Fast, needs no model and gives the same result on every run.
    """

    def __init__(self, model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self.embedded_texts: list[str] = []
        self.fail = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise EmbeddingError("The fake embedder was told to fail")
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
