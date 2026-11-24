from pathlib import Path

import pytest

from paper_similarity.embeddings import fastembed_embedder
from paper_similarity.embeddings.base import EmbeddingError
from paper_similarity.embeddings.fastembed_embedder import FastEmbedEmbedder


def test_unavailable_model_is_reported_as_embedding_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(*args: object, **kwargs: object) -> str:
        raise ConnectionError("no network")

    monkeypatch.setattr(fastembed_embedder, "snapshot_download", fail)

    with pytest.raises(EmbeddingError, match="Could not load"):
        FastEmbedEmbedder(tmp_path, max_text_tokens=100)
