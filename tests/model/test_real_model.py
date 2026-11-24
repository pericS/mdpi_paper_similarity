"""Checks of the real embedding model. Run with: uv run pytest --run-model"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pydantic import SecretStr

from paper_similarity.config import Settings
from paper_similarity.db.models import EMBEDDING_DIMENSIONS
from paper_similarity.embeddings.base import Embedder, TextTooLongError, UnembeddableTextError
from paper_similarity.embeddings.fastembed_embedder import FastEmbedEmbedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    # The configured cache folder is used, but these tests need no database.
    settings = Settings(database_url=SecretStr("postgresql+psycopg://unused"))
    return FastEmbedEmbedder(settings.embedding_cache_dir, max_text_tokens=2000)


def test_vectors_fit_the_database_and_have_length_one(embedder: Embedder) -> None:
    (vector,) = embedder.embed(["Graph neural networks"])

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert np.linalg.norm(vector) == pytest.approx(1)


def test_related_text_is_more_similar_than_unrelated_text(embedder: Embedder) -> None:
    query, related, unrelated = embedder.embed(
        [
            "Deep learning for image classification with convolutional networks.",
            "Convolutional neural networks recognise objects in photographs.",
            "Soil erosion rates in Mediterranean vineyards.",
        ]
    )

    assert np.dot(query, related) > np.dot(query, unrelated)


def test_text_after_the_model_limit_is_not_ignored(embedder: Embedder) -> None:
    beginning = "protein folding " * 400
    ending = "quantum gravity string theory " * 60

    without, with_ending = embedder.embed([beginning, beginning + ending])

    assert not np.allclose(without, with_ending)


def test_result_does_not_depend_on_other_texts_in_the_call(embedder: Embedder) -> None:
    short, long = "Graph neural networks", "A longer abstract about physics. " * 300

    (alone,) = embedder.embed([short])
    together = embedder.embed([long, short])[1]

    assert alone == pytest.approx(together, abs=1e-6)


def test_same_text_gives_the_same_vector_from_many_threads(embedder: Embedder) -> None:
    texts = [f"Paper number {i} about topic {i % 7}" for i in range(40)]
    expected = embedder.embed(texts)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda text: embedder.embed([text])[0], texts))

    np.testing.assert_allclose(results, expected, atol=1e-6)


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected(embedder: Embedder, text: str) -> None:
    with pytest.raises(UnembeddableTextError):
        embedder.embed(["A valid text", text])


def test_text_above_the_token_limit_is_rejected(embedder: Embedder) -> None:
    embedder.embed(["word " * 2000])

    with pytest.raises(TextTooLongError, match="2001 tokens, the limit is 2000"):
        embedder.embed(["word " * 2001])


def test_no_texts_give_no_vectors(embedder: Embedder) -> None:
    assert embedder.embed([]) == []
