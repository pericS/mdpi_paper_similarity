import numpy as np

from paper_similarity.embeddings.base import Embedder
from tests.fakes import FakeEmbedder


def test_fake_embedder_ranks_texts_with_shared_words_higher() -> None:
    embedder: Embedder = FakeEmbedder()

    query, similar, different = embedder.embed(
        ["neural network training", "neural network design", "soil erosion in vineyards"]
    )

    assert np.dot(query, similar) > np.dot(query, different)
