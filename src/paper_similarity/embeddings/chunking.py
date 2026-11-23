"""Splits long text into pieces the model can read, and combines their vectors."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from paper_similarity.embeddings.base import EmbeddingError


@dataclass(frozen=True)
class Chunk:
    text: str
    token_count: int


def split_into_chunks(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[Chunk]:
    """Split text into consecutive chunks of at most max_tokens, between words.

    The tokenizer must not truncate. Special tokens are not counted.
    """
    encoding = tokenizer.encode(text, add_special_tokens=False)
    word_ids, offsets = encoding.word_ids, encoding.offsets
    total = len(offsets)

    chunks = []
    start = 0
    while start < total:
        end = min(start + max_tokens, total)
        end = _move_back_to_word_boundary(word_ids, start, end)
        first_character, last_character = offsets[start][0], offsets[end - 1][1]
        chunks.append(Chunk(text[first_character:last_character], end - start))
        start = end
    return chunks


def _move_back_to_word_boundary(word_ids: Sequence[int | None], start: int, end: int) -> int:
    """Avoid ending a chunk inside a word, unless that word alone fills the chunk."""
    if end == len(word_ids):
        return end
    for candidate in range(end, start, -1):
        inside_word = (
            word_ids[candidate] is not None and word_ids[candidate] == word_ids[candidate - 1]
        )
        if not inside_word:
            return candidate
    return end


def combine_vectors(
    vectors: Sequence[NDArray[np.number]], weights: Sequence[int]
) -> NDArray[np.float64]:
    """Return the weighted average of the vectors, scaled to length 1."""
    average = np.average(np.asarray(vectors, dtype=np.float64), axis=0, weights=weights)
    length = np.linalg.norm(average)
    if not np.isfinite(length) or length == 0:
        raise EmbeddingError("The model produced an unusable vector")
    normalized: NDArray[np.float64] = average / length
    return normalized
