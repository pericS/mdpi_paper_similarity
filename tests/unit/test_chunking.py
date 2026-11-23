import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from paper_similarity.embeddings.base import EmbeddingError
from paper_similarity.embeddings.chunking import Chunk, combine_vectors, split_into_chunks


@pytest.fixture
def tokenizer() -> Tokenizer:
    """A tiny tokenizer: a, b, c are one token; abc is three (a ##b ##c)."""
    vocabulary = ["[UNK]", "a", "b", "c", "##b", "##c"]
    model = WordPiece({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]")  # noqa: S106
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_short_text_is_one_chunk(tokenizer: Tokenizer) -> None:
    assert split_into_chunks("a b", tokenizer, max_tokens=4) == [Chunk("a b", 2)]


def test_text_of_exactly_the_limit_is_one_chunk(tokenizer: Tokenizer) -> None:
    assert split_into_chunks("a b c", tokenizer, max_tokens=3) == [Chunk("a b c", 3)]


def test_long_text_is_split_without_losing_anything(tokenizer: Tokenizer) -> None:
    chunks = split_into_chunks("a b c a b", tokenizer, max_tokens=2)

    assert chunks == [Chunk("a b", 2), Chunk("c a", 2), Chunk("b", 1)]


def test_words_are_not_split(tokenizer: Tokenizer) -> None:
    # A limit of 4 would end inside the second "abc".
    chunks = split_into_chunks("abc abc a", tokenizer, max_tokens=4)

    assert chunks == [Chunk("abc", 3), Chunk("abc a", 4)]


def test_word_longer_than_a_chunk_is_split_anyway(tokenizer: Tokenizer) -> None:
    chunks = split_into_chunks("abc", tokenizer, max_tokens=2)

    assert chunks == [Chunk("ab", 2), Chunk("c", 1)]


@pytest.mark.parametrize("text", ["", "  \n "])
def test_blank_text_has_no_chunks(tokenizer: Tokenizer, text: str) -> None:
    assert split_into_chunks(text, tokenizer, max_tokens=4) == []


def test_single_vector_is_only_normalized() -> None:
    result = combine_vectors([np.array([3.0, 4.0])], [10])

    assert result.tolist() == pytest.approx([0.6, 0.8])


def test_longer_chunks_have_more_influence() -> None:
    result = combine_vectors([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [3, 1])

    assert result[0] / result[1] == pytest.approx(3)
    assert np.linalg.norm(result) == pytest.approx(1)


def test_vectors_that_cancel_out_are_rejected() -> None:
    with pytest.raises(EmbeddingError):
        combine_vectors([np.array([1.0, 0.0]), np.array([-1.0, 0.0])], [1, 1])


def test_not_a_number_is_rejected() -> None:
    with pytest.raises(EmbeddingError):
        combine_vectors([np.array([np.nan, 1.0])], [1])
