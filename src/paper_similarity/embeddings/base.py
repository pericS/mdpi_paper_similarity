"""What the rest of the application needs to know about embeddings."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingError(Exception):
    """An embedding could not be produced, e.g. the model could not be loaded."""


class UnembeddableTextError(EmbeddingError):
    """The text itself is the problem: it contains nothing to embed."""


class TextTooLongError(UnembeddableTextError):
    """The text has more tokens than the configured limit."""


class Embedder(Protocol):
    """Any object with these members can be used as an embedder."""

    @property
    def model_id(self) -> str:
        """Identifies the model. Vectors with different model IDs are not comparable."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector of length 1 per text, in the same order.

        Raises EmbeddingError. One bad text fails the whole call.
        """
        ...
