"""Embeds text on the CPU with a local model, through FastEmbed."""

import logging
from collections.abc import Sequence
from pathlib import Path

from fastembed import TextEmbedding
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

from paper_similarity.embeddings.base import EmbeddingError, UnembeddableTextError
from paper_similarity.embeddings.chunking import combine_vectors, split_into_chunks

logger = logging.getLogger(__name__)

# These values belong together, so they are constants and not settings: changing the
# model changes the vector size (a migration) and requires re-embedding everything.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REPOSITORY = "Qdrant/bge-small-en-v1.5-onnx-Q"  # the same model in ONNX format
MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
MODEL_ID = f"{MODEL_NAME}@{MODEL_REVISION[:12]}"

MAX_MODEL_TOKENS = 512
SPECIAL_TOKENS = 2  # the model adds [CLS] at the start and [SEP] at the end
MAX_CHUNK_TOKENS = MAX_MODEL_TOKENS - SPECIAL_TOKENS
BATCH_SIZE = 16


def download_model(cache_dir: Path) -> Path:
    """Return the folder with the pinned model files, downloading them if needed."""
    try:
        # Try the cache first, so a cached model never needs the network.
        return _model_snapshot(cache_dir, local_files_only=True)
    except LocalEntryNotFoundError:
        logger.info("Downloading embedding model %s to %s", MODEL_ID, cache_dir)
        return _model_snapshot(cache_dir, local_files_only=False)


def _model_snapshot(cache_dir: Path, *, local_files_only: bool) -> Path:
    return Path(
        snapshot_download(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


class FastEmbedEmbedder:
    """Implements the Embedder protocol. Create one per process: loading is expensive."""

    def __init__(self, cache_dir: Path) -> None:
        try:
            model_path = download_model(cache_dir.expanduser())
            self._model = TextEmbedding(
                MODEL_NAME,
                specific_model_path=str(model_path),
                providers=["CPUExecutionProvider"],
            )
            # FastEmbed's own tokenizer silently cuts text after 512 tokens. This
            # separate one does not, so it can split long text into chunks.
            self._tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        except Exception as error:
            raise EmbeddingError(f"Could not load embedding model {MODEL_ID}") from error

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        chunks_per_text = [
            split_into_chunks(text, self._tokenizer, MAX_CHUNK_TOKENS) for text in texts
        ]
        if not all(chunks_per_text):
            raise UnembeddableTextError("The text contains nothing to embed")

        all_chunks = [chunk.text for chunks in chunks_per_text for chunk in chunks]
        try:
            vectors = iter(list(self._model.embed(all_chunks, batch_size=BATCH_SIZE)))
        except Exception as error:
            raise EmbeddingError("The embedding model failed") from error

        result = []
        for chunks in chunks_per_text:
            weights = [chunk.token_count for chunk in chunks]
            text_vectors = [next(vectors) for _ in chunks]
            result.append(combine_vectors(text_vectors, weights).tolist())
        return result
