# Paper Similarity Service

Finds indexed arXiv papers similar to a submitted manuscript. Work in progress.

## Development

Requires [uv](https://docs.astral.sh/uv/); it provides Python 3.14 if needed.

```bash
uv sync
uv run uvicorn paper_similarity.main:create_app --factory --reload
```

Then open http://127.0.0.1:8000/docs.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
