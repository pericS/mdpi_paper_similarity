# Paper Similarity Service

Finds indexed arXiv papers similar to a submitted manuscript. Work in progress.

## Development

Requires [uv](https://docs.astral.sh/uv/) (it provides Python 3.14 if needed) and Docker with Compose.

```bash
cp .env.example .env
uv sync
docker compose up -d --wait        # PostgreSQL with pgvector
uv run alembic upgrade head        # create the tables
uv run uvicorn paper_similarity.main:create_app --factory --reload
```

Then open http://127.0.0.1:8000/docs.

`docker compose down` keeps the stored data. `docker compose down -v` deletes it.

The embedding model (about 65 MB) is downloaded on first use and then works offline.
Use `uv run paper-similarity download-model` to download it in advance.

## Ingest arXiv papers

```bash
uv run paper-similarity ingest --query "cat:cs.CL AND abs:retrieval" --max-results 100
```

`--query` uses the [arXiv search syntax](https://info.arxiv.org/help/api/user-manual.html#query_details).
The newest papers are read first. Running the command again processes only changes.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest                        # needs Docker
uv run pytest -m "not integration"   # fast, without Docker
uv run pytest --run-model            # also tests the real embedding model
```
