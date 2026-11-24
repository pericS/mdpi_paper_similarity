# Paper Similarity Service

A FastAPI service that indexes arXiv papers with vector embeddings, accepts a manuscript, and
returns the most similar indexed papers. It runs locally and needs no paid service or API key.

**Contents:** [Quick start](#quick-start) · [API](#api) · [Ingestion](#ingestion) ·
[Configuration](#configuration) · [Development and tests](#development-and-tests) ·
[Architecture](#architecture) · [Decisions](#technical-decisions-and-trade-offs) ·
[Scores](#how-to-interpret-scores) · [Limitations](#known-limitations) · [AI tools](#use-of-ai-tools)

## Quick start

**Prerequisites:** Docker with Compose, and internet access for the first build and for arXiv.
The first build takes about two minutes and the image needs about 720 MB.

```bash
cp .env.example .env
docker compose up --build -d --wait
docker compose run --rm ingest --query "cat:cs.CL AND abs:retrieval" --max-results 50
```

Compose starts PostgreSQL, applies the migrations and starts the API. The image already contains
the embedding model. Generated API documentation: http://127.0.0.1:8000/docs.

If a port is taken, set `API_PORT` or `DB_PORT` in `.env`. Stop with `docker compose down`, which
keeps all data; `docker compose down -v` deletes it.

## API

**Submit a paper** (the title is optional):

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/papers \
  -H "Content-Type: application/json" \
  -d '{"title": "Protein structure", "text": "We predict three-dimensional protein structures and binding sites from amino acid sequences."}'
```

```json
{"id": "01a20ebf-2443-7387-9926-06ab2f05e58a", "status": "created"}
```

**Retrieve similar papers** with the returned ID (`limit` is 1 to 100, 10 by default):

```bash
curl -s "http://127.0.0.1:8000/api/v1/posts/<id>/similar?limit=5"
```

```json
{
  "paper_id": "01a20ebf-2443-7387-9926-06ab2f05e58a",
  "embedding_model": "BAAI/bge-small-en-v1.5@aa8f8b060edb",
  "matches": [
    {
      "arxiv_id": "2610.08937",
      "title": "A Shortcut to Structure in AlphaFold 3",
      "authors": ["Jonathan Feldman", "Jeffrey Skolnick"],
      "primary_category": "q-bio.BM",
      "published_at": "2026-10-06T18:04:17Z",
      "url": "https://arxiv.org/abs/2610.08937",
      "score": 0.831
    }
  ]
}
```

Results depend on the ingested papers; this example needs `--query "cat:q-bio.BM"`. Further matches
are left out and the score is shortened.

**Health:** `GET /health/live` (the process runs) and `GET /health/ready` (the database is usable).

| Status | When |
|---|---|
| 200, 201 | Success. No matches is `200` with an empty list |
| 400, 413 | Unreadable body; body over 2 MiB |
| 404, 405 | Unknown paper or path; wrong method |
| 422 | Invalid field, limit or ID; text too long |
| 503 | Database or embedding model unavailable, or not ready |
| 500 | Unexpected error |

All errors have one form. The `request_id` is also in the `X-Request-ID` header and in the log lines
of that request:

```json
{"error": {"code": "invalid_request", "message": "The request is invalid",
  "details": [{"field": "body.text", "message": "Field required"}], "request_id": "a6dbdb20..."}}
```

**Assumptions about the specification**

- It creates papers at `/papers` but retrieves similar ones at `/posts/{id}/similar`. Both paths are
  implemented as written, and `{id}` is the ID of the submitted paper. In a real project I would
  confirm the naming and prefer `/papers/{id}/similar`.
- Every valid `POST` creates a new paper, also for a text that was sent before.

## Ingestion

```bash
# With Docker
docker compose run --rm ingest --query "cat:cs.CL AND abs:retrieval" --max-results 100
# Without Docker
uv run paper-similarity ingest --query "cat:cs.CL AND abs:retrieval" --max-results 100
```

`--query` uses the [arXiv search syntax](https://info.arxiv.org/help/api/user-manual.html#query_details),
for example `cat:astro-ph.GA` or `ti:"graph neural network"`. `--max-results` is 1 to 2000 (100 by
default), newest papers first.

- **Repeatable:** a paper is identified by its arXiv ID without version. Only new or changed text
  is embedded again, and an older version never replaces a newer one.
- **Failures:** each page of up to 100 papers is one transaction, so earlier pages stay saved.
  Timeouts, `429`, `5xx` and broken responses are retried with growing delays. A rejected query is
  not retried.
- **Polite:** one connection, at least 3 seconds between requests, one ingestion at a time.
- **Exit code:** 0 success, 1 failure or no usable papers, 2 wrong arguments.

It is a command because it is a batch job that depends on an external service: it can be repeated
and scheduled, and problems with arXiv do not affect the API.

## Configuration

Environment variables or `.env` (see `.env.example`). Wrong values stop the start.

| Variable | Default | Meaning |
|---|---|---|
| `PAPER_SIMILARITY_DATABASE_URL` | required | `postgresql+psycopg://...`; Compose sets it for containers |
| `PAPER_SIMILARITY_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `PAPER_SIMILARITY_MAX_TEXT_TOKENS` | `20000` | Longest text that is embedded |
| `PAPER_SIMILARITY_EMBEDDING_CACHE_DIR` | `~/.cache/paper-similarity` | Model files (set in the image) |
| `POSTGRES_PASSWORD` | required by Compose | Used in a URL: letters and digits are simplest |
| `API_PORT`, `DB_PORT` | `8000`, `5432` | Published by Compose on 127.0.0.1 only |

## Development and tests

Needs [uv](https://docs.astral.sh/uv/), which provides Python 3.14, and Docker for the database.

```bash
cp .env.example .env
uv sync
docker compose up -d --wait db
uv run alembic upgrade head
uv run uvicorn paper_similarity.main:create_app --factory --reload
```

If you change `DB_PORT`, change the port in `PAPER_SIMILARITY_DATABASE_URL` too. Stop the API container
(`docker compose stop api`) before using port 8000 here.

The model (about 65 MB) is downloaded on first use and then works offline.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest                        # needs Docker, starts its own PostgreSQL
uv run pytest -m "not integration"   # fast, without Docker
uv run pytest --run-model            # also tests the real embedding model
```

**Test strategy.** Tests follow risk: duplicates, ranking, failures of arXiv, the model and the
database, invalid input, and confidential text in logs. Most run against a real PostgreSQL in a
container, because upserts, constraints and vector search cannot be tested meaningfully without it.
arXiv is simulated, partly with saved real responses, and the model is replaced by a small
deterministic fake. During development, code was broken on purpose to check that a test fails.

**Test limitations.** The real model is tested only with `--run-model`. Real `429` answers and
timeouts from arXiv, load, and ranking against expert judgement are not covered.

## Architecture

```
arXiv API ──> ArxivClient ──> IngestionService ──┐
                                 │               ├──> repositories ──> PostgreSQL + pgvector
HTTP ──> FastAPI routes ──> Submission- and ─────┘                     arxiv_papers (index)
                            SimilarityService                          submitted_papers
                                 │
                              Embedder (bge-small-en-v1.5, local)
```

| Package in `src/paper_similarity` | Responsibility |
|---|---|
| `api` | Routes, schemas, errors, middleware |
| `services` | Application rules; know nothing about HTTP |
| `repositories`, `db` | Queries; models, sessions, locks |
| `arxiv` | HTTP client and feed parser |
| `embeddings` | Text preparation, chunking, model |
| `cli.py`, `main.py`, `config.py` | Commands, application, settings |

- **Submit:** validate, embed without holding a database connection, store text and vector in one
  transaction.
- **Search:** read the stored vector and run one query. The model is not used.

## Technical decisions and trade-offs

| Decision | Why | Trade-off and rejected options |
|---|---|---|
| PostgreSQL + pgvector | Paper and vector are saved in one transaction; one system to operate | Needs Docker. A separate vector database must be kept in sync; SQLite fits concurrent writers less well |
| `bge-small-en-v1.5` through FastEmbed (ONNX) | Local, free, CPU only, 384 dimensions, 512 tokens | SPECTER2 is trained on scientific papers and may rank better, but needs a heavier setup. `all-MiniLM-L6-v2` cuts text after 256 tokens. External APIs cost money and receive manuscripts |
| Pinned model revision | The files behind a name can change | Updates are manual. The revision is stored with each vector; only equal models are compared |
| Same text construction for both sides: title, blank line, abstract or text, whitespace collapsed | Otherwise formatting would influence similarity | Authors and categories are not embedded |
| Long text: chunks of up to 510 tokens, average weighted by length | The model silently ignores text after 512 tokens | An average mixes the topics of a long manuscript |
| Cosine similarity, exact search | Finds the true nearest papers, no tuning | Grows with the index: 24 ms for 10,000 and 102 ms for 100,000 papers in a test. HNSW is the next step |
| Two tables | Submitted manuscripts cannot appear in results | Two repeated columns |
| Synchronous code | Embedding is CPU work; FastAPI uses threads | About 40 requests at once per process |
| Limits in characters and tokens | Tokens decide the cost: 100,000 symbols took 17.8 s, English 3.7 s | Very long manuscripts are rejected |
| Migration as a separate step | Several API processes must not migrate at once | One more command without Compose |
| Own arXiv client with an explicit retry loop | Every failure rule is visible and tested | More own code than with the `arxiv` package |

## How to interpret scores

`score` is the cosine similarity, from -1 to 1. Use it to **order** results of one query. It is not a
percentage or a probability. With 468 papers from several fields I measured:

| Pairs of papers | 5% below | Median | 95% below |
|---|---|---|---|
| Different fields | 0.54 | 0.62 | 0.74 |
| Same field | 0.62 | 0.71 | 0.80 |
| A paper and its nearest neighbour | 0.79 | 0.84 | 0.89 |

Unrelated papers therefore score about 0.6. Search always returns something: a pizza recipe got a best
score of 0.527. The nearest paper was from the same field in 82.5% of cases (41.4% by chance). This
shows that the search works, but it is not an expert evaluation.

## Known limitations

- Ranking quality was not compared with another model or with expert judgement. The model is English.
- Long texts share the CPU: with 20 maximum-size texts at once the slowest took 54 s. There is no
  limit on parallel embeddings and no rate limiting.
- No authentication, by scope: whoever has an ID can retrieve its results. Manuscripts are stored
  until deleted manually.
- No minimum score, filters or paging beyond 100 results.
- After a model change, the index must be ingested again and manuscripts submitted again. There is no
  command that re-embeds everything.
- If arXiv results change during ingestion, a paper can rarely be missed. Running it again helps.
- Withdrawn papers are not removed. No metrics or tracing. Tested on Linux x86-64 only.

**Before production** I would add authentication, rate limiting, metrics, backups, separate database
users for migrations and API, and an evaluation with labelled data.

## Use of AI tools

- **Tool:** Claude Code with the model Claude Opus 5.5.
- **Used for:** discussing the design, writing code, tests and documentation, and running checks and
  measurements, in ten steps from an agreed plan.
- **Review:** I discussed and approved the plan, reviewed each step before committing it, and made all
  commits myself.
- **Validation:** strict type checks, linting, automated tests, deliberately broken code, and runs
  against real arXiv, PostgreSQL and the real model. This corrected several mistakes in generated work,
  for example an embedding batch size that used far too much memory, a bad request that caused a `500`,
  a benchmark with unrealistic data, and manuscript text in error logs.
