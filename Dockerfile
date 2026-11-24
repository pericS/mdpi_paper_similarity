# Stage 1: install the locked dependencies and the application.
FROM python:3.14-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies change rarely, so this layer is reused when only our code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
COPY src ./src
# Not editable: the package is really installed, which also tests the packaging.
RUN uv sync --locked --no-editable

# Stage 2: download the pinned embedding model, so containers start without a network.
FROM builder AS model
RUN /app/.venv/bin/python -c "from pathlib import Path; \
from paper_similarity.embeddings.fastembed_embedder import download_model; \
download_model(Path('/opt/model'))" \
    # The application runs as another user, who must be able to read the files.
    && chmod -R a+rX /opt/model

# Stage 3: the final image, without uv and build files.
FROM python:3.14-slim-trixie
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=model /opt/model /opt/model
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    PAPER_SIMILARITY_EMBEDDING_CACHE_DIR=/opt/model
# By number, so that platforms can verify that it is not root.
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/health/live', timeout=4)"]
CMD ["uvicorn", "paper_similarity.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
