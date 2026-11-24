from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, func, select
from typer.testing import CliRunner, Result

from paper_similarity import cli
from paper_similarity.arxiv.errors import ArxivQueryError, ArxivUnavailableError
from paper_similarity.arxiv.feed import ArxivPage
from paper_similarity.config import Settings, get_settings
from paper_similarity.db.locks import INGESTION_LOCK_KEY, advisory_lock
from paper_similarity.db.models import ArxivPaper
from tests.fakes import FakeEmbedder, FakePaperSource, make_page, make_record


@pytest.fixture
def settings(database_url: str) -> Settings:
    return Settings(_env_file=None, database_url=SecretStr(database_url))


@pytest.fixture(autouse=True)
def replace_expensive_parts(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "FastEmbedEmbedder", lambda cache_dir: FakeEmbedder())


def run_ingest(
    monkeypatch: pytest.MonkeyPatch, *pages: ArxivPage | Exception, arguments: tuple[str, ...] = ()
) -> Result:
    monkeypatch.setattr(cli, "ArxivClient", lambda http_client: FakePaperSource(*pages))
    return CliRunner().invoke(cli.app, ["ingest", "--query", "cat:cs.CL", *arguments])


def count_papers(engine: Engine) -> int:
    with engine.connect() as connection:
        return connection.scalar(select(func.count()).select_from(ArxivPaper)) or 0


def test_ingest_saves_papers_and_prints_the_result(
    monkeypatch: pytest.MonkeyPatch, committed_database: Engine
) -> None:
    page = make_page(make_record("2401.00001"), make_record("2401.00002"), skipped=1)

    first = run_ingest(monkeypatch, page)
    second = run_ingest(monkeypatch, page)

    assert first.exit_code == 0
    assert "Result: 2 created, 0 updated, 0 unchanged, 1 skipped" in first.output
    assert second.exit_code == 0
    assert "Result: 0 created, 0 updated, 2 unchanged, 1 skipped" in second.output
    assert count_papers(committed_database) == 2


def test_failure_is_reported_together_with_what_was_saved(
    monkeypatch: pytest.MonkeyPatch, committed_database: Engine
) -> None:
    result = run_ingest(monkeypatch, make_page(make_record()), ArxivUnavailableError("no answer"))

    assert result.exit_code == 1
    assert "Error: arXiv: no answer" in result.output
    assert "Result: 1 created" in result.output
    assert count_papers(committed_database) == 1


def test_rejected_query_shows_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_ingest(monkeypatch, ArxivQueryError("Invalid query string"))

    assert result.exit_code == 1
    assert "Invalid query string" in result.output


@pytest.mark.parametrize("page", [make_page(), make_page(skipped=5)])
def test_no_usable_papers_is_a_failure(monkeypatch: pytest.MonkeyPatch, page: ArxivPage) -> None:
    result = run_ingest(monkeypatch, page)

    assert result.exit_code == 1
    assert "No usable papers were found" in result.output


def test_second_ingestion_at_the_same_time_is_refused(
    monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    with advisory_lock(engine, INGESTION_LOCK_KEY):
        result = run_ingest(monkeypatch, make_page(make_record()))

    assert result.exit_code == 1
    assert "already running" in result.output
    assert count_papers(engine) == 0


def test_unreachable_database_is_reported_without_the_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None, database_url=SecretStr("postgresql+psycopg://u:top-secret@127.0.0.1:1/x")
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    result = run_ingest(monkeypatch, make_page(make_record()))

    assert result.exit_code == 1
    assert "Is it running" in result.output
    assert "top-secret" not in result.output


def test_missing_tables_are_explained(
    monkeypatch: pytest.MonkeyPatch, empty_database_engine: Engine
) -> None:
    url = empty_database_engine.url.render_as_string(hide_password=False)
    settings = Settings(_env_file=None, database_url=SecretStr(url))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    result = run_ingest(monkeypatch, make_page(make_record()))

    assert result.exit_code == 1
    assert "alembic upgrade head" in result.output


def test_invalid_configuration_is_explained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env file here
    monkeypatch.delenv("PAPER_SIMILARITY_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(cli, "get_settings", get_settings)

    result = run_ingest(monkeypatch, make_page(make_record()))

    assert result.exit_code == 1
    assert "Invalid configuration" in result.output
    assert "database_url" in result.output


@pytest.mark.parametrize(
    "arguments", [("--max-results", "0"), ("--max-results", "2001"), ("--query", " ")]
)
def test_invalid_arguments_are_rejected(
    monkeypatch: pytest.MonkeyPatch, arguments: tuple[str, ...]
) -> None:
    result = run_ingest(monkeypatch, make_page(make_record()), arguments=arguments)

    assert result.exit_code == 2
