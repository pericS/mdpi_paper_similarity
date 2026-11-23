import pytest
from pydantic import ValidationError

from paper_similarity.config import LogLevel, Settings

DATABASE_URL = "postgresql+psycopg://user:top-secret@localhost/papers"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_SIMILARITY_DATABASE_URL", DATABASE_URL)
    monkeypatch.delenv("PAPER_SIMILARITY_LOG_LEVEL", raising=False)


def test_log_level_defaults_to_info() -> None:
    assert Settings(_env_file=None).log_level is LogLevel.INFO


def test_log_level_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_SIMILARITY_LOG_LEVEL", "DEBUG")

    assert Settings(_env_file=None).log_level is LogLevel.DEBUG


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_SIMILARITY_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError, match="log_level"):
        Settings(_env_file=None)


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAPER_SIMILARITY_DATABASE_URL")

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)


def test_wrong_database_driver_is_rejected_without_leaking_the_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PAPER_SIMILARITY_DATABASE_URL", "postgresql://user:top-secret@localhost/papers"
    )

    with pytest.raises(ValidationError, match="database_url") as error:
        Settings(_env_file=None)

    assert "top-secret" not in str(error.value)


def test_password_is_hidden_when_settings_are_printed() -> None:
    settings = Settings(_env_file=None)

    assert "top-secret" not in repr(settings)
    assert settings.database_url.get_secret_value() == DATABASE_URL
