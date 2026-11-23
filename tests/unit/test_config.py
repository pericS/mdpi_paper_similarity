import pytest
from pydantic import ValidationError

from paper_similarity.config import LogLevel, Settings


def test_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAPER_SIMILARITY_LOG_LEVEL", raising=False)

    assert Settings(_env_file=None).log_level is LogLevel.INFO


def test_log_level_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_SIMILARITY_LOG_LEVEL", "DEBUG")

    assert Settings(_env_file=None).log_level is LogLevel.DEBUG


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_SIMILARITY_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError, match="log_level"):
        Settings(_env_file=None)
