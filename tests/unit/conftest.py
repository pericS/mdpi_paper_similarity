import pytest
from pydantic import SecretStr

from paper_similarity.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings that do not depend on the developer's environment or .env file."""
    return Settings(
        _env_file=None,
        # Nothing listens here: unit tests must not need a database.
        database_url=SecretStr("postgresql+psycopg://user:unused@127.0.0.1:1/unused"),
    )
