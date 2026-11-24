import pytest
from sqlalchemy import Engine

from paper_similarity.db.locks import LockNotAvailableError, advisory_lock

KEY = 1


def test_lock_can_be_held_only_once(engine: Engine) -> None:
    with (
        advisory_lock(engine, KEY),
        pytest.raises(LockNotAvailableError),
        advisory_lock(engine, KEY),
    ):
        pass


def test_different_locks_are_independent(engine: Engine) -> None:
    with advisory_lock(engine, KEY), advisory_lock(engine, KEY + 1):
        pass


def test_lock_is_released_after_an_error(engine: Engine) -> None:
    with pytest.raises(RuntimeError), advisory_lock(engine, KEY):
        raise RuntimeError

    with advisory_lock(engine, KEY):
        pass
