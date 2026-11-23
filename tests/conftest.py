import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-model",
        action="store_true",
        help="also run tests that use the real embedding model (downloads about 70 MB once)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-model"):
        return
    skip = pytest.mark.skip(reason="needs --run-model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
