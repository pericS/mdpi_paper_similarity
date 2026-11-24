"""Logging setup, shared by the API and the command line."""

import logging

from paper_similarity.config import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level.value,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # It logs every request with its full URL at INFO.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
