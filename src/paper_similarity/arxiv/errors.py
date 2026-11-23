"""Errors raised while reading from arXiv."""


class ArxivError(Exception):
    """Base class: catch this to handle every arXiv problem."""


class ArxivQueryError(ArxivError):
    """arXiv rejected the request. Repeating it would not help."""


class ArxivUnavailableError(ArxivError):
    """arXiv could not be reached, or kept failing after all attempts."""


class ArxivResponseError(ArxivError):
    """arXiv answered with something that is not a usable feed."""
