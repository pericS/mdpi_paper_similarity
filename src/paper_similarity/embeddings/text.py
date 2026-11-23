"""Builds the text that is embedded, in the same way for arXiv and submitted papers."""


def build_embedding_text(title: str | None, body: str) -> str:
    """Join title and body (abstract or manuscript), collapsing all whitespace.

    arXiv wraps titles and abstracts over several lines. Collapsing whitespace makes
    the result independent of that formatting.
    """
    parts = [" ".join((title or "").split()), " ".join(body.split())]
    return "\n\n".join(part for part in parts if part)
