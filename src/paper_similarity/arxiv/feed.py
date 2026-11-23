"""Converts arXiv's Atom XML into Python objects."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from xml.etree.ElementTree import Element, ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import fromstring

from paper_similarity.arxiv.errors import ArxivQueryError, ArxivResponseError

logger = logging.getLogger(__name__)

NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
FEED_TAG = f"{{{NAMESPACES['atom']}}}feed"

# New style: 2401.12345v2. Old style: hep-th/9901001v3, sometimes math.GT/0309136v1.
ARXIV_ID_URL = re.compile(
    r"https?://arxiv\.org/abs/"
    r"(?:(?P<new>\d{4}\.\d{4,5})|(?P<archive>[a-z-]+)(?:\.[A-Za-z-]+)?/(?P<number>\d{7}))"
    r"v(?P<version>[1-9]\d*)"
)
ERROR_ID = re.compile(r"https?://arxiv\.org/api/errors")


@dataclass(frozen=True)
class ArxivRecord:
    """Metadata of one paper, as provided by arXiv."""

    arxiv_id: str  # without the version
    version: int
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str
    published_at: datetime
    revised_at: datetime

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


@dataclass(frozen=True)
class ArxivPage:
    """One page of search results."""

    total_results: int  # for the whole query, not only this page
    records: tuple[ArxivRecord, ...]
    skipped: int  # entries that could not be used

    @property
    def entry_count(self) -> int:
        return len(self.records) + self.skipped


def parse_feed(content: bytes) -> ArxivPage:
    """Parse a response. One unusable entry is skipped; an unusable feed raises."""
    try:
        feed = fromstring(content, forbid_dtd=True)
    except (ParseError, DefusedXmlException) as error:
        raise ArxivResponseError("arXiv returned invalid XML") from error
    if feed.tag != FEED_TAG:
        raise ArxivResponseError("arXiv returned XML that is not an Atom feed")

    entries = feed.findall("atom:entry", NAMESPACES)
    for entry in entries:
        _raise_if_error_entry(entry)

    records = []
    skipped = 0
    for entry in entries:
        try:
            records.append(_parse_entry(entry))
        except ValueError as error:
            skipped += 1
            entry_id = entry.findtext("atom:id", "(no id)", NAMESPACES)
            logger.warning("Skipping arXiv entry %r: %s", entry_id, error)
    return ArxivPage(_total_results(feed), tuple(records), skipped)


def _total_results(feed: Element) -> int:
    try:
        return int(_text(feed, "opensearch:totalResults"))
    except ValueError as error:
        raise ArxivResponseError("arXiv returned a feed without a result count") from error


def _raise_if_error_entry(entry: Element) -> None:
    """arXiv reports errors as a feed with one entry that describes the problem."""
    if ERROR_ID.match(entry.findtext("atom:id", "", NAMESPACES)):
        message = entry.findtext("atom:summary", "", NAMESPACES).strip()
        raise ArxivQueryError(message or "arXiv reported an unknown error")


def _parse_entry(entry: Element) -> ArxivRecord:
    arxiv_id, version = parse_id_url(_text(entry, "atom:id"))
    authors = entry.findall("atom:author", NAMESPACES)
    categories = entry.findall("atom:category", NAMESPACES)
    return ArxivRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=_text(entry, "atom:title"),
        abstract=_text(entry, "atom:summary"),
        authors=tuple(_text(author, "atom:name") for author in authors),
        categories=tuple(_term(category) for category in categories),
        primary_category=_term(entry.find("arxiv:primary_category", NAMESPACES)),
        published_at=_timestamp(entry, "atom:published"),
        revised_at=_timestamp(entry, "atom:updated"),
    )


def parse_id_url(url: str) -> tuple[str, int]:
    """Split http://arxiv.org/abs/2401.12345v2 into ("2401.12345", 2)."""
    match = ARXIV_ID_URL.fullmatch(url)
    if match is None:
        raise ValueError(f"unexpected arXiv ID {url!r}")
    arxiv_id = match["new"] or f"{match['archive']}/{match['number']}"
    return arxiv_id, int(match["version"])


def _text(parent: Element, path: str) -> str:
    """Return the required text of a child element, with whitespace collapsed."""
    text = " ".join(parent.findtext(path, "", NAMESPACES).split())
    if not text:
        raise ValueError(f"{path} is missing")
    return text


def _term(category: Element | None) -> str:
    term = category.get("term", "").strip() if category is not None else ""
    if not term:
        raise ValueError("a category is missing")
    return term


def _timestamp(entry: Element, path: str) -> datetime:
    timestamp = datetime.fromisoformat(_text(entry, path))
    if timestamp.tzinfo is None:
        raise ValueError(f"{path} has no time zone")
    return timestamp
