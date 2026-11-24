from datetime import UTC, datetime
from pathlib import Path

import pytest

from paper_similarity.arxiv.errors import ArxivQueryError, ArxivResponseError
from paper_similarity.arxiv.feed import parse_feed, parse_id_url

# Real responses, saved from the arXiv API.
FIXTURES = Path(__file__).parents[1] / "fixtures" / "arxiv"


def read_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_real_response_is_parsed() -> None:
    page = parse_feed(read_fixture("two_papers.xml"))

    assert page.total_results == 125980
    assert page.skipped == 0
    first, second = page.records
    assert first.arxiv_id == "2611.27973"
    assert first.version == 1
    assert first.title == "MemOnce: A Scalable Memory Architecture for Large Language Models"
    assert first.abstract.startswith("Deploying large language models")
    assert first.authors[:2] == ("Chang Nie", "Chaoyou Fu")
    assert first.url == "https://arxiv.org/abs/2611.27973"
    assert first.published_at == datetime(2026, 11, 20, 18, 51, 29, tzinfo=UTC)
    assert second.primary_category == "cs.SE"
    assert second.categories == ("cs.SE", "cs.AI", "cs.CL")


def test_old_and_new_ids_are_parsed() -> None:
    page = parse_feed(read_fixture("old_and_new_ids.xml"))

    assert [(record.arxiv_id, record.version) for record in page.records] == [
        ("2401.12345", 1),
        ("hep-th/9901001", 3),
    ]


def test_no_results_is_not_an_error() -> None:
    page = parse_feed(read_fixture("no_results.xml"))

    assert (page.total_results, page.records, page.skipped) == (0, (), 0)


def test_error_entry_raises_with_the_message_from_arxiv() -> None:
    with pytest.raises(ArxivQueryError, match="Invalid query string"):
        parse_feed(read_fixture("invalid_query.xml"))


@pytest.mark.parametrize(
    ("original", "broken"),
    [
        (b"http://arxiv.org/abs/2611.27973v1", b"http://arxiv.org/abs/unexpected"),
        (b"<title>MemOnce", b"<other>MemOnce"),
        (b'<arxiv:primary_category term="cs.CL"/>', b""),
        (b"<published>2026-11-20T18:51:29Z", b"<published>yesterday"),
        (b"<published>2026-11-20T18:51:29Z", b"<published>2026-11-20T18:51:29"),
    ],
)
def test_one_unusable_entry_is_skipped(original: bytes, broken: bytes) -> None:
    content = read_fixture("two_papers.xml")
    assert original in content
    content = content.replace(original, broken, 1)
    if broken.startswith(b"<other>"):
        content = content.replace(b"Large Language Models</title>", b"Models</other>", 1)

    page = parse_feed(content)

    assert [record.arxiv_id for record in page.records] == ["2611.27895"]
    assert page.skipped == 1


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"<html><body>Service unavailable</body></html>",
        b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>",  # no result count
        read_fixture("two_papers.xml")[:2000],  # incomplete
        b'<?xml version="1.0" encoding="unknown-1"?><feed/>',  # LookupError
        b'<?xml version="1.0" encoding="utf-32"?><feed/>',  # ValueError
    ],
)
def test_unusable_response_raises(content: bytes) -> None:
    with pytest.raises(ArxivResponseError):
        parse_feed(content)


@pytest.mark.parametrize(
    "doctype",
    [
        b'<!DOCTYPE feed [<!ENTITY a "lol">]>',  # the start of a "billion laughs" attack
        b'<!DOCTYPE feed SYSTEM "http://example.org/feed.dtd">',  # an external file
    ],
)
def test_xml_with_a_document_type_is_rejected(doctype: bytes) -> None:
    content = read_fixture("no_results.xml")
    start_of_feed = content.index(b"<feed")

    with pytest.raises(ArxivResponseError, match="invalid XML"):
        parse_feed(doctype + content[start_of_feed:])


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://arxiv.org/abs/2401.12345v2", ("2401.12345", 2)),
        ("https://arxiv.org/abs/0704.0001v12", ("0704.0001", 12)),
        ("http://arxiv.org/abs/hep-th/9901001v3", ("hep-th/9901001", 3)),
        ("http://arxiv.org/abs/math.GT/0309136v1", ("math/0309136", 1)),
    ],
)
def test_id_is_split_into_paper_and_version(url: str, expected: tuple[str, int]) -> None:
    assert parse_id_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://arxiv.org/abs/2401.12345",  # no version
        "http://arxiv.org/abs/2401.12345v0",
        "http://arxiv.org/abs/2401.123v1",
        "http://example.org/abs/2401.12345v1",
        "http://arxiv.org/abs/2401.12345v1/extra",
        "",
    ],
)
def test_unexpected_id_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="unexpected arXiv ID"):
        parse_id_url(url)
