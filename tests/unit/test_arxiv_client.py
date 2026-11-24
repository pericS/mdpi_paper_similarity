from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx2
import pytest

from paper_similarity.arxiv.client import ArxivClient
from paper_similarity.arxiv.errors import ArxivError, ArxivQueryError, ArxivUnavailableError

FIXTURES = Path(__file__).parents[1] / "fixtures" / "arxiv"

Reply = httpx2.Response | Exception


def feed(count: int, total: int, first: int = 1) -> httpx2.Response:
    """Build a response with `count` papers, numbered from `first`."""
    entries = "".join(
        f"""<entry>
          <id>http://arxiv.org/abs/2401.{number:05}v1</id>
          <title>Title {number}</title><summary>Abstract {number}</summary>
          <published>2024-01-01T00:00:00Z</published><updated>2024-01-01T00:00:00Z</updated>
          <author><name>An Author</name></author>
          <category term="cs.CL"/><arxiv:primary_category term="cs.CL"/>
        </entry>"""
        for number in range(first, first + count)
    )
    return httpx2.Response(
        200,
        text=f"""<feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
          <opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>""",
    )


class FakeArxiv:
    """Answers with prepared replies; records requests and waiting, without real time."""

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.requests: list[httpx2.Request] = []
        self.request_times: list[float] = []
        self.sleeps: list[float] = []
        # A new client first waits. Most tests start when that wait is already over.
        self.now = -3.0
        self.client = self.create_client()
        self.now = 0.0

    def create_client(self) -> ArxivClient:
        return ArxivClient(
            httpx2.Client(transport=httpx2.MockTransport(self._answer)),
            sleep=self._sleep,
            clock=lambda: self.now,
        )

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.request_times.append(self.now)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def read_ids(self, max_results: int = 10, page_size: int = 10) -> list[str]:
        pages = self.client.iter_pages("cat:cs.CL", max_results, page_size)
        return [record.arxiv_id for page in pages for record in page.records]

    def parameters(self, name: str) -> list[str]:
        return [request.url.params[name] for request in self.requests]


def test_query_is_sent_to_arxiv() -> None:
    arxiv = FakeArxiv(feed(1, total=1))
    list(arxiv.client.iter_pages('cat:cs.CL AND ti:"deep & wide"', 5))

    url = arxiv.requests[0].url
    assert str(url).startswith("https://export.arxiv.org/api/query?")
    assert url.params["search_query"] == 'cat:cs.CL AND ti:"deep & wide"'
    assert url.params["sortBy"] == "submittedDate"


def test_pages_are_read_until_max_results() -> None:
    arxiv = FakeArxiv(feed(2, total=100, first=1), feed(1, total=100, first=3))

    assert arxiv.read_ids(max_results=3, page_size=2) == ["2401.00001", "2401.00002", "2401.00003"]
    assert arxiv.parameters("start") == ["0", "2"]
    assert arxiv.parameters("max_results") == ["2", "1"]


def test_reading_stops_when_there_are_no_more_results() -> None:
    arxiv = FakeArxiv(feed(2, total=3, first=1), feed(1, total=3, first=3))

    assert len(arxiv.read_ids(max_results=50, page_size=2)) == 3


def test_short_page_does_not_lose_papers() -> None:
    arxiv = FakeArxiv(feed(1, total=3, first=1), feed(2, total=3, first=2))

    assert arxiv.read_ids(page_size=3) == ["2401.00001", "2401.00002", "2401.00003"]
    assert arxiv.parameters("start") == ["0", "1"]


def test_no_results_gives_one_empty_page() -> None:
    arxiv = FakeArxiv(feed(0, total=0))

    assert arxiv.read_ids() == []
    assert len(arxiv.requests) == 1


def test_requests_are_at_least_three_seconds_apart() -> None:
    arxiv = FakeArxiv(*[feed(1, total=3, first=n) for n in (1, 2, 3)])
    arxiv.read_ids(page_size=1)

    assert arxiv.request_times == [0, 3, 6]


def test_next_job_also_waits_three_seconds() -> None:
    arxiv = FakeArxiv(feed(1, total=1), feed(1, total=1))
    arxiv.read_ids()
    arxiv.now += 1  # saving the last page
    arxiv.client = arxiv.create_client()
    arxiv.now += 0.5  # loading the model
    arxiv.read_ids()

    # Only the 0.5 s count: the new client cannot know when the last request was.
    assert arxiv.request_times == [0, 4]
    assert arxiv.sleeps == [2.5]


def test_no_extra_waiting_when_the_caller_was_slow() -> None:
    arxiv = FakeArxiv(feed(1, total=2, first=1), feed(1, total=2, first=2))
    pages = arxiv.client.iter_pages("cat:cs.CL", 10, 1)
    next(pages)
    arxiv.now += 10  # e.g. embedding and saving the first page
    next(pages)

    assert arxiv.sleeps == []


@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ConnectError("refused"),
        httpx2.ReadTimeout("too slow"),
        httpx2.RemoteProtocolError("disconnected"),
        httpx2.Response(429),
        httpx2.Response(500),
        httpx2.Response(503),
        httpx2.Response(200, text="<feed"),
        httpx2.Response(200, content=b'<?xml version="1.0" encoding="unknown-1"?><feed/>'),
        feed(0, total=5),  # empty although there are results
    ],
    ids=[
        "connect",
        "timeout",
        "disconnect",
        "429",
        "500",
        "503",
        "broken-xml",
        "unknown-encoding",
        "empty-page",
    ],
)
def test_temporary_failure_is_retried(failure: Reply) -> None:
    arxiv = FakeArxiv(failure, feed(1, total=1))

    assert arxiv.read_ids() == ["2401.00001"]
    assert arxiv.sleeps == [5]


def test_retries_wait_longer_each_time_and_then_give_up() -> None:
    arxiv = FakeArxiv(*[httpx2.Response(503) for _ in range(4)])

    with pytest.raises(ArxivUnavailableError, match=r"failed 4 times.*HTTP 503"):
        arxiv.read_ids()

    assert arxiv.sleeps == [5, 10, 20]


def test_failure_on_a_later_page_keeps_earlier_pages() -> None:
    arxiv = FakeArxiv(feed(1, total=2), *[httpx2.ReadTimeout("slow") for _ in range(4)])
    pages = arxiv.client.iter_pages("cat:cs.CL", 10, 1)

    assert len(next(pages).records) == 1
    with pytest.raises(ArxivUnavailableError):
        next(pages)


def test_longer_retry_after_is_respected() -> None:
    arxiv = FakeArxiv(httpx2.Response(429, headers={"Retry-After": "42"}), feed(1, total=1))
    arxiv.read_ids()

    assert arxiv.sleeps == [42]


def test_retry_after_as_a_date_is_respected() -> None:
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
    arxiv = FakeArxiv(httpx2.Response(503, headers={"Retry-After": date}), feed(1, total=1))
    arxiv.read_ids()

    assert arxiv.sleeps == [pytest.approx(60, abs=2)]


@pytest.mark.parametrize("value", ["0", "soon", "-5", ""])
def test_useless_retry_after_is_ignored(value: str) -> None:
    arxiv = FakeArxiv(httpx2.Response(503, headers={"Retry-After": value}), feed(1, total=1))
    arxiv.read_ids()

    assert arxiv.sleeps == [5]


def test_very_long_retry_after_gives_up_immediately() -> None:
    arxiv = FakeArxiv(httpx2.Response(429, headers={"Retry-After": "3600"}))

    with pytest.raises(ArxivUnavailableError, match="3600"):
        arxiv.read_ids()

    assert arxiv.sleeps == []


def test_invalid_query_is_not_retried_and_shows_the_reason() -> None:
    content = (FIXTURES / "invalid_query.xml").read_bytes()
    arxiv = FakeArxiv(httpx2.Response(400, content=content))

    with pytest.raises(ArxivQueryError, match="Invalid query string"):
        arxiv.read_ids()

    assert len(arxiv.requests) == 1


@pytest.mark.parametrize(
    "content", [b"Bad request", b'<?xml version="1.0" encoding="unknown-1"?><feed/>']
)
def test_rejected_request_without_explanation(content: bytes) -> None:
    arxiv = FakeArxiv(httpx2.Response(400, content=content))

    with pytest.raises(ArxivQueryError, match="without an explanation"):
        arxiv.read_ids()


@pytest.mark.parametrize("status", [301, 403, 404])
def test_other_statuses_are_not_retried(status: int) -> None:
    arxiv = FakeArxiv(httpx2.Response(status))

    with pytest.raises(ArxivError, match=str(status)):
        arxiv.read_ids()

    assert len(arxiv.requests) == 1


@pytest.mark.parametrize(
    ("query", "max_results", "page_size"), [(" ", 1, 1), ("cat:cs.CL", 0, 1), ("cat:cs.CL", 1, 0)]
)
def test_invalid_arguments_are_rejected(query: str, max_results: int, page_size: int) -> None:
    arxiv = FakeArxiv()

    with pytest.raises(ValueError, match="must"):
        next(arxiv.client.iter_pages(query, max_results, page_size))

    assert arxiv.requests == []
