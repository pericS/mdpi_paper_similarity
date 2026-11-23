"""Polite and fault-tolerant access to the arXiv API."""

import logging
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx2

from paper_similarity.arxiv.errors import (
    ArxivError,
    ArxivQueryError,
    ArxivResponseError,
    ArxivUnavailableError,
)
from paper_similarity.arxiv.feed import ArxivPage, parse_feed

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
MIN_SECONDS_BETWEEN_REQUESTS = 3.0  # required by arXiv's terms of use
MAX_ATTEMPTS = 4
FIRST_RETRY_DELAY_SECONDS = 5.0
MAX_RETRY_DELAY_SECONDS = 120.0
DEFAULT_PAGE_SIZE = 100


def create_http_client() -> httpx2.Client:
    """Create the HTTP client for arXiv. Close it after use, e.g. with `with`."""
    return httpx2.Client(
        headers={"User-Agent": "paper-similarity"},
        timeout=httpx2.Timeout(30.0, connect=10.0),
        limits=httpx2.Limits(max_connections=1),  # required by arXiv's terms of use
    )


class _RetryableError(Exception):
    """A problem that may disappear when the request is repeated."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ArxivClient:
    """Reads search results page by page. Not safe to share between threads."""

    def __init__(
        self,
        http_client: httpx2.Client,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http_client = http_client
        # Tests replace these, so that they do not really wait.
        self._sleep = sleep
        self._clock = clock
        self._last_request_finished: float | None = None

    def iter_pages(
        self, query: str, max_results: int, page_size: int = DEFAULT_PAGE_SIZE
    ) -> Iterator[ArxivPage]:
        """Yield pages, newest papers first, until max_results entries were read.

        Pages are requested one at a time, when the caller asks for the next one.
        """
        if not query.strip():
            raise ValueError("query must not be blank")
        if max_results < 1 or page_size < 1:
            raise ValueError("max_results and page_size must be at least 1")

        start = 0
        while start < max_results:
            page = self._fetch_page(query, start, min(page_size, max_results - start))
            yield page
            # A page may be shorter than requested, so continue after what was received.
            start += page.entry_count
            if start >= page.total_results or page.entry_count == 0:
                return

    def _fetch_page(self, query: str, start: int, max_results: int) -> ArxivPage:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._try_fetch_page(query, start, max_results)
            except _RetryableError as error:
                if attempt == MAX_ATTEMPTS:
                    raise ArxivUnavailableError(
                        f"arXiv failed {MAX_ATTEMPTS} times. Last problem: {error}"
                    ) from error
                delay = _retry_delay(attempt, error)
                logger.warning(
                    "arXiv attempt %d of %d failed (%s), retrying in %.0f s",
                    attempt,
                    MAX_ATTEMPTS,
                    error,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _try_fetch_page(self, query: str, start: int, max_results: int) -> ArxivPage:
        self._wait_for_turn()
        try:
            response = self._http_client.get(
                API_URL,
                params={
                    "search_query": query,
                    "start": start,
                    "max_results": max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
        except httpx2.RequestError as error:
            raise _RetryableError(f"{type(error).__name__}: {error}") from error
        finally:
            self._last_request_finished = self._clock()

        _raise_for_status(response)
        try:
            page = parse_feed(response.content)
        except ArxivResponseError as error:
            # Usually a temporary problem, such as an incomplete response.
            raise _RetryableError(str(error)) from error

        if page.entry_count == 0 and start < page.total_results:
            raise _RetryableError("arXiv returned an empty page although there are more results")
        return page

    def _wait_for_turn(self) -> None:
        if self._last_request_finished is None:
            return
        elapsed = self._clock() - self._last_request_finished
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            self._sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)


def _raise_for_status(response: httpx2.Response) -> None:
    status = response.status_code
    if status == 200:
        return
    if status == 429 or status >= 500:
        raise _RetryableError(f"HTTP {status}", _retry_after_seconds(response))
    if status == 400:
        raise ArxivQueryError(_error_message(response))
    raise ArxivError(f"arXiv returned unexpected HTTP status {status}")


def _error_message(response: httpx2.Response) -> str:
    """Return arXiv's explanation of a rejected request, if there is one."""
    try:
        parse_feed(response.content)
    except ArxivQueryError as error:
        return str(error)
    except ArxivResponseError:
        pass
    return "arXiv rejected the request without an explanation"


def _retry_delay(attempt: int, error: _RetryableError) -> float:
    """Double the delay after each attempt: 5, 10, 20 s; never less than arXiv asks for."""
    delay: float = FIRST_RETRY_DELAY_SECONDS * 2 ** (attempt - 1)
    if error.retry_after is not None:
        delay = max(delay, error.retry_after)
    if delay > MAX_RETRY_DELAY_SECONDS:
        raise ArxivUnavailableError(
            f"arXiv asked to wait {delay:.0f} s. Try again later"
        ) from error
    return delay


def _retry_after_seconds(response: httpx2.Response) -> float | None:
    """Read the Retry-After header, which holds either seconds or a date."""
    value = response.headers.get("Retry-After", "").strip()
    if value.isdecimal():
        return float(value)
    try:
        moment = parsedate_to_datetime(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - datetime.now(UTC)).total_seconds())
