from datetime import datetime

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from paper_similarity.arxiv.errors import ArxivUnavailableError
from paper_similarity.arxiv.feed import ArxivPage
from paper_similarity.db.models import ArxivPaper
from paper_similarity.embeddings.base import EmbeddingError
from paper_similarity.repositories.arxiv_papers import ArxivPaperRepository, EmbeddedPaper
from paper_similarity.services.ingestion import IngestionReport, IngestionService
from tests.fakes import FakeEmbedder, FakePaperSource, make_page, make_record

Sessions = sessionmaker[Session]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def ingest(
    sessions: Sessions, embedder: FakeEmbedder, *pages: ArxivPage | Exception
) -> IngestionReport:
    service = IngestionService(FakePaperSource(*pages), embedder, sessions)
    return sum(service.ingest("cat:cs.CL", 100), IngestionReport())


def stored_papers(sessions: Sessions) -> dict[str, ArxivPaper]:
    with sessions() as session:
        return {paper.arxiv_id: paper for paper in session.scalars(select(ArxivPaper))}


def set_updated_at_to_the_past(sessions: Sessions) -> datetime:
    """now() is the same during the whole test transaction, so make a change visible."""
    past = datetime.fromisoformat("2020-01-01T00:00:00+00:00")
    with sessions.begin() as session:
        session.execute(update(ArxivPaper).values(updated_at=past))
    return past


def test_new_papers_are_stored_with_their_vectors(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    report = ingest(
        session_factory, embedder, make_page(make_record("2401.00001"), make_record("2401.00002"))
    )

    assert report == IngestionReport(created=2)
    papers = stored_papers(session_factory)
    assert papers.keys() == {"2401.00001", "2401.00002"}
    paper = papers["2401.00001"]
    assert paper.authors == ["First Author", "Second Author"]
    assert paper.url == "https://arxiv.org/abs/2401.00001"
    assert paper.embedding == pytest.approx(embedder.embed(["A title\n\nAn abstract"])[0])
    assert paper.embedding_model == "fake-model"
    assert papers["2401.00001"].id != papers["2401.00002"].id


def test_second_run_changes_nothing_and_embeds_nothing(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    page = make_page(make_record("2401.00001"), make_record("2401.00002"))
    ingest(session_factory, embedder, page)
    past = set_updated_at_to_the_past(session_factory)
    embedder.embedded_texts.clear()

    report = ingest(session_factory, embedder, page)

    assert report == IngestionReport(unchanged=2)
    assert embedder.embedded_texts == []
    papers = stored_papers(session_factory)
    assert len(papers) == 2
    assert {paper.updated_at for paper in papers.values()} == {past}


def test_new_version_updates_the_same_row_and_its_vector(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    ingest(session_factory, embedder, make_page(make_record(abstract="old words")))
    before = stored_papers(session_factory)["2401.00001"]
    past = set_updated_at_to_the_past(session_factory)

    report = ingest(
        session_factory, embedder, make_page(make_record(version=2, abstract="other content"))
    )

    assert report == IngestionReport(updated=1)
    after = stored_papers(session_factory)["2401.00001"]
    assert after.id == before.id
    assert after.created_at == before.created_at
    assert after.updated_at > past
    assert (after.version, after.abstract) == (2, "other content")
    assert after.embedding != before.embedding


def test_changed_metadata_is_saved_without_a_new_vector(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    ingest(session_factory, embedder, make_page(make_record()))
    past = set_updated_at_to_the_past(session_factory)
    embedder.embedded_texts.clear()

    report = ingest(
        session_factory, embedder, make_page(make_record(categories=("cs.CL", "cs.AI")))
    )

    assert report == IngestionReport(updated=1)
    assert embedder.embedded_texts == []
    paper = stored_papers(session_factory)["2401.00001"]
    assert paper.categories == ["cs.CL", "cs.AI"]
    assert paper.updated_at > past


def test_older_version_is_ignored(session_factory: Sessions, embedder: FakeEmbedder) -> None:
    ingest(session_factory, embedder, make_page(make_record(version=2, title="New")))

    report = ingest(session_factory, embedder, make_page(make_record(version=1, title="Old")))

    assert report == IngestionReport(unchanged=1)
    assert stored_papers(session_factory)["2401.00001"].title == "New"


def test_another_model_causes_new_vectors(session_factory: Sessions) -> None:
    page = make_page(make_record())
    ingest(session_factory, FakeEmbedder("old-model"), page)

    report = ingest(session_factory, FakeEmbedder("new-model"), page)

    assert report == IngestionReport(updated=1)
    assert stored_papers(session_factory)["2401.00001"].embedding_model == "new-model"


def test_paper_repeated_in_a_page_is_stored_once_in_its_newest_version(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    page = make_page(make_record(version=1), make_record(version=2), make_record(version=1))

    report = ingest(session_factory, embedder, page)

    assert report == IngestionReport(created=1, unchanged=2)
    assert stored_papers(session_factory)["2401.00001"].version == 2


def test_paper_repeated_on_the_next_page_is_stored_once(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    report = ingest(session_factory, embedder, make_page(make_record()), make_page(make_record()))

    assert report == IngestionReport(created=1, unchanged=1)


def test_failure_on_a_later_page_keeps_earlier_pages(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    with pytest.raises(ArxivUnavailableError):
        ingest(
            session_factory,
            embedder,
            make_page(make_record("2401.00001")),
            ArxivUnavailableError("down"),
        )

    assert stored_papers(session_factory).keys() == {"2401.00001"}


def test_failed_embedding_saves_nothing_from_that_page(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    embedder.fail = True

    with pytest.raises(EmbeddingError):
        ingest(session_factory, embedder, make_page(make_record()))

    assert stored_papers(session_factory) == {}


def test_paper_that_cannot_be_embedded_is_skipped(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    page = make_page(
        make_record("2401.00001"),
        make_record("2401.00002", title=" ", abstract=" "),
        skipped=3,
    )

    report = ingest(session_factory, embedder, page)

    assert report == IngestionReport(created=1, skipped=4)
    assert stored_papers(session_factory).keys() == {"2401.00001"}


def test_empty_page_is_not_a_problem(session_factory: Sessions, embedder: FakeEmbedder) -> None:
    assert ingest(session_factory, embedder, make_page()) == IngestionReport()


def test_database_itself_refuses_to_replace_a_newer_version(session_factory: Sessions) -> None:
    """Covers a change by another process, after our comparison with stored papers."""

    def embedded(version: int) -> EmbeddedPaper:
        record = make_record(version=version, title=f"Version {version}")
        vector = [1.0] + [0.0] * 383
        return EmbeddedPaper(record, "0" * 64, vector, "fake-model")

    with session_factory.begin() as session:
        ArxivPaperRepository(session).upsert([embedded(2)])
        ArxivPaperRepository(session).upsert([embedded(1)])

    with session_factory() as session:
        assert session.scalar(select(ArxivPaper.title)) == "Version 2"
        assert session.scalar(select(func.count()).select_from(ArxivPaper)) == 1


def test_metadata_is_not_combined_with_a_vector_of_other_text(
    session_factory: Sessions, embedder: FakeEmbedder
) -> None:
    ingest(session_factory, embedder, make_page(make_record(title="Stored")))

    with session_factory.begin() as session:
        ArxivPaperRepository(session).update_metadata(
            make_record(title="Changed"), content_hash="f" * 64, embedding_model="fake-model"
        )

    assert stored_papers(session_factory)["2401.00001"].title == "Stored"
