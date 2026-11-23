from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from paper_similarity.db.models import EMBEDDING_DIMENSIONS, ArxivPaper, SubmittedPaper


def unit_vector(axis: int) -> list[float]:
    """A vector of length 1 that points along one axis."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[axis] = 1.0
    return vector


def make_arxiv_paper(arxiv_id: str = "2401.12345", axis: int = 0) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=1,
        title="A title",
        abstract="An abstract",
        authors=["First Author", "Second Author"],
        categories=["cs.CL", "cs.AI"],
        primary_category="cs.CL",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2024, 1, 23, tzinfo=UTC),
        revised_at=datetime(2024, 1, 24, tzinfo=UTC),
        content_hash="0" * 64,
        embedding=unit_vector(axis),
        embedding_model="test-model",
    )


def make_submitted_paper(text: str = "Some text") -> SubmittedPaper:
    return SubmittedPaper(text=text, embedding=unit_vector(0), embedding_model="test-model")


def test_arxiv_paper_is_stored_and_read_back(session: Session) -> None:
    session.add(make_arxiv_paper())
    session.commit()
    session.expunge_all()  # forget the Python object, so that it is read from the database

    paper = session.scalars(select(ArxivPaper)).one()

    assert paper.authors == ["First Author", "Second Author"]
    assert paper.published_at == datetime(2024, 1, 23, tzinfo=UTC)
    assert paper.created_at.tzinfo is not None
    assert paper.embedding == unit_vector(0)
    assert paper.id.version == 7


def test_submitted_paper_may_have_no_title(session: Session) -> None:
    session.add(make_submitted_paper())
    session.commit()
    session.expunge_all()

    assert session.scalars(select(SubmittedPaper)).one().title is None


def test_cosine_distance_is_calculated_by_the_database(session: Session) -> None:
    session.add_all(
        [make_arxiv_paper("2401.00001", axis=0), make_arxiv_paper("2401.00002", axis=1)]
    )
    session.commit()

    distance = ArxivPaper.embedding.cosine_distance(unit_vector(0))
    rows = session.execute(select(ArxivPaper.arxiv_id, distance).order_by(distance)).all()

    assert [(row[0], row[1]) for row in rows] == [("2401.00001", 0.0), ("2401.00002", 1.0)]


def test_duplicate_arxiv_id_is_rejected(session: Session) -> None:
    session.add_all([make_arxiv_paper(), make_arxiv_paper()])

    with pytest.raises(IntegrityError, match="uq_arxiv_papers_arxiv_id"):
        session.commit()


def test_wrong_embedding_size_is_rejected(session: Session) -> None:
    paper = make_submitted_paper()
    paper.embedding = [1.0, 2.0]
    session.add(paper)

    with pytest.raises(StatementError, match="expected 384 dimensions"):
        session.commit()


def test_zero_embedding_is_rejected(session: Session) -> None:
    paper = make_submitted_paper()
    paper.embedding = [0.0] * EMBEDDING_DIMENSIONS
    session.add(paper)

    with pytest.raises(IntegrityError, match="embedding_not_zero"):
        session.commit()


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_submitted_text_is_rejected(session: Session, text: str) -> None:
    session.add(make_submitted_paper(text=text))

    with pytest.raises(IntegrityError, match="text_not_blank"):
        session.commit()


def test_query_parameters_are_hidden_in_error_messages(session: Session) -> None:
    paper = make_submitted_paper(text="confidential manuscript")
    paper.embedding = [1.0, 2.0]
    session.add(paper)

    with pytest.raises(StatementError) as error:
        session.commit()

    assert "confidential" not in str(error.value)
