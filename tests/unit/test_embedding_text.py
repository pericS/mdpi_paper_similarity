import pytest

from paper_similarity.embeddings.text import build_embedding_text


@pytest.mark.parametrize(
    ("title", "body", "expected"),
    [
        ("A title", "An abstract.", "A title\n\nAn abstract."),
        ("A title\n  on two lines", "An\n abstract. ", "A title on two lines\n\nAn abstract."),
        (None, "Only text.", "Only text."),
        ("   ", "Only text.", "Only text."),
        ("Only title", " ", "Only title"),
        (None, " \n ", ""),
    ],
)
def test_title_and_body_are_combined(title: str | None, body: str, expected: str) -> None:
    assert build_embedding_text(title, body) == expected
