"""ORM models: the Python description of the database tables."""

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID, uuid7

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import ARRAY, CheckConstraint, DateTime, MetaData, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Must match the embedding model. Another size needs a migration and re-embedding.
EMBEDDING_DIMENSIONS = 384


class Base(DeclarativeBase):
    # Predictable constraint names let migrations refer to them.
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        datetime: DateTime(timezone=True),
        str: Text(),
        list[str]: ARRAY(Text()),
        list[float]: VECTOR(EMBEDDING_DIMENSIONS),
    }


class ArxivPaper(Base):
    """A paper from arXiv. These rows are the searchable index."""

    __tablename__ = "arxiv_papers"
    __table_args__ = (
        CheckConstraint("version >= 1", name="version_positive"),
        CheckConstraint("vector_norm(embedding) > 0", name="embedding_not_zero"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    # Without the version suffix, so every version of a paper maps to one row.
    arxiv_id: Mapped[str] = mapped_column(unique=True)
    version: Mapped[int]
    title: Mapped[str]
    abstract: Mapped[str]
    authors: Mapped[list[str]]
    categories: Mapped[list[str]]
    primary_category: Mapped[str]
    url: Mapped[str]
    # When arXiv published the first version and the current version.
    published_at: Mapped[datetime]
    revised_at: Mapped[datetime]
    # SHA-256 of the embedded text: tells ingestion whether re-embedding is needed.
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]]
    embedding_model: Mapped[str]
    # When this row was created and last changed.
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class SubmittedPaper(Base):
    """A manuscript submitted through the API. Never part of the searchable index."""

    __tablename__ = "submitted_papers"
    __table_args__ = (
        CheckConstraint("vector_norm(embedding) > 0", name="embedding_not_zero"),
        CheckConstraint("text ~ '\\S'", name="text_not_blank"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    title: Mapped[str | None]
    text: Mapped[str]
    embedding: Mapped[list[float]]
    embedding_model: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
