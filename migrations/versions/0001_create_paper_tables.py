"""create paper tables

Revision ID: 0001
Revises:
Created: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Written out instead of imported: a migration must not change when the models do.
DIMENSIONS = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "arxiv_papers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("arxiv_id", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("authors", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("categories", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("primary_category", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revised_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", VECTOR(DIMENSIONS), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "vector_norm(embedding) > 0",
            name=op.f("ck_arxiv_papers_embedding_not_zero"),
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_arxiv_papers_version_positive")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_arxiv_papers")),
        sa.UniqueConstraint("arxiv_id", name=op.f("uq_arxiv_papers_arxiv_id")),
    )

    op.create_table(
        "submitted_papers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(DIMENSIONS), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "vector_norm(embedding) > 0",
            name=op.f("ck_submitted_papers_embedding_not_zero"),
        ),
        sa.CheckConstraint("text ~ '\\S'", name=op.f("ck_submitted_papers_text_not_blank")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submitted_papers")),
    )


def downgrade() -> None:
    op.drop_table("submitted_papers")
    op.drop_table("arxiv_papers")
    # The vector extension stays: it may have existed before this migration.
