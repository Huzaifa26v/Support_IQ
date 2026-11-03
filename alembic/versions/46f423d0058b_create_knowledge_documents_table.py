"""create knowledge documents table

Revision ID: 46f423d0058b
Revises: 562a8eb5031c
Create Date: 2026-11-03 18:45:55.191779

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '46f423d0058b'
down_revision: str | Sequence[str] | None = '562a8eb5031c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    """Upgrade schema."""
    # Create the knowledge_documents table.
    op.create_table(
        'knowledge_documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=20), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('checksum', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for document checksum and ID.
    op.create_index(
        op.f('ix_knowledge_documents_checksum'),
        'knowledge_documents',
        ['checksum'],
        unique=True,
    )
    op.create_index(
        op.f('ix_knowledge_documents_id'),
        'knowledge_documents',
        ['id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove indexes and table in reverse order.
    op.drop_index(
        op.f('ix_knowledge_documents_id'),
        table_name='knowledge_documents',
    )
    op.drop_index(
        op.f('ix_knowledge_documents_checksum'),
        table_name='knowledge_documents',
    )
    op.drop_table('knowledge_documents')