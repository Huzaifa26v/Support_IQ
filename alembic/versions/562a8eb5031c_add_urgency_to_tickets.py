"""add urgency to tickets

Revision ID: 562a8eb5031c
Revises: 9a6756006f87
Create Date: 2026-11-01 18:56:35.460983
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Revision identifiers, used by Alembic.
revision: str = "562a8eb5031c"
down_revision: Union[str, Sequence[str], None] = "9a6756006f87"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add urgency column to existing tickets."""

    op.add_column(
        "tickets",
        sa.Column(
            "urgency",
            sa.String(length=20),
            nullable=False,
            server_default="medium",
        ),
    )


def downgrade() -> None:
    """Remove urgency column from tickets."""

    op.drop_column("tickets", "urgency")