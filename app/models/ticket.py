# Import datetime utilities to store ticket creation and update times.
from datetime import datetime, timezone

# Import SQLAlchemy column types.
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
)

# Import ORM tools for defining columns and relationships.
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Import the shared SQLAlchemy Base class.
from app.db.base import Base


# ==========================================================
# TICKET DATABASE MODEL
# ==========================================================

class Ticket(Base):
    """
    Database model representing a customer support ticket.

    Each ticket belongs to a customer and contains information
    about the customer's support request.
    """

    # Define the database table name.
    __tablename__ = "tickets"

    # ------------------------------------------------------
    # 1. PRIMARY KEY
    # ------------------------------------------------------

    # Every ticket receives a unique ID.
    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    # ------------------------------------------------------
    # 2. TICKET INFORMATION
    # ------------------------------------------------------

    # Short title describing the customer's issue.
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    # Detailed explanation of the customer's problem.
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ML-predicted category of the customer support ticket.
    # The default value supports existing tickets.
    category: Mapped[str] = mapped_column(
        String(50),
        default="other",
        nullable=False,
    )

    # ------------------------------------------------------
    # 3. TICKET STATUS
    # ------------------------------------------------------

    # New tickets are created with the status "open".
    status: Mapped[str] = mapped_column(
        String(30),
        default="open",
        nullable=False,
    )

    # ------------------------------------------------------
    # 4. TICKET PRIORITY
    # ------------------------------------------------------

    # Default priority is "medium".
    priority: Mapped[str] = mapped_column(
        String(20),
        default="medium",
        nullable=False,
    )

    # ------------------------------------------------------
    # 5. CUSTOMER RELATIONSHIP
    # ------------------------------------------------------

    # Foreign key connects each ticket to a customer in users.
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    # Establish an ORM relationship with the User model.
    # This allows access through ticket.customer.
    customer = relationship(
        "User",
        back_populates="tickets",
    )

    # ------------------------------------------------------
    # 6. TIMESTAMPS
    # ------------------------------------------------------

    # Automatically record when the ticket is created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Automatically record when the ticket is updated.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ------------------------------------------------------
    # 7. OBJECT REPRESENTATION
    # ------------------------------------------------------

    def __repr__(self) -> str:
        """Return a readable representation of a ticket."""

        return (
            f"<Ticket(id={self.id}, "
            f"title='{self.title}', "
            f"status='{self.status}')>"
        )