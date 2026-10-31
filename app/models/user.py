# Import datetime utilities for the account creation timestamp.
from datetime import datetime, timezone

# Import SQLAlchemy column types.
from sqlalchemy import Boolean, DateTime, String

# Import ORM tools for defining columns and relationships.
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Import the shared SQLAlchemy Base class.
from app.db.base import Base


# ==========================================================
# USER DATABASE MODEL
# ==========================================================

class User(Base):
    """
    Database model representing a SupportIQ user.

    A user can be a customer, support agent, or administrator.
    A customer can also have multiple support tickets.
    """

    __tablename__ = "users"

    # ------------------------------------------------------
    # 1. PRIMARY KEY
    # ------------------------------------------------------

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    # ------------------------------------------------------
    # 2. USER INFORMATION
    # ------------------------------------------------------

    full_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    # ------------------------------------------------------
    # 3. USER ROLE
    # ------------------------------------------------------

    role: Mapped[str] = mapped_column(
        String(30),
        default="customer",
        nullable=False,
    )

    # ------------------------------------------------------
    # 4. ACCOUNT STATUS
    # ------------------------------------------------------

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # ------------------------------------------------------
    # 5. USER-TICKET RELATIONSHIP
    # ------------------------------------------------------

    # One user can have multiple support tickets.
    # back_populates connects this relationship to
    # the customer relationship inside the Ticket model.
    #
    # cascade="all, delete-orphan" means tickets belonging
    # to a user are also deleted if that user is deleted.
    tickets: Mapped[list["Ticket"]] = relationship(
        "Ticket",
        back_populates="customer",
        cascade="all, delete-orphan",
    )

    # ------------------------------------------------------
    # 6. TIMESTAMP
    # ------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ------------------------------------------------------
    # 7. OBJECT REPRESENTATION
    # ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<User(id={self.id}, "
            f"email='{self.email}', role='{self.role}')>"
        )