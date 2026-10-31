# Import the shared SQLAlchemy Base class.
from app.db.base import Base

# Import the database engine.
from app.db.session import engine

# Import both models.
# This ensures SQLAlchemy registers their tables
# before creating the database schema.
from app.models import User, Ticket  # noqa: F401


# ==========================================================
# DATABASE INITIALIZATION
# ==========================================================

def init_db() -> None:
    """
    Initialize the database by creating all registered tables.

    This is suitable for our current development setup.
    We will introduce Alembic migrations later.
    """

    # Create any tables that do not already exist.
    Base.metadata.create_all(bind=engine)