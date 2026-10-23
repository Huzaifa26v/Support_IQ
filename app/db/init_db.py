from app.db.base import Base
from app.db.session import engine

# Import models so SQLAlchemy registers their tables.
from app.models import User  # noqa: F401


def init_db() -> None:
    """Create registered database tables for local development."""
    Base.metadata.create_all(bind=engine)
    