# Import the User database model.
from app.models.user import User
from app.models.ticket import Ticket
from app.models.knowledge_document import KnowledgeDocument  # noqa: F401

# Import the Ticket database model.
from app.models.ticket import Ticket

from app.models.knowledge_document import KnowledgeDocument

# Define which models are available when importing this package.
__all__ = [
    "User",
    "Ticket",
    "KnowledgeDocument",
]