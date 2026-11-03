# Export database models from this package.
from app.models.knowledge_document import KnowledgeDocument
from app.models.ticket import Ticket
from app.models.user import User

# Define which models are available when importing this package.
__all__ = [
    "User",
    "Ticket",
    "KnowledgeDocument",
]