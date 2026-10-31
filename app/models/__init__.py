# Import the User database model.
from app.models.user import User

# Import the Ticket database model.
from app.models.ticket import Ticket


# Define which models are available when importing this package.
__all__ = [
    "User",
    "Ticket",
]