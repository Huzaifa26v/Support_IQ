# Import SQLAlchemy Session for database communication.
from sqlalchemy.orm import Session

# Import the User database model.
from app.models.user import User

# Import repository functions.
# The repository handles actual database operations.
from app.repositories import user_repository

# Import the UserCreate schema for validated input.
from app.schemas.user import UserCreate


# ==========================================================
# 1. CREATE USER
# ==========================================================

def create_user(db: Session, user_data: UserCreate) -> User:
    """
    Create a new user after checking whether the email exists.

    Args:
        db: Active database session.
        user_data: Validated user information.

    Returns:
        Newly created User object.

    Raises:
        ValueError: If the email is already registered.
    """

    # Check whether another user already has this email.
    existing_user = user_repository.get_user_by_email(
        db,
        str(user_data.email),
    )

    # Prevent duplicate user registrations.
    if existing_user:
        raise ValueError(
            "A user with this email already exists."
        )

    # If the email is unique, ask the repository to create the user.
    return user_repository.create_user(
        db=db,
        full_name=user_data.full_name,
        email=str(user_data.email),
        role=user_data.role,
    )


# ==========================================================
# 2. GET USER BY ID
# ==========================================================

def get_user(db: Session, user_id: int) -> User | None:
    """
    Retrieve a user using their unique ID.
    """

    # Delegate the database search to the repository.
    return user_repository.get_user_by_id(
        db,
        user_id,
    )


# ==========================================================
# 3. GET ALL USERS
# ==========================================================

def list_users(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> list[User]:
    """
    Retrieve a list of users with pagination.
    """

    # Delegate the database retrieval to the repository.
    return user_repository.get_users(
        db,
        skip=skip,
        limit=limit,
    )