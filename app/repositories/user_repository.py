# Import select to construct SQL queries using SQLAlchemy.
from sqlalchemy import select

# Import Session to interact with the database.
from sqlalchemy.orm import Session

# Import our User database model.
from app.models.user import User


# ==========================================================
# 1. GET USER BY EMAIL
# ==========================================================

def get_user_by_email(db: Session, email: str) -> User | None:
    """
    Find a user using their email address.

    Args:
        db: Active database session.
        email: Email address to search for.

    Returns:
        User object if found, otherwise None.
    """

    # Construct a SELECT query to find a matching email.
    statement = select(User).where(User.email == email)

    # Execute the query and return the matching user.
    return db.scalar(statement)


# ==========================================================
# 2. GET USER BY ID
# ==========================================================

def get_user_by_id(db: Session, user_id: int) -> User | None:
    """
    Find a user using their unique database ID.
    """

    # Construct a query that searches by user ID.
    statement = select(User).where(User.id == user_id)

    # Execute the query and return the user if found.
    return db.scalar(statement)


# ==========================================================
# 3. GET ALL USERS
# ==========================================================

def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> list[User]:
    """
    Retrieve multiple users from the database.

    Args:
        db: Active database session.
        skip: Number of records to skip.
        limit: Maximum number of records to retrieve.

    Returns:
        A list of User objects.
    """

    # Construct a query to retrieve users.
    statement = (
        select(User)
        .order_by(User.id)  # Keep results in ID order.
        .offset(skip)       # Skip the requested number of records.
        .limit(limit)       # Restrict the number of returned records.
    )

    # Execute the query and convert the results into a list.
    return list(db.scalars(statement).all())


# ==========================================================
# 4. CREATE NEW USER
# ==========================================================

def create_user(
    db: Session,
    full_name: str,
    email: str,
    role: str,
) -> User:
    """
    Insert a new user into the database.

    Args:
        db: Active database session.
        full_name: User's full name.
        email: User's email address.
        role: User's assigned role.

    Returns:
        The newly created User object.
    """

    # Create a new SQLAlchemy User object.
    user = User(
        full_name=full_name,
        email=email,
        role=role,
    )

    # Add the new user to the database session.
    db.add(user)

    # Save the changes permanently in the database.
    db.commit()

    # Refresh the object to retrieve generated database values,
    # such as the automatically assigned user ID.
    db.refresh(user)

    # Return the newly created user.
    return user