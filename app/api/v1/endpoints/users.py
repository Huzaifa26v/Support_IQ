# Import Annotated to combine Python types with FastAPI dependencies.
from typing import Annotated

# Import FastAPI tools for creating routes, handling errors,
# validating query parameters, and defining HTTP status codes.
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

# Import SQLAlchemy Session to communicate with the database.
from sqlalchemy.orm import Session

# Import the database dependency that provides a database session.
from app.db.session import get_db

# Import Pydantic schemas for validating user input
# and formatting user responses.
from app.schemas.user import UserCreate, UserResponse

# Import the user service containing our business logic.
from app.services import user_service


# Create an API router for all user-related endpoints.
# prefix="/users" means every route starts with /users.
# tags=["Users"] groups these endpoints in Swagger UI.
router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# Create a reusable database dependency.
# FastAPI will automatically provide a database session
# whenever an endpoint uses DatabaseSession.
DatabaseSession = Annotated[Session, Depends(get_db)]


# ==========================================================
# 1. CREATE USER
# ==========================================================

# POST request creates a new user in the database.
# response_model defines the structure of the returned data.
# status_code=201 indicates successful resource creation.
@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    user_data: UserCreate,  # Receives and validates user information.
    db: DatabaseSession,    # Provides access to the database.
):
    try:
        # Call the service layer to create the user.
        return user_service.create_user(db, user_data)

    except ValueError as exc:
        # Return HTTP 409 if a user with this email already exists.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


# ==========================================================
# 2. GET ALL USERS
# ==========================================================

# GET request retrieves a list of users from the database.
@router.get(
    "/",
    response_model=list[UserResponse],
)
def get_all_users(
    db: DatabaseSession,

    # skip controls how many records should be skipped.
    # It is useful for pagination.
    skip: Annotated[int, Query(ge=0)] = 0,

    # limit controls the maximum number of users returned.
    # Minimum is 1 and maximum is 100.
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    # Call the service layer to retrieve users.
    return user_service.list_users(
        db,
        skip=skip,
        limit=limit,
    )


# ==========================================================
# 3. GET USER BY ID
# ==========================================================

# GET request retrieves one user using their unique ID.
@router.get(
    "/{user_id}",
    response_model=UserResponse,
)
def get_user_by_id(
    user_id: int,       # Receives the user ID from the URL.
    db: DatabaseSession,  # Provides database access.
):
    # Ask the service layer to find the requested user.
    user = user_service.get_user(db, user_id)

    # If the user does not exist, return HTTP 404.
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # Return the requested user.
    return user