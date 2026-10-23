from sqlalchemy.orm import Session

from app.models.user import User
from app.repositories import user_repository
from app.schemas.user import UserCreate


def create_user(db: Session, user_data: UserCreate) -> User:
    existing_user = user_repository.get_user_by_email(
        db,
        user_data.email,
    )

    if existing_user:
        raise ValueError("A user with this email already exists.")

    return user_repository.create_user(
        db=db,
        full_name=user_data.full_name,
        email=str(user_data.email),
        role=user_data.role,
    )


def get_user(db: Session, user_id: int) -> User | None:
    return user_repository.get_user_by_id(db, user_id)


def list_users(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> list[User]:
    return user_repository.get_users(
        db,
        skip=skip,
        limit=limit,
    )
