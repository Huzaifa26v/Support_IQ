from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User


def get_user_by_email(db: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    return db.scalar(statement)


def get_user_by_id(db: Session, user_id: int) -> User | None:
    statement = select(User).where(User.id == user_id)
    return db.scalar(statement)


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> list[User]:
    statement = (
        select(User)
        .order_by(User.id)
        .offset(skip)
        .limit(limit)
    )
    return list(db.scalars(statement).all())


def create_user(
    db: Session,
    full_name: str,
    email: str,
    role: str,
) -> User:
    user = User(
        full_name=full_name,
        email=email,
        role=role,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user
