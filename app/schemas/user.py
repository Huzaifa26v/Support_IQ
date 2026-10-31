# Import datetime to define the format of the user's creation date.
from datetime import datetime

# BaseModel is used to create Pydantic schemas.
# ConfigDict allows us to configure how Pydantic handles data.
# EmailStr validates email addresses.
# Field allows us to add validation rules.
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ==========================================================
# 1. USER CREATE SCHEMA
# ==========================================================

class UserCreate(BaseModel):
    """
    Schema used when a new user is created.

    It validates the data received from the API request.
    """

    # User's full name must contain between 2 and 100 characters.
    full_name: str = Field(
        min_length=2,
        max_length=100,
    )

    # EmailStr checks whether the supplied email has a valid format.
    email: EmailStr

    # Users can have one of three roles:
    # customer, agent, or admin.
    # The default role is customer.
    role: str = Field(
        default="customer",
        pattern="^(customer|agent|admin)$",
    )


# ==========================================================
# 2. USER RESPONSE SCHEMA
# ==========================================================

class UserResponse(BaseModel):
    """
    Schema used when returning user information through the API.

    It defines which fields the API will send back to the client.
    """

    # Unique database ID assigned to the user.
    id: int

    # User's full name.
    full_name: str

    # User's email address.
    email: EmailStr

    # User's assigned role.
    role: str

    # Indicates whether the user account is active.
    is_active: bool

    # Date and time when the user account was created.
    created_at: datetime

    # Allows Pydantic to read data directly from SQLAlchemy objects.
    # This is useful when returning database records through FastAPI.
    model_config = ConfigDict(
        from_attributes=True,
    )