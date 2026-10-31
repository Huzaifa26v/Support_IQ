# Import APIRouter to combine multiple API routes.
from fastapi import APIRouter

# Import the router containing our user-management endpoints.
from app.api.v1.endpoints.users import router as users_router


# Create the main API router.
# This router will contain all version 1 API endpoints.
api_router = APIRouter()


# Register the user-management routes.
# This makes the user endpoints available through the main router.
api_router.include_router(users_router)