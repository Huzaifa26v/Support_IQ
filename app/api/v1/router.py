# Import APIRouter to combine multiple API routes.
from fastapi import APIRouter

# Import the User management router.
from app.api.v1.endpoints.users import router as users_router

# Import the Ticket management router.
from app.api.v1.endpoints.tickets import router as tickets_router


# ==========================================================
# MAIN API ROUTER
# ==========================================================

# Create the main router for API version 1.
api_router = APIRouter()


# ==========================================================
# REGISTER USER ENDPOINTS
# ==========================================================

# Include all user-management endpoints.
api_router.include_router(users_router)


# ==========================================================
# REGISTER TICKET ENDPOINTS
# ==========================================================

# Include all ticket-management endpoints.
api_router.include_router(tickets_router)