# Import asynccontextmanager to manage application startup and shutdown.
from contextlib import asynccontextmanager

# Import FastAPI to create our backend application.
from fastapi import FastAPI

# Import application settings from the configuration file.
from app.core.config import settings

# Import the main API router containing all version 1 endpoints.
from app.api.v1.router import api_router


# ==========================================================
# APPLICATION LIFESPAN
# ==========================================================

# This function manages application startup and shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Database tables are managed through Alembic migrations.
    # We no longer create tables automatically during startup.

    # Keep the application running.
    yield

    # Future cleanup operations can be added after yield.


# ==========================================================
# CREATE FASTAPI APPLICATION
# ==========================================================

# Initialize the main FastAPI application.
app = FastAPI(

    # Application name comes from the configuration file.
    title=settings.app_name,

    # Current application version.
    version="0.1.0",

    # Description displayed in Swagger documentation.
    description=(
        "SupportIQ is an AI-powered customer support "
        "platform for ticket management and assistance."
    ),

    # Enable or disable debugging based on application settings.
    debug=settings.debug,

    # Register the startup and shutdown lifecycle function.
    lifespan=lifespan,
)


# ==========================================================
# ROOT ENDPOINT
# ==========================================================

# This endpoint checks whether the application is running.
@app.get("/", tags=["General"])
def root():

    return {
        "message": "Welcome to SupportIQ!",
        "application": settings.app_name,
        "status": "running",
    }


# ==========================================================
# HEALTH CHECK ENDPOINT
# ==========================================================

# This endpoint provides a basic application health status.
@app.get("/health", tags=["Health"])
def health_check():

    return {
        "status": "healthy",
        "application": settings.app_name,
    }


# ==========================================================
# REGISTER API ROUTES
# ==========================================================

# Connect all version 1 API endpoints to the main application.
# Every route inside api_router will receive the /api/v1 prefix.
app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)