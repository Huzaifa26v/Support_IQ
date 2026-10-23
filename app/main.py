
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.v1.router import api_router

from app.core.config import settings
from app.db.init_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database when the application starts."""
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "SupportIQ is an AI-powered customer support "
        "platform for ticket management and assistance."
    ),
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to SupportIQ!",
        "application": settings.app_name,
        "status": "running",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name,
    }


app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)