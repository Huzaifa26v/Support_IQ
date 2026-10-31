# Import pytest to create reusable testing fixtures.
import pytest

# Import TestClient to send HTTP requests to our FastAPI application.
from fastapi.testclient import TestClient

# Import SQLAlchemy tools to create a separate testing database.
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import the FastAPI application.
from app.main import app

# Import the shared SQLAlchemy Base class.
from app.db.base import Base

# Import the database dependency used by our API endpoints.
from app.db.session import get_db


# ==========================================================
# 1. TEST DATABASE CONFIGURATION
# ==========================================================

# Use an in-memory SQLite database for testing.
# This database exists only while the tests are running.
TEST_DATABASE_URL = "sqlite://"

# Create a separate database engine for tests.
test_engine = create_engine(
    TEST_DATABASE_URL,

    # SQLite requires this setting when used across test threads.
    connect_args={"check_same_thread": False},

    # Keep the same in-memory database connection available
    # throughout the testing session.
    poolclass=StaticPool,
)

# Create a session factory connected to the test database.
TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


# ==========================================================
# 2. TEST DATABASE DEPENDENCY
# ==========================================================

def override_get_db():
    """
    Provide a database session connected to the test database.

    This replaces the application's normal database dependency
    while automated tests are running.
    """

    # Create a new test database session.
    db = TestingSessionLocal()

    try:
        # Provide the session to the API endpoint.
        yield db

    finally:
        # Close the session after the request finishes.
        db.close()


# ==========================================================
# 3. PYTEST DATABASE FIXTURE
# ==========================================================

@pytest.fixture
def client():
    """
    Create a clean test environment for each test.

    This fixture:
    1. Creates database tables.
    2. Overrides the application's database dependency.
    3. Provides a FastAPI test client.
    4. Cleans up after the test finishes.
    """

    # Create all registered database tables in the test database.
    Base.metadata.create_all(bind=test_engine)

    # Replace the normal database dependency with the test version.
    app.dependency_overrides[get_db] = override_get_db

    # Create a client for sending API requests.
    # We do not use the context manager here because we do not
    # want the application's normal startup lifespan to initialize
    # the development database during these isolated tests.
    test_client = TestClient(app)

    # Provide the client to the test function.
    yield test_client

    # Remove dependency overrides after the test finishes.
    app.dependency_overrides.clear()

    # Delete all test database tables.
    Base.metadata.drop_all(bind=test_engine)