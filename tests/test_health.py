# ==========================================================
# SUPPORTIQ HEALTH CHECK TESTS
# ==========================================================

# Pytest automatically provides the client fixture
# defined in tests/conftest.py.


# ==========================================================
# 1. TEST HEALTH ENDPOINT
# ==========================================================

def test_health_check(client):
    """
    Verify that the health endpoint returns a successful response.
    """

    # Send a GET request to the health endpoint.
    response = client.get("/health")

    # Verify that the server returns HTTP 200.
    assert response.status_code == 200

    # Verify the response contains the expected health information.
    assert response.json() == {
        "status": "healthy",
        "application": "SupportIQ",
    }


# ==========================================================
# 2. TEST ROOT ENDPOINT
# ==========================================================

def test_root_endpoint(client):
    """
    Verify that the root endpoint returns the welcome message.
    """

    # Send a GET request to the root endpoint.
    response = client.get("/")

    # Verify that the request was successful.
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    data = response.json()

    # Verify the welcome message.
    assert data["message"] == "Welcome to SupportIQ!"

    # Verify the application name.
    assert data["application"] == "SupportIQ"

    # Verify that the application reports its running status.
    assert data["status"] == "running"