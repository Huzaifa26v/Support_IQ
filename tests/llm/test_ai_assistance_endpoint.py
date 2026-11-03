# Import pytest for mocking and exception testing.

# Import the AI assistance endpoint module for mocking.
from app.api.v1.endpoints import ai_assistance

# ==========================================================
# 1. TEST SUCCESSFUL AI QUESTION REQUEST
# ==========================================================

def test_ai_assistance_success(client, monkeypatch):
    """Test successful AI question processing through the API."""

    # Define a sample response from the AI service.
    expected_response = {
        "query": "How can I reset my password?",
        "answer": "Follow the password reset instructions.",
        "sources": [
            {
                "source": "user_guide.pdf",
                "page": 2,
            }
        ],
        "total_results": 1,
    }

    # Replace the real AI service with a mock.
    def mock_ask_question(question):
        assert question == "How can I reset my password?"
        return expected_response

    monkeypatch.setattr(
        ai_assistance,
        "ask_question",
        mock_ask_question,
    )

    # Send a POST request to the AI assistance endpoint.
    response = client.post(
        "/api/v1/ai-assistance/ask",
        json={"question": "How can I reset my password?"},
    )

    # Verify the HTTP response.
    assert response.status_code == 200
    assert response.json() == expected_response


# ==========================================================
# 2. TEST INVALID QUESTION
# ==========================================================

def test_ai_assistance_rejects_short_question(client):
    """Test that questions shorter than 3 characters are rejected."""

    response = client.post(
        "/api/v1/ai-assistance/ask",
        json={"question": "Hi"},
    )

    # Pydantic validation should reject the request.
    assert response.status_code == 422


# ==========================================================
# 3. TEST AI SERVICE FAILURE
# ==========================================================

def test_ai_assistance_service_failure(client, monkeypatch):
    """Test that unexpected service errors return HTTP 503."""

    # Simulate an AI service failure.
    def mock_ask_question(question):
        raise RuntimeError("Simulated AI service failure.")

    monkeypatch.setattr(
        ai_assistance,
        "ask_question",
        mock_ask_question,
    )

    # Send a valid question.
    response = client.post(
        "/api/v1/ai-assistance/ask",
        json={"question": "How can I reset my password?"},
    )

    # Verify that the API returns a safe error response.
    assert response.status_code == 503
    assert response.json() == {
        "detail": "AI assistance is temporarily unavailable."
    }