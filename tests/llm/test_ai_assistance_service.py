import pytest

from app.services import ai_assistance_service

# ==========================================================
# 1. TEST SUCCESSFUL QUESTION PROCESSING
# ==========================================================

def test_ask_question_successfully(monkeypatch):
    """Test that a question is passed to the AI agent."""

    # Define the expected response from the AI agent.
    expected_response = {
        "query": "How can I reset my password?",
        "answer": "Follow the password reset instructions.",
        "sources": [
            {"source": "user_guide.pdf", "page": 2}
        ],
        "total_results": 1,
    }

    # Mock the agent to avoid making a real OpenAI API call.
    def mock_run_agent(question):
        assert question == "How can I reset my password?"
        return expected_response

    # Replace the actual agent with our mock function.
    monkeypatch.setattr(
        ai_assistance_service,
        "run_agent",
        mock_run_agent,
    )

    # Call the service.
    result = ai_assistance_service.ask_question(
        "How can I reset my password?"
    )

    # Verify the returned response.
    assert result == expected_response


# ==========================================================
# 2. TEST EMPTY QUESTION VALIDATION
# ==========================================================

def test_ask_question_rejects_empty_question():
    """Test that an empty question raises a ValueError."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        ai_assistance_service.ask_question("   ")


# ==========================================================
# 3. TEST INVALID QUESTION TYPE
# ==========================================================

def test_ask_question_rejects_invalid_type():
    """Test that non-string questions raise a TypeError."""

    with pytest.raises(TypeError, match="Question must be a string"):
        ai_assistance_service.ask_question(123)