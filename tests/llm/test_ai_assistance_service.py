# Import pytest for testing and exception handling.
import pytest

# Import the AI assistance service.
from app.services import ai_assistance_service

# ==========================================================
# 1. TEST SUCCESSFUL QUESTION PROCESSING
# ==========================================================

def test_ask_question_successfully(monkeypatch):
    """Test that a question is passed to the RAG response generator."""

    # Create a sample response without calling the real OpenAI API.
    expected_response = {
        "query": "How can I reset my password?",
        "answer": "Follow the password reset instructions.",
        "sources": [
            {"source": "user_guide.pdf", "page": 2}
        ],
        "total_results": 1,
    }

    # Mock the response generator.
    def mock_generate_response(question):
        assert question == "How can I reset my password?"
        return expected_response

    monkeypatch.setattr(
        ai_assistance_service,
        "generate_rag_response",
        mock_generate_response,
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