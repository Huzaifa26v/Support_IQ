# Import pytest for testing validation errors.
import pytest

# Import Pydantic's ValidationError.
from pydantic import ValidationError

# Import the AI request and response schemas.
from app.schemas.ai_analysis import (
    AIAnswerResponse,
    AIQuestionRequest,
)

# ==========================================================
# 1. TEST AI QUESTION REQUEST
# ==========================================================

def test_valid_ai_question():
    """Test that a valid customer question is accepted."""

    request = AIQuestionRequest(
        question="How can I reset my password?"
    )

    assert request.question == "How can I reset my password?"


def test_ai_question_cannot_be_empty_or_too_short():
    """Test that questions shorter than 3 characters are rejected."""

    with pytest.raises(ValidationError):
        AIQuestionRequest(question="Hi")


def test_ai_question_cannot_exceed_maximum_length():
    """Test that questions longer than 2000 characters are rejected."""

    with pytest.raises(ValidationError):
        AIQuestionRequest(question="A" * 2001)


# ==========================================================
# 2. TEST AI ANSWER RESPONSE
# ==========================================================

def test_valid_ai_answer_response():
    """Test that a valid AI response with sources is accepted."""

    response = AIAnswerResponse(
        query="How can I reset my password?",
        answer="Follow the password reset instructions.",
        sources=[
            {
                "source": "user_guide.pdf",
                "page": 2,
            }
        ],
        total_results=1,
    )

    assert response.query == "How can I reset my password?"
    assert response.total_results == 1
    assert response.sources[0]["source"] == "user_guide.pdf"


def test_ai_response_rejects_negative_total_results():
    """Test that total_results cannot be negative."""

    with pytest.raises(ValidationError):
        AIAnswerResponse(
            query="Test question",
            answer="Test answer",
            sources=[],
            total_results=-1,
        )