# Import FastAPI components for creating API routes and handling errors.
# Import logging for recording unexpected application errors.
import logging

from fastapi import APIRouter, HTTPException, status

# Import AI request and response schemas.
from app.schemas.ai_analysis import (
    AIAnswerResponse,
    AIQuestionRequest,
)

# Import the AI assistance service.
from app.services.ai_assistance_service import ask_question

# Configure the module logger.
logger = logging.getLogger(__name__)

# Create the AI assistance router.
router = APIRouter(
    prefix="/ai-assistance",
    tags=["AI Assistance"],
)


@router.post(
    "/ask",
    response_model=AIAnswerResponse,
    status_code=status.HTTP_200_OK,
)
def ask_ai(question_data: AIQuestionRequest) -> AIAnswerResponse:
    """
    Accept a customer question and generate an AI-powered answer.

    The service retrieves relevant knowledge-base information
    and generates a response using the configured language model.
    """

    try:
        # Process the question through the AI assistance service.
        result = ask_question(question_data.question)

        # Return the generated answer and supporting sources.
        return AIAnswerResponse(**result)

    except Exception:
        # Record the full error internally for debugging.
        logger.exception("AI assistance request failed.")

        # Return a safe error message without exposing internal details.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI assistance is temporarily unavailable.",
        ) from None