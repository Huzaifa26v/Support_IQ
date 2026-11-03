# Import the existing RAG response generator.
from app.llm.response_generator import generate_rag_response


def ask_question(question: str) -> dict:
    """
    Process a customer question using the RAG and LLM pipeline.

    The response generator retrieves relevant knowledge-base
    information and uses the configured LLM to generate an answer.

    Args:
        question: Customer's support question.

    Returns:
        Dictionary containing the answer and supporting sources.
    """

    # Validate the customer's question.
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Generate an answer using the existing RAG pipeline.
    result = generate_rag_response(
        question=question.strip(),
    )

    # Return the generated answer and supporting sources.
    return result