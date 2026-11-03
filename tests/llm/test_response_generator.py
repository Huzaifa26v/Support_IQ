from types import SimpleNamespace

import pytest

# Import the response generator we want to test.
from app.llm import response_generator


def test_generate_rag_response_success(monkeypatch):
    """Verify that a response is generated using retrieved context."""

    # Mock the retriever to avoid accessing the actual ChromaDB.
    def mock_retriever(query, top_k):
        return {
            "query": query,
            "context": "Customers can reset passwords through the website.",
            "chunks": [
                {
                    "content": "Customers can reset passwords through the website.",
                    "metadata": {"source": "support_guide.pdf"},
                    "distance": 0.2,
                }
            ],
            "total_results": 1,
        }

    # Mock the OpenAI response structure.
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="You can reset your password through our website."
                )
            )
        ]
    )

    # Create a mock OpenAI client.
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: mock_response
            )
        )
    )

    # Replace actual dependencies with our mock functions.
    monkeypatch.setattr(
        response_generator,
        "retrieve_relevant_context",
        mock_retriever,
    )

    monkeypatch.setattr(
        response_generator,
        "get_openai_client",
        lambda: mock_client,
    )

    # Generate the customer-support response.
    result = response_generator.generate_rag_response(
        "How can I reset my password?"
    )

    # Verify the generated answer and its source.
    assert result["answer"] == (
        "You can reset your password through our website."
    )
    assert result["total_results"] == 1
    assert result["sources"] == [
        {"source": "support_guide.pdf"}
    ]


def test_generate_rag_response_without_context(monkeypatch):
    """Verify that OpenAI is not called when no context is available."""

    # Return an empty retrieval result.
    def mock_retriever(query, top_k):
        return {
            "query": query,
            "context": "",
            "chunks": [],
            "total_results": 0,
        }

    # Replace the retriever with the empty-result mock.
    monkeypatch.setattr(
        response_generator,
        "retrieve_relevant_context",
        mock_retriever,
    )

    # Track whether the OpenAI client is initialized.
    def unexpected_client_call():
        pytest.fail("OpenAI client should not be initialized.")

    monkeypatch.setattr(
        response_generator,
        "get_openai_client",
        unexpected_client_call,
    )

    # Generate a response without relevant knowledge.
    result = response_generator.generate_rag_response(
        "How can I reset my password?"
    )

    # Verify that the fallback response is returned.
    assert "could not find relevant information" in result["answer"]
    assert result["sources"] == []
    assert result["total_results"] == 0


def test_generate_rag_response_rejects_empty_question():
    """Verify that an empty question is rejected."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        response_generator.generate_rag_response("")


def test_generate_rag_response_rejects_invalid_question_type():
    """Verify that a non-string question is rejected."""

    with pytest.raises(TypeError, match="Question must be a string"):
        response_generator.generate_rag_response(123)