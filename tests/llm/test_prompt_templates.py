import pytest

# Import the prompt builder that we want to test.
from app.llm.prompt_templates import build_rag_messages


def test_build_rag_messages_returns_system_and_user_messages():
    """Verify that the prompt builder creates both required messages."""

    # Prepare sample customer input and retrieved knowledge.
    question = "How can I reset my password?"
    context = "Customers can reset passwords through the Forgot Password page."

    # Generate the messages.
    messages = build_rag_messages(question, context)

    # Verify that both messages are returned in the correct order.
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Verify that the question and context are included.
    assert question in messages[1]["content"]
    assert context in messages[1]["content"]


def test_build_rag_messages_rejects_empty_question():
    """Verify that an empty customer question is rejected."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_rag_messages("", "Some knowledge-base context.")


def test_build_rag_messages_rejects_invalid_question_type():
    """Verify that a non-string question raises a TypeError."""

    with pytest.raises(TypeError, match="Question must be a string"):
        build_rag_messages(123, "Some knowledge-base context.")


def test_build_rag_messages_rejects_invalid_context_type():
    """Verify that a non-string context raises a TypeError."""

    with pytest.raises(TypeError, match="Context must be a string"):
        build_rag_messages("How can I reset my password?", 123)