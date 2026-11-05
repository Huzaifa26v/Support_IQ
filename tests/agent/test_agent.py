import pytest

from app.agent import agent

# ==========================================================
# 1. TEST SUCCESSFUL AGENT EXECUTION
# ==========================================================

def test_run_agent_successfully(monkeypatch):
    """Test that the agent returns the expected response."""

    expected_result = {
        "answer": "You can reset your password through the help page.",
        "sources": [{"source": "help.pdf"}],
        "total_results": 1,
    }

    def mock_workflow(question):
        assert question == "How do I reset my password?"
        return expected_result

    monkeypatch.setattr(
        agent,
        "run_agent_workflow",
        mock_workflow,
    )

    result = agent.run_agent("How do I reset my password?")

    assert result == {
        "query": "How do I reset my password?",
        "answer": expected_result["answer"],
        "sources": expected_result["sources"],
        "total_results": 1,
    }


# ==========================================================
# 2. TEST QUESTION WHITESPACE HANDLING
# ==========================================================

def test_run_agent_strips_question_whitespace(monkeypatch):
    """Test that unnecessary whitespace is removed."""

    def mock_workflow(question):
        assert question == "How can I contact support?"

        return {
            "answer": "Contact customer support.",
            "sources": [],
            "total_results": 0,
        }

    monkeypatch.setattr(
        agent,
        "run_agent_workflow",
        mock_workflow,
    )

    result = agent.run_agent("  How can I contact support?  ")

    assert result["query"] == "How can I contact support?"


# ==========================================================
# 3. TEST EMPTY QUESTION VALIDATION
# ==========================================================

def test_run_agent_rejects_empty_question():
    """Test that an empty question raises ValueError."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        agent.run_agent("   ")


# ==========================================================
# 4. TEST INVALID QUESTION TYPE
# ==========================================================

def test_run_agent_rejects_non_string_question():
    """Test that non-string questions raise TypeError."""

    with pytest.raises(TypeError, match="Question must be a string"):
        agent.run_agent(123)