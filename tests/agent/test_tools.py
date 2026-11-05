
import pytest

from app.agent import tools

# ==========================================================
# 1. TEST KNOWLEDGE BASE SEARCH
# ==========================================================

def test_search_knowledge_base_successfully(monkeypatch):
    """Test successful knowledge-base retrieval."""

    expected_result = {
        "query": "password reset",
        "context": "Password reset instructions.",
        "chunks": [],
        "total_results": 0,
    }

    def mock_retriever(query, top_k):
        assert query == "password reset"
        assert top_k == 5
        return expected_result

    monkeypatch.setattr(
        tools,
        "retrieve_relevant_context",
        mock_retriever,
    )

    result = tools.search_knowledge_base("password reset")

    assert result == expected_result


def test_search_knowledge_base_rejects_empty_query():
    """Test that empty search queries are rejected."""

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        tools.search_knowledge_base("   ")


def test_search_knowledge_base_rejects_invalid_top_k():
    """Test that invalid result limits are rejected."""

    with pytest.raises(ValueError, match="top_k must be between 1 and 10"):
        tools.search_knowledge_base("password reset", top_k=20)


# ==========================================================
# 2. TEST CATEGORY PREDICTION
# ==========================================================

def test_predict_category_successfully(monkeypatch):
    """Test successful ticket category prediction."""

    def mock_predictor(text):
        assert text == "Unable to access my account"
        return "account_access"

    monkeypatch.setattr(
        tools,
        "predict_category",
        mock_predictor,
    )

    result = tools.predict_category_tool(
        "Unable to access my account"
    )

    assert result == {"category": "account_access"}


def test_predict_category_rejects_empty_text():
    """Test that empty ticket descriptions are rejected."""

    with pytest.raises(ValueError, match="Ticket text cannot be empty"):
        tools.predict_category_tool("   ")


# ==========================================================
# 3. TEST URGENCY PREDICTION
# ==========================================================

def test_predict_urgency_successfully(monkeypatch):
    """Test successful ticket urgency prediction."""

    def mock_predictor(text):
        assert text == "Payment transaction failed"
        return "high"

    monkeypatch.setattr(
        tools,
        "predict_urgency",
        mock_predictor,
    )

    result = tools.predict_urgency_tool(
        "Payment transaction failed"
    )

    assert result == {"urgency": "high"}


def test_predict_urgency_rejects_empty_text():
    """Test that empty ticket descriptions are rejected."""

    with pytest.raises(ValueError, match="Ticket text cannot be empty"):
        tools.predict_urgency_tool("   ")


# ==========================================================
# 4. TEST TOOL EXECUTION
# ==========================================================

def test_execute_tool_rejects_unknown_tool():
    """Test that unregistered tools cannot be executed."""

    with pytest.raises(ValueError, match="Unknown agent tool"):
        tools.execute_tool("unknown_tool", {})


def test_execute_tool_rejects_invalid_arguments():
    """Test that tool arguments must be dictionaries."""

    with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
        tools.execute_tool("search_knowledge_base", "invalid")

