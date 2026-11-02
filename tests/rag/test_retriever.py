import pytest

from app.rag import retriever


def test_retrieve_relevant_context(monkeypatch):
    def mock_search(query, n_results):
        return {
            "documents": [[
                "Refunds are available within 30 days.",
                "Billing issues go to the billing department.",
            ]],
            "metadatas": [[
                {"document_id": "doc-1"},
                {"document_id": "doc-2"},
            ]],
            "distances": [[0.2, 0.7]],
        }

    monkeypatch.setattr(
        retriever,
        "search_similar_chunks",
        mock_search,
    )

    result = retriever.retrieve_relevant_context(
        "What is the refund policy?",
        top_k=2,
    )

    assert result["total_results"] == 2
    assert "Refunds are available" in result["context"]
    assert result["query"] == "What is the refund policy?"


def test_retrieve_returns_empty_context(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "search_similar_chunks",
        lambda query, n_results: {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        },
    )

    result = retriever.retrieve_relevant_context("Refund policy")

    assert result["total_results"] == 0
    assert result["context"] == ""
    assert result["chunks"] == []


def test_retrieve_rejects_empty_query():
    with pytest.raises(ValueError):
        retriever.retrieve_relevant_context("")


def test_retrieve_rejects_invalid_query_type():
    with pytest.raises(TypeError):
        retriever.retrieve_relevant_context(None)


def test_retrieve_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        retriever.retrieve_relevant_context(
            "Refund policy",
            top_k=0,
        )


def test_retrieve_rejects_invalid_top_k_type():
    with pytest.raises(TypeError):
        retriever.retrieve_relevant_context(
            "Refund policy",
            top_k="five",
        )