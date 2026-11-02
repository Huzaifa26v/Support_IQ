import pytest

from app.rag import vector_store


@pytest.fixture
def test_collection(tmp_path, monkeypatch):
    """
    Create an isolated temporary ChromaDB collection
    for each test.
    """

    monkeypatch.setattr(
        vector_store,
        "VECTOR_DB_PATH",
        tmp_path / "vector_db",
    )

    return vector_store.get_vector_store()


def test_get_vector_store_creates_collection(test_collection):
    assert test_collection.name == vector_store.COLLECTION_NAME
    assert test_collection.count() == 0


def test_add_chunks_stores_documents(test_collection, monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "generate_embeddings",
        lambda texts: [[0.1] * 384 for _ in texts],
    )

    chunks = [
        {
            "chunk_index": 0,
            "text": "Customers can request refunds within 30 days.",
        },
        {
            "chunk_index": 1,
            "text": "Billing issues go to the billing department.",
        },
    ]

    ids = vector_store.add_chunks(
        chunks,
        {"document_id": "test-document"},
    )

    assert len(ids) == 2
    assert test_collection.count() == 2


def test_add_chunks_stores_metadata(test_collection, monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "generate_embeddings",
        lambda texts: [[0.1] * 384 for _ in texts],
    )

    chunks = [
        {
            "chunk_index": 0,
            "text": "Refunds are available within 30 days.",
        }
    ]

    vector_store.add_chunks(
        chunks,
        {"document_id": "document-123"},
    )

    results = test_collection.get(
        include=["documents", "metadatas"]
    )

    assert results["documents"][0] == chunks[0]["text"]
    assert results["metadatas"][0]["document_id"] == "document-123"
    assert results["metadatas"][0]["chunk_index"] == 0


def test_add_chunks_rejects_empty_input():
    with pytest.raises(ValueError):
        vector_store.add_chunks([])


def test_add_chunks_rejects_invalid_chunk():
    with pytest.raises(TypeError):
        vector_store.add_chunks(["invalid chunk"])


def test_search_retrieves_relevant_document(
    test_collection,
    monkeypatch,
):
    monkeypatch.setattr(
        vector_store,
        "generate_embeddings",
        lambda texts: [[0.1] * 384 for _ in texts],
    )

    monkeypatch.setattr(
        vector_store,
        "generate_query_embedding",
        lambda query: [0.1] * 384,
    )

    chunks = [
        {
            "chunk_index": 0,
            "text": "Customers can request refunds within 30 days.",
        }
    ]

    vector_store.add_chunks(chunks)

    results = vector_store.search_similar_chunks(
        "What is the refund policy?"
    )

    assert len(results["documents"][0]) == 1
    assert (
        results["documents"][0][0]
        == chunks[0]["text"]
    )


def test_search_returns_empty_results_for_empty_collection(
    test_collection,
    monkeypatch,
):
    monkeypatch.setattr(
        vector_store,
        "generate_query_embedding",
        lambda query: [0.1] * 384,
    )

    results = vector_store.search_similar_chunks(
        "What is the refund policy?"
    )

    assert results["documents"] == [[]]


def test_search_rejects_empty_query():
    with pytest.raises(ValueError):
        vector_store.search_similar_chunks("")


def test_search_rejects_invalid_result_count():
    with pytest.raises(ValueError):
        vector_store.search_similar_chunks(
            "Refund policy",
            n_results=0,
        )