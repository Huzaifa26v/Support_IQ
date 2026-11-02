import numpy as np
import pytest

from app.rag import embeddings


class FakeEmbeddingModel:
    """Simulates the embedding model without downloading it."""

    def encode(self, texts, **kwargs):
        return np.array([
            [0.1] * 384 for _ in texts
        ])


def test_generate_embeddings_returns_vectors(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    result = embeddings.generate_embeddings([
        "Customer support",
        "Refund policy",
    ])

    assert len(result) == 2
    assert len(result[0]) == 384


def test_generate_embeddings_rejects_empty_list():
    with pytest.raises(ValueError):
        embeddings.generate_embeddings([])


def test_generate_embeddings_rejects_invalid_input():
    with pytest.raises(TypeError):
        embeddings.generate_embeddings("Customer support")


def test_generate_embeddings_rejects_non_string_items():
    with pytest.raises(TypeError):
        embeddings.generate_embeddings(["Support", 123])


def test_generate_embeddings_rejects_empty_text():
    with pytest.raises(ValueError):
        embeddings.generate_embeddings(["Support", " "])


def test_generate_query_embedding_returns_vector(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    result = embeddings.generate_query_embedding(
        "How can I get a refund?"
    )

    assert isinstance(result, list)
    assert len(result) == 384


def test_generate_query_embedding_rejects_empty_query():
    with pytest.raises(ValueError):
        embeddings.generate_query_embedding("")


def test_generate_query_embedding_rejects_invalid_type():
    with pytest.raises(TypeError):
        embeddings.generate_query_embedding(None)