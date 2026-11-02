import pytest

from app.rag.chunker import chunk_text


def test_chunk_text_returns_chunks():
    text = "SupportIQ provides customer support services."

    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    assert all("text" in chunk for chunk in chunks)


def test_chunk_indexes_are_sequential():
    text = "This is a sample document for testing chunk indexes."

    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

    indexes = [chunk["chunk_index"] for chunk in chunks]

    assert indexes == list(range(len(chunks)))


def test_chunks_preserve_all_text():
    text = "SupportIQ handles customer tickets and support requests."

    chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)

    combined_text = " ".join(chunk["text"] for chunk in chunks)

    for word in text.split():
        assert word in combined_text


def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        chunk_text("")


def test_invalid_input_type_raises_error():
    with pytest.raises(TypeError):
        chunk_text(None)


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=0)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=10, chunk_overlap=10)