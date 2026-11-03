from pathlib import Path
from uuid import uuid4

import chromadb

from app.rag.embeddings import generate_embeddings, generate_query_embedding

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Persistent storage directory
VECTOR_DB_PATH = BASE_DIR / "vector_db"

# ChromaDB collection name
COLLECTION_NAME = "supportiq_knowledge"


def get_vector_store():
    """
    Initialize ChromaDB with persistent local storage.

    Returns:
        ChromaDB collection.
    """

    # Ensure the storage directory exists
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    # Get an existing collection or create a new one
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SupportIQ knowledge base documents",
            "hnsw:space": "cosine",
        },
    )

    return collection


def add_chunks(
    chunks: list[dict],
    document_metadata: dict | None = None,
) -> list[str]:
    """
    Generate embeddings and store document chunks in ChromaDB.

    Args:
        chunks: List of dictionaries containing chunk_index and text.
        document_metadata: Optional metadata about the source document.

    Returns:
        List of generated chunk IDs.
    """

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Chunks must be a non-empty list.")

    if not all(isinstance(chunk, dict) for chunk in chunks):
        raise TypeError("Every chunk must be a dictionary.")

    texts = []

    for chunk in chunks:
        if not isinstance(chunk.get("text"), str):
            raise TypeError("Every chunk must contain text as a string.")

        if not chunk["text"].strip():
            raise ValueError("Chunk text cannot be empty.")

        texts.append(chunk["text"])

    # Generate embeddings for all chunks
    embeddings = generate_embeddings(texts)

    # Generate unique IDs
    chunk_ids = [str(uuid4()) for _ in chunks]

    # Prepare metadata for ChromaDB
    metadatas = []

    for chunk in chunks:
        metadata = dict(document_metadata or {})

        metadata["chunk_index"] = int(chunk["chunk_index"])

        metadatas.append(metadata)

    # Get the ChromaDB collection
    collection = get_vector_store()

    # Store chunks, embeddings, and metadata
    collection.add(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return chunk_ids

def delete_chunks(chunk_ids: list[str]) -> None:
    """
    Delete previously stored chunks from ChromaDB.

    This cleanup function helps prevent orphaned vectors
    when a document's database update fails.
    """

    # Validate the supplied chunk IDs.
    if not isinstance(chunk_ids, list):
        raise TypeError("chunk_ids must be a list.")

    if not chunk_ids:
        return

    if not all(isinstance(chunk_id, str) for chunk_id in chunk_ids):
        raise TypeError("Every chunk ID must be a string.")

    # Access the existing ChromaDB collection.
    collection = get_vector_store()

    # Delete the specified chunks.
    collection.delete(ids=chunk_ids)

def search_similar_chunks(
    query: str,
    n_results: int = 5,
) -> dict:
    """
    Search for semantically similar document chunks.

    Args:
        query: User's search question.
        n_results: Maximum number of results.

    Returns:
        ChromaDB search results.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(n_results, int) or isinstance(n_results, bool):
        raise TypeError("n_results must be an integer.")

    if n_results <= 0:
        raise ValueError("n_results must be greater than zero.")

    # Get the collection
    collection = get_vector_store()

    # Check whether documents exist
    total_documents = collection.count()

    if total_documents == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # Convert the query into an embedding
    query_embedding = generate_query_embedding(query)

    # Search the vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total_documents),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results