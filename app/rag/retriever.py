from app.rag.vector_store import search_similar_chunks


def retrieve_relevant_context(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Retrieve relevant knowledge-base chunks for a user query.

    Args:
        query: Customer's question.
        top_k: Maximum number of relevant chunks to retrieve.

    Returns:
        Dictionary containing retrieved context and source metadata.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Search ChromaDB for similar document chunks.
    results = search_similar_chunks(
        query=query,
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Organize retrieved information.
    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved_chunks.append({
            "content": document,
            "metadata": metadata or {},
            "distance": distance,
        })

    # Combine retrieved chunks into context for the LLM.
    context = "\n\n".join(
        chunk["content"]
        for chunk in retrieved_chunks
    )

    return {
        "query": query,
        "context": context,
        "chunks": retrieved_chunks,
        "total_results": len(retrieved_chunks),
    }