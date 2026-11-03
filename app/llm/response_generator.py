# Application settings, including model configuration.
from app.core.config import settings

# Get the configured OpenAI client.
from app.llm.client import get_openai_client

# Build system and user messages for the LLM.
from app.llm.prompt_templates import build_rag_messages

# Retrieve relevant knowledge-base information from ChromaDB.
from app.rag.retriever import retrieve_relevant_context


def generate_rag_response(
    question: str,
    top_k: int = 5,
) -> dict:
    """
    Generate a customer-support answer using Retrieval-Augmented Generation.

    The function retrieves relevant knowledge-base chunks,
    builds an LLM prompt, and generates an answer using OpenAI.

    Args:
        question: Customer's support question.
        top_k: Maximum number of relevant chunks to retrieve.

    Returns:
        Dictionary containing the generated answer and its sources.
    """

    # Validate the customer's question.
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Retrieve relevant information from the knowledge base.
    retrieval_result = retrieve_relevant_context(
        query=question,
        top_k=top_k,
    )

    # Extract the retrieved context and source chunks.
    context = retrieval_result["context"]
    chunks = retrieval_result["chunks"]

    # Avoid generating an unsupported answer when no context is found.
    if not context.strip():
        return {
            "query": question,
            "answer": (
                "I could not find relevant information in the "
                "knowledge base to answer your question."
            ),
            "sources": [],
            "total_results": 0,
        }

    # Prepare messages using the retrieved knowledge.
    messages = build_rag_messages(
        question=question,
        context=context,
    )

    # Initialize the configured OpenAI client.
    client = get_openai_client()

    # Generate an answer using the selected language model.
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    # Extract the generated answer from the API response.
    answer = response.choices[0].message.content

    # Convert retrieved metadata into a clean source list.
    sources = [
        chunk["metadata"]
        for chunk in chunks
    ]

    # Return the answer with its supporting source information.
    return {
        "query": question,
        "answer": answer or "",
        "sources": sources,
        "total_results": retrieval_result["total_results"],
    }