from sentence_transformers import SentenceTransformer


# Default embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cache the model so it is loaded only once.
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load and return the embedding model.

    The model is loaded only once and reused
    for subsequent embedding operations.
    """

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple text documents into numerical vectors.

    Args:
        texts: List of text strings.

    Returns:
        A list of embedding vectors.
    """

    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings.")

    if not texts:
        raise ValueError("Text list cannot be empty.")

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Every item must be a string.")

    if not all(text.strip() for text in texts):
        raise ValueError("Text items cannot be empty.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def generate_query_embedding(text: str) -> list[float]:
    """
    Convert a single user query into an embedding vector.

    Args:
        text: User's search query.

    Returns:
        A numerical embedding vector.
    """

    if not isinstance(text, str):
        raise TypeError("Query must be a string.")

    if not text.strip():
        raise ValueError("Query cannot be empty.")

    embeddings = generate_embeddings([text])

    return embeddings[0]