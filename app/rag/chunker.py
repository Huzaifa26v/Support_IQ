def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Divide text into smaller overlapping chunks without
    splitting words at normal chunk boundaries.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Find a word boundary near the end of the chunk.
        if end < text_length:
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk,
            })

        if end >= text_length:
            break

        # Calculate the approximate overlapping position.
        next_start = end - chunk_overlap

        # Move forward to the next word boundary.
        if next_start > start:
            boundary = text.find(" ", next_start, end)

            if boundary != -1:
                next_start = boundary + 1
            else:
                next_start = end
        else:
            next_start = end

        # Guarantee forward progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks