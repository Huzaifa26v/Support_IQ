# Import regular expressions for text cleaning.
import re


def clean_text(text: str) -> str:
    """
    Clean customer support ticket text before machine learning.

    Args:
        text (str): Original customer ticket text.

    Returns:
        str: Cleaned customer ticket text.
    """

    # Handle missing or invalid input safely.
    if not isinstance(text, str):
        return ""

    # Convert all text to lowercase.
    text = text.lower()

    # Remove URLs such as https://example.com.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses.
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text,
    )

    # Remove user mentions such as @support.
    text = re.sub(r"@\w+", " ", text)

    # Preserve letters, numbers, spaces, and useful punctuation.
    # We keep !, ?, and apostrophes because they can carry meaning.
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)

    # Replace multiple spaces or line breaks with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces from the beginning and end.
    text = text.strip()

    return text