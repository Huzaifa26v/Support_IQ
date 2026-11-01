# Import the function we want to test.
from ml.preprocessing.clean_text import clean_text


def test_clean_text():
    """Test whether customer ticket text is cleaned correctly."""

    # Original customer support message.
    original_text = (
        "URGENT!!! Please check https://example.com "
        "@support. My refund is NOT processed!"
    )

    # Apply text cleaning.
    cleaned_text = clean_text(original_text)

    # Verify important words are preserved.
    assert "urgent" in cleaned_text
    assert "refund" in cleaned_text
    assert "not" in cleaned_text
    assert "processed" in cleaned_text

    # Verify URLs and mentions are removed.
    assert "https://" not in cleaned_text
    assert "@support" not in cleaned_text


def test_empty_text():
    """Test how the function handles empty text."""

    assert clean_text("") == ""


def test_invalid_input():
    """Test how the function handles invalid input."""

    assert clean_text(None) == ""


def test_lowercase_conversion():
    """Test whether uppercase text is converted to lowercase."""

    assert clean_text("HELLO SUPPORT") == "hello support"