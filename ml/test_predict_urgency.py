# Import pytest for automated testing.
import pytest

# Import the urgency prediction module.
from ml.predict_urgency import (
    predict_urgency,
    load_urgency_model,
)


# ==========================================================
# TEST MODEL LOADING
# ==========================================================

def test_load_urgency_model():
    """
    Verify that the saved urgency model loads successfully.
    """

    model_artifact = load_urgency_model()

    assert model_artifact is not None
    assert "model" in model_artifact
    assert "vectorizer" in model_artifact
    assert "target_column" in model_artifact

    assert model_artifact["target_column"] == "urgency"


# ==========================================================
# TEST VALID PREDICTIONS
# ==========================================================

def test_predict_urgency_returns_valid_class():
    """
    Verify that predictions belong to the expected
    urgency categories.
    """

    ticket_text = (
        "My account is not working and I cannot log in."
    )

    prediction = predict_urgency(ticket_text)

    assert prediction in ["high", "medium", "low"]


def test_predict_urgency_with_payment_issue():
    """
    Verify that the model can process payment-related text.
    """

    ticket_text = (
        "I cannot complete my payment because the "
        "application keeps showing an error."
    )

    prediction = predict_urgency(ticket_text)

    assert prediction in ["high", "medium", "low"]


# ==========================================================
# TEST INVALID INPUT
# ==========================================================

def test_predict_urgency_with_empty_string():
    """
    Verify that empty input raises a ValueError.
    """

    with pytest.raises(ValueError):
        predict_urgency("")


def test_predict_urgency_with_whitespace():
    """
    Verify that whitespace-only input raises a ValueError.
    """

    with pytest.raises(ValueError):
        predict_urgency("   ")


def test_predict_urgency_with_none():
    """
    Verify that None input raises a TypeError.
    """

    with pytest.raises(TypeError):
        predict_urgency(None)


def test_predict_urgency_with_numeric_input():
    """
    Verify that numeric input raises a TypeError.
    """

    with pytest.raises(TypeError):
        predict_urgency(12345)