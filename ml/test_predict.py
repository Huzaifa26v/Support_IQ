# Import pytest for writing automated tests.
import pytest

# Import prediction functions from our module.
from ml.predict import (
    load_model,
    predict_category,
)


# ==========================================================
# TEST MODEL LOADING
# ==========================================================

def test_load_model():
    """
    Verify that the trained model and vectorizer load correctly.
    """

    model, vectorizer = load_model()

    # Check that both objects were loaded.
    assert model is not None
    assert vectorizer is not None

    # Check the classifier type.
    assert model.__class__.__name__ == "LinearSVC"


# ==========================================================
# TEST CATEGORY PREDICTIONS
# ==========================================================

def test_predict_account_category():
    """
    Verify account-related ticket classification.
    """

    message = "I cannot log into my account."

    prediction = predict_category(message)

    assert prediction == "account"


def test_predict_billing_category():
    """
    Verify billing-related ticket classification.
    """

    message = "I was charged twice for my subscription."

    prediction = predict_category(message)

    assert prediction == "billing"


def test_predict_technical_category():
    """
    Verify technical ticket classification.
    """

    message = "The application keeps crashing."

    prediction = predict_category(message)

    assert prediction == "technical"


def test_predict_feature_request_category():
    """
    Verify feature request classification.
    """

    message = "Please add a dark mode feature."

    prediction = predict_category(message)

    assert prediction == "feature_request"


# ==========================================================
# TEST INPUT VALIDATION
# ==========================================================

def test_predict_empty_message():
    """
    Verify that empty messages are rejected.
    """

    with pytest.raises(ValueError):
        predict_category("")


def test_predict_whitespace_message():
    """
    Verify that whitespace-only messages are rejected.
    """

    with pytest.raises(ValueError):
        predict_category("   ")


def test_predict_invalid_input_type():
    """
    Verify that invalid input types are rejected.
    """

    with pytest.raises(ValueError):
        predict_category(None)