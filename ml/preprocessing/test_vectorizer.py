# Import the functions we want to test.
from ml.preprocessing.vectorizer import (
    create_vectorizer,
    fit_transform_text,
    transform_text,
)


def test_create_vectorizer():
    """Check whether the vectorizer is created correctly."""

    vectorizer = create_vectorizer()

    assert vectorizer.max_features == 10000
    assert vectorizer.ngram_range == (1, 2)


def test_fit_transform_text():
    """Check whether training text is converted into numerical features."""

    training_texts = [
        "My refund is not processed",
        "I have a technical problem",
        "Thank you for resolving my issue",
        "My account is locked",
    ]

    vectorizer = create_vectorizer()

    features = fit_transform_text(vectorizer, training_texts)

    # Four input messages should produce four feature rows.
    assert features.shape[0] == 4

    # The vectorizer should learn a vocabulary.
    assert len(vectorizer.vocabulary_) > 0


def test_transform_text():
    """Check whether new messages use the existing vocabulary."""

    training_texts = [
        "My refund is not processed",
        "I have a technical problem",
        "My account is locked",
    ]

    new_texts = [
        "My refund is still pending",
        "I cannot access my account",
    ]

    vectorizer = create_vectorizer()

    # Learn vocabulary using training data.
    fit_transform_text(vectorizer, training_texts)

    # Transform new data without fitting again.
    features = transform_text(vectorizer, new_texts)

    assert features.shape[0] == 2

    # The number of columns must match the training vocabulary.
    assert features.shape[1] == len(vectorizer.vocabulary_)


def test_vectorizer_does_not_change_vocabulary():
    """Ensure transforming new data does not change vocabulary."""

    training_texts = [
        "refund problem",
        "account problem",
        "technical issue",
    ]

    vectorizer = create_vectorizer()

    fit_transform_text(vectorizer, training_texts)

    original_vocabulary = vectorizer.vocabulary_.copy()

    transform_text(vectorizer, ["completely new vocabulary"])

    assert vectorizer.vocabulary_ == original_vocabulary