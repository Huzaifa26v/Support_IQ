# Import TF-IDF vectorizer from scikit-learn.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our reusable text-cleaning function.
from ml.preprocessing.clean_text import clean_text


def create_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
) -> TfidfVectorizer:
    """
    Create and configure a TF-IDF vectorizer.

    Args:
        max_features: Maximum number of vocabulary features.
        ngram_range: Consider single words and two-word phrases.
        min_df: Minimum document frequency for a term.

    Returns:
        Configured TF-IDF vectorizer.
    """

    # Configure TF-IDF for customer support messages.
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )

    return vectorizer


def fit_transform_text(
    vectorizer: TfidfVectorizer,
    texts: list[str],
):
    """
    Learn vocabulary from training text and transform it.

    Important:
    Fit only on training data to prevent data leakage.
    """

    return vectorizer.fit_transform(texts)


def transform_text(
    vectorizer: TfidfVectorizer,
    texts: list[str],
):
    """
    Transform validation or test text using the
    vocabulary already learned from training data.
    """

    return vectorizer.transform(texts)