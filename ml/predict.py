# Import Path to manage project file paths.
from pathlib import Path

# Import Joblib to load our saved machine learning model.
import joblib


# ==========================================================
# CONFIGURATION
# ==========================================================

# Get the root directory of the SupportIQ project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the path of our saved model.
MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "category_classifier.joblib"
)


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

def load_model():
    """
    Load the saved classifier and TF-IDF vectorizer.
    """

    # Check whether the model file exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}. "
            "Run 'python -m ml.save_model' first."
        )

    # Load the saved model artifact.
    model_artifact = joblib.load(MODEL_PATH)

    # Extract the classifier and vectorizer.
    model = model_artifact["model"]
    vectorizer = model_artifact["vectorizer"]

    return model, vectorizer


# ==========================================================
# PREDICT TICKET CATEGORY
# ==========================================================

def predict_category(text: str) -> str:
    """
    Predict the category of a customer support message.

    Args:
        text: Customer's support message.

    Returns:
        Predicted ticket category.
    """

    # Validate the input.
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Please provide a non-empty customer message."
        )

    # Load the trained model and vectorizer.
    model, vectorizer = load_model()

    # Convert the customer message into TF-IDF features.
    text_features = vectorizer.transform([text])

    # Predict the ticket category.
    prediction = model.predict(text_features)

    # Return the predicted category.
    return str(prediction[0])


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    # Example customer support messages.
    sample_messages = [
        "I cannot log into my account.",
        "I was charged twice for my subscription.",
        "The application keeps crashing.",
        "Please add a dark mode feature.",
    ]

    print("=" * 55)
    print("SUPPORTIQ - TICKET CATEGORY PREDICTION")
    print("=" * 55)

    # Predict the category of each message.
    for message in sample_messages:

        category = predict_category(message)

        print(f"\nCustomer message: {message}")
        print(f"Predicted category: {category}")