# Import Path to manage file paths.
from pathlib import Path

# Import Joblib to load the saved ML model.
import joblib


# ==========================================================
# CONFIGURATION
# ==========================================================

# Find the SupportIQ project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the saved urgency model path.
MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "urgency_classifier.joblib"
)


# ==========================================================
# LOAD URGENCY MODEL
# ==========================================================

def load_urgency_model():
    """
    Load the trained urgency classifier and TF-IDF vectorizer.

    Returns:
        dict: Saved model artifacts.
    """

    # Check whether the model file exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Urgency model not found at: {MODEL_PATH}. "
            "Please run 'python -m ml.save_urgency_model' first."
        )

    # Load the saved model artifacts.
    model_artifact = joblib.load(MODEL_PATH)

    return model_artifact


# ==========================================================
# PREDICT TICKET URGENCY
# ==========================================================

def predict_urgency(text: str) -> str:
    """
    Predict the urgency level of a customer support ticket.

    Args:
        text (str): Customer ticket title and description.

    Returns:
        str: Predicted urgency (high, medium, or low).
    """

    # Validate the input.
    if not isinstance(text, str):
        raise TypeError("Ticket text must be a string.")

    # Reject empty or whitespace-only input.
    if not text.strip():
        raise ValueError("Ticket text cannot be empty.")

    # Load the trained model and vectorizer.
    model_artifact = load_urgency_model()

    model = model_artifact["model"]
    vectorizer = model_artifact["vectorizer"]

    # Convert ticket text into TF-IDF features.
    text_features = vectorizer.transform([text])

    # Predict the urgency level.
    prediction = model.predict(text_features)

    # Return the predicted class as a string.
    return str(prediction[0])


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    # Example customer support tickets.
    sample_tickets = [
        "My account has been hacked and I cannot access it.",
        "I have a question about my monthly subscription.",
        "My application is not responding and I cannot complete my payment.",
    ]

    print("=" * 60)
    print("SUPPORTIQ - URGENCY PREDICTION")
    print("=" * 60)

    # Predict urgency for each sample ticket.
    for ticket in sample_tickets:

        urgency = predict_urgency(ticket)

        print(f"\nTicket: {ticket}")
        print(f"Predicted urgency: {urgency}")