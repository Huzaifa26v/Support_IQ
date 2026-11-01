# Import Path to manage project directories.
from pathlib import Path

# Import Joblib to save machine learning artifacts.
import joblib

# Import Linear SVM classifier.
from sklearn.svm import LinearSVC

# Import evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Import dataset loading functions.
from ml.data_loader import (
    load_supportiq_dataset,
    extract_features_and_labels,
)

# Import TF-IDF vectorization functions.
from ml.preprocessing.vectorizer import (
    create_vectorizer,
    fit_transform_text,
    transform_text,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

# Find the root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the model artifacts directory.
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

# Define the urgency model output path.
MODEL_PATH = ARTIFACTS_DIR / "urgency_classifier.joblib"

# Define the urgency metadata output path.
METADATA_PATH = ARTIFACTS_DIR / "urgency_model_metadata.joblib"

# Specify the prediction target.
TARGET_COLUMN = "urgency"

# Use the best C value identified during experiments.
MODEL_CONFIG = {
    "C": 0.5,
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 5000,
}


# ==========================================================
# TRAIN AND SAVE URGENCY MODEL
# ==========================================================

def train_and_save_urgency_model():
    """
    Train the final urgency classifier using the combined
    training and validation datasets.

    Evaluate the final model using the untouched test set.

    Save the trained model, vectorizer and metadata.
    """

    print("=" * 65)
    print("SUPPORTIQ - FINAL URGENCY MODEL TRAINING")
    print("=" * 65)

    # Step 1: Create the artifacts directory.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: Load the dataset.
    print("\nLoading dataset...")

    dataset = load_supportiq_dataset()

    # Step 3: Extract training data.
    X_train_text, y_train = extract_features_and_labels(
        dataset,
        split_name="train",
        target_column=TARGET_COLUMN,
    )

    # Step 4: Extract validation data.
    X_val_text, y_val = extract_features_and_labels(
        dataset,
        split_name="validation",
        target_column=TARGET_COLUMN,
    )

    # Step 5: Extract test data.
    X_test_text, y_test = extract_features_and_labels(
        dataset,
        split_name="test",
        target_column=TARGET_COLUMN,
    )

    print(f"Training records: {len(X_train_text)}")
    print(f"Validation records: {len(X_val_text)}")
    print(f"Test records: {len(X_test_text)}")

    # Step 6: Convert Hugging Face Columns into Python lists.
    # This allows training and validation data to be combined.

    X_train_text = list(X_train_text)
    X_val_text = list(X_val_text)

    y_train = list(y_train)
    y_val = list(y_val)

    # Combine training and validation data.
    # The test dataset remains completely separate.

    X_final_train_text = X_train_text + X_val_text
    y_final_train = y_train + y_val

    print(
        f"\nFinal training records: "
        f"{len(X_final_train_text)}"
    )

    # Step 7: Create the TF-IDF vectorizer.
    vectorizer = create_vectorizer()

    # Learn the vocabulary using combined training data.
    print("\nCreating TF-IDF features...")

    X_final_train = fit_transform_text(
        vectorizer,
        X_final_train_text,
    )

    # Transform test data using the fitted vectorizer.
    # Do not fit the vectorizer on test data.

    X_test = transform_text(
        vectorizer,
        X_test_text,
    )

    print(f"Training feature shape: {X_final_train.shape}")
    print(f"Test feature shape: {X_test.shape}")

    # Step 8: Create and train the final Linear SVM.
    print("\nTraining final urgency classifier...")

    model = LinearSVC(**MODEL_CONFIG)

    model.fit(X_final_train, y_final_train)

    print("Training completed successfully!")

    # Step 9: Evaluate using the untouched test dataset.
    print("\nEvaluating final model on test dataset...")

    y_test_predictions = model.predict(X_test)

    # Calculate test accuracy.
    test_accuracy = accuracy_score(
        y_test,
        y_test_predictions,
    )

    # Calculate test macro F1-score.
    test_macro_f1 = f1_score(
        y_test,
        y_test_predictions,
        average="macro",
        zero_division=0,
    )

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1-score: {test_macro_f1:.4f}")

    # Generate classification report.
    print("\nClassification Report:")

    report = classification_report(
        y_test,
        y_test_predictions,
        zero_division=0,
    )

    print(report)

    # Generate confusion matrix.
    print("Confusion Matrix:")

    matrix = confusion_matrix(
        y_test,
        y_test_predictions,
        labels=model.classes_,
    )

    print(matrix)

    # Step 10: Save the trained model and vectorizer.
    model_artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "target_column": TARGET_COLUMN,
        "model_name": "Linear SVM",
        "model_config": MODEL_CONFIG,
    }

    joblib.dump(model_artifact, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")

    # Step 11: Save evaluation metadata.
    metadata = {
        "model_name": "Linear SVM",
        "algorithm": "Linear Support Vector Machine",
        "feature_extraction": "TF-IDF",
        "target_column": TARGET_COLUMN,
        "best_c_value": MODEL_CONFIG["C"],
        "training_records": len(X_final_train_text),
        "test_records": len(X_test_text),
        "classes": list(model.classes_),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "validation_accuracy_best": 0.7090,
        "validation_macro_f1_best": 0.7057,
    }

    joblib.dump(metadata, METADATA_PATH)

    print(f"Metadata saved at: {METADATA_PATH}")

    print("\nFinal urgency model artifacts created successfully!")

    return model_artifact, metadata


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    train_and_save_urgency_model()