# Import Path to manage file and folder paths.
from pathlib import Path

# Import Joblib to save trained machine learning objects.
import joblib

# Import Linear SVM classifier.
from sklearn.svm import LinearSVC

# Import our dataset loading functions.
from ml.data_loader import (
    load_supportiq_dataset,
    extract_features_and_labels,
)

# Import TF-IDF vectorization functions.
from ml.preprocessing.vectorizer import (
    create_vectorizer,
    fit_transform_text,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

# Find the root directory of the SupportIQ project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define where trained models will be stored.
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

# Define the output model file.
MODEL_PATH = ARTIFACTS_DIR / "category_classifier.joblib"

# Define the model metadata file.
METADATA_PATH = ARTIFACTS_DIR / "category_model_metadata.joblib"

# Prediction target.
TARGET_COLUMN = "category"

# Configure the selected Linear SVM model.
MODEL_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 5000,
}


# ==========================================================
# TRAIN AND SAVE MODEL
# ==========================================================

def train_and_save_model():
    """
    Train the selected Linear SVM model and save
    both the classifier and TF-IDF vectorizer.
    """

    print("=" * 60)
    print("SUPPORTIQ - MODEL TRAINING AND SAVING")
    print("=" * 60)

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

    print(f"Training records: {len(X_train_text)}")

    # Step 4: Create the TF-IDF vectorizer.
    vectorizer = create_vectorizer()

    # Learn vocabulary using training data only.
    print("\nConverting training text into TF-IDF features...")

    X_train = fit_transform_text(
        vectorizer,
        X_train_text,
    )

    print(f"Training feature shape: {X_train.shape}")

    # Step 5: Create and train Linear SVM.
    print("\nTraining Linear SVM...")

    model = LinearSVC(**MODEL_CONFIG)

    model.fit(X_train, y_train)

    print("Training completed!")

    # Step 6: Save model and vectorizer together.
    model_artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "target_column": TARGET_COLUMN,
        "model_name": "Linear SVM",
    }

    joblib.dump(model_artifact, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")

    # Step 7: Save metadata separately.
    metadata = {
        "model_name": "Linear SVM",
        "algorithm": "Linear Support Vector Machine",
        "feature_extraction": "TF-IDF",
        "target_column": TARGET_COLUMN,
        "training_records": len(X_train_text),
        "classes": list(model.classes_),
        "validation_accuracy": 0.8595,
        "validation_macro_f1": 0.8568,
    }

    joblib.dump(metadata, METADATA_PATH)

    print(f"Metadata saved at: {METADATA_PATH}")

    print("\nModel artifacts created successfully!")

    return model_artifact, metadata


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    train_and_save_model()