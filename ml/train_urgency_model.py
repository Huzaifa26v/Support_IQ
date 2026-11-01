# Import Linear Support Vector Classifier.
from sklearn.svm import LinearSVC

# Import evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
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
# MODEL CONFIGURATION
# ==========================================================

# Define the prediction target.
TARGET_COLUMN = "urgency"

# Configure Linear SVM.
MODEL_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 5000,
}


# ==========================================================
# MODEL TRAINING
# ==========================================================

def train_svm_urgency_model():
    """
    Train a ticket urgency classification model
    using TF-IDF and Linear SVM.
    """

    print("=" * 60)
    print("SUPPORTIQ - LINEAR SVM URGENCY CLASSIFICATION")
    print("=" * 60)

    # Step 1: Load the dataset.
    print("\nLoading dataset...")

    dataset = load_supportiq_dataset()

    # Step 2: Extract training data.
    X_train_text, y_train = extract_features_and_labels(
        dataset,
        split_name="train",
        target_column=TARGET_COLUMN,
    )

    # Step 3: Extract validation data.
    X_val_text, y_val = extract_features_and_labels(
        dataset,
        split_name="validation",
        target_column=TARGET_COLUMN,
    )

    print(f"Training records: {len(X_train_text)}")
    print(f"Validation records: {len(X_val_text)}")

    # Display available urgency classes.
    print("\nUrgency classes:")
    print(sorted(set(y_train)))

    # Step 4: Create TF-IDF vectorizer.
    vectorizer = create_vectorizer()

    # Learn vocabulary only from training data.
    print("\nConverting training text into TF-IDF features...")

    X_train = fit_transform_text(
        vectorizer,
        X_train_text,
    )

    # Transform validation data using the same vocabulary.
    X_val = transform_text(
        vectorizer,
        X_val_text,
    )

    print(f"Training feature shape: {X_train.shape}")
    print(f"Validation feature shape: {X_val.shape}")

    # Step 5: Create Linear SVM model.
    model = LinearSVC(**MODEL_CONFIG)

    # Train the model.
    print("\nTraining Linear SVM urgency model...")

    model.fit(X_train, y_train)

    print("Model training completed!")

    # Step 6: Predict validation labels.
    print("\nEvaluating model on validation data...")

    y_pred = model.predict(X_val)

    # Step 7: Calculate accuracy.
    accuracy = accuracy_score(y_val, y_pred)

    print("\n" + "=" * 60)
    print("LINEAR SVM URGENCY VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            y_pred,
            labels=model.classes_,
        )
    )

    # Return trained components for later comparison and saving.
    return model, vectorizer, accuracy


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    train_svm_urgency_model()