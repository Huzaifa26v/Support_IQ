# Import Linear Support Vector Classifier.
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
# EXPERIMENT CONFIGURATION
# ==========================================================

# Define the target column.
TARGET_COLUMN = "urgency"

# Different C values to experiment with.
C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

# Common configuration for every model.
COMMON_CONFIG = {
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 5000,
}


# ==========================================================
# MODEL COMPARISON
# ==========================================================

def compare_urgency_models():
    """
    Train and compare multiple Linear SVM models
    using different C values.

    The best model is selected using validation
    macro F1-score.
    """

    print("=" * 65)
    print("SUPPORTIQ - URGENCY MODEL HYPERPARAMETER EXPERIMENTS")
    print("=" * 65)

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

    # Step 4: Create TF-IDF vectorizer.
    vectorizer = create_vectorizer()

    # Fit the vectorizer only on training data.
    print("\nCreating TF-IDF features...")

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

    # Store experiment results.
    results = []

    # Store the best model.
    best_model = None
    best_c = None
    best_f1 = -1.0

    # Step 5: Train models with different C values.
    for c_value in C_VALUES:

        print("\n" + "-" * 65)
        print(f"Training Linear SVM with C = {c_value}")
        print("-" * 65)

        # Create the model.
        model = LinearSVC(
            C=c_value,
            **COMMON_CONFIG,
        )

        # Train using training data only.
        model.fit(X_train, y_train)

        # Predict validation labels.
        y_pred = model.predict(X_val)

        # Calculate evaluation metrics.
        accuracy = accuracy_score(y_val, y_pred)

        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # Store results.
        results.append({
            "C": c_value,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")

        # Keep the model with the highest macro F1.
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model
            best_c = c_value

    # Step 6: Display comparison results.
    print("\n" + "=" * 65)
    print("URGENCY MODEL COMPARISON RESULTS")
    print("=" * 65)

    print(
        f"\n{'C Value':<15}"
        f"{'Accuracy':<20}"
        f"{'Macro F1-score':<20}"
    )

    print("-" * 55)

    # Sort results by macro F1-score.
    sorted_results = sorted(
        results,
        key=lambda result: result["macro_f1"],
        reverse=True,
    )

    for result in sorted_results:
        print(
            f"{result['C']:<15}"
            f"{result['accuracy']:<20.4f}"
            f"{result['macro_f1']:<20.4f}"
        )

    # Step 7: Display the best model.
    print("\n" + "=" * 65)
    print("BEST URGENCY MODEL")
    print("=" * 65)

    print(f"Best C value: {best_c}")
    print(f"Best validation Macro F1: {best_f1:.4f}")

    # Evaluate the best model in more detail.
    best_predictions = best_model.predict(X_val)

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            best_predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            best_predictions,
            labels=best_model.classes_,
        )
    )

    # Return experiment results and the best model.
    return best_model, best_c, sorted_results


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    compare_urgency_models()