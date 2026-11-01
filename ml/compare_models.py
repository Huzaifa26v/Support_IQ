# Import both classification algorithms.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Import evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

TARGET_COLUMN = "category"

# Define the models we want to compare.
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
    ),
}


# ==========================================================
# MODEL COMPARISON
# ==========================================================

def compare_models():
    """
    Train and compare classification algorithms
    using the same training and validation datasets.
    """

    print("=" * 65)
    print("SUPPORTIQ - MACHINE LEARNING MODEL COMPARISON")
    print("=" * 65)

    # Step 1: Load dataset.
    dataset = load_supportiq_dataset()

    # Step 2: Extract training data.
    X_train_text, y_train = extract_features_and_labels(
        dataset,
        "train",
        TARGET_COLUMN,
    )

    # Step 3: Extract validation data.
    X_val_text, y_val = extract_features_and_labels(
        dataset,
        "validation",
        TARGET_COLUMN,
    )

    print(f"\nTraining records: {len(X_train_text)}")
    print(f"Validation records: {len(X_val_text)}")

    # Step 4: Create TF-IDF features.
    vectorizer = create_vectorizer()

    X_train = fit_transform_text(
        vectorizer,
        X_train_text,
    )

    X_val = transform_text(
        vectorizer,
        X_val_text,
    )

    # Store model evaluation results.
    results = []

    # Step 5: Train and evaluate each model.
    for model_name, model in MODELS.items():

        print(f"\nTraining {model_name}...")

        # Train using training data only.
        model.fit(X_train, y_train)

        # Predict validation labels.
        y_pred = model.predict(X_val)

        # Calculate evaluation metrics.
        accuracy = accuracy_score(y_val, y_pred)

        precision = precision_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # Save the results.
        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    # Step 6: Display comparison table.
    print("\n" + "=" * 65)
    print("MODEL COMPARISON RESULTS")
    print("=" * 65)

    print(
        f"\n{'Model':<25}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
    )

    print("-" * 73)

    for result in results:

        print(
            f"{result['model']:<25}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1_score']:<12.4f}"
        )

    # Step 7: Identify the model with the highest F1-score.
    best_model = max(
        results,
        key=lambda result: result["f1_score"],
    )

    print("\n" + "=" * 65)
    print("BEST PERFORMING MODEL")
    print("=" * 65)

    print(f"Model: {best_model['model']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"F1-score: {best_model['f1_score']:.4f}")

    return results, best_model


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    compare_models()