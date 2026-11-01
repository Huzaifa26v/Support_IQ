# Import Path to manage file and folder paths.
from pathlib import Path

# Import load_from_disk to load our locally saved Hugging Face dataset.
from datasets import load_from_disk


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

# Find the root directory of the SupportIQ project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the location of our downloaded dataset.
DATASET_PATH = PROJECT_ROOT / "ml" / "data" / "raw" / "triageiq"

# Define the columns used for training.
TEXT_COLUMN = "text"

# These are the three prediction targets.
TARGET_COLUMNS = [
    "category",
    "sentiment",
    "urgency",
]


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_supportiq_dataset():
    """
    Load the locally saved SupportIQ dataset.

    Returns:
        DatasetDict: Dataset containing train, validation, and test splits.
    """

    # Check whether the dataset directory exists.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}. "
            "Run ml/download_dataset.py first."
        )

    # Load the dataset from local storage.
    dataset = load_from_disk(str(DATASET_PATH))

    # Check that all required splits are available.
    required_splits = {"train", "validation", "test"}

    missing_splits = required_splits - set(dataset.keys())

    if missing_splits:
        raise ValueError(
            f"Missing dataset splits: {sorted(missing_splits)}"
        )

    # Check that all required columns exist in every split.
    required_columns = {TEXT_COLUMN, *TARGET_COLUMNS}

    for split_name, split_data in dataset.items():
        missing_columns = required_columns - set(split_data.column_names)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split_name}: "
                f"{sorted(missing_columns)}"
            )

    return dataset


# ==========================================================
# EXTRACT TEXT AND LABELS
# ==========================================================

def extract_features_and_labels(dataset, split_name: str, target_column: str):
    """
    Extract ticket text and one target label from a dataset split.

    Args:
        dataset: Loaded dataset containing all splits.
        split_name: Name of the split (train, validation, or test).
        target_column: Target column to predict.

    Returns:
        tuple: Ticket texts and corresponding labels.
    """

    # Validate the requested split.
    if split_name not in dataset:
        raise ValueError(f"Unknown dataset split: {split_name}")

    # Validate the requested prediction target.
    if target_column not in TARGET_COLUMNS:
        raise ValueError(
            f"Invalid target: {target_column}. "
            f"Choose one of {TARGET_COLUMNS}."
        )

    # Select the requested split.
    split_data = dataset[split_name]

    # Extract ticket messages.
    texts = split_data[TEXT_COLUMN]

    # Extract the corresponding labels.
    labels = split_data[target_column]

    return texts, labels