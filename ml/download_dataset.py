# Import the Hugging Face function used to download datasets.
from datasets import load_dataset

# Import Path to manage project directories and file paths.
from pathlib import Path


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

# Hugging Face dataset repository ID.
DATASET_ID = "coldstart88/triageiq-dataset"

# Find the root directory of the SupportIQ project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define where the downloaded dataset will be stored.
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "triageiq"


# ==========================================================
# DOWNLOAD DATASET
# ==========================================================

def download_dataset():

    # Create the output directory if it does not exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SupportIQ training dataset...")

    # Download the dataset from Hugging Face.
    dataset = load_dataset(DATASET_ID)

    # Display the available dataset splits.
    print("\nAvailable dataset splits:")

    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} records")

    # Save the downloaded dataset locally.
    dataset.save_to_disk(str(OUTPUT_DIR))

    print("\nDataset downloaded successfully!")
    print(f"Saved location: {OUTPUT_DIR}")


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    download_dataset()