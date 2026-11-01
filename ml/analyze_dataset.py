# Import Path to manage project directories.
from pathlib import Path

# Import pandas for dataset analysis.
import pandas as pd

# Import load_from_disk to load our local Hugging Face dataset.
from datasets import load_from_disk


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "ml" / "data" / "raw" / "triageiq"


# ==========================================================
# LOAD DATASET
# ==========================================================

dataset = load_from_disk(str(DATASET_PATH))


# ==========================================================
# DATASET ANALYSIS
# ==========================================================

def analyze_dataset():

    print("=" * 60)
    print("SUPPORTIQ DATASET ANALYSIS")
    print("=" * 60)

    for split_name, split_data in dataset.items():

        # Convert the Hugging Face split into a pandas DataFrame.
        df = split_data.to_pandas()

        print("\n" + "=" * 60)
        print(f"{split_name.upper()} DATASET")
        print("=" * 60)

        # Display total records and columns.
        print(f"\nTotal records: {len(df)}")
        print(f"Total columns: {len(df.columns)}")

        # Check for missing values.
        print("\nMISSING VALUES:")
        print(df.isnull().sum())

        # Check for duplicate records.
        print("\nDUPLICATE RECORDS:")
        print(df.duplicated().sum())

        # Analyze category distribution.
        print("\nCATEGORY DISTRIBUTION:")
        print(df["category"].value_counts())

        # Analyze sentiment distribution.
        print("\nSENTIMENT DISTRIBUTION:")
        print(df["sentiment"].value_counts())

        # Analyze urgency distribution.
        print("\nURGENCY DISTRIBUTION:")
        print(df["urgency"].value_counts())


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    analyze_dataset()