# Import Path to manage file locations.
from pathlib import Path

# Import load_from_disk to read our locally saved dataset.
from datasets import load_from_disk


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

# Find the root directory of the SupportIQ project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the location of our downloaded dataset.
DATASET_PATH = PROJECT_ROOT / "ml" / "data" / "raw" / "triageiq"


# ==========================================================
# LOAD DATASET
# ==========================================================

# Load the dataset from local storage.
dataset = load_from_disk(str(DATASET_PATH))


# ==========================================================
# DISPLAY DATASET INFORMATION
# ==========================================================

print("=" * 60)
print("SUPPORTIQ DATASET INSPECTION")
print("=" * 60)

# Display the complete dataset structure.
print("\nDATASET STRUCTURE:")
print(dataset)

# Display available splits.
print("\nAVAILABLE SPLITS:")
print(list(dataset.keys()))


# ==========================================================
# INSPECT EACH SPLIT
# ==========================================================

for split_name, split_data in dataset.items():

    print("\n" + "=" * 60)
    print(f"SPLIT: {split_name.upper()}")
    print("=" * 60)

    # Display the number of records.
    print(f"\nTotal records: {len(split_data)}")

    # Display column names.
    print("\nCOLUMN NAMES:")
    print(split_data.column_names)

    # Display dataset features and data types.
    print("\nDATA TYPES:")
    print(split_data.features)

    # Display the first record.
    print("\nFIRST RECORD:")
    print(split_data[0])

    # Display the second record if available.
    if len(split_data) > 1:
        print("\nSECOND RECORD:")
        print(split_data[1])