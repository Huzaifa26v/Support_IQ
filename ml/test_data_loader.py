# Import the functions we want to test.
from ml.data_loader import (
    load_supportiq_dataset,
    extract_features_and_labels,
    TARGET_COLUMNS,
)


def test_load_supportiq_dataset():
    """Check that the dataset loads with all required splits."""

    dataset = load_supportiq_dataset()

    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset


def test_dataset_has_required_columns():
    """Check that every split contains the required columns."""

    dataset = load_supportiq_dataset()

    for split_name in ["train", "validation", "test"]:
        columns = dataset[split_name].column_names

        assert "text" in columns

        for target in TARGET_COLUMNS:
            assert target in columns


def test_extract_features_and_labels():
    """Check that ticket texts and category labels are extracted."""

    dataset = load_supportiq_dataset()

    texts, labels = extract_features_and_labels(
        dataset,
        split_name="train",
        target_column="category",
    )

    assert len(texts) == len(labels)
    assert len(texts) > 0
    assert isinstance(texts[0], str)
    assert isinstance(labels[0], str)


def test_invalid_target_raises_error():
    """Check that an unsupported prediction target is rejected."""

    dataset = load_supportiq_dataset()

    try:
        extract_features_and_labels(
            dataset,
            split_name="train",
            target_column="invalid_target",
        )
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for invalid target."


def test_invalid_split_raises_error():
    """Check that an unknown dataset split is rejected."""

    dataset = load_supportiq_dataset()

    try:
        extract_features_and_labels(
            dataset,
            split_name="unknown",
            target_column="category",
        )
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for invalid split."