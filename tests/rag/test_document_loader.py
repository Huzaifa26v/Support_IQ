from pathlib import Path

import pytest

from app.rag.document_loader import load_document


def test_load_txt_document(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "SupportIQ customer support policy.",
        encoding="utf-8",
    )

    result = load_document(file_path)

    assert result["text"] == "SupportIQ customer support policy."
    assert result["metadata"]["file_name"] == "sample.txt"
    assert result["metadata"]["file_type"] == ".txt"


def test_load_missing_document(tmp_path: Path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(file_path)


def test_unsupported_file_type(tmp_path: Path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("Sample content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


def test_empty_document(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no extractable text"):
        load_document(file_path)


def test_load_directory_instead_of_file(tmp_path: Path):
    with pytest.raises(ValueError, match="Path is not a file"):
        load_document(tmp_path)