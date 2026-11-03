from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import KnowledgeDocument
from app.rag.ingestion_service import (
    calculate_checksum,
    find_existing_document,
)

# ==========================================================
# TEST DATABASE FIXTURE
# ==========================================================

@pytest.fixture
def db_session():
    """
    Create an isolated in-memory SQLite database for testing.

    StaticPool keeps the same in-memory database connection
    available throughout the test.
    """

    # Create a temporary SQLite database in memory.
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all registered database tables.
    Base.metadata.create_all(bind=test_engine)

    # Create a session factory for database operations.
    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    # Open a database session for the test.
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        # Roll back changes and close the test session.
        db.rollback()
        db.close()

        # Remove the temporary database tables.
        Base.metadata.drop_all(bind=test_engine)

        # Close the test database engine.
        test_engine.dispose()


# ==========================================================
# CHECKSUM TESTS
# ==========================================================

def test_calculate_checksum(tmp_path: Path):
    """Verify that a document receives a SHA-256 checksum."""

    # Create a temporary document.
    file_path = tmp_path / "sample.txt"
    file_path.write_text("SupportIQ test document", encoding="utf-8")

    # Calculate the document checksum.
    checksum = calculate_checksum(file_path)

    # SHA-256 checksums contain exactly 64 hexadecimal characters.
    assert len(checksum) == 64
    assert all(character in "0123456789abcdef" for character in checksum)


def test_identical_files_have_same_checksum(tmp_path: Path):
    """Verify that identical file contents produce identical checksums."""

    # Create two files containing identical content.
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("SupportIQ knowledge base", encoding="utf-8")
    second_file.write_text("SupportIQ knowledge base", encoding="utf-8")

    # Calculate checksums for both files.
    first_checksum = calculate_checksum(first_file)
    second_checksum = calculate_checksum(second_file)

    # Both checksums should be identical.
    assert first_checksum == second_checksum


def test_different_files_have_different_checksums(tmp_path: Path):
    """Verify that different file contents produce different checksums."""

    # Create two files with different contents.
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("Customer support", encoding="utf-8")
    second_file.write_text("Technical support", encoding="utf-8")

    # Calculate both checksums.
    first_checksum = calculate_checksum(first_file)
    second_checksum = calculate_checksum(second_file)

    # Different contents should produce different checksums.
    assert first_checksum != second_checksum


def test_checksum_missing_file_raises_error(tmp_path: Path):
    """Verify that missing documents are rejected."""

    # Define a path that does not exist.
    missing_file = tmp_path / "missing.txt"

    # The checksum function should raise an error.
    with pytest.raises(FileNotFoundError):
        calculate_checksum(missing_file)


# ==========================================================
# DUPLICATE DOCUMENT TESTS
# ==========================================================

def test_find_existing_document(db_session):
    """Verify that an existing document can be found by checksum."""

    # Create a document with a unique checksum.
    checksum = "a" * 64

    document = KnowledgeDocument(
        file_name="policy.txt",
        file_type="txt",
        storage_path="data/policy.txt",
        file_size=100,
        checksum=checksum,
    )

    # Save the document in the test database.
    db_session.add(document)
    db_session.flush()

    # Search for the document using its checksum.
    result = find_existing_document(db_session, checksum)

    # Verify that the correct document was retrieved.
    assert result is not None
    assert result.file_name == "policy.txt"
    assert result.checksum == checksum


def test_find_non_existing_document(db_session):
    """Verify that an unknown checksum returns None."""

    # Search using a checksum that is not stored.
    result = find_existing_document(db_session, "b" * 64)

    # No matching document should be returned.
    assert result is None


    # ==========================================================
# DOCUMENT INGESTION TESTS
# ==========================================================

def test_ingest_document_success(
    tmp_path,
    db_session,
    monkeypatch,
):
    """Verify that a document is successfully ingested."""

    from app.rag import ingestion_service

    # Create a temporary text document.
    file_path = tmp_path / "support_policy.txt"
    file_path.write_text(
        "SupportIQ provides customer support services.",
        encoding="utf-8",
    )

    # Mock document loading to avoid external dependencies.
    monkeypatch.setattr(
        ingestion_service,
        "load_document",
        lambda path: {
            "text": "SupportIQ provides customer support services.",
            "metadata": {
                "file_name": "support_policy.txt",
                "file_type": ".txt",
            },
        },
    )

    # Mock vector storage and return sample chunk IDs.
    monkeypatch.setattr(
        ingestion_service,
        "add_chunks",
        lambda chunks, document_metadata=None: ["chunk-1", "chunk-2"],
    )

    # Mock vector cleanup.
    monkeypatch.setattr(
        ingestion_service,
        "delete_chunks",
        lambda chunk_ids: None,
    )

    # Process the document.
    document = ingestion_service.ingest_document(
        db_session,
        file_path,
    )

    # Verify the completed document record.
    assert document.status == "completed"
    assert document.file_name == "support_policy.txt"
    assert document.chunk_count == 2
    assert document.processed_at is not None
    assert document.error_message is None


def test_ingest_document_rejects_duplicate(
    tmp_path,
    db_session,
):
    """Verify that duplicate documents are rejected."""

    from app.rag.ingestion_service import ingest_document

    # Create a temporary document.
    file_path = tmp_path / "duplicate.txt"
    file_path.write_text(
        "Duplicate document content.",
        encoding="utf-8",
    )

    # Ingest the document for the first time.
    ingest_document(db_session, file_path)

    # Attempt to ingest the same document again.
    with pytest.raises(ValueError, match="already exists"):
        ingest_document(db_session, file_path)


def test_ingest_document_marks_failure(
    tmp_path,
    db_session,
    monkeypatch,
):
    """Verify that processing failures update document status."""

    from app.rag import ingestion_service

    # Create a temporary document.
    file_path = tmp_path / "invalid_content.txt"
    file_path.write_text(
        "Sample document content.",
        encoding="utf-8",
    )

    # Simulate a document-loading failure.
    def mock_load_document(path):
        raise ValueError("Document extraction failed.")

    monkeypatch.setattr(
        ingestion_service,
        "load_document",
        mock_load_document,
    )

    # Attempt ingestion and expect the original error.
    with pytest.raises(ValueError, match="Document extraction failed"):
        ingestion_service.ingest_document(
            db_session,
            file_path,
        )

    # Retrieve the failed document record.
    checksum = ingestion_service.calculate_checksum(file_path)

    failed_document = ingestion_service.find_existing_document(
        db_session,
        checksum,
    )

    # Verify the failure status and error message.
    assert failed_document is not None
    assert failed_document.status == "failed"
    assert failed_document.error_message == "Document extraction failed."
    assert failed_document.chunk_count == 0