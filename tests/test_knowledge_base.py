# Import datetime utilities for document timestamps.
from datetime import UTC, datetime

# Import Path for temporary document storage.
# Import pytest for fixtures and mocking.
# Import the Knowledge Base endpoint module for mocking.
from app.api.v1.endpoints import knowledge_base

# Import the KnowledgeDocument database model.
from app.models.knowledge_document import KnowledgeDocument

# Import the custom exception for duplicate documents.
from app.rag.ingestion_service import DuplicateDocumentError

# ==========================================================
# 1. SUCCESSFUL DOCUMENT UPLOAD
# ==========================================================

def test_upload_knowledge_document(client, monkeypatch, tmp_path):
    """Test that a valid document uploads successfully."""

    # Use a temporary directory instead of permanent storage.
    monkeypatch.setattr(
        knowledge_base.settings,
        "knowledge_upload_dir",
        tmp_path,
    )

    # Create a sample database document for the mocked service.
    now = datetime.now(UTC)

    document = KnowledgeDocument(
        id=1,
        file_name="guide.txt",
        file_type="txt",
        storage_path=str(tmp_path / "guide.txt"),
        file_size=20,
        checksum="a" * 64,
        status="completed",
        chunk_count=2,
        created_at=now,
        updated_at=now,
        processed_at=now,
    )

    # Mock ingestion so the test does not call embeddings or ChromaDB.
    monkeypatch.setattr(
        knowledge_base,
        "ingest_document",
        lambda **kwargs: document,
    )

    # Send a sample TXT document to the upload endpoint.
    response = client.post(
        "/api/v1/knowledge-base/upload",
        files={
            "file": (
                "guide.txt",
                b"This is a knowledge base guide.",
                "text/plain",
            )
        },
    )

    # Verify that the API returns HTTP 201.
    assert response.status_code == 201

    # Verify the returned document information.
    assert response.json()["file_name"] == "guide.txt"
    assert response.json()["status"] == "completed"


# ==========================================================
# 2. UNSUPPORTED DOCUMENT FORMAT
# ==========================================================

def test_upload_unsupported_file_format(client):
    """Test that unsupported file formats are rejected."""

    # Send an unsupported file extension.
    response = client.post(
        "/api/v1/knowledge-base/upload",
        files={
            "file": (
                "document.docx",
                b"Unsupported document",
                "application/octet-stream",
            )
        },
    )

    # Verify HTTP 415 Unsupported Media Type.
    assert response.status_code == 415


# ==========================================================
# 3. EMPTY DOCUMENT
# ==========================================================

def test_upload_empty_document(client, monkeypatch, tmp_path):
    """Test that an empty document is rejected."""

    # Configure temporary document storage.
    monkeypatch.setattr(
        knowledge_base.settings,
        "knowledge_upload_dir",
        tmp_path,
    )

    # Track whether ingestion is called.
    ingestion_called = False

    def mock_ingestion(**kwargs):
        nonlocal ingestion_called
        ingestion_called = True

    # Replace the real ingestion service with the mock.
    monkeypatch.setattr(
        knowledge_base,
        "ingest_document",
        mock_ingestion,
    )

    # Upload an empty TXT file.
    response = client.post(
        "/api/v1/knowledge-base/upload",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    # Verify HTTP 400 Bad Request.
    assert response.status_code == 400

    # Verify ingestion was never started.
    assert ingestion_called is False


# ==========================================================
# 4. DOCUMENT EXCEEDING MAXIMUM SIZE
# ==========================================================

def test_upload_document_exceeding_size_limit(
    client,
    monkeypatch,
    tmp_path,
):
    """Test that oversized documents are rejected."""

    # Configure temporary document storage.
    monkeypatch.setattr(
        knowledge_base.settings,
        "knowledge_upload_dir",
        tmp_path,
    )

    # Set a small size limit to simplify the test.
    monkeypatch.setattr(
        knowledge_base.settings,
        "knowledge_max_upload_size",
        10,
    )

    # Track whether ingestion is called.
    ingestion_called = False

    def mock_ingestion(**kwargs):
        nonlocal ingestion_called
        ingestion_called = True

    # Replace the real ingestion service.
    monkeypatch.setattr(
        knowledge_base,
        "ingest_document",
        mock_ingestion,
    )

    # Upload a document larger than 10 bytes.
    response = client.post(
        "/api/v1/knowledge-base/upload",
        files={
            "file": (
                "large.txt",
                b"This document exceeds the configured limit.",
                "text/plain",
            )
        },
    )

    # Verify HTTP 413 Content Too Large.
    assert response.status_code == 413

    # Verify ingestion was never started.
    assert ingestion_called is False


# ==========================================================
# 5. DUPLICATE DOCUMENT
# ==========================================================

def test_upload_duplicate_document(client, monkeypatch, tmp_path):
    """Test that duplicate documents return HTTP 409."""

    # Configure temporary document storage.
    monkeypatch.setattr(
        knowledge_base.settings,
        "knowledge_upload_dir",
        tmp_path,
    )

    # Mock ingestion to raise the custom duplicate exception.
    def mock_ingestion(**kwargs):
        raise DuplicateDocumentError(
            "Document already exists."
        )

    monkeypatch.setattr(
        knowledge_base,
        "ingest_document",
        mock_ingestion,
    )

    # Upload a document that already exists.
    response = client.post(
        "/api/v1/knowledge-base/upload",
        files={
            "file": (
                "duplicate.txt",
                b"This document already exists.",
                "text/plain",
            )
        },
    )

    # Verify HTTP 409 Conflict.
    assert response.status_code == 409


# ==========================================================
# 6. LIST KNOWLEDGE DOCUMENTS
# ==========================================================

def test_get_knowledge_documents(client):
    """Test that the API returns the knowledge document list."""

    # Request all knowledge documents.
    response = client.get(
        "/api/v1/knowledge-base/"
    )

    # Verify HTTP 200 OK.
    assert response.status_code == 200

    # A fresh test database should initially contain no documents.
    assert response.json() == []