
# Standard library imports.
import hashlib
from datetime import UTC, datetime
from pathlib import Path

# SQLAlchemy imports.
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Project imports.
from app.models.knowledge_document import KnowledgeDocument
from app.rag.chunker import chunk_text
from app.rag.document_loader import load_document
from app.rag.vector_store import add_chunks, delete_chunks


# Custom exception for duplicate knowledge documents.
class DuplicateDocumentError(ValueError):
    """Raised when a document already exists in the knowledge base."""


# ==========================================================
# 1. CALCULATE DOCUMENT CHECKSUM
# ==========================================================

def calculate_checksum(file_path: str | Path) -> str:
    """
    Generate a SHA-256 checksum for a document.

    Reading the file in chunks avoids loading large files
    entirely into memory.
    """

    path = Path(file_path)

    # Ensure the document exists.
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    # Initialize SHA-256 hashing.
    sha256 = hashlib.sha256()

    # Read the document in 8 KB blocks.
    with path.open("rb") as document:
        for chunk in iter(lambda: document.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================================
# 2. FIND EXISTING DOCUMENT
# ==========================================================

def find_existing_document(
    db: Session,
    checksum: str,
) -> KnowledgeDocument | None:
    """
    Find an existing knowledge document using its checksum.

    Returns the existing document if found.
    Otherwise, returns None.
    """

    statement = select(KnowledgeDocument).where(
        KnowledgeDocument.checksum == checksum
    )

    return db.scalar(statement)


# ==========================================================
# 3. INGEST KNOWLEDGE DOCUMENT
# ==========================================================

def ingest_document(
    db: Session,
    file_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> KnowledgeDocument:
    """
    Process a document and store its chunks in ChromaDB.

    Workflow:
        1. Calculate document checksum.
        2. Reject duplicate documents.
        3. Create a SQLite document record.
        4. Extract document text.
        5. Divide text into chunks.
        6. Store chunks and embeddings in ChromaDB.
        7. Update the SQLite record.

    If processing fails, mark the document as failed and
    remove any vectors already inserted.
    """

    path = Path(file_path)

    # Calculate the document checksum.
    checksum = calculate_checksum(path)

    # Check whether this document already exists.
    existing_document = find_existing_document(db, checksum)

    if existing_document:
        raise DuplicateDocumentError(
            f"Document already exists: {existing_document.file_name}"
        )

    # Create the initial database record.
    document = KnowledgeDocument(
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        storage_path=str(path.resolve()),
        file_size=path.stat().st_size,
        checksum=checksum,
        status="processing",
    )

    db.add(document)

    try:
        # Commit to obtain a permanent document ID.
        db.commit()
        db.refresh(document)

    except IntegrityError as exc:
        # Handle duplicate insertion from another request.
        db.rollback()

        raise DuplicateDocumentError(
            "This document has already been registered."
        ) from exc

    # Track inserted vectors for possible cleanup.
    chunk_ids = []

    try:
        # Extract text from the uploaded document.
        loaded_document = load_document(path)

        # Divide extracted text into smaller chunks.
        chunks = chunk_text(
            loaded_document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Prepare metadata for ChromaDB.
        metadata = {
            "document_id": str(document.id),
            "file_name": document.file_name,
            "file_type": document.file_type,
        }

        # Generate embeddings and store document chunks.
        chunk_ids = add_chunks(
            chunks,
            document_metadata=metadata,
        )

        # Update the database record after successful processing.
        document.status = "completed"
        document.chunk_count = len(chunk_ids)
        document.error_message = None
        document.processed_at = datetime.now(UTC)

        db.commit()
        db.refresh(document)

        return document

    except Exception as exc:
        # Remove vectors if they were already inserted.
        if chunk_ids:
            try:
                delete_chunks(chunk_ids)
            except Exception:
                # Preserve the original processing exception.
                pass

        # Roll back incomplete database operations.
        db.rollback()

        # Retrieve the document record again.
        failed_document = db.get(
            KnowledgeDocument,
            document.id,
        )

        if failed_document is not None:
            # Record the processing failure.
            failed_document.status = "failed"
            failed_document.error_message = str(exc)[:2000]
            failed_document.chunk_count = 0

            db.commit()

        # Re-raise the original exception.
        raise