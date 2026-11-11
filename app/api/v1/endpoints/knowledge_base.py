# Import Path for safe document storage.
# Import shutil for reliable upload-directory cleanup.
import shutil
from pathlib import Path

# Import Annotated for FastAPI dependency declarations.
from typing import Annotated

# Import UUID to generate unique storage directories.
from uuid import uuid4

# Import FastAPI tools for file uploads and HTTP responses.
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

# Import SQLAlchemy tools for database queries.
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

# Import application settings.
from app.core.config import settings

# Import database dependency.
from app.db.session import get_db

# Import the knowledge document database model.
from app.models.knowledge_document import KnowledgeDocument

# Import the RAG ingestion service and duplicate exception.
from app.rag.ingestion_service import (
    DuplicateDocumentError,
    ingest_document,
)

# Import the document response schema.
from app.schemas.knowledge_document import KnowledgeDocumentResponse

# Create the Knowledge Base API router.
router = APIRouter(
    prefix="/knowledge-base",
    tags=["Knowledge Base"],
)

# Create a reusable database dependency.
DatabaseSession = Annotated[Session, Depends(get_db)]

# Supported document formats.
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Read uploaded documents in 1 MB blocks.
UPLOAD_CHUNK_SIZE = 1024 * 1024


# ==========================================================
# 1. UPLOAD KNOWLEDGE DOCUMENT
# ==========================================================
@router.post(
    "/upload",
    response_model=KnowledgeDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_knowledge_document(
    file: Annotated[UploadFile, File(description="PDF or TXT document")],
    db: DatabaseSession,
) -> KnowledgeDocument:
    """
    Upload and process a PDF or TXT knowledge document.

    The document is stored safely before being passed
    to the existing RAG ingestion service.
    """

    # Validate that a filename was provided.
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    # Normalize Windows and Linux path separators.
    original_filename = Path(file.filename.replace("\\", "/")).name

    # Extract and validate the document extension.
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and TXT documents are supported.",
        )

    # Validate the filename length against the database column.
    if len(original_filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot exceed 255 characters.",
        )

    # Create a unique directory for this uploaded document.
    document_directory = settings.knowledge_upload_dir / uuid4().hex
    document_path = document_directory / original_filename

    # Create the storage directory.
    document_directory.mkdir(parents=True, exist_ok=True)

    # Track the total uploaded bytes.
    total_size = 0

    try:
        # Save the uploaded document in small blocks.
        with document_path.open("wb") as destination:
            while True:
                chunk = file.file.read(UPLOAD_CHUNK_SIZE)

                # Stop when the upload is complete.
                if not chunk:
                    break

                total_size += len(chunk)

                # Reject files exceeding the configured limit.
                if total_size > settings.knowledge_max_upload_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="Document exceeds the maximum upload size of 10 MB.",
                    )

                # Write the current block to storage.
                destination.write(chunk)

        # Reject empty documents before ingestion.
        if total_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded document is empty.",
            )

        # Process the document and save it in the database.
        document = ingest_document(
            file_path=document_path,
            db=db,
        )

        # Return the processed document to FastAPI.
        return document

    except DuplicateDocumentError as exc:
        # Remove the newly uploaded duplicate document.
        shutil.rmtree(document_directory, ignore_errors=True)

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document already exists in the knowledge base.",
        ) from exc

    except HTTPException:
        # Clean up files after upload validation failures.
        shutil.rmtree(document_directory, ignore_errors=True)
        raise

    except ValueError as exc:
        # Clean up files after document-processing failures.
        shutil.rmtree(document_directory, ignore_errors=True)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document could not be processed.",
        ) from exc

    except Exception:
        # Clean up files after unexpected errors.
        shutil.rmtree(document_directory, ignore_errors=True)
        raise

    finally:
        # Close the uploaded file in every situation.
        file.file.close()
# ==========================================================
# 2. GET ALL KNOWLEDGE DOCUMENTS
# ==========================================================

@router.get(
    "/",
    response_model=list[KnowledgeDocumentResponse],
)
def get_knowledge_documents(
    db: DatabaseSession,

    # Number of records to skip.
    skip: Annotated[int, Query(ge=0)] = 0,

    # Maximum number of records to return.
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """
    Retrieve uploaded knowledge documents.

    Results are ordered from newest to oldest.
    """

    # Build the database query.
    statement = (
        select(KnowledgeDocument)
        .order_by(desc(KnowledgeDocument.created_at))
        .offset(skip)
        .limit(limit)
    )

    # Execute the query and return document records.
    return db.scalars(statement).all()