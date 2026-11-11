# Import datetime for document timestamps.
from datetime import datetime

# Import Literal to restrict document status to valid values.
from typing import Literal

# Import Pydantic tools for schema creation and validation.
from pydantic import BaseModel, ConfigDict

# ==========================================================
# KNOWLEDGE DOCUMENT RESPONSE SCHEMA
# ==========================================================

class KnowledgeDocumentResponse(BaseModel):
    """
    Schema used when returning knowledge document information.

    It defines which document details the API exposes.
    """

    # Unique document ID from the database.
    id: int

    # Original name of the uploaded document.
    file_name: str

    # Document format, such as pdf or txt.
    file_type: str

    # Original document size in bytes.
    file_size: int

    # Current document processing status.
    status: Literal["processing", "completed", "failed"]

    # Number of text chunks created from the document.
    chunk_count: int

    # Document creation timestamp.
    created_at: datetime

    # Last update timestamp.
    updated_at: datetime

    # Timestamp when document processing completed.
    # This can be None if processing has not completed.
    processed_at: datetime | None

    # Allow Pydantic to read values directly from SQLAlchemy objects.
    model_config = ConfigDict(
        from_attributes=True,
    )