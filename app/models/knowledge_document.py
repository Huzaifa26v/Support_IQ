from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class KnowledgeDocument(Base):
    """
    Database model representing a document in the SupportIQ
    knowledge base.

    Documents are processed, divided into chunks, and stored
    in the vector database for RAG-based retrieval.
    """

    __tablename__ = "knowledge_documents"

    # Primary key
    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    # Original uploaded filename
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Supported document type: pdf or txt
    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Actual location of the stored document
    storage_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    # Document size in bytes
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # SHA-256 checksum for duplicate detection
    checksum: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    # Processing status: processing, completed, or failed
    status: Mapped[str] = mapped_column(
        String(30),
        default="processing",
        nullable=False,
    )

    # Number of chunks generated from this document
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Error details if document processing fails
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Record creation timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Record update timestamp
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Processing completion timestamp
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<KnowledgeDocument(id={self.id}, "
            f"file_name='{self.file_name}', "
            f"status='{self.status}')>"
        )