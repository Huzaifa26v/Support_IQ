from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: str | Path) -> dict:
    """
    Load a PDF or TXT document and extract its text.

    Returns:
        A dictionary containing the extracted text and metadata.
    """

    path = Path(file_path)

    # Validate file existence
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Validate file extension
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    # Extract text according to file type
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDF documents are not supported.")

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    else:
        text = path.read_text(encoding="utf-8-sig")

    # Reject empty documents
    text = text.strip()

    if not text:
        raise ValueError("The document contains no extractable text.")

    return {
        "text": text,
        "metadata": {
            "file_name": path.name,
            "file_type": path.suffix.lower(),
            "file_path": str(path.resolve()),
        },
    }