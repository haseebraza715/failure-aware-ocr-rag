from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


class PdfPreprocessingError(RuntimeError):
    """Raised when Docling cannot produce a usable document representation."""


def convert_pdf_with_docling(pdf_path: Path) -> Any:
    """Convert one PDF with the pinned Docling runtime."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Docling input PDF is missing: {pdf_path}")
    document_converter = import_module("docling.document_converter")
    converter = document_converter.DocumentConverter()
    result = converter.convert(pdf_path)
    document = getattr(result, "document", None)
    if document is None:
        raise PdfPreprocessingError(f"Docling returned no document for {pdf_path}")
    return document


def export_docling_markdown(pdf_path: Path, output_path: Path) -> Path:
    """Convert a PDF and persist Docling's structured Markdown export."""
    document = convert_pdf_with_docling(pdf_path)
    markdown = document.export_to_markdown()
    if not str(markdown).strip():
        raise PdfPreprocessingError(f"Docling produced empty Markdown for {pdf_path}")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(markdown), encoding="utf-8")
    return output_path
