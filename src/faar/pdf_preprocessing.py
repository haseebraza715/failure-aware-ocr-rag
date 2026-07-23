from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any


class PdfPreprocessingError(RuntimeError):
    """Raised when Docling cannot produce a usable document representation."""


def convert_pdf_with_docling(pdf_path: Path) -> Any:
    """Convert one PDF with the pinned Docling runtime and return the conversion result."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Docling input PDF is missing: {pdf_path}")
    document_converter = import_module("docling.document_converter")
    converter = document_converter.DocumentConverter()

    document_model = import_module("docling.datamodel.document")
    if hasattr(document_model, "DocumentConversionInput"):
        conversion_input = document_model.DocumentConversionInput.from_paths([pdf_path])
        results = list(converter.convert(conversion_input))
        if not results:
            raise PdfPreprocessingError(f"Docling returned no conversion result for {pdf_path}")
        result = results[0]
    else:
        result = converter.convert(pdf_path)

    status = getattr(result, "status", None)
    status_name = getattr(status, "name", str(status) if status is not None else "")
    pages = getattr(result, "pages", None)
    document = getattr(result, "document", None)
    if status_name and status_name != "SUCCESS":
        raise PdfPreprocessingError(f"Docling conversion status {status_name!r} for {pdf_path}")
    if document is None and not pages:
        raise PdfPreprocessingError(f"Docling returned no document/pages for {pdf_path}")
    return result


def _result_to_markdown(result: Any) -> str:
    if hasattr(result, "render_as_markdown"):
        return str(result.render_as_markdown())
    document = getattr(result, "document", None)
    if document is not None and hasattr(document, "export_to_markdown"):
        return str(document.export_to_markdown())
    return ""


def _result_to_audit_payload(result: Any, pdf_path: Path) -> dict[str, Any]:
    status = getattr(result, "status", None)
    pages = getattr(result, "pages", []) or []
    page_summaries = []
    for page in pages:
        page_summaries.append(
            {
                "page_no": getattr(page, "page_no", None),
                "page_hash": getattr(page, "page_hash", None),
                "width": getattr(getattr(page, "size", None), "width", None),
                "height": getattr(getattr(page, "size", None), "height", None),
                "cell_count": len(getattr(page, "cells", []) or []),
            }
        )
    payload: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "status": getattr(status, "name", str(status) if status is not None else None),
        "page_count": len(pages),
        "pages": page_summaries,
        "errors": [str(err) for err in (getattr(result, "errors", []) or [])],
    }
    return payload


def export_docling_markdown(pdf_path: Path, output_path: Path) -> Path:
    """Convert a PDF and persist Docling's structured audit Markdown (+ JSON sidecar)."""
    pdf_path = pdf_path.expanduser().resolve()
    result = convert_pdf_with_docling(pdf_path)
    markdown = _result_to_markdown(result).strip()
    audit = _result_to_audit_payload(result, pdf_path)
    if not markdown:
        # Image-heavy pages may yield empty markdown while still converting successfully.
        markdown = (
            f"<!-- docling audit: status={audit.get('status')} "
            f"pages={audit.get('page_count')} -->\n"
        )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown if markdown.endswith("\n") else markdown + "\n", encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    # Keep paths project-relative when possible for portability of the sidecar.
    try:
        from .asset_paths import to_relative_project_path

        project_root = Path.cwd().resolve()
        audit["pdf_path"] = to_relative_project_path(pdf_path, project_root)
    except Exception:
        audit["pdf_path"] = pdf_path.name
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit.get("page_count", 0) <= 0 and getattr(result, "document", None) is None:
        raise PdfPreprocessingError(f"Docling produced no pages for {pdf_path}")
    return output_path
