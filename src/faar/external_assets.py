from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = {"mpdocvqa", "arxivqa"}

_ALIASES = {
    "id": ("example_id", "question_id", "qa_id", "id", "ID"),
    "question": ("question", "questions", "query"),
    "answer": ("correct_answer", "answers", "answer", "ground_truth", "gold_answer"),
    "document": ("doc_name", "document_id", "doc_id", "document", "document_name", "file_name"),
    "page": ("page_ids", "page_id", "page_no", "page", "page_number", "evidence_page_no"),
    "image": ("image_paths", "image_path", "page_images", "page_image", "image"),
    "ocr": ("ocr_text_paths", "ocr_text_path", "ocr_paths", "ocr_path", "ocr_file", "ocr"),
}

_PLACEHOLDER_VALUES = {
    "-",
    "?",
    "n/a",
    "na",
    "none",
    "null",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
}


class ExternalAssetError(ValueError):
    """Raised when external benchmark data cannot form a paper-ready manifest."""


def normalise_dataset_name(dataset: str) -> str:
    key = dataset.lower().replace("-", "").replace("_", "")
    aliases = {"mpdocvqa": "mpdocvqa", "arxivqa": "arxivqa"}
    try:
        return aliases[key]
    except KeyError as exc:
        raise ExternalAssetError(
            f"Unsupported external dataset {dataset!r}; expected MP-DocVQA or ArXivQA."
        ) from exc


def _first(row: dict[str, Any], kind: str) -> Any:
    for key in _ALIASES[kind]:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    return (
        lowered in _PLACEHOLDER_VALUES
        or "placeholder" in lowered
        or lowered.startswith(("path/to/", "your/path", "your_"))
        or bool(re.fullmatch(r"<[^>]+>", text))
    )


def _required_text(row: dict[str, Any], kind: str, row_label: str) -> str:
    value = _first(row, kind)
    if kind == "answer" and isinstance(value, list):
        value = next((item for item in value if not _is_placeholder(item)), None)
    if isinstance(value, dict):
        value = value.get("id") or value.get("name") or value.get("value")
    if _is_placeholder(value):
        raise ExternalAssetError(f"{row_label} has a missing or placeholder {kind}.")
    return str(value).strip()


def _page_number(value: Any, row_label: str) -> int:
    if isinstance(value, dict):
        value = _first(value, "page")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExternalAssetError(f"{row_label} has invalid page value {value!r}; expected an integer.") from exc


def _resolve_asset_path(value: Any, asset_root: Path, row_label: str, kind: str) -> str:
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("uri")
    if _is_placeholder(value):
        raise ExternalAssetError(f"{row_label} has a missing or placeholder {kind} path.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = asset_root / path
    path = path.resolve()
    if not path.is_file():
        raise ExternalAssetError(f"{row_label} references a missing {kind} file: {path}")
    return str(path)


def _extract_val_rows(payload: Any) -> list[dict[str, Any]]:
    split_proven = False
    rows: Any = payload
    if isinstance(payload, dict):
        declared_split = payload.get("split") or payload.get("dataset_split")
        if declared_split is not None:
            if str(declared_split).strip().lower() != "val":
                raise ExternalAssetError(f"External benchmark source must be exactly the val split, not {declared_split!r}.")
            split_proven = True
        if "val" in payload:
            rows = payload["val"]
            split_proven = True
        else:
            rows = payload.get("records", payload.get("data", payload.get("examples")))
    if not isinstance(rows, list) or not rows:
        raise ExternalAssetError("External benchmark source must contain a non-empty list of val records.")
    if not all(isinstance(row, dict) for row in rows):
        raise ExternalAssetError("Every external benchmark record must be a JSON object.")

    row_splits = {str(row.get("split") or row.get("dataset_split")).strip().lower() for row in rows if row.get("split") or row.get("dataset_split")}
    if row_splits:
        if row_splits != {"val"}:
            raise ExternalAssetError(f"External benchmark records must all be val; found splits {sorted(row_splits)}.")
        if len(row_splits) == 1 and all(row.get("split") or row.get("dataset_split") for row in rows):
            split_proven = True
    if not split_proven:
        raise ExternalAssetError(
            "Could not verify the exact val split. Use a top-level 'val' key, set top-level split='val', "
            "or mark every record split='val'."
        )
    return rows


def _page_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    nested_pages = row.get("pages") or row.get("document_pages")
    if isinstance(nested_pages, list) and nested_pages:
        merged: list[dict[str, Any]] = []
        for page in nested_pages:
            if not isinstance(page, dict):
                raise ExternalAssetError("Nested page entries must be JSON objects.")
            item = {**row, **page}
            for kind, canonical_key in (("page", "page_id"), ("image", "image_path"), ("ocr", "ocr_text_path")):
                page_value = _first(page, kind)
                item[canonical_key] = page_value if page_value is not None else _first(row, kind)
            merged.append(item)
        return merged

    pages = _as_list(_first(row, "page"))
    images = _as_list(_first(row, "image"))
    ocr_paths = _as_list(_first(row, "ocr"))
    count = max(len(pages), len(images), len(ocr_paths))
    if count == 0:
        return [row]
    for values, label in ((pages, "page"), (images, "image"), (ocr_paths, "OCR")):
        if len(values) not in {1, count}:
            raise ExternalAssetError(f"Mismatched {label} list length: expected 1 or {count}, got {len(values)}.")
    expanded = []
    for index in range(count):
        item = dict(row)
        item["page_id"] = pages[index if len(pages) > 1 else 0] if pages else None
        item["image_path"] = images[index if len(images) > 1 else 0] if images else None
        item["ocr_text_path"] = ocr_paths[index if len(ocr_paths) > 1 else 0] if ocr_paths else None
        expanded.append(item)
    return expanded


def build_external_asset_manifest(
    source: Path,
    dataset: str,
    *,
    asset_root: Path | None = None,
) -> dict[str, Any]:
    """Normalize an MP-DocVQA/ArXivQA val JSON file into FAAR's asset schema."""
    dataset_key = normalise_dataset_name(dataset)
    source = source.resolve()
    if not source.is_file():
        raise ExternalAssetError(f"External benchmark source does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalAssetError(f"Could not read external benchmark JSON {source}: {exc}") from exc
    rows = _extract_val_rows(payload)
    root = (asset_root or source.parent).resolve()
    if not root.is_dir():
        raise ExternalAssetError(f"Asset root does not exist or is not a directory: {root}")

    records: list[dict[str, Any]] = []
    corpus_by_id: dict[str, dict[str, Any]] = {}
    seen_examples: set[str] = set()
    for row_index, row in enumerate(rows):
        row_label = f"record {row_index}"
        example_id = _required_text(row, "id", row_label)
        if example_id in seen_examples:
            raise ExternalAssetError(f"Duplicate example id {example_id!r}.")
        seen_examples.add(example_id)
        question = _required_text(row, "question", example_id)
        answer = _required_text(row, "answer", example_id)
        document_id = _required_text(row, "document", example_id)

        query_page_ids: list[int] = []
        query_images: list[str] = []
        query_ocr: list[str] = []
        corpus_ids: list[str] = []
        for page_row in _page_rows(row):
            page_id = _page_number(_first(page_row, "page"), example_id)
            image_path = _resolve_asset_path(_first(page_row, "image"), root, example_id, "image")
            ocr_path = _resolve_asset_path(_first(page_row, "ocr"), root, example_id, "OCR")
            corpus_id = f"{document_id}:p{page_id}"
            corpus_page = {
                "corpus_id": corpus_id,
                "doc_name": document_id,
                "page_id": page_id,
                "ocr_text_path": ocr_path,
                "image_path": image_path,
            }
            existing = corpus_by_id.get(corpus_id)
            if existing is not None and existing != corpus_page:
                raise ExternalAssetError(f"Conflicting assets for corpus page {corpus_id!r}.")
            corpus_by_id[corpus_id] = corpus_page
            query_page_ids.append(page_id)
            query_images.append(image_path)
            query_ocr.append(ocr_path)
            corpus_ids.append(corpus_id)

        records.append(
            {
                "example_id": example_id,
                "doc_name": document_id,
                "question": question,
                "correct_answer": answer,
                "page_ids": query_page_ids,
                "ocr_text_path": query_ocr[0],
                "ocr_text_paths": query_ocr,
                "image_paths": query_images,
                "corpus_ids": corpus_ids,
                "metadata": {"dataset": dataset_key, "split": "val", "document_id": document_id},
            }
        )

    return {
        "dataset": dataset_key,
        "split": "val",
        "source": str(source),
        "records": records,
        "corpus_pages": [corpus_by_id[key] for key in sorted(corpus_by_id)],
    }


def validate_manifest_payload(payload: dict[str, Any]) -> None:
    """Validate a normalized payload without relying on the source JSON."""
    if payload.get("split") != "val":
        raise ExternalAssetError("External asset manifest split must be exactly 'val'.")
    if normalise_dataset_name(str(payload.get("dataset", ""))) not in SUPPORTED_DATASETS:
        raise ExternalAssetError("External asset manifest has an unsupported dataset.")
    records = payload.get("records")
    pages = payload.get("corpus_pages")
    if not isinstance(records, list) or not records or not isinstance(pages, list) or not pages:
        raise ExternalAssetError("External asset manifest requires non-empty records and corpus_pages lists.")
    page_ids = {page.get("corpus_id") for page in pages if isinstance(page, dict)}
    for record in records:
        if not isinstance(record, dict) or not set(record.get("corpus_ids", [])).issubset(page_ids):
            raise ExternalAssetError("Every query record must reference registered corpus_pages.")
