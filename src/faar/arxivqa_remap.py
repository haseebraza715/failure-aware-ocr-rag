from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArXivQARemapError(ValueError):
    """Raised when ArXivQA cannot be mapped to a complete paper corpus."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArXivQARemapError(f"Could not read JSON source {path}: {exc}") from exc


def _rows(payload: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("val") or payload.get("records") or payload.get("data")
    if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
        raise ArXivQARemapError(f"{label} must contain a non-empty list of JSON objects.")
    return payload


def _text(row: dict[str, Any], keys: tuple[str, ...], label: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ArXivQARemapError(f"{label} is missing one of: {', '.join(keys)}.")


def _page_id(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArXivQARemapError(f"{label} has invalid page id {value!r}.") from exc


def build_full_paper_source(
    qa_source: Path,
    paper_inventory: Path,
    figure_mapping: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    """Build a strict full-paper ArXivQA source for external asset registration.

    The figure mapping is deliberately explicit. It must identify the paper and
    answer page for every QA row; retrieval later receives every page in the
    paper inventory, not only the mapped evidence page.
    """
    qa_rows = _rows(_read_json(qa_source), "ArXivQA source")
    inventory_payload = _read_json(paper_inventory)
    inventory_rows = inventory_payload.get("papers") if isinstance(inventory_payload, dict) else None
    if not isinstance(inventory_rows, list) or not inventory_rows:
        raise ArXivQARemapError("Paper inventory must contain a non-empty 'papers' list.")
    mapping_rows = _rows(_read_json(figure_mapping), "figure mapping")

    papers: dict[str, dict[str, Any]] = {}
    for index, paper in enumerate(inventory_rows):
        if not isinstance(paper, dict):
            raise ArXivQARemapError(f"papers[{index}] must be a JSON object.")
        paper_id = _text(paper, ("paper_id", "doc_id", "id"), f"papers[{index}]")
        pages = paper.get("pages")
        if not isinstance(pages, list) or not pages:
            raise ArXivQARemapError(f"Paper {paper_id!r} has no complete page inventory.")
        normalized_pages: list[dict[str, Any]] = []
        seen_pages: set[int] = set()
        for page in pages:
            if not isinstance(page, dict):
                raise ArXivQARemapError(f"Paper {paper_id!r} has a non-object page entry.")
            page_id = _page_id(page.get("page_id", page.get("page_number")), paper_id)
            if page_id in seen_pages:
                raise ArXivQARemapError(f"Paper {paper_id!r} repeats page {page_id}.")
            image_path = page.get("image_path") or page.get("page_image")
            ocr_path = page.get("ocr_text_path") or page.get("ocr_path") or page.get("ocr_file")
            if not image_path or not ocr_path:
                raise ArXivQARemapError(f"Paper {paper_id!r} page {page_id} needs image_path and ocr_text_path.")
            seen_pages.add(page_id)
            normalized_pages.append(
                {"page_id": page_id, "image_path": str(image_path), "ocr_text_path": str(ocr_path)}
            )
        papers[paper_id] = {"paper_id": paper_id, "pages": sorted(normalized_pages, key=lambda p: p["page_id"])}

    mappings: dict[str, dict[str, Any]] = {}
    for index, mapping in enumerate(mapping_rows):
        qa_id = _text(mapping, ("qa_id", "example_id", "id"), f"figure mapping[{index}]")
        if qa_id in mappings:
            raise ArXivQARemapError(f"Figure mapping repeats QA id {qa_id!r}.")
        paper_id = _text(mapping, ("paper_id", "doc_id", "document_id"), qa_id)
        if paper_id not in papers:
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} references unknown paper {paper_id!r}.")
        raw_pages = mapping.get("evidence_page_ids") or mapping.get("answer_page_ids") or mapping.get("page_ids")
        if not isinstance(raw_pages, list) or not raw_pages:
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} needs explicit evidence_page_ids.")
        evidence_pages = [_page_id(value, qa_id) for value in raw_pages]
        known_pages = {page["page_id"] for page in papers[paper_id]["pages"]}
        if not set(evidence_pages).issubset(known_pages):
            raise ArXivQARemapError(f"Figure mapping {qa_id!r} points outside paper {paper_id!r}'s page inventory.")
        mappings[qa_id] = {
            "paper_id": paper_id,
            "evidence_page_ids": evidence_pages,
            "figure_id": mapping.get("figure_id") or mapping.get("figure_image"),
        }

    records: list[dict[str, Any]] = []
    for index, row in enumerate(qa_rows):
        qa_id = _text(row, ("example_id", "qa_id", "question_id", "id"), f"ArXivQA row {index}")
        mapping = mappings.get(qa_id)
        if mapping is None:
            raise ArXivQARemapError(f"ArXivQA row {qa_id!r} has no explicit figure-to-paper mapping.")
        paper_id = mapping["paper_id"]
        question = _text(row, ("question", "questions", "query"), qa_id)
        answer = _text(row, ("answer", "answers", "ground_truth", "correct_answer"), qa_id)
        if isinstance(row.get("answers"), list):
            answer = _text({"answer": next((item for item in row["answers"] if str(item).strip()), None)}, ("answer",), qa_id)
        pages = papers[paper_id]["pages"]
        records.append(
            {
                "id": qa_id,
                "question": question,
                "answer": answer,
                "document": paper_id,
                "evidence_page_ids": mapping["evidence_page_ids"],
                "metadata": {
                    "remapping_method": "full_paper",
                    "source_qa_id": qa_id,
                    "figure_id": mapping["figure_id"],
                },
            }
        )

    payload = {
        "split": "val",
        "remapping": {
            "method": "full_paper",
            "qa_source": str(qa_source),
            "paper_inventory": str(paper_inventory),
            "figure_mapping": str(figure_mapping),
        },
        "data": records,
        "documents": [
            {"doc_id": paper["paper_id"], "pages": paper["pages"]}
            for paper in sorted(papers.values(), key=lambda item: item["paper_id"])
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
