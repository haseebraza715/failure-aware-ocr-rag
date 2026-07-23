from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .chunking import build_page_chunks
from .data import DatasetUnavailableError
from .settings import RetrievalSettings
from .types import Phase0Example


def _normalise_dataset(dataset: str) -> str:
    return dataset.lower().replace("-", "").replace("_", "")


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


class BenchmarkRepository:
    """Loads the complete, explicitly registered assets required for a paper run."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        corpus_pages: list[dict[str, Any]],
        project_root: Path,
        dataset: str,
        split: str,
    ) -> None:
        self.project_root = project_root
        self.dataset = dataset
        self.split = split
        self._records = {str(record["example_id"]): record for record in records}
        self._corpus_pages = corpus_pages
        if not self._records:
            raise DatasetUnavailableError(f"No records were registered for {dataset} split {split}.")
        if not self._corpus_pages:
            raise DatasetUnavailableError(f"No shared corpus pages were registered for {dataset} split {split}.")
        self._validate_assets()

    def _validate_assets(self) -> None:
        missing: list[str] = []
        for page in self._corpus_pages:
            page_id = str(page.get("corpus_id", "<missing corpus id>"))
            text_path = Path(page["ocr_text_path"]) if page.get("ocr_text_path") else None
            image_path = Path(page["image_path"]) if page.get("image_path") else None
            if not str(page.get("text", "")).strip() and (text_path is None or not text_path.exists()):
                missing.append(f"{page_id}: OCR text")
            if image_path is None or not image_path.exists():
                missing.append(f"{page_id}: page image")
        if missing:
            preview = "\n".join(missing[:10])
            raise DatasetUnavailableError(
                f"{self.dataset} {self.split} is not ready for a paper run. "
                "Register complete GOT-OCR text and page images for every selected example.\n"
                f"Missing assets (first 10):\n{preview}"
            )

    def list_example_ids(self) -> list[str]:
        return sorted(self._records)

    def get_example(self, example_id: str) -> Phase0Example:
        record = self._records[example_id]
        return Phase0Example(
            example_id=example_id,
            doc_name=str(record.get("doc_name", "")),
            question=str(record["question"]),
            correct_answer=str(record["correct_answer"]),
            page_ids=[int(page_id) for page_id in record.get("page_ids", [])],
            ocr_text="",
            ocr_text_path=self.project_root / ".query-only",
            gt_text_path=None,
            image_paths=[],
            metadata=dict(record.get("metadata") or {}),
        )

    def get_corpus_chunks(self, settings: RetrievalSettings):
        chunks = []
        for page in self._corpus_pages:
            text = str(page.get("text", ""))
            if not text and page.get("ocr_text_path"):
                text = Path(page["ocr_text_path"]).read_text(errors="ignore")
            chunks.extend(
                build_page_chunks(
                    example_id=str(page["corpus_id"]),
                    doc_name=str(page["doc_name"]),
                    page_id=int(page["page_id"]),
                    page_text=text,
                    settings=settings,
                    image_path=str(page["image_path"]),
                )
            )
        if not chunks:
            raise DatasetUnavailableError(f"The registered {self.dataset} corpus produced no retrievable chunks.")
        return chunks

    def corpus_image_paths(self) -> list[Path]:
        return [Path(page["image_path"]) for page in self._corpus_pages]


def load_benchmark_repository(project_root: Path, dataset: str, split: str) -> BenchmarkRepository:
    dataset_key = _normalise_dataset(dataset)
    manifest_path = project_root / "data/benchmark_assets" / dataset_key / f"{split}.json"
    if not manifest_path.exists():
        raise DatasetUnavailableError(
            f"Missing benchmark asset manifest: {manifest_path}. "
            "Create it with register_benchmark_assets.py after GOT-OCR and page rendering are complete."
        )
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else None
    corpus_pages = payload.get("corpus_pages") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not isinstance(corpus_pages, list):
        raise DatasetUnavailableError(f"Asset manifest {manifest_path} must contain records and corpus_pages lists.")
    return BenchmarkRepository(records, corpus_pages, project_root, dataset_key, split)


def build_ohr_asset_manifest(project_root: Path, split: str, ocr_dir: Path, image_dir: Path) -> dict[str, list[dict[str, Any]]]:
    split_payload = json.loads((project_root / "split.json").read_text())
    if split not in split_payload["splits"]:
        raise ValueError(f"Unknown OHR-Bench split {split!r}.")
    selected_ids = set(split_payload["splits"][split])
    source_rows = json.loads((project_root / "OHR-Bench/data/qas_v2.json").read_text())
    source_by_id = {str(row["ID"]): row for row in source_rows}
    records: list[dict[str, Any]] = []
    for example_id in sorted(selected_ids):
        row = source_by_id[example_id]
        page_ids = [int(page) for page in _listify(row.get("evidence_page_no"))]
        records.append(
            {
                "example_id": example_id,
                "doc_name": row["doc_name"],
                "question": row["questions"],
                "correct_answer": row["answers"],
                "page_ids": page_ids,
                "corpus_ids": [f"{row['doc_name']}:p{page_id}" for page_id in page_ids],
                "metadata": {
                    "doc_type": row.get("doc_type", ""),
                    "evidence_source": row.get("evidence_source", ""),
                },
            }
        )
    selected_documents = {str(record["doc_name"]) for record in records}
    corpus_pages = _load_ohr_corpus_pages(ocr_dir, image_dir, selected_documents)
    return {"records": records, "corpus_pages": corpus_pages}


def _load_ohr_corpus_pages(
    ocr_dir: Path,
    image_dir: Path,
    selected_documents: set[str],
) -> list[dict[str, Any]]:
    pages_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(ocr_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("pages", payload.get("data", []))
        doc_name = str(path.relative_to(ocr_dir).with_suffix(""))
        if doc_name not in selected_documents:
            continue
        for row in rows:
            page_id = int(row.get("page_idx", row.get("page_no", row.get("page_id", 0))))
            text = str(row.get("text", ""))
            image_path = _find_page_image(image_dir, doc_name, page_id)
            corpus_id = f"{doc_name}:p{page_id}"
            pages_by_id[corpus_id] = {
                "corpus_id": corpus_id,
                "doc_name": doc_name,
                "page_id": page_id,
                "text": text,
                "image_path": str(image_path.resolve()) if image_path else "",
            }
    for path in sorted(ocr_dir.rglob("*.txt")):
        relative = path.relative_to(ocr_dir)
        match = re.match(r"(.+?)(?:_page_|_p)(\d+)$", relative.stem)
        if not match:
            continue
        doc_name = str(relative.with_name(match.group(1)).with_suffix(""))
        if doc_name not in selected_documents:
            continue
        page_id = int(match.group(2))
        image_path = _find_page_image(image_dir, doc_name, page_id)
        corpus_id = f"{doc_name}:p{page_id}"
        page = {
            "corpus_id": corpus_id,
            "doc_name": doc_name,
            "page_id": page_id,
            "ocr_text_path": str(path.resolve()),
            "image_path": str(image_path.resolve()) if image_path else "",
        }
        existing = pages_by_id.get(corpus_id)
        if existing is None or not str(existing.get("text", "")).strip():
            pages_by_id[corpus_id] = page
    return [pages_by_id[corpus_id] for corpus_id in sorted(pages_by_id)]


def _find_page_image(image_dir: Path, doc_name: str, page_id: int) -> Path | None:
    base = image_dir / doc_name
    candidates = [
        base.with_name(f"{base.name}_page_{page_id}.png"),
        base.with_name(f"{base.name}_page_{page_id}.jpg"),
        base.with_name(f"{base.name}_p{page_id}.png"),
        base.with_name(f"{base.name}_p{page_id}.jpg"),
    ]
    return next((path for path in candidates if path.exists()), None)
