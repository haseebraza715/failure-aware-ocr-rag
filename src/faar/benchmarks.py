from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import DatasetUnavailableError
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

    def __init__(self, records: list[dict[str, Any]], project_root: Path, dataset: str, split: str) -> None:
        self.project_root = project_root
        self.dataset = dataset
        self.split = split
        self._records = {str(record["example_id"]): record for record in records}
        if not self._records:
            raise DatasetUnavailableError(f"No records were registered for {dataset} split {split}.")
        self._validate_assets()

    def _validate_assets(self) -> None:
        missing: list[str] = []
        for example_id, record in self._records.items():
            ocr_path = Path(record["ocr_text_path"])
            image_paths = [Path(path) for path in record.get("image_paths", [])]
            if not ocr_path.exists():
                missing.append(f"{example_id}: OCR text {ocr_path}")
            if not image_paths or any(not path.exists() for path in image_paths):
                missing.append(f"{example_id}: one or more page images")
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
        ocr_path = Path(record["ocr_text_path"])
        return Phase0Example(
            example_id=example_id,
            doc_name=str(record.get("doc_name", "")),
            question=str(record["question"]),
            correct_answer=str(record["correct_answer"]),
            page_ids=[int(page) for page in record.get("page_ids", [])],
            ocr_text=ocr_path.read_text(errors="ignore"),
            ocr_text_path=ocr_path,
            gt_text_path=None,
            image_paths=[Path(path) for path in record["image_paths"]],
            metadata=dict(record.get("metadata") or {}),
        )


def load_benchmark_repository(project_root: Path, dataset: str, split: str) -> BenchmarkRepository:
    dataset_key = _normalise_dataset(dataset)
    manifest_path = project_root / "data/benchmark_assets" / dataset_key / f"{split}.json"
    if not manifest_path.exists():
        raise DatasetUnavailableError(
            f"Missing benchmark asset manifest: {manifest_path}. "
            "Create it with register_benchmark_assets.py after GOT-OCR and page rendering are complete."
        )
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise DatasetUnavailableError(f"Asset manifest {manifest_path} must contain a records list.")
    return BenchmarkRepository(records, project_root, dataset_key, split)


def build_ohr_asset_manifest(project_root: Path, split: str, ocr_dir: Path, image_dir: Path) -> list[dict[str, Any]]:
    split_payload = json.loads((project_root / "split.json").read_text())
    if split not in split_payload["splits"]:
        raise ValueError(f"Unknown OHR-Bench split {split!r}.")
    selected_ids = set(split_payload["splits"][split])
    source_rows = json.loads((project_root / "OHR-Bench/data/qas_v2.json").read_text())
    source_by_id = {str(row["ID"]): row for row in source_rows}
    records: list[dict[str, Any]] = []
    for example_id in sorted(selected_ids):
        row = source_by_id[example_id]
        image_paths = sorted(str(path.resolve()) for path in image_dir.glob(f"{example_id}_*.png"))
        page_number = row.get("evidence_page_no", 0)
        records.append(
            {
                "example_id": example_id,
                "doc_name": row["doc_name"],
                "question": row["questions"],
                "correct_answer": row["answers"],
                "page_ids": _listify(page_number),
                "ocr_text_path": str((ocr_dir / f"{example_id}.txt").resolve()),
                "image_paths": image_paths,
                "metadata": {
                    "doc_type": row.get("doc_type", ""),
                    "evidence_source": row.get("evidence_source", ""),
                },
            }
        )
    return records
