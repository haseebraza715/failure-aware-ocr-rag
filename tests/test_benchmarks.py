import json
from pathlib import Path

import pytest

from faar.benchmarks import BenchmarkRepository, build_ohr_asset_manifest
from faar.data import DatasetUnavailableError


def test_benchmark_repository_requires_complete_ocr_and_images(tmp_path: Path) -> None:
    record = {
        "example_id": "e1",
        "question": "Q",
        "correct_answer": "A",
        "ocr_text_path": str(tmp_path / "missing.txt"),
        "image_paths": [],
    }
    with pytest.raises(DatasetUnavailableError, match="not ready for a paper run"):
        BenchmarkRepository([record], tmp_path, "ohrbench", "test")


def test_ohr_manifest_uses_immutable_split(tmp_path: Path) -> None:
    (tmp_path / "OHR-Bench/data").mkdir(parents=True)
    (tmp_path / "ocr").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "split.json").write_text(json.dumps({"splits": {"test": ["e1"], "val": [], "train": []}}))
    (tmp_path / "OHR-Bench/data/qas_v2.json").write_text(
        json.dumps(
            [{"ID": "e1", "doc_name": "doc", "questions": "Q", "answers": "A", "evidence_page_no": 2}]
        )
    )
    (tmp_path / "ocr/e1.txt").write_text("OCR")
    (tmp_path / "images/e1_2.png").write_text("image")
    records = build_ohr_asset_manifest(tmp_path, "test", tmp_path / "ocr", tmp_path / "images")
    assert records[0]["example_id"] == "e1"
    assert records[0]["ocr_text_path"].endswith("ocr/e1.txt")
