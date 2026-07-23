from __future__ import annotations

import json
from pathlib import Path

import pytest

from faar.benchmarks import BenchmarkRepository, build_ohr_asset_manifest
from faar.data import DatasetUnavailableError
from faar.settings import RetrievalSettings


def _write_ohr_fixture(tmp_path: Path, *, pages: range, evidence_page: int = 2) -> None:
    (tmp_path / "OHR-Bench/data/retrieval_base/gt").mkdir(parents=True)
    (tmp_path / "ocr").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "split.json").write_text(json.dumps({"splits": {"test": ["e1"], "val": [], "train": []}}))
    (tmp_path / "OHR-Bench/data/qas_v2.json").write_text(
        json.dumps(
            [
                {
                    "ID": "e1",
                    "doc_name": "doc",
                    "questions": "Q",
                    "answers": "A",
                    "evidence_page_no": evidence_page,
                }
            ]
        )
    )
    inventory_rows = [{"text": f"gt {page}", "page_idx": page} for page in pages]
    (tmp_path / "OHR-Bench/data/retrieval_base/gt/doc.json").write_text(json.dumps(inventory_rows))
    for page in pages:
        (tmp_path / "ocr" / f"doc_page_{page}.txt").write_text(f"OCR {page}")
        (tmp_path / "images" / f"doc_page_{page}.png").write_bytes(b"image")


def test_benchmark_repository_requires_complete_ocr_and_images(tmp_path: Path) -> None:
    record = {
        "example_id": "e1",
        "doc_name": "doc",
        "question": "Q",
        "correct_answer": "A",
        "page_ids": [1],
        "corpus_ids": ["doc:p1"],
    }
    corpus_pages = [
        {
            "corpus_id": "doc:p1",
            "doc_name": "doc",
            "page_id": 1,
            "ocr_text_path": "missing.txt",
            "image_path": "",
        }
    ]
    with pytest.raises(DatasetUnavailableError, match="not ready for a paper run|incomplete"):
        BenchmarkRepository(
            [record],
            corpus_pages,
            tmp_path,
            "ohrbench",
            "test",
            document_inventory={"doc": [1]},
        )


def test_ohr_manifest_keeps_evidence_metadata_but_indexes_all_document_pages(tmp_path: Path) -> None:
    _write_ohr_fixture(tmp_path, pages=range(1, 6), evidence_page=2)

    manifest = build_ohr_asset_manifest(tmp_path, "test", tmp_path / "ocr", tmp_path / "images")

    assert manifest["records"][0]["example_id"] == "e1"
    assert manifest["records"][0]["page_ids"] == [2]
    assert manifest["records"][0]["metadata"]["evidence_page_ids"] == [2]
    assert manifest["records"][0]["corpus_ids"] == [f"doc:p{page}" for page in range(1, 6)]
    assert [page["page_id"] for page in manifest["corpus_pages"]] == [1, 2, 3, 4, 5]
    assert all(not Path(page["image_path"]).is_absolute() for page in manifest["corpus_pages"])
    assert all(not Path(page["ocr_text_path"]).is_absolute() for page in manifest["corpus_pages"])
    assert manifest["document_inventory"]["doc"] == [1, 2, 3, 4, 5]

    repository = BenchmarkRepository(
        manifest["records"],
        manifest["corpus_pages"],
        tmp_path,
        "ohrbench",
        "test",
        document_inventory=manifest["document_inventory"],
    )
    chunks = repository.get_corpus_chunks(RetrievalSettings(chunk_size_words=8, chunk_overlap_words=0))
    assert {chunk.page_id for chunk in chunks} == {1, 2, 3, 4, 5}


def test_ohr_manifest_rejects_evidence_only_assets(tmp_path: Path) -> None:
    _write_ohr_fixture(tmp_path, pages=range(1, 6), evidence_page=2)
    # Keep inventory for pages 1-5, but delete non-evidence OCR/image assets.
    for page in (1, 3, 4, 5):
        (tmp_path / "ocr" / f"doc_page_{page}.txt").unlink()
        (tmp_path / "images" / f"doc_page_{page}.png").unlink()

    with pytest.raises(DatasetUnavailableError, match="incomplete relative to the document inventory"):
        build_ohr_asset_manifest(tmp_path, "test", tmp_path / "ocr", tmp_path / "images")


def test_load_rejects_path_traversal_in_manifest(tmp_path: Path) -> None:
    asset_dir = tmp_path / "data/benchmark_assets/ohrbench"
    asset_dir.mkdir(parents=True)
    (tmp_path / "safe.png").write_bytes(b"png")
    (tmp_path / "safe.txt").write_text("OCR")
    payload = {
        "records": [
            {
                "example_id": "e1",
                "doc_name": "doc",
                "question": "Q",
                "correct_answer": "A",
                "page_ids": [1],
                "corpus_ids": ["doc:p1"],
            }
        ],
        "corpus_pages": [
            {
                "corpus_id": "doc:p1",
                "doc_name": "doc",
                "page_id": 1,
                "ocr_text_path": "../outside.txt",
                "image_path": "safe.png",
            }
        ],
        "document_inventory": {"doc": [1]},
    }
    (asset_dir / "test.json").write_text(json.dumps(payload))

    from faar.benchmarks import load_benchmark_repository
    from faar.asset_paths import AssetPathError

    with pytest.raises((DatasetUnavailableError, AssetPathError, ValueError), match="\\.\\.|escapes|not ready"):
        load_benchmark_repository(tmp_path, "ohrbench", "test")
