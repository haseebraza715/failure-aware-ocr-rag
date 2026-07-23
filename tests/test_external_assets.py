from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from faar.benchmarks import BenchmarkRepository
from faar.external_assets import ExternalAssetError, build_external_asset_manifest
from faar.settings import RetrievalSettings


def _asset_files(root: Path) -> None:
    (root / "assets").mkdir()
    (root / "assets/page-1.png").write_bytes(b"png")
    (root / "assets/page-1.txt").write_text("OCR page one")
    (root / "assets/page-2.png").write_bytes(b"png")
    (root / "assets/page-2.txt").write_text("OCR page two")


def test_normalizes_aliases_and_deduplicates_corpus_pages(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "data": [
                    {
                        "question_id": "q1",
                        "query": "What is the total?",
                        "answers": ["42", "forty two"],
                        "doc_id": "doc-a",
                        "page_no": 1,
                        "image_path": "assets/page-1.png",
                        "ocr_path": "assets/page-1.txt",
                    },
                    {
                        "id": "q2",
                        "question": "Which document?",
                        "answer": "A",
                        "document_id": "doc-a",
                        "page": 1,
                        "image": "assets/page-1.png",
                        "ocr": "assets/page-1.txt",
                    },
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "MP-DocVQA")

    assert manifest["dataset"] == "mpdocvqa"
    assert manifest["split"] == "val"
    assert len(manifest["records"]) == 2
    assert manifest["records"][0]["correct_answer"] == "42"
    assert manifest["records"][0]["corpus_ids"] == ["doc-a:p1"]
    assert len(manifest["corpus_pages"]) == 1
    assert Path(manifest["corpus_pages"][0]["image_path"]).is_absolute()

    repository = BenchmarkRepository(
        manifest["records"], manifest["corpus_pages"], tmp_path, "mpdocvqa", "val"
    )
    chunks = repository.get_corpus_chunks(RetrievalSettings(chunk_size_words=10, chunk_overlap_words=0))
    assert chunks[0].example_id == "doc-a:p1"


def test_normalizes_nested_arxiv_pages(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "arxivqa.json"
    source.write_text(
        json.dumps(
            {
                "val": [
                    {
                        "qa_id": "a1",
                        "questions": "Where is the result?",
                        "ground_truth": "Page two",
                        "document": {"id": "paper-7"},
                        "pages": [
                            {"page_number": 1, "page_image": "assets/page-1.png", "ocr_file": "assets/page-1.txt"},
                            {"page_number": 2, "page_image": "assets/page-2.png", "ocr_file": "assets/page-2.txt"},
                        ],
                    }
                ]
            }
        )
    )

    manifest = build_external_asset_manifest(source, "arxivqa")

    assert manifest["records"][0]["page_ids"] == [1, 2]
    assert manifest["records"][0]["corpus_ids"] == ["paper-7:p1", "paper-7:p2"]
    assert len(manifest["corpus_pages"]) == 2


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"split": "test", "data": [{"id": "q"}]}, "exactly the val split"),
        ({"data": [{"id": "q"}]}, "Could not verify the exact val split"),
    ],
)
def test_rejects_unverified_or_non_val_sources(tmp_path: Path, payload: dict, message: str) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ExternalAssetError, match=message):
        build_external_asset_manifest(source, "mpdocvqa")


@pytest.mark.parametrize(
    "changed_key, changed_value, message",
    [
        ("question", "TBD", "placeholder question"),
        ("image_path", "path/to/image.png", "placeholder image path"),
        ("ocr_path", "missing.txt", "missing OCR file"),
    ],
)
def test_rejects_placeholders_and_missing_assets(
    tmp_path: Path, changed_key: str, changed_value: str, message: str
) -> None:
    _asset_files(tmp_path)
    row = {
        "id": "q1",
        "question": "Question",
        "answer": "Answer",
        "document": "doc",
        "page": 1,
        "image_path": "assets/page-1.png",
        "ocr_path": "assets/page-1.txt",
    }
    row[changed_key] = changed_value
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"val": [row]}))
    with pytest.raises(ExternalAssetError, match=message):
        build_external_asset_manifest(source, "mpdocvqa")


def test_cli_writes_query_and_corpus_manifest(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "val": [
                    {
                        "id": "q1",
                        "question": "Question",
                        "answer": "Answer",
                        "document": "doc",
                        "page": 1,
                        "image": "assets/page-1.png",
                        "ocr": "assets/page-1.txt",
                    }
                ]
            }
        )
    )
    out = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parents[1] / "register_external_assets.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--dataset", "arxivqa", "--source", str(source), "--out", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text())
    assert payload["records"][0]["example_id"] == "q1"
    assert payload["corpus_pages"][0]["corpus_id"] == "doc:p1"
