from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from faar.benchmarks import BenchmarkRepository
from faar.external_assets import ExternalAssetError, build_external_asset_manifest
from faar.settings import RetrievalSettings


def _asset_files(root: Path, pages: range = range(1, 3)) -> None:
    (root / "assets").mkdir(exist_ok=True)
    for page in pages:
        (root / "assets" / f"page-{page}.png").write_bytes(b"png")
        (root / "assets" / f"page-{page}.txt").write_text(f"OCR page {page}")


def _documents_inventory(doc_id: str, pages: range) -> list[dict]:
    return [
        {
            "doc_id": doc_id,
            "pages": [
                {
                    "page_id": page,
                    "image_path": f"assets/page-{page}.png",
                    "ocr_text_path": f"assets/page-{page}.txt",
                }
                for page in pages
            ],
        }
    ]


def test_corpus_includes_all_document_pages_when_answer_is_on_page_two(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(1, 6))
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "documents": _documents_inventory("doc-a", range(1, 6)),
                "data": [
                    {
                        "question_id": "q1",
                        "query": "What is on page two?",
                        "answers": ["answer-page-2"],
                        "doc_id": "doc-a",
                        "evidence_page_no": 2,
                    }
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "MP-DocVQA", project_root=tmp_path)

    assert manifest["records"][0]["page_ids"] == [2]
    assert manifest["records"][0]["corpus_ids"] == [f"doc-a:p{page}" for page in range(1, 6)]
    assert [page["page_id"] for page in manifest["corpus_pages"]] == [1, 2, 3, 4, 5]
    assert all(not Path(page["image_path"]).is_absolute() for page in manifest["corpus_pages"])
    assert manifest["document_inventory"]["doc-a"] == [1, 2, 3, 4, 5]

    repository = BenchmarkRepository(
        manifest["records"],
        manifest["corpus_pages"],
        tmp_path,
        "mpdocvqa",
        "val",
        document_inventory=manifest["document_inventory"],
    )
    chunks = repository.get_corpus_chunks(RetrievalSettings(chunk_size_words=10, chunk_overlap_words=0))
    assert {chunk.page_id for chunk in chunks} == {1, 2, 3, 4, 5}


def test_rejects_evidence_only_source_without_document_inventory(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "data": [
                    {
                        "id": "q1",
                        "question": "What is the total?",
                        "answer": "42",
                        "document": "doc-a",
                        "page_no": 1,
                        "image_path": "assets/page-1.png",
                        "ocr_path": "assets/page-1.txt",
                    }
                ],
            }
        )
    )

    with pytest.raises(ExternalAssetError, match="Complete document pages cannot be proven"):
        build_external_asset_manifest(source, "MP-DocVQA", project_root=tmp_path)


def test_normalizes_aliases_with_complete_inventory(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "documents": _documents_inventory("doc-a", range(1, 3)),
                "data": [
                    {
                        "question_id": "q1",
                        "query": "What is the total?",
                        "answers": ["42", "forty two"],
                        "doc_id": "doc-a",
                        "evidence_page_no": 1,
                    },
                    {
                        "id": "q2",
                        "question": "Which document?",
                        "answer": "A",
                        "document_id": "doc-a",
                        "page": 1,
                    },
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "MP-DocVQA", project_root=tmp_path)

    assert manifest["dataset"] == "mpdocvqa"
    assert len(manifest["records"]) == 2
    assert manifest["records"][0]["correct_answer"] == "42"
    assert len(manifest["corpus_pages"]) == 2
    assert manifest["corpus_pages"][0]["image_path"] == "assets/page-1.png"


def test_normalizes_nested_arxiv_pages_with_inventory(tmp_path: Path) -> None:
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
                        "evidence_page_ids": [2],
                    }
                ],
                "documents": [
                    {
                        "doc_id": "paper-7",
                        "pages": [
                            {
                                "page_number": 1,
                                "page_image": "assets/page-1.png",
                                "ocr_file": "assets/page-1.txt",
                            },
                            {
                                "page_number": 2,
                                "page_image": "assets/page-2.png",
                                "ocr_file": "assets/page-2.txt",
                            },
                        ],
                    }
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "arxivqa", project_root=tmp_path)

    assert manifest["records"][0]["page_ids"] == [2]
    assert manifest["records"][0]["corpus_ids"] == ["paper-7:p1", "paper-7:p2"]
    assert len(manifest["corpus_pages"]) == 2


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"split": "test", "documents": [], "data": [{"id": "q"}]}, "exactly the val split"),
        ({"data": [{"id": "q"}]}, "Could not verify the exact val split"),
    ],
)
def test_rejects_unverified_or_non_val_sources(tmp_path: Path, payload: dict, message: str) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ExternalAssetError, match=message):
        build_external_asset_manifest(source, "mpdocvqa", project_root=tmp_path)


def test_rejects_placeholders_and_missing_inventory_assets(tmp_path: Path) -> None:
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
                        "evidence_page_no": 1,
                    }
                ],
                "documents": [
                    {
                        "doc_id": "doc",
                        "pages": [
                            {
                                "page_id": 1,
                                "image_path": "path/to/image.png",
                                "ocr_text_path": "assets/page-1.txt",
                            }
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(ExternalAssetError, match="placeholder image path"):
        build_external_asset_manifest(source, "mpdocvqa", project_root=tmp_path)


def test_cli_writes_portable_query_and_corpus_manifest(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(1, 3))
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
                        "evidence_page_no": 1,
                    }
                ],
                "documents": _documents_inventory("doc", range(1, 3)),
            }
        )
    )
    out = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parents[1] / "register_external_assets.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "arxivqa",
            "--source",
            str(source),
            "--project-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text())
    assert payload["records"][0]["example_id"] == "q1"
    assert {page["corpus_id"] for page in payload["corpus_pages"]} == {"doc:p1", "doc:p2"}
    assert all(not Path(page["image_path"]).is_absolute() for page in payload["corpus_pages"])
