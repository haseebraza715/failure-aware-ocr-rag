from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PREPARE_PATH = Path(__file__).resolve().parents[1] / "cluster/prepare_assets.py"
PREPARE_SPEC = spec_from_file_location("prepare_assets", PREPARE_PATH)
assert PREPARE_SPEC and PREPARE_SPEC.loader
prepare = module_from_spec(PREPARE_SPEC)
sys.modules["prepare_assets"] = prepare
PREPARE_SPEC.loader.exec_module(prepare)

DOCS = ["academic/doc_a", "academic/doc_b", "manual/doc_c"]
PAGES = [0, 1]

MODEL_LOCK = {
    "models": {
        "got_ocr": {
            "repository": "stepfun-ai/GOT-OCR-2.0-hf",
            "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
        }
    }
}


def write_lock(project: Path) -> None:
    config = project / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "model_revisions.json").write_text(json.dumps(MODEL_LOCK), encoding="utf-8")
    (config / "split_checksums.json").write_text(
        json.dumps(
            {
                "split_sha256": prepare.sha256_file(project / "split.json"),
                "qas_v2_sha256": prepare.sha256_file(project / "OHR-Bench/data/qas_v2.json"),
            }
        ),
        encoding="utf-8",
    )


def build_project(tmp_path: Path, docs: list[str] = DOCS) -> Path:
    project = tmp_path
    (project / "OHR-Bench/data").mkdir(parents=True, exist_ok=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(
        json.dumps([{"ID": f"e{i}", "doc_name": doc} for i, doc in enumerate(docs)])
    )
    (project / "split.json").write_text(
        json.dumps(
            {
                "splits": {
                    "val": [f"e{i}" for i in range(len(docs))],
                    "test": [f"e{i}" for i in range(len(docs))],
                }
            }
        )
    )
    gt = project / "OHR-Bench/data/retrieval_base/gt"
    for doc in docs:
        inventory = gt / f"{doc}.json"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(json.dumps([{"text": "x", "page_idx": page} for page in PAGES]))
        pdf = project / "pdfs" / f"{doc}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(f"%PDF-1.4\n{doc}\n".encode())
    write_lock(project)
    return project


def install_fakes(monkeypatch: pytest.MonkeyPatch, ocr_calls: list[str] | None = None) -> None:
    import faar.asset_preparation as prep

    class FakePageCountDocument:
        def __init__(self, _path: str) -> None:
            self.page_count = len(PAGES)

        def __len__(self) -> int:
            return self.page_count

        def close(self) -> None:
            pass

    def fake_docling(pdf_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# audit\n", encoding="utf-8")
        return output_path

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_id]))
        return {"pdf_page_count": len(page_ids), "render_runtime_sec": 0.01, "render_scale": scale}

    def fake_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        if ocr_calls is not None:
            ocr_calls.append(image_path.name)
        return f"OCR for {image_path.name}"

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    monkeypatch.setitem(
        sys.modules,
        "pypdfium2",
        __import__("types").SimpleNamespace(PdfDocument=FakePageCountDocument),
    )
    return fake_ocr, fake_docling


def run_shard(project: Path, *args, monkeypatch=None, ocr_calls=None, **kwargs):
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    return prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
        **kwargs,
    )


def test_deterministic_sharding_and_full_coverage() -> None:
    docs = sorted({f"{category}/{name}" for category in ("academic", "manual", "paper") for name in ("a", "b")})
    shards = [prepare.select_shard(docs, index, 4) for index in range(4)]
    assert all(not set(shard) & set(other) for shard in shards for other in shards if shard is not other)
    assert sorted({doc for shard in shards for doc in shard}) == sorted(docs)
    first = prepare.select_shard(docs, 1, 4)
    for _ in range(3):
        assert prepare.select_shard(docs, 1, 4) == first


def test_select_shard_rejects_invalid_metadata() -> None:
    docs = sorted({f"{category}/{name}" for category in ("academic", "manual", "paper") for name in ("a", "b")})
    for bad_index in (-1, 4, 7):
        with pytest.raises(ValueError, match="out of range"):
            prepare.select_shard(docs, bad_index, 4)
    with pytest.raises(ValueError, match="num_shards"):
        prepare.select_shard(docs, 0, 0)
    with pytest.raises(ValueError, match="num_shards"):
        prepare.select_shard(docs, 0, True)
    with pytest.raises(ValueError, match="shard_index"):
        prepare.select_shard(docs, False, 4)
    with pytest.raises(ValueError, match="duplicates"):
        prepare.select_shard([doc for doc in DOCS] + [DOCS[0]], 0, 1)


def test_bounds_apply_deterministically() -> None:
    page_ids_by_doc = {doc: PAGES for doc in DOCS}
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_documents=2) == (DOCS[:2], True)
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_documents=10) == (DOCS, False)
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_pages=3) == (DOCS[:1], True)
    assert prepare.bound_documents(DOCS, page_ids_by_doc, max_pages=6) == (DOCS, False)
    assert prepare.bound_documents(DOCS, page_ids_by_doc) == (DOCS, False)


def test_run_shard_prepares_all_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls)
    assert summary["failed"] == []
    assert summary["pages_completed"] == len(DOCS) * len(PAGES)
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert set(checkpoint["completed"]) == set(DOCS)
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_completed"] == DOCS
    assert manifest["documents_failed"] == {}
    assert manifest["split"] == "val"
    assert len(ocr_calls) == len(DOCS) * len(PAGES)
    for doc in DOCS:
        provenance = json.loads((project / "out/provenance" / f"{doc}.json").read_text())
        assert provenance["pdf_sha256"] == manifest.get("pdf_sha256") or provenance["pages"]


def test_rerun_is_idempotent_and_skips_valid_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls)
    first = json.loads((project / "out/checkpoint.json").read_text())
    ocr_calls.clear()
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["skipped"] == DOCS
    assert ocr_calls == []
    second = json.loads((project / "out/checkpoint.json").read_text())
    assert first["completed"] == second["completed"]
    assert first["updated_at_utc"] != second["updated_at_utc"]


def test_rerun_without_resume_refuses_existing_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit, match="pass --resume"):
        run_shard(project, monkeypatch=monkeypatch)


def test_resume_after_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    calls = {"count": 0}
    original_check = prepare.check_termination

    def interrupting_check() -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise SystemExit(143)

    monkeypatch.setattr(prepare, "check_termination", interrupting_check)
    with pytest.raises(SystemExit) as excinfo:
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=fake_ocr,
            export_docling_fn=fake_docling,
        )
    assert excinfo.value.code == 143
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert set(checkpoint["completed"]) == {DOCS[0]}
    assert not (project / "out/shard_manifest.json").is_file()
    monkeypatch.setattr(prepare, "check_termination", original_check)
    ocr_calls.clear()
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["failed"] == []
    assert summary["skipped"] == [DOCS[0]]
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["resumed"] is True
    assert len(ocr_calls) == len(DOCS[1:]) * len(PAGES)


def test_corrupt_checkpoint_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    (project / "out").mkdir(parents=True)
    (project / "out/checkpoint.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable JSON"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)
    (project / "out/checkpoint.json").write_text(json.dumps({"completed": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid 'completed' section"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)
    (project / "out/checkpoint.json").write_text(
        json.dumps({"completed": {}, "split": "test"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Refusing to resume"):
        run_shard(project, monkeypatch=monkeypatch, resume=True)


def test_recoverable_document_error_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)

    def failing_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        if "doc_b" in str(image_path):
            raise ValueError("simulated recoverable OCR failure")
        return "text"

    fake_docling = install_fakes(monkeypatch)[1]
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=failing_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary["failed"] == ["academic/doc_b"]
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert set(checkpoint["completed"]) == {"academic/doc_a", "manual/doc_c"}
    assert "simulated recoverable OCR failure" in checkpoint["failed"]["academic/doc_b"]["error"]
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_completed"] == ["academic/doc_a", "manual/doc_c"]
    assert manifest["documents_failed"]["academic/doc_b"]["error"].startswith("simulated")


def test_fatal_memory_error_aborts_with_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_MAX_RSS_GB", "0.000001")
    fake_ocr, fake_docling = install_fakes(monkeypatch)
    with pytest.raises(MemoryError):
        prepare.run_shard(
            project_root=project,
            split="val",
            doc_names=DOCS,
            page_ids_by_doc={doc: PAGES for doc in DOCS},
            out_root=project / "out",
            checkpoint_path=project / "out/checkpoint.json",
            manifest_path=project / "out/shard_manifest.json",
            pdf_root=project / "pdfs",
            extract_got_ocr_fn=fake_ocr,
            export_docling_fn=fake_docling,
        )
    assert not (project / "out/shard_manifest.json").is_file()
    monkeypatch.delenv("FAAR_MAX_RSS_GB")
    monkeypatch.setenv("FAAR_MAX_RSS_GB", "0.000001")
    with pytest.raises(MemoryError):
        run_shard(project, monkeypatch=monkeypatch)


def test_empty_shard_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path)
    fake_ocr, fake_docling = install_fakes(monkeypatch)
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=[],
        page_ids_by_doc={},
        out_root=project / "out",
        checkpoint_path=project / "out/checkpoint.json",
        manifest_path=project / "out/shard_manifest.json",
        shard_index=2,
        num_shards=3,
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary["failed"] == []
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_assigned"] == []
    assert manifest["documents_completed"] == []
    assert manifest["shard_index"] == 2 and manifest["num_shards"] == 3


def test_merge_shard_manifests_validates_coverage(tmp_path: Path) -> None:
    documents = [["academic/doc_a"], ["academic/doc_b"], ["manual/doc_c"]]
    paths: list[Path] = []
    for index, docs in enumerate(documents):
        payload = {
            "schema_version": 1,
            "kind": "ohr-asset-shard-manifest",
            "dataset": "ohrbench",
            "split": "val",
            "shard_index": index,
            "num_shards": 3,
            "created_at_utc": "t",
            "finished_at_utc": "t",
            "resumed": False,
            "bounds": {"max_documents": None, "max_pages": None, "bounds_hit": False},
            "documents_assigned": docs,
            "documents_completed": docs,
            "documents_failed": {},
            "retried_documents": [],
            "pages_assigned": 2,
            "pages_completed": 2,
            "runtime_sec": 1.0,
            "checkpoint": "c",
            "out_root": "o",
        }
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    merged = prepare.merge_shard_manifests(paths, expected_documents=DOCS)
    assert merged["documents"] == DOCS
    assert merged["pages_completed"] == 6
    with pytest.raises(SystemExit, match="missing shard indices"):
        prepare.merge_shard_manifests(paths[:2], expected_documents=DOCS)
    with pytest.raises(SystemExit, match="declared by"):
        prepare.merge_shard_manifests([paths[0], paths[1], paths[2], paths[2]], expected_documents=DOCS)


def test_merge_rejects_uncompleted_documents(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "ohr-asset-shard-manifest",
        "dataset": "ohrbench",
        "split": "val",
        "shard_index": 0,
        "num_shards": 1,
        "created_at_utc": "t",
        "finished_at_utc": "t",
        "resumed": False,
        "bounds": {"max_documents": None, "max_pages": None, "bounds_hit": False},
        "documents_assigned": DOCS,
        "documents_completed": DOCS[:1],
        "documents_failed": {"manual/doc_c": {"error": "x"}},
        "retried_documents": [],
        "pages_assigned": 6,
        "pages_completed": 2,
        "runtime_sec": 1.0,
        "checkpoint": "c",
        "out_root": "o",
    }
    path = tmp_path / "shard0.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="uncompleted documents"):
        prepare.merge_shard_manifests([path], expected_documents=DOCS)


def test_path_traversal_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        prepare._safe_output_path("../escape", Path("/tmp/root"))
    with pytest.raises(ValueError, match="non-empty"):
        prepare._safe_output_path("", Path("/tmp/root"))


def test_corrupt_checkpoint_outputs_recovered_by_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    run_shard(project, monkeypatch=monkeypatch)
    evil_entry = dict(json.loads((project / "out/checkpoint.json").read_text()))
    evil_entry["completed"]["academic/doc_a"]["outputs"]["images"][0] = "../escaped.png"
    (project / "out/checkpoint.json").write_text(json.dumps(evil_entry))
    ocr_calls: list[str] = []
    summary = run_shard(project, monkeypatch=monkeypatch, ocr_calls=ocr_calls, resume=True)
    assert summary["failed"] == []
    assert ocr_calls == []
    manifest = json.loads((project / "out/shard_manifest.json").read_text())
    assert manifest["documents_completed"] == DOCS
    checkpoint = json.loads((project / "out/checkpoint.json").read_text())
    assert checkpoint["completed"]["academic/doc_a"]["outputs"]["images"][0] != "../escaped.png"


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    out_root = project / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--split",
            "val",
            "--shard-index",
            "1",
            "--num-shards",
            "2",
            "--out-root",
            str(out_root),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["documents_assigned"] == ["manual/doc_c"]
    assert payload["shard_label"] == "shard2of2"
    assert not out_root.exists()


def test_cli_dry_run_applies_bounds(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--max-documents",
            "1",
            "--max-pages",
            "2",
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["documents_assigned"] == DOCS[:1]
    assert payload["bounds"]["bounds_hit"] is True


def test_cli_rejects_train_split_and_bad_shards(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    for extra in (["--split", "train"], ["--num-shards", "0"], ["--shard-index", "5", "--num-shards", "2"]):
        completed = subprocess.run(
            [
                sys.executable,
                str(PREPARE_PATH),
                "--project-root",
                str(project),
                "--out-root",
                str(project / "out"),
                "--dry-run",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=project,
        )
        assert completed.returncode != 0, extra


def test_cli_rejects_missing_split_lock(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    (project / "config/split_checksums.json").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode != 0
    assert "split checksum lock" in completed.stderr


def test_paths_with_spaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = build_project(tmp_path / "my project with spaces")
    ocr_calls: list[str] = []
    fake_ocr, fake_docling = install_fakes(monkeypatch, ocr_calls)
    summary = prepare.run_shard(
        project_root=project,
        split="val",
        doc_names=DOCS,
        page_ids_by_doc={doc: PAGES for doc in DOCS},
        out_root=project / "out space",
        checkpoint_path=project / "out space/checkpoint.json",
        manifest_path=project / "out space/shard_manifest.json",
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )
    assert summary["failed"] == []
    assert len(ocr_calls) == len(DOCS) * len(PAGES)


def test_split_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0
    with (project / "split.json").open("a") as handle:
        handle.write(" ")
    tampered = subprocess.run(
        [
            sys.executable,
            str(PREPARE_PATH),
            "--project-root",
            str(project),
            "--out-root",
            str(project / "out"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert tampered.returncode != 0
    assert "integrity check failed" in tampered.stderr
