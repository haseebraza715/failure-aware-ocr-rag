from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from faar.asset_preparation import (
    execute_document_preparation,
    page_image_name,
    page_ocr_name,
    plan_document_work,
)
from prepare_benchmark_assets import page_image_name as cli_page_image_name


def test_deterministic_output_naming() -> None:
    assert page_image_name("academic/demo", 3) == "academic/demo_page_3.png"
    assert page_ocr_name("academic/demo", 3) == "academic/demo_page_3.txt"
    assert cli_page_image_name("academic/demo", 3) == "academic/demo_page_3.png"


def test_smoke_plan_is_resumable_and_portable(tmp_path: Path) -> None:
    pdf = tmp_path / "prepared/pdfs/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    out_root = tmp_path / "prepared"
    work = plan_document_work(
        project_root=tmp_path,
        doc_rel="academic/demo",
        pdf_path=pdf,
        page_ids=[0, 1],
        out_root=out_root,
        got_ocr={"repository": "stepfun-ai/GOT-OCR-2.0-hf", "revision": "a" * 40},
    )

    assert work["pdf"] == "prepared/pdfs/academic/demo.pdf"
    assert work["stages"]["render_pages"]["outputs"] == [
        "prepared/images/academic/demo_page_0.png",
        "prepared/images/academic/demo_page_1.png",
    ]
    assert work["stages"]["got_ocr"]["outputs"] == [
        "prepared/ocr/academic/demo_page_0.txt",
        "prepared/ocr/academic/demo_page_1.txt",
    ]
    assert work["checkpoint"]["render_pages"] is False


def test_execute_one_document_with_injected_backends(tmp_path: Path) -> None:
    project = tmp_path
    (project / "config").mkdir()
    (project / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    }
                }
            }
        )
    )
    pdf_src = project / "pdfs/academic/demo.pdf"
    pdf_src.parent.mkdir(parents=True)
    # Minimal valid-enough PDF bytes are not required because we mock render/OCR/Docling,
    # but filesystem source must exist for copy.
    pdf_src.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")

    out_root = project / "prepared"
    revisions_seen: list[str | None] = []

    def fake_docling(pdf_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# docling audit\n", encoding="utf-8")
        return output_path

    def fake_ocr(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        revisions_seen.append(revision)
        return f"OCR for {image_path.name}"

    # Monkeypatch render via writing PNGs ourselves by patching render_pdf_pages.
    import faar.asset_preparation as prep

    def fake_render(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id, path in image_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_id]))
        return {"pdf_page_count": len(page_ids), "render_runtime_sec": 0.01, "render_scale": scale}

    prep.render_pdf_pages = fake_render  # type: ignore[assignment]

    result = execute_document_preparation(
        project_root=project,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=out_root,
        pdf_root=project / "pdfs",
        extract_got_ocr_fn=fake_ocr,
        export_docling_fn=fake_docling,
    )

    assert result.got_ocr_revision == "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
    assert revisions_seen == ["d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"]
    assert (out_root / "images/academic/demo_page_0.png").stat().st_size > 0
    assert (out_root / "ocr/academic/demo_page_0.txt").read_text().startswith("OCR for")
    assert not Path(result.outputs["images"][0]).is_absolute()
    provenance = json.loads((out_root / "provenance/academic/demo.json").read_text())
    assert provenance["got_ocr_revision"] == result.got_ocr_revision
    assert "Users" not in json.dumps(provenance)


def test_execute_document_checks_memory_at_every_expensive_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faar.asset_preparation as prep

    (tmp_path / "config").mkdir()
    (tmp_path / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "pdfs/academic/demo.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    stages: list[str] = []

    def fake_docling(_pdf: Path, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("audit", encoding="utf-8")
        return output

    def fake_render(_pdf: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0):
        for page_id in page_ids:
            image_paths[page_id].parent.mkdir(parents=True, exist_ok=True)
            image_paths[page_id].write_bytes(b"png")
        return {"pdf_page_count": 1, "render_runtime_sec": 0.0, "render_scale": scale}

    monkeypatch.setattr(prep, "render_pdf_pages", fake_render)
    monkeypatch.setattr(prep, "enforce_memory_budget", stages.append)

    execute_document_preparation(
        project_root=tmp_path,
        doc_rel="academic/demo",
        page_ids=[0],
        out_root=tmp_path / "prepared",
        pdf_root=tmp_path / "pdfs",
        extract_got_ocr_fn=lambda *_args, **_kwargs: "text",
        export_docling_fn=fake_docling,
    )

    assert stages == [
        "asset preparation before PDF extraction",
        "asset preparation after PDF extraction",
        "asset preparation before Docling audit",
        "asset preparation after Docling audit",
        "asset preparation before page rendering",
        "asset preparation after page rendering",
        "asset preparation before GOT-OCR page 0",
        "asset preparation after GOT-OCR page 0",
    ]


def test_cli_smoke_writes_plan_without_execution(tmp_path: Path) -> None:
    inventory = tmp_path / "OHR-Bench/data/retrieval_base/gt/academic"
    inventory.mkdir(parents=True)
    (inventory / "demo.json").write_text(json.dumps([{"text": "x", "page_idx": 0}]))
    pdf = tmp_path / "pdfs/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {
                        "repository": "stepfun-ai/GOT-OCR-2.0-hf",
                        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
                    }
                }
            }
        )
    )
    out_root = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "prepare_benchmark_assets.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "ohrbench",
            "--pdf-root",
            str(tmp_path / "pdfs"),
            "--document-inventory",
            str(tmp_path / "OHR-Bench/data/retrieval_base/gt"),
            "--out-root",
            str(out_root),
            "--smoke-doc",
            "academic/demo",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "plan_only"
    assert payload["page_ids"] == [0]
    assert (out_root / "prepare_checkpoint.json").is_file()


def test_plan_imports_do_not_load_inference_runtime() -> None:
    """Plan-only imports must not pull Torch, Transformers, or ColPali."""
    repo_root = Path(__file__).resolve().parents[1]
    probe = r"""
import sys
from pathlib import Path
root = Path(%r)
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))
from faar.asset_preparation import load_locked_got_ocr, plan_document_work, page_image_name
import prepare_benchmark_assets
banned = {"torch", "transformers", "colpali_engine"}
loaded = sorted({name.split(".")[0] for name in sys.modules if name.split(".")[0] in banned})
assert not loaded, loaded
assert page_image_name("doc", 0).endswith("_page_0.png")
print("PLAN_IMPORTS_CLEAN")
""" % (str(repo_root),)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "PLAN_IMPORTS_CLEAN" in completed.stdout
