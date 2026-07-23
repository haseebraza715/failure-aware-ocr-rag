from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from prepare_benchmark_assets import page_image_name, page_ocr_name, plan_document_work


def test_deterministic_output_naming() -> None:
    assert page_image_name("academic/demo", 3) == "academic/demo_page_3.png"
    assert page_ocr_name("academic/demo", 3) == "academic/demo_page_3.txt"


def test_smoke_plan_is_resumable_and_portable(tmp_path: Path) -> None:
    pdf = tmp_path / "data/ohr_bench_raw/pdfs_sample/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    out_root = tmp_path / "prepared"
    work = plan_document_work(
        project_root=tmp_path,
        doc_rel="academic/demo",
        pdf_path=pdf,
        page_ids=[1, 2],
        out_root=out_root,
    )

    assert work["pdf"] == "data/ohr_bench_raw/pdfs_sample/academic/demo.pdf"
    assert work["stages"]["render_pages"]["outputs"] == [
        "prepared/images/academic/demo_page_1.png",
        "prepared/images/academic/demo_page_2.png",
    ]
    assert work["stages"]["got_ocr"]["outputs"] == [
        "prepared/ocr/academic/demo_page_1.txt",
        "prepared/ocr/academic/demo_page_2.txt",
    ]
    assert work["checkpoint"]["render_pages"] is False


def test_cli_smoke_writes_plan_without_execution(tmp_path: Path) -> None:
    pdf = tmp_path / "pdfs/academic/demo.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
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
            "--out-root",
            str(out_root),
            "--smoke-doc",
            "academic/demo",
            "--page-ids",
            "1,2",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "plan_only"
    assert (out_root / "prepare_checkpoint.json").is_file()
    plan = json.loads((out_root / "prepare_plan_academic__demo.json").read_text())
    assert plan["page_ids"] == [1, 2]
