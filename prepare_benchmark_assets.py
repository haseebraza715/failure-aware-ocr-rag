from __future__ import annotations

"""
Resumable FAAR benchmark asset preparation.

Pipeline stages per document:
  1. Safe PDF extract/copy from zip or filesystem
  2. Docling structured audit Markdown
  3. Deterministic page PNG rendering
  4. Pinned GOT-OCR text extraction

Full-dataset execution remains disabled. Use `--smoke-doc` for one document.
"""

import argparse
import json
import os
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.asset_paths import to_relative_project_path
from faar.asset_preparation import (
    STAGE_ORDER,
    execute_document_preparation,
    load_locked_got_ocr,
    page_image_name,
    page_ocr_name,
    plan_document_work,
    resolve_pdf_source,
)
from faar.ohr_inventory import (
    diagnose_ohr_inventory_gaps,
    load_resolved_ohr_document_inventory,
    resolve_ohr_inventory_path,
)
from faar.operations import install_graceful_termination_handler
from faar.run_io import atomic_write_text


def _rel(path: Path, project_root: Path) -> str:
    try:
        return to_relative_project_path(path, project_root)
    except Exception:
        return path.as_posix()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return {"completed": {}, "updated_at_utc": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Preparation checkpoint is unreadable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("completed"), dict):
        raise SystemExit(f"Preparation checkpoint has an invalid structure: {path}")
    return payload


def save_checkpoint(path: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _huggingface_cache_bytes() -> int | None:
    raw = os.getenv("HF_HOME")
    cache_root = Path(raw).expanduser() if raw and raw.strip() else Path.home() / ".cache/huggingface"
    if not cache_root.is_dir():
        return 0
    total = 0
    try:
        for path in cache_root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        return None
    return total


def _inventory_page_ids(project_root: Path, doc_rel: str, inventory_dir: Path, page_ids_arg: str | None) -> list[int]:
    if page_ids_arg:
        return [int(part.strip()) for part in page_ids_arg.split(",") if part.strip()]
    inventory, _resolutions = load_resolved_ohr_document_inventory(inventory_dir, {doc_rel})
    if doc_rel not in inventory:
        path, resolved, kind = resolve_ohr_inventory_path(inventory_dir, doc_rel)
        raise SystemExit(
            f"No complete gt inventory for {doc_rel!r} (resolved={resolved!r}, diagnosis={kind}, path={path})."
        )
    return inventory[doc_rel]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute one-document PDF → Docling → PNG → GOT-OCR asset preparation."
    )
    parser.add_argument("--dataset", required=True, choices=["ohrbench", "mpdocvqa", "arxivqa"])
    parser.add_argument("--pdf-root", type=Path, help="Optional filesystem root containing source PDFs.")
    parser.add_argument(
        "--pdf-zip",
        type=Path,
        default=None,
        help="PDF archive (default: data/ohr_bench_raw/pdfs.zip).",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for audit/images/ocr.")
    parser.add_argument(
        "--smoke-doc",
        type=str,
        help="Document id for one-document planning/execution (e.g. academic/DUDE_...).",
    )
    parser.add_argument(
        "--page-ids",
        type=str,
        default=None,
        help="Optional explicit page ids. Default: complete gt inventory pages for the document.",
    )
    parser.add_argument(
        "--document-inventory",
        type=Path,
        default=None,
        help="OHR gt inventory directory (default: OHR-Bench/data/retrieval_base/gt).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the one-document pipeline (local Docling/GOT-OCR). Full-dataset mode remains blocked.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Resumable checkpoint JSON path.")
    parser.add_argument(
        "--diagnose-inventory",
        action="store_true",
        help="Print OHR inventory gap diagnosis for qas_v2 and exit.",
    )
    args = parser.parse_args()
    install_graceful_termination_handler()

    def _env_path(name: str) -> Path | None:
        raw = os.getenv(name)
        return Path(raw).expanduser() if raw and raw.strip() else None

    project_root = Path.cwd().resolve()
    pdf_root = args.pdf_root or _env_path("FAAR_PDF_ROOT")
    pdf_zip = args.pdf_zip or _env_path("FAAR_PDF_ZIP")
    inventory_dir = (
        args.document_inventory
        or _env_path("FAAR_DOCUMENT_INVENTORY")
        or (project_root / "OHR-Bench/data/retrieval_base/gt")
    )
    if not inventory_dir.is_absolute():
        inventory_dir = project_root / inventory_dir
    if pdf_root is not None and not pdf_root.is_dir():
        raise SystemExit(f"FAAR_PDF_ROOT / --pdf-root is not a directory: {pdf_root}")
    if pdf_zip is not None and not pdf_zip.is_file():
        raise SystemExit(f"FAAR_PDF_ZIP / --pdf-zip is not a file: {pdf_zip}")
    if not inventory_dir.is_dir():
        raise SystemExit(f"document inventory is not a directory: {inventory_dir}")

    if args.diagnose_inventory:
        diagnose_zip = pdf_zip or (project_root / "data/ohr_bench_raw/pdfs.zip")
        pdf_names: set[str] = set()
        if diagnose_zip.is_file():
            with zipfile.ZipFile(diagnose_zip) as archive:
                pdf_names = {name for name in archive.namelist() if name.endswith(".pdf")}
        report = diagnose_ohr_inventory_gaps(
            qas_path=project_root / "OHR-Bench/data/qas_v2.json",
            inventory_dir=inventory_dir,
            pdf_names=pdf_names,
        )
        print(json.dumps(report, indent=2))
        return

    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (out_root / "prepare_checkpoint.json")

    if not args.smoke_doc:
        raise SystemExit(
            "Full-dataset preparation is intentionally not enabled. "
            "Pass --smoke-doc <doc_rel> to plan or execute one document."
        )
    if args.dataset != "ohrbench":
        raise SystemExit("One-document execution is currently implemented for --dataset ohrbench only.")

    doc_rel = args.smoke_doc.strip().removesuffix(".pdf")
    page_ids = _inventory_page_ids(project_root, doc_rel, inventory_dir, args.page_ids)
    if not page_ids:
        raise SystemExit("--page-ids / inventory must contain at least one page id.")

    kind, source = resolve_pdf_source(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_root=pdf_root,
        pdf_zip=pdf_zip,
        inventory_dir=inventory_dir,
    )
    if kind == "missing":
        raise SystemExit(f"PDF for {doc_rel!r} was not found in --pdf-root or pdfs.zip.")

    pdf_path = out_root / "pdfs" / f"{doc_rel}.pdf"
    locked = load_locked_got_ocr(project_root)
    work = plan_document_work(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_path=pdf_path,
        page_ids=page_ids,
        out_root=out_root,
        got_ocr=locked,
    )
    work["pdf_source"] = {
        "kind": kind,
        "source": source.as_posix() if isinstance(source, Path) else None,
    }
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.setdefault("completed", {})
    checkpoint["dataset"] = args.dataset
    checkpoint["smoke_doc"] = doc_rel
    checkpoint["plan"] = work
    checkpoint["resumed"] = bool(checkpoint["completed"]) or bool(checkpoint.get("failed"))
    checkpoint["cache_path"] = str(
        (Path(os.getenv("HF_HOME")) if os.getenv("HF_HOME") else Path.home() / ".cache/huggingface").expanduser()
    )

    plan_path = out_root / f"prepare_plan_{doc_rel.replace('/', '__')}.json"
    atomic_write_text(plan_path, json.dumps(work, indent=2) + "\n")

    if not args.execute:
        save_checkpoint(checkpoint_path, checkpoint)
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "dataset": args.dataset,
                    "smoke_doc": doc_rel,
                    "page_ids": page_ids,
                    "page_count": len(page_ids),
                    "pdf_source": work["pdf_source"],
                    "got_ocr": locked,
                    "expected_outputs": {
                        "pdf": work["pdf"],
                        "docling_audit": work["stages"]["docling_audit"]["output"],
                        "images": work["stages"]["render_pages"]["outputs"],
                        "ocr": work["stages"]["got_ocr"]["outputs"],
                    },
                    "plan": _rel(plan_path, project_root),
                    "checkpoint": _rel(checkpoint_path, project_root),
                    "stages": STAGE_ORDER,
                },
                indent=2,
            )
        )
        return

    runtime_started = time.perf_counter()
    if checkpoint.get("cache_bytes_before") is None:
        checkpoint["cache_bytes_before"] = _huggingface_cache_bytes()
    result = execute_document_preparation(
        project_root=project_root,
        doc_rel=doc_rel,
        page_ids=page_ids,
        out_root=out_root,
        pdf_root=pdf_root,
        pdf_zip=pdf_zip,
        inventory_dir=inventory_dir,
    )
    checkpoint["calibration"] = {
        "runtime_sec_total_wall": round(time.perf_counter() - runtime_started, 3),
        "resumed": bool(checkpoint.get("resumed")),
        "cache_bytes_before": checkpoint.get("cache_bytes_before"),
        "cache_bytes_after": _huggingface_cache_bytes(),
        "cache_path": checkpoint.get("cache_path"),
    }
    checkpoint["completed"][doc_rel] = {
        "page_ids": page_ids,
        "pdf_sha256": result.pdf_sha256,
        "outputs": result.outputs,
        "metrics": result.metrics,
    }
    checkpoint["plan"] = plan_document_work(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_path=result.pdf_path,
        page_ids=page_ids,
        out_root=out_root,
        got_ocr=locked,
    )
    save_checkpoint(checkpoint_path, checkpoint)
    result_path = out_root / f"prepare_result_{doc_rel.replace('/', '__')}.json"
    atomic_write_text(
        result_path,
        json.dumps(
            {
                "doc_name": result.doc_name,
                "page_ids": result.page_ids,
                "pdf_sha256": result.pdf_sha256,
                "got_ocr_repository": result.got_ocr_repository,
                "got_ocr_revision": result.got_ocr_revision,
                "device": result.device,
                "outputs": result.outputs,
                "metrics": result.metrics,
            },
            indent=2,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "mode": "execute",
                "dataset": args.dataset,
                "smoke_doc": doc_rel,
                "page_count": len(page_ids),
                "device": result.device,
                "got_ocr_revision": result.got_ocr_revision,
                "result": _rel(result_path, project_root),
                "outputs": result.outputs,
                "metrics": {
                    "pdf_sha256": result.pdf_sha256,
                    "render_runtime_sec": result.metrics.get("render_runtime_sec"),
                    "docling_runtime_sec": result.metrics.get("docling_runtime_sec"),
                    "got_ocr_runtime_sec_total": result.metrics.get("got_ocr_runtime_sec_total"),
                    "per_page_got_ocr_runtime_sec": result.metrics.get("per_page_got_ocr_runtime_sec"),
                    "peak_rss_bytes": result.metrics.get("peak_rss_bytes"),
                    "storage": result.metrics.get("storage"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
