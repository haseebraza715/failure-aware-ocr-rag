from __future__ import annotations

"""
Resumable FAAR benchmark asset preparation (design + smoke harness).

Pipeline stages per document:
  1. PDF discovery
  2. Docling audit Markdown
  3. Deterministic page PNG rendering
  4. Pinned GOT-OCR text extraction

Full-dataset execution is intentionally disabled. Use `--smoke-doc` to write a
one-document plan and checkpoint. Do not use this to start Phase 1 or paid calls.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.asset_paths import to_relative_project_path


STAGE_ORDER = ("docling_audit", "render_pages", "got_ocr")


def page_image_name(doc_rel: str, page_id: int) -> str:
    return f"{doc_rel}_page_{page_id}.png"


def page_ocr_name(doc_rel: str, page_id: int) -> str:
    return f"{doc_rel}_page_{page_id}.txt"


def docling_audit_name(doc_rel: str) -> str:
    return f"{doc_rel}.docling.md"


def _rel(path: Path, project_root: Path) -> str:
    try:
        return to_relative_project_path(path, project_root)
    except Exception:
        return path.as_posix()


def plan_document_work(
    *,
    project_root: Path,
    doc_rel: str,
    pdf_path: Path,
    page_ids: list[int],
    out_root: Path,
) -> dict:
    """Deterministic output naming for one document (no heavy work)."""
    image_dir = out_root / "images"
    ocr_dir = out_root / "ocr"
    audit_dir = out_root / "docling"
    audit_output = audit_dir / docling_audit_name(doc_rel)
    image_outputs = [image_dir / page_image_name(doc_rel, page_id) for page_id in page_ids]
    ocr_outputs = [ocr_dir / page_ocr_name(doc_rel, page_id) for page_id in page_ids]

    stages = {
        "docling_audit": {
            "input": _rel(pdf_path, project_root),
            "output": _rel(audit_output, project_root),
            "done": audit_output.exists(),
        },
        "render_pages": {
            "outputs": [_rel(path, project_root) for path in image_outputs],
            "done": all(path.exists() for path in image_outputs),
        },
        "got_ocr": {
            "outputs": [_rel(path, project_root) for path in ocr_outputs],
            "done": all(path.exists() for path in ocr_outputs),
        },
    }
    return {
        "doc_name": doc_rel,
        "pdf": _rel(pdf_path, project_root),
        "page_ids": page_ids,
        "stages": stages,
        "checkpoint": {stage: stages[stage]["done"] for stage in STAGE_ORDER},
    }


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return {"completed": {}, "updated_at_utc": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or smoke-test resumable PDF → Docling → PNG → GOT-OCR asset preparation."
    )
    parser.add_argument("--dataset", required=True, choices=["ohrbench", "mpdocvqa", "arxivqa"])
    parser.add_argument("--pdf-root", type=Path, help="Root containing source PDFs.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for audit/images/ocr.")
    parser.add_argument(
        "--smoke-doc",
        type=str,
        help="Document id for one-document smoke planning (e.g. academic/2403.20330v2).",
    )
    parser.add_argument(
        "--page-ids",
        type=str,
        default="1",
        help="Comma-separated page ids for smoke planning when inventory is unavailable (default: 1).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for future one-document execution. Without this flag, only plans/checkpoints are written.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Resumable checkpoint JSON path.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (out_root / "prepare_checkpoint.json")

    if not args.smoke_doc:
        raise SystemExit(
            "Full-dataset preparation is intentionally not enabled yet. "
            "Pass --smoke-doc <doc_rel> to plan a one-document smoke run."
        )

    doc_rel = args.smoke_doc.strip().removesuffix(".pdf")
    pdf_root = args.pdf_root or (project_root / "data/ohr_bench_raw/pdfs_sample")
    pdf_path = Path(pdf_root)
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path
    pdf_path = pdf_path / f"{doc_rel}.pdf"

    page_ids = [int(part.strip()) for part in args.page_ids.split(",") if part.strip()]
    if not page_ids:
        raise SystemExit("--page-ids must contain at least one integer page id.")

    work = plan_document_work(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_path=pdf_path,
        page_ids=page_ids,
        out_root=out_root,
    )
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.setdefault("completed", {})
    checkpoint["dataset"] = args.dataset
    checkpoint["smoke_doc"] = doc_rel
    checkpoint["plan"] = work

    if args.execute:
        raise SystemExit(
            "Execute mode is intentionally blocked in this Phase 0 audit commit. "
            "The plan and checkpoint were prepared; run Docling/GOT-OCR later after asset access is approved."
        )

    save_checkpoint(checkpoint_path, checkpoint)
    plan_path = out_root / f"prepare_plan_{doc_rel.replace('/', '__')}.json"
    plan_path.write_text(json.dumps(work, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": "plan_only",
                "dataset": args.dataset,
                "smoke_doc": doc_rel,
                "plan": _rel(plan_path, project_root),
                "checkpoint": _rel(checkpoint_path, project_root),
                "stages": STAGE_ORDER,
                "note": "Full PDF→Docling→PNG→GOT-OCR execution is not run in this audit.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
