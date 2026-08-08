from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.arxivqa_prepare import (
    DEFAULT_DOWNLOAD_DELAY_SEC,
    VIDORE_EXPECTED_ROWS,
    build_paper_inventory,
    build_pdf_download_plan,
    build_qa_source,
    download_pdfs,
    extract_figures_from_parquet,
    finalize_inventory_ocr_readiness,
    load_verified_staged,
    match_evidence_pages,
    read_json,
    render_all_pdf_pages,
    run_prepare_pipeline,
    stage_vidore_against_official,
    write_json,
    write_staging_manifest,
)


def _default_paths(project_root: Path) -> dict[str, Path]:
    return {
        "parquet": project_root / "data/external/arxivqa/vidore/data/test-00000-of-00001.parquet",
        "jsonl": project_root / "data/external/arxivqa/raw/arxivqa.jsonl",
        "out_root": project_root / "data/benchmark_prep/arxivqa",
    }


def _add_shared_paths(parser: argparse.ArgumentParser, project_root: Path) -> None:
    defaults = _default_paths(project_root)
    parser.add_argument("--parquet", type=Path, default=defaults["parquet"], help="ViDoRe 500-row parquet.")
    parser.add_argument("--jsonl", type=Path, default=defaults["jsonl"], help="Official MMInstruction JSONL.")
    parser.add_argument("--out-root", type=Path, default=defaults["out_root"], help="Preparation output root.")
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=VIDORE_EXPECTED_ROWS,
        help=f"Optional compatibility check. It must equal the locked count ({VIDORE_EXPECTED_ROWS}).",
    )


def main(argv: list[str] | None = None) -> None:
    project_root = Path.cwd().resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ArXivQA full-paper remapping inputs from the fixed ViDoRe 500-row subset. "
            "Never fabricates figure mappings; evidence pages never define retrieval."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stage_p = sub.add_parser(
        "stage",
        help="Stage ViDoRe rows against official JSONL by image+question+answer+normalized options.",
    )
    _add_shared_paths(stage_p, project_root)

    extract_p = sub.add_parser("extract-figures", help="Extract staged figure images from the parquet.")
    _add_shared_paths(extract_p, project_root)

    plans_p = sub.add_parser("write-plans", help="Write deterministic QA source and PDF download plan.")
    _add_shared_paths(plans_p, project_root)

    download_p = sub.add_parser("download-pdfs", help="Resumable polite PDF downloads from an existing plan.")
    _add_shared_paths(download_p, project_root)
    download_p.add_argument("--delay-sec", type=float, default=DEFAULT_DOWNLOAD_DELAY_SEC)

    render_p = sub.add_parser("render-pages", help="Render all pages for downloaded PDFs and write inventory.")
    _add_shared_paths(render_p, project_root)

    match_p = sub.add_parser(
        "match-evidence",
        help="Score evidence-page candidates for required human confirmation.",
    )
    _add_shared_paths(match_p, project_root)

    all_p = sub.add_parser("prepare", help="Stage, extract figures, and write plans (optional download/render/match).")
    _add_shared_paths(all_p, project_root)
    all_p.add_argument("--download", action="store_true", help="Download PDFs after writing the plan.")
    all_p.add_argument("--render", action="store_true", help="Render PDF pages after download/presence.")
    all_p.add_argument("--match", action="store_true", help="Match evidence pages after render.")
    all_p.add_argument("--delay-sec", type=float, default=DEFAULT_DOWNLOAD_DELAY_SEC)
    finalize_p = sub.add_parser(
        "finalize-ocr-inventory",
        help="Verify pinned GOT-OCR files and provenance, then write an OCR-ready inventory.",
    )
    _add_shared_paths(finalize_p, project_root)
    finalize_p.add_argument("--ocr-provenance-dir", type=Path, required=True)
    finalize_p.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    parquet = args.parquet if args.parquet.is_absolute() else project_root / args.parquet
    jsonl = args.jsonl if args.jsonl.is_absolute() else project_root / args.jsonl
    source_lock = project_root / "config/arxivqa_source_lock.json"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.command == "stage":
        staged = stage_vidore_against_official(
            parquet, jsonl, source_lock_path=source_lock, expected_rows=args.expected_rows
        )
        payload = write_staging_manifest(
            staged,
            out_root / "staging_manifest.json",
            parquet_path=parquet,
            jsonl_path=jsonl,
            project_root=project_root,
            source_lock_path=source_lock,
        )
        print(
            json.dumps(
                {
                    "rows": payload["row_count"],
                    "papers": payload["paper_count"],
                    "out": str(out_root / "staging_manifest.json"),
                },
                indent=2,
            )
        )
        return

    if args.command == "extract-figures":
        staged = load_verified_staged(
            out_root / "staging_manifest.json",
            parquet,
            jsonl,
            source_lock_path=source_lock,
        )
        manifest = extract_figures_from_parquet(
            parquet, staged, out_root / "figures", project_root=project_root
        )
        print(
            json.dumps(
                {
                    "figures": manifest["figure_count"],
                    "out": str(out_root / "figures/figures_manifest.json"),
                },
                indent=2,
            )
        )
        return

    if args.command == "write-plans":
        staged = load_verified_staged(
            out_root / "staging_manifest.json",
            parquet,
            jsonl,
            source_lock_path=source_lock,
        )
        qa = build_qa_source(
            staged,
            out_root / "qa_source.json",
            parquet_path=parquet,
            jsonl_path=jsonl,
            project_root=project_root,
            source_lock_path=source_lock,
        )
        plan = build_pdf_download_plan(
            staged,
            out_root / "pdf_download_plan.json",
            pdf_dir=out_root / "pdfs",
            project_root=project_root,
            source_lock_path=source_lock,
        )
        print(
            json.dumps(
                {
                    "qa_rows": len(qa["data"]),
                    "papers": plan["paper_count"],
                    "qa_source": str(out_root / "qa_source.json"),
                    "pdf_download_plan": str(out_root / "pdf_download_plan.json"),
                },
                indent=2,
            )
        )
        return

    if args.command == "download-pdfs":
        staged = load_verified_staged(
            out_root / "staging_manifest.json",
            parquet,
            jsonl,
            source_lock_path=source_lock,
        )
        plan = build_pdf_download_plan(
            staged,
            out_root / "pdf_download_plan.json",
            pdf_dir=out_root / "pdfs",
            project_root=project_root,
            source_lock_path=source_lock,
        )
        summary = download_pdfs(
            plan,
            pdf_dir=out_root / "pdfs",
            delay_sec=args.delay_sec,
            project_root=project_root,
        )
        print(
            json.dumps(
                {
                    "downloaded": summary["downloaded"],
                    "skipped_existing": summary["skipped_existing"],
                    "failed": summary["failed"],
                    "papers": plan["paper_count"],
                    "pdf_override_count": plan["pdf_override_count"],
                    "out": str(out_root / "pdfs/download_results.json"),
                },
                indent=2,
            )
        )
        return

    if args.command == "render-pages":
        plan = read_json(out_root / "pdf_download_plan.json")
        paper_renders = []
        for paper in plan["papers"]:
            paper_id = paper["paper_id"]
            pdf_path = out_root / "pdfs" / f"{paper_id.replace('/', '__')}.pdf"
            if not pdf_path.is_file():
                raise SystemExit(f"Planned PDF is missing and inventory cannot be complete: {pdf_path}")
            paper_renders.append(
                render_all_pdf_pages(
                    pdf_path,
                    out_root / "images",
                    paper_id=paper_id,
                    project_root=project_root,
                )
            )
        inventory = build_paper_inventory(
            paper_renders,
            out_root / "paper_inventory.json",
            project_root=project_root,
            expected_paper_ids=[str(paper["paper_id"]) for paper in plan["papers"]],
        )
        print(
            json.dumps(
                {
                    "rendered_papers": len(paper_renders),
                    "inventory_papers": len(inventory["papers"]),
                    "out": str(out_root / "paper_inventory.json"),
                },
                indent=2,
            )
        )
        return

    if args.command == "match-evidence":
        staged = load_verified_staged(
            out_root / "staging_manifest.json",
            parquet,
            jsonl,
            source_lock_path=source_lock,
        )
        figures = read_json(out_root / "figures/figures_manifest.json")
        inventory = read_json(out_root / "paper_inventory.json")
        evidence = match_evidence_pages(staged, figures, inventory, project_root=project_root)
        write_json(out_root / "evidence_candidates.json", evidence)
        print(
            json.dumps(
                {
                    "candidate_count": evidence["candidate_count"],
                    "unresolved": evidence["unresolved_count"],
                    "candidate_report": str(out_root / "evidence_candidates.json"),
                    "human_review_required": True,
                },
                indent=2,
            )
        )
        return

    if args.command == "finalize-ocr-inventory":
        inventory = read_json(out_root / "paper_inventory.json")
        provenance_dir = (
            args.ocr_provenance_dir
            if args.ocr_provenance_dir.is_absolute()
            else project_root / args.ocr_provenance_dir
        )
        ready = finalize_inventory_ocr_readiness(
            inventory, project_root=project_root, ocr_provenance_dir=provenance_dir
        )
        destination = args.out or (out_root / "paper_inventory_ocr_ready.json")
        if not destination.is_absolute():
            destination = project_root / destination
        write_json(destination, ready)
        print(json.dumps({"state": ready["state"], "out": str(destination)}, indent=2))
        return

    if args.command == "prepare":
        summary = run_prepare_pipeline(
            parquet_path=parquet,
            jsonl_path=jsonl,
            out_root=out_root,
            project_root=project_root,
            expected_rows=args.expected_rows,
            source_lock_path=source_lock,
            download=args.download,
            render=args.render,
            match=args.match,
            delay_sec=args.delay_sec,
        )
        print(json.dumps(summary, indent=2))
        return

    raise SystemExit(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
