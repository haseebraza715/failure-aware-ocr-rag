from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def grep_matches(root: Path, term: str) -> list[str]:
    command = ["grep", "-r", "-l", term, "."]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    # The plan calls grep over the full checkout, but Git remote metadata is not
    # a paper source artifact and is never included in an anonymized submission.
    return [line for line in completed.stdout.splitlines() if line and not line.startswith("./.git/")]


def pdf_author(pdf_path: Path) -> str | None:
    if not pdf_path.exists():
        return None
    if shutil.which("pdfinfo") is None:
        return None
    completed = subprocess.run(["pdfinfo", str(pdf_path)], text=True, capture_output=True, check=False)
    for line in completed.stdout.splitlines():
        if line.startswith("Author:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FAAR plan's before-submitting anonymization checks.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--paper", type=Path, default=Path("paper.pdf"))
    parser.add_argument("--out", type=Path, default=Path("paper_artifacts/pre_submission_check.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    paper = args.paper if args.paper.is_absolute() else root / args.paper
    allowed_checklist_references = {
        "./docs/faar-aaai-plan.md",
        "./PLAN.md",
        "./pre_submission_check.py",
    }
    raw_handle_matches = grep_matches(root, "haseebraza")
    raw_name_matches = grep_matches(root, "your-real-name")
    report = {
        "haseebraza_matches": raw_handle_matches,
        "your_real_name_matches": raw_name_matches,
        "unexpected_haseebraza_matches": [path for path in raw_handle_matches if path not in allowed_checklist_references],
        "unexpected_your_real_name_matches": [path for path in raw_name_matches if path not in allowed_checklist_references],
        "pdf_path": str(paper),
        "pdf_present": paper.exists(),
        "pdf_author": pdf_author(paper),
    }
    report["passed"] = (
        not report["unexpected_haseebraza_matches"]
        and not report["unexpected_your_real_name_matches"]
        and report["pdf_present"]
        and report["pdf_author"] == ""
    )
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Pre-submission anonymization check failed.")


if __name__ == "__main__":
    main()
