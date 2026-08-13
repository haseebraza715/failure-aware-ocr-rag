from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.arxivqa_remap import build_full_paper_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable full-paper ArXivQA source data.")
    parser.add_argument("--qa", type=Path, required=True, help="Raw ArXivQA val QA JSON.")
    parser.add_argument("--paper-inventory", type=Path, required=True, help="Complete paper/page inventory JSON.")
    parser.add_argument("--figure-mapping", type=Path, required=True, help="Explicit QA-to-paper/page mapping JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output source JSON consumed by manifest registration.")
    args = parser.parse_args()
    payload = build_full_paper_source(args.qa, args.paper_inventory, args.figure_mapping, output=args.out)
    print(f"Wrote {len(payload['data'])} QA rows across {len(payload['documents'])} complete papers to {args.out}")


if __name__ == "__main__":
    main()
