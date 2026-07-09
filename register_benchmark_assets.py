from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.benchmarks import build_ohr_asset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Register complete GOT-OCR and page-image assets for a FAAR paper run.")
    parser.add_argument("--dataset", required=True, choices=["ohrbench"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = Path.cwd()
    records = build_ohr_asset_manifest(root, args.split, args.ocr_dir, args.image_dir)
    out = args.out or root / "data/benchmark_assets/ohrbench" / f"{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "ohrbench",
        "split": args.split,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(out), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
