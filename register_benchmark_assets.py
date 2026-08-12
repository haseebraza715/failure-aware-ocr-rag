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
from faar.run_io import atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register complete GOT-OCR and page-image assets for a FAAR paper run.")
    parser.add_argument("--dataset", required=True, choices=["ohrbench"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--document-inventory",
        type=Path,
        help="Directory of complete per-document page inventories (default: OHR-Bench/data/retrieval_base/gt).",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    manifest = build_ohr_asset_manifest(
        root,
        args.split,
        args.ocr_dir,
        args.image_dir,
        document_inventory_dir=args.document_inventory,
    )
    out = args.out or root / "data/benchmark_assets/ohrbench" / f"{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "ohrbench",
        "split": args.split,
        "created_at_utc": datetime.now(UTC).isoformat(),
        **manifest,
    }
    atomic_write_text(out, json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(out.relative_to(root)) if out.is_relative_to(root) else str(out),
                "records": len(manifest["records"]),
                "corpus_pages": len(manifest["corpus_pages"]),
                "documents": len(manifest["document_inventory"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
