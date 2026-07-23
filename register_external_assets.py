from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.external_assets import build_external_asset_manifest, validate_manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register MP-DocVQA or ArXivQA validation assets for a FAAR paper run."
    )
    parser.add_argument("--dataset", required=True, choices=["mpdocvqa", "arxivqa"])
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--source", type=Path, required=True, help="Source val JSON file.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Base directory for relative OCR/image paths; defaults to the source JSON directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root used for portable relative paths; defaults to the current working directory.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = (args.project_root or Path.cwd()).resolve()
    payload = build_external_asset_manifest(
        args.source,
        args.dataset,
        asset_root=args.asset_root,
        project_root=root,
    )
    payload["created_at_utc"] = datetime.now(UTC).isoformat()
    validate_manifest_payload(payload, project_root=root)
    out = args.out or root / "data/benchmark_assets" / args.dataset / "val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out.relative_to(root)) if out.is_relative_to(root) else str(out),
                "records": len(payload["records"]),
                "corpus_pages": len(payload["corpus_pages"]),
                "documents": len(payload["document_inventory"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
