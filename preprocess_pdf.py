from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.pdf_preprocessing import export_docling_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF to structured Markdown with Docling.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = export_docling_markdown(args.pdf, args.out)
    print(destination)


if __name__ == "__main__":
    main()
