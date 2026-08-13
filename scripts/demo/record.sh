#!/usr/bin/env bash
# record.sh — regenerate assets/demo/demo.cast (the terminal recording).
# Then render with: MEDIA_VENV=<media venv> scripts/demo/render.sh
# Tool-path detection with graceful errors; safe to re-run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ASCIINEMA="${ASCIINEMA:-$HOME/.local/bin/asciinema}"
if [ ! -x "$ASCIINEMA" ]; then
  if command -v asciinema >/dev/null 2>&1; then
    ASCIINEMA="$(command -v asciinema)"
  else
    echo "ERROR: asciinema not found (expected at $HOME/.local/bin/asciinema or on PATH)" >&2
    exit 1
  fi
fi

VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: project venv not found at $VENV — create it first (e.g. 'uv venv .venv && uv pip install -e .')" >&2
  exit 1
fi

mkdir -p assets/demo
export PATH="$VENV/bin:$PATH"

echo "Recording to assets/demo/demo.cast (100x30)…"
"$ASCIINEMA" rec -q --overwrite --cols 100 --rows 30 -c "bash scripts/demo/demo_body.sh" assets/demo/demo.cast
echo "Done: assets/demo/demo.cast ($(wc -c < assets/demo/demo.cast 2>/dev/null) bytes)"
echo "Render next: scripts/demo/render.sh assets/demo/demo.cast assets/demo"
