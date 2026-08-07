#!/usr/bin/env bash
# FailSafeRAG — one-command research demo.
#
# Fully offline and deterministic:
#   * local-hash embeddings (no model download / no GPU)
#   * mock VLM backend (no API cost)
#   * Hugging Face forced offline (ByT5 word-level correction uses local cache)
#
# It runs four representative examples through the real `faar-demo` CLI, then
# renders the run logs as a narrated walkthrough that makes the "failure-aware
# cost control" story explicit: retrieve text-first, spend on typed recovery
# ONLY when a quality signal says the evidence is unreliable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
FAAR_DEMO="$VENV/bin/faar-demo"

BOLD="$(printf '\033[1m')"
CYAN="$(printf '\033[36m')"
GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"
RED="$(printf '\033[31m')"
RESET="$(printf '\033[0m')"
DIM="$(printf '\033[2m')"

say()  { printf '%s%s%s\n' "$CYAN" "$*" "$RESET"; }
ok()   { printf '%s[✓] %s%s\n' "$GREEN" "$*" "$RESET"; }
info() { printf '%s%s%s\n' "$DIM" "$*" "$RESET"; }
die()  { printf '%s[✗] %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }

# ── locate the CLI ──────────────────────────────────────────────────────────
if [ ! -x "$PYTHON" ]; then
    die "virtualenv not found at $VENV — create it first (e.g. 'uv venv .venv')"
fi
if [ ! -x "$FAAR_DEMO" ]; then
    info "faar-demo entry point missing — installing editable package into .venv"
    if ! command -v uv >/dev/null 2>&1; then
        die "uv not found; please run 'python -m pip install -e \"$ROOT\"' first"
    fi
    uv pip install --python "$PYTHON" -e "$ROOT" >/dev/null
fi

# Force offline model access so the demo is reproducible anywhere.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE

# If the ByT5 model is not in the local HF cache, word-level correction degrades
# to a guarded skip (recorded in the run log) instead of a network fetch.
if [ -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--google--byt5-small" ]; then
    info "ByT5 cache: present (word-level recovery fully available)"
else
    info "ByT5 cache: MISSING — word-level recovery will log 'byt5_model_unavailable' and skip;"
    info "             the rest of the demo is unaffected (pre-fetch with 'huggingface-cli download google/byt5-small')"
fi

TMP="${TMPDIR:-/tmp}"
TMP="${TMP%/}"
OUT_DIR="$(mktemp -d "$TMP/faar-demo.XXXXXX")"

run_example() {
    local example_id="$1"
    local label="$2"
    local out="$3"
    info "running example ${example_id:0:8} (${label}) …"
    "$FAAR_DEMO" run-example \
        --example-id "$example_id" \
        --project-root "$ROOT" \
        --vlm-backend mock \
        --seed 42 \
        --output "$out" >/dev/null
    ok "${label} complete"
}

# ── banner ──────────────────────────────────────────────────────────────────
printf '\n%s' "$CYAN"
printf '╔══════════════════════════════════════════════════════════════════════════════════╗\n'
printf '║  FAAR · Failure-Aware Agentic Recovery for OCR-RAG                               ║\n'
printf '║  Text-first retrieval → quality gate → typed recovery ONLY when signals fire     ║\n'
printf '╚══════════════════════════════════════════════════════════════════════════════════╝\n'
printf '%s\n\n' "$RESET"

say "Step 1 · Four real questions over OCR documents (offline, deterministic)"
info "  embeddings  : local-hash-v1   (no model download, no GPU)"
info "  VLM backend : mock            (no API cost)"
info "  seed        : 42"
echo

say "Step 2 · Run the end-to-end pipeline via the faar-demo CLI"
info "  example 1 · manual/dgx_a100 p.27   — bundled example (quality gate fires: weak lexical evidence)"
run_example "446d159e-b5c2-45dc-91cc-faaa931f3649" "bundled example" "$OUT_DIR/01_bundled.json"
info "  example 2 · law agreement p.93     — healthy evidence, expect a clean direct pass"
run_example "1dc1d806-6dda-4c9e-a391-cffa15ae14f2" "clean-pass contrast" "$OUT_DIR/02_clean_pass.json"
info "  example 3 · finance table p.16     — OCR word-noise, expect word-level recovery"
run_example "8960b388-5154-469f-86f5-c3e0c3b82238" "word-level contrast" "$OUT_DIR/03_word_level.json"
info "  example 4 · administration chart p.8 — layout/table corruption, expect structural/visual recovery"
run_example "a9afa87a-d181-49bf-aa09-9a46f25a3c68" "structural contrast" "$OUT_DIR/04_structural.json"
echo

say "Step 3 · Narrated pipeline walkthrough"
echo
"$PYTHON" "$ROOT/scripts/demo_render.py" "$OUT_DIR"

say "Demo complete. Run logs kept for inspection:"
info "  $OUT_DIR"
