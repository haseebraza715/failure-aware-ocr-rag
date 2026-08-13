#!/usr/bin/env bash
# demo_body.sh — drives the recorded demo session (deterministic, offline).
# Run from the repository root. Set PATH to the project venv so commands look clean.
# NOTE: edit this file to change the demo; then run scripts/demo/record.sh to regenerate.
set -uo pipefail

PROMPT='\033[1;32m❯\033[0m '

header() { printf '\033[1;36m%s\033[0m\n' "$1"; sleep 1.2; }
run() {
  printf "${PROMPT}%s\n" "$*"
  sleep 1.2
  "$@"
  sleep 2.6
}
pause() { sleep "$1"; }

header "FAAR · Failure-Aware Agentic Recovery for OCR-RAG"
header "Text-first retrieval -> quality gate -> diagnose -> recover -> answer; vision only when needed"

header "Step 1 · the demo corpus (committed, offline, deterministic)"
run ls -R examples/demo_corpus

header "Step 2 · two pages, same facts — one clean, one corrupted by OCR word noise"
run head -n 2 examples/demo_corpus/artifacts/phase0/ocr_text/noisy_threshold.txt
run head -n 2 examples/demo_corpus/artifacts/phase0/ocr_text/clean_threshold.txt

header "Step 3 · run the real pipeline over the corpus (local-hash embeddings, mock VLM, seed 42)"
run python scripts/demo/demo_run.py

header "Story: retrieval lands, the quality gate flags the corrupted evidence (word_noise_alert),"
header "the failure is diagnosed and routed to word-level recovery, and the answer is delivered"
header "with measured confidence (support_score 9 vs 10 on the clean page) — zero multimodal spend."

pause 2.5
