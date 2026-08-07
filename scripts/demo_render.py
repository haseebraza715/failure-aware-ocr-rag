#!/usr/bin/env python3
"""Render FAAR `run-example` JSON logs as a narrated, human-readable walkthrough.

Usage:
    python scripts/demo_render.py <run1.json> [run2.json ...]
    python scripts/demo_render.py <directory-of-runs/>

Reads the structured output written by `faar-demo run-example --output <path>`
and prints a console-friendly narration of each run followed by a routing
summary that makes the "failure-aware cost" story explicit.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

WIDTH = 92

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"

SIGNAL_LABELS = {
    "low_quality_score": "quality score below confidence threshold",
    "layout_alert": "layout-structure signals (tables/columns) detected",
    "word_noise_alert": "OCR word-noise / corruption detected",
    "low_lexical_score": "lexical (BM25) evidence too weak to trust",
    "low_dense_score": "dense-embedding evidence too weak to trust",
    "no_retrieval_hits": "retrieval returned no evidence",
}

RECOVERY_LABELS = {
    "semantic": ("retry_retrieval", "semantic recovery — re-query with context-anchored evidence"),
    "word_level": ("correct_text", "word-level recovery — correct OCR noise before answering"),
    "structural": ("invoke_vlm", "structural recovery — selective visual fallback"),
}


def _banner(text: str, char: str = "=") -> None:
    print(BOLD + CYAN + char * WIDTH)
    print(f"{char} {text}".ljust(WIDTH - 1) + char)
    print(char * WIDTH + RESET)


def _rule(char: str = "-") -> None:
    print(DIM + char * WIDTH + RESET)


def _field(label: str, value: str, color: str = RESET) -> None:
    print(f"  {BOLD}{label:<11}{RESET} {color}{value}{RESET}")


def _cell(plain: str, color: str, width: int) -> str:
    pad = " " * max(width - len(plain), 0)
    return f"{color}{plain}{pad}{RESET}"


def _wrap(text: str, indent: str = "  ") -> list[str]:
    return textwrap.wrap(text.replace("\n", " "), width=WIDTH - len(indent) - 2)


def _print_wrapped(label: str, text: str) -> None:
    lines = _wrap(text)
    print(f"  {BOLD}{label:<11}{RESET} {lines[0]}")
    for line in lines[1:]:
        print(f"{'':<14}{line}")


def _ok(value: str) -> str:
    return GREEN + value + RESET


def _warn(value: str) -> str:
    return YELLOW + value + RESET


def _info(value: str) -> str:
    return CYAN + value + RESET


def _fail(value: str) -> str:
    return RED + value + RESET


def _gate_verdict(payload: dict) -> tuple[str, str]:
    gate = payload.get("gate", {})
    reasons = gate.get("reasons", [])
    if not reasons:
        return _ok("PASS"), "evidence looks healthy — answer directly"
    headline = SIGNAL_LABELS.get(reasons[0], reasons[0])
    extra = f" (+{len(reasons) - 1} more)" if len(reasons) > 1 else ""
    return _fail("FAIL"), f"{headline}{extra} — do not trust a direct answer"


def _multimodal_spend(payload: dict) -> str:
    action = payload.get("policy_action")
    visual = payload.get("visual_result") or {}
    if action == "invoke_vlm":
        if visual.get("status") == "succeeded":
            return _warn("SPENT (VLM invoked)")
        return _ok("0  (mock backend — would invoke VLM in production)")
    return _ok("0  (text-only pipeline)")


def _render_run(payload: dict) -> None:
    meta = payload.get("run_metadata", {})
    model_config = meta.get("model_config", {})
    top = (payload.get("top_hits") or [{}])[0]
    chunk = top.get("chunk", {})
    gate = payload.get("gate", {})
    verdict_color, verdict_note = _gate_verdict(payload)
    failure = payload.get("failure_type")
    action = payload.get("policy_action")
    if action in ("correct_text", "retry_retrieval", "invoke_vlm"):
        _, recovery_desc = RECOVERY_LABELS.get(failure, (action, action))
    else:
        recovery_desc = "none — direct answer"
    retry_query = payload.get("semantic_retry_query") or ""
    outcome = payload.get("action_outcome", {})
    answer_meta = payload.get("answer_meta", {})

    _banner(f"{payload.get('example_id', '')[:8]}  ·  {chunk.get('doc_name', '')}  p.{chunk.get('page_id', '?')}")
    _field("QUESTION", payload.get("question", ""))
    gold = payload.get("correct_answer", "")
    pred = payload.get("answer", "")
    match = str(pred).strip().lower() == str(gold).strip().lower()
    _field("GROUND TRUTH", str(gold))
    _field("ANSWER", str(pred), GREEN if match else YELLOW)
    _print_wrapped("EVIDENCE", (chunk.get("text", "") or "")[:340] + (" …" if len(chunk.get("text", "")) > 340 else ""))
    _rule()

    _field("1 RETRIEVE", f"text-first hybrid (BM25 + local-hash dense)  ·  {model_config.get('embedding_backend', '?')} backend")
    print(f"{'':<13} top hit fused {top.get('fused_score', 0):.3f}  ·  lexical {top.get('bm25_score', 0):.2f}  ·  dense {top.get('dense_score', 0):.2f}")

    qs = gate.get("quality_score", 0)
    corruption = gate.get("corruption_score", 0)
    layout_count = gate.get("layout_signal_count", 0)
    print(f"  {BOLD}2 GATE{RESET}      quality {qs:.3f}  ·  OCR corruption {corruption:.3f}  ·  layout alerts {layout_count}")
    print(f"{'':<13}{verdict_color} {verdict_note}")
    if gate.get("layout_signals"):
        active = [k for k, v in gate.get("layout_signals", {}).items() if v]
        if active:
            print(f"{'':<13} layout signals: {', '.join(active)}")

    if action in ("correct_text", "retry_retrieval", "invoke_vlm"):
        print(f"  {BOLD}3 DIAGNOSE{RESET}  failure type = {_info(str(failure))}  →  {BOLD}{_info(recovery_desc)}{RESET}")
        if retry_query:
            _print_wrapped("4 RETRY", f"re-query: {retry_query}")
        if outcome:
            print(f"{'':<13} outcome: {outcome.get('action')} · {outcome.get('status')} · {outcome.get('reason')}")
        label = "5 ANSWER"
    else:
        print(f"  {BOLD}3 DECIDE{RESET}    no failure diagnosed — {_ok('answer directly')}, recovery skipped")
        label = "4 ANSWER"

    print(
        f"  {BOLD}{label}{RESET}     {str(pred)}  "
        f"({answer_meta.get('answer_mode', '?')}, support {answer_meta.get('support_score', 0)})"
    )
    print(f"  {BOLD}COST{RESET}       multimodal tokens: {_multimodal_spend(payload)}")
    print()


def _render_summary(runs: list[dict]) -> None:
    _banner("ROUTING SUMMARY — FAILURE-AWARE COST CONTROL")
    header = f"{'example':<10}{'gate':<6}{'signal fired':<44}{'recovery':<20}{'multimodal':<30}"
    print(BOLD + header + RESET)
    _rule()
    for payload in runs:
        eid = payload.get("example_id", "?")[:8]
        gate = payload.get("gate", {})
        reasons = gate.get("reasons", [])
        verdict = "PASS" if not reasons else "FAIL"
        signal = reasons[0] if reasons else "— (healthy)"
        if len(reasons) > 1:
            signal += f" +{len(reasons) - 1}"
        action = payload.get("policy_action", "answer_direct")
        if action == "answer_direct":
            recovery = "none"
        else:
            recovery = f"{payload.get('failure_type')} ({action})"
        spend = _multimodal_spend(payload)
        gate_cell = _cell(verdict, GREEN if verdict == "PASS" else RED, 5)
        print(f"{_cell(eid, RESET, 10)}{gate_cell}{_cell(signal, RESET, 44)}{_cell(recovery, RESET, 20)}{spend}")
    print()
    _print_wrapped(
        "READ",
        "A naive always-on visual pipeline sends every page image to a VLM and pays "
        "multimodal cost on every question. FAAR retrieves text-first and only routes "
        "to visual (or word-level / semantic) recovery when the quality gate reports a "
        "specific failure — so clean questions cost nothing, and difficult OCR questions "
        "pay only for the recovery they actually need.",
    )
    print()


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: demo_render.py <run.json> [run.json ...] or <dir/>")
        sys.exit(2)
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob("*.json"))
    runs: list[dict] = []
    for path in paths:
        runs.append(json.loads(path.read_text()))
    if not runs:
        print("no run logs found")
        sys.exit(1)

    print()
    _banner("FAAR · FAILURE-AWARE AGENTIC RECOVERY FOR OCR-RAG")
    print(
        textwrap.fill(
            "A document-QA research pipeline that retrieves text-first, then spends on "
            "typed recovery (semantic retry · word-level OCR correction · structural/visual "
            "fallback) ONLY when quality signals say the evidence is unreliable.",
            width=WIDTH,
        )
    )
    print()
    first_meta = runs[0].get("run_metadata", {})
    cfg = first_meta.get("model_config", {})
    print(f"  embeddings : {cfg.get('embedding_backend', '?')}  (deterministic, no model download)")
    print(f"  VLM backend: {cfg.get('vlm_backend', '?')}  (offline, no API cost)")
    print(f"  seed       : {first_meta.get('seed', '?')}  ·  python {first_meta.get('python_version', '?')}")
    print()
    _rule("·")

    for payload in runs:
        _render_run(payload)

    _render_summary(runs)


if __name__ == "__main__":
    main()
