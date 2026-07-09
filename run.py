from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import evaluate_results
from faar.experiment_runner import run_profile
from faar.gate_tuning import load_locked_threshold
from faar.results_aggregator import summarize_examples
from faar.settings import AppSettings


BASELINE_MAP = {
    ("off", "off"): ("B0", "naive_rag", "Text-only RAG"),
    ("off", "always_vlm"): ("B1", "faar_full", "Always-VLM"),
    ("on", "random_type"): ("B2", "faar_no_diagnosis", "Random recovery"),
}


def _require_key_for_paid_vlm(vlm_backend: str) -> None:
    if vlm_backend in {"claude-sonnet-4-5", "anthropic", "claude"} and not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing required key: ANTHROPIC_API_KEY for VLM_BACKEND=claude-sonnet-4-5.")
    if vlm_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing required key: OPENAI_API_KEY for VLM_BACKEND=openai.")


def _settings_from_args(args: argparse.Namespace) -> AppSettings:
    settings = AppSettings(project_root=Path.cwd())
    settings.recovery.vlm_backend = args.vlm or os.getenv("VLM_BACKEND", settings.recovery.vlm_backend)
    if settings.recovery.vlm_backend.startswith("claude"):
        settings.recovery.anthropic_model = settings.recovery.vlm_backend
    settings.retrieval.embedding_model = args.embed or os.getenv("EMBED_MODEL", settings.retrieval.embedding_model)
    settings.retrieval.reranker = args.reranker or os.getenv("RERANKER", settings.retrieval.reranker)
    settings.recovery.ocr_engine = args.ocr or os.getenv("OCR_ENGINE", settings.recovery.ocr_engine)
    settings.recovery.log_vlm_calls = os.getenv("LOG_VLM_CALLS", "false").lower() == "true"
    locked_threshold = load_locked_threshold(settings.gate_threshold_path)
    if locked_threshold is not None:
        settings.gate.quality_threshold = locked_threshold
    return settings


def _run_profile_to_result(
    settings: AppSettings,
    profile: str,
    out: Path,
    label: str,
    max_examples: int | None,
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    rows = run_profile(settings, profile_name=profile, max_examples=max_examples, output_dir=out.parent / f"{out.stem}_rows")
    summary = summarize_examples(rows)
    payload = {
        "label": label,
        "profile": profile,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_spec": run_spec,
        "summary": {
            "EM": summary["em"],
            "F1": summary["f1"],
            "vlm_rate": summary["vlm_rate"],
            "harm_rate": 0.0,
            "cost_usd": _read_vlm_cost(settings.project_root / "logs/vlm_calls.jsonl"),
            "runtime_sec": round(time.perf_counter() - start, 4),
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _read_vlm_cost(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if line.strip():
            total += float(json.loads(line).get("cost_usd", 0.0))
    return round(total, 6)


def _unsupported_external_runtime(args: argparse.Namespace) -> None:
    raise SystemExit(
        f"{args.mode or args.dataset} requires external assets/runtime not present locally. "
        "Do not report placeholder numbers; add the required dataset/model dependencies and rerun."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FAAR AAAI experiment runner compatibility wrapper.")
    parser.add_argument("--gate", choices=["on", "off"])
    parser.add_argument("--recovery", choices=["off", "always_vlm", "random_type"])
    parser.add_argument("--mode", choices=["faar", "colpali", "visrag"])
    parser.add_argument("--dataset", default="ohrbench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ocr")
    parser.add_argument("--embed")
    parser.add_argument("--reranker")
    parser.add_argument("--vlm")
    parser.add_argument("--ablate", choices=["no_gate", "no_diagnosis", "no_wordlevel_llm", "no_semantic_retry"])
    parser.add_argument("--wordlevel_fallback")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    settings = _settings_from_args(args)
    settings.validate_runtime_paths()
    run_spec = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "gate_threshold": settings.gate.quality_threshold,
        "embedding_model": settings.retrieval.embedding_model,
        "reranker": settings.retrieval.reranker,
        "ocr_engine": settings.recovery.ocr_engine,
        "vlm_backend": settings.recovery.vlm_backend,
    }

    if args.mode in {"colpali", "visrag"}:
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        _unsupported_external_runtime(args)

    if args.mode == "faar":
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_profile_to_result(
            settings, "faar_full", args.out, f"FAAR {args.dataset} {args.split}", args.max_examples, run_spec
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    if args.ablate:
        ablation_profile = {
            "no_gate": "faar_full",
            "no_diagnosis": "faar_no_diagnosis",
            "no_wordlevel_llm": "faar_full",
            "no_semantic_retry": "faar_no_backtrack",
        }[args.ablate]
        if args.ablate == "no_gate":
            _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_profile_to_result(settings, ablation_profile, args.out, args.ablate, args.max_examples, run_spec)
        print(json.dumps(payload["summary"], indent=2))
        return

    key = (args.gate or "on", args.recovery or "off")
    if key not in BASELINE_MAP:
        raise SystemExit(f"Unsupported baseline combination: gate={args.gate}, recovery={args.recovery}")
    baseline_id, profile, label = BASELINE_MAP[key]
    if baseline_id == "B1":
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
    payload = _run_profile_to_result(settings, profile, args.out, label, args.max_examples, run_spec)
    if baseline_id == "B2" and Path("results/b0.json").exists():
        payload["summary"]["harm_rate"] = evaluate_results(args.out, baseline_path=Path("results/b0.json"))["harm_rate"]
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
