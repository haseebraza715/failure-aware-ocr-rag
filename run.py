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
from faar.api_logging import openai_cost_rates
from faar.benchmarks import load_benchmark_repository
from faar.experiment_runner import run_profile
from faar.gate_tuning import load_locked_threshold
from faar.results_aggregator import summarize_examples
from faar.run_io import atomic_write_text, select_shard, shard_label
from faar.settings import AppSettings
from faar.visual_baselines import run_visual_baseline


BASELINE_MAP = {
    ("off", "off"): ("B0", "naive_rag", "Text-only RAG"),
    ("off", "always_vlm"): ("B1", "faar_always_vlm", "Always-VLM"),
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
    if settings.recovery.vlm_backend == "openai":
        settings.recovery.openai_model = os.getenv("OPENAI_MODEL", settings.recovery.openai_model)
    settings.retrieval.embedding_model = args.embed or os.getenv("EMBED_MODEL", settings.retrieval.embedding_model)
    settings.retrieval.reranker = args.reranker or os.getenv("RERANKER", settings.retrieval.reranker)
    settings.recovery.ocr_engine = args.ocr or os.getenv("OCR_ENGINE", settings.recovery.ocr_engine)
    settings.recovery.log_vlm_calls = True
    settings.experiment.random_seed = args.seed
    if args.wordlevel_fallback:
        settings.experiment.wordlevel_fallback = args.wordlevel_fallback
        settings.recovery.wordlevel_fallback = args.wordlevel_fallback
    locked_threshold = load_locked_threshold(settings.gate_threshold_path)
    if locked_threshold is not None:
        settings.gate.quality_threshold = locked_threshold
    settings.validate_openai_snapshot()
    settings.validate_model_revisions(include_visual=args.mode if args.mode in {"colpali", "visrag"} else None)
    return settings


def _run_profile_to_result(
    settings: AppSettings,
    profile: str,
    out: Path,
    label: str,
    max_examples: int | None,
    run_spec: dict[str, Any],
    dataset: str,
    split: str,
    *,
    resume: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    usage_path = settings.project_root / "logs/vlm_calls.jsonl"
    start_usage = _read_vlm_usage(usage_path)
    rows = run_profile(
        settings,
        profile_name=profile,
        max_examples=max_examples,
        output_dir=out.parent / f"{out.stem}_rows",
        dataset=dataset,
        split=split,
        resume=resume,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    summary = summarize_examples(rows)
    end_usage = _read_vlm_usage(usage_path)
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
            "api_requests": end_usage["api_requests"] - start_usage["api_requests"],
            "prompt_tokens": end_usage["prompt_tokens"] - start_usage["prompt_tokens"],
            "completion_tokens": end_usage["completion_tokens"] - start_usage["completion_tokens"],
            "cost_usd": round(end_usage["cost_usd"] - start_usage["cost_usd"], 6),
            "runtime_sec": round(time.perf_counter() - start, 4),
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(payload, indent=2) + "\n")
    return payload


def _read_vlm_usage(path: Path) -> dict[str, int | float]:
    if not path.exists():
        return {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    usage: dict[str, int | float] = {
        "api_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "started":
                usage["api_requests"] += 1
            usage["prompt_tokens"] += int(record.get("prompt_tokens", 0))
            usage["completion_tokens"] += int(record.get("completion_tokens", 0))
            usage["cost_usd"] += float(record.get("cost_usd", 0.0))
    usage["cost_usd"] = round(float(usage["cost_usd"]), 6)
    return usage


def _apply_baseline_harm(
    payload: dict[str, Any],
    output_path: Path,
    baseline_path: Path | None,
    *,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    if baseline_path is None:
        return payload
    if not baseline_path.is_file():
        raise SystemExit(f"Matching B0 result does not exist: {baseline_path}")
    baseline_payload = json.loads(baseline_path.read_text())
    baseline_spec = baseline_payload.get("run_spec", {}) if isinstance(baseline_payload, dict) else {}
    for key, expected in (("dataset", dataset), ("split", split)):
        observed = baseline_spec.get(key)
        if observed is not None and observed != expected:
            raise SystemExit(
                f"Baseline mismatch: expected {key}={expected!r}, but {baseline_path} records {observed!r}."
            )
    payload["summary"]["harm_rate"] = evaluate_results(output_path, baseline_path=baseline_path)["harm_rate"]
    atomic_write_text(output_path, json.dumps(payload, indent=2) + "\n")
    return payload


def _run_visual_baseline_to_result(
    settings: AppSettings,
    mode: str,
    out: Path,
    max_examples: int | None,
    run_spec: dict[str, Any],
    dataset: str,
    split: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    usage_path = settings.project_root / "logs/vlm_calls.jsonl"
    start_usage = _read_vlm_usage(usage_path)
    repo = load_benchmark_repository(settings.project_root, dataset, split)
    result = run_visual_baseline(settings, repo, mode, max_examples=max_examples)
    if isinstance(result, dict):
        payload = result
    else:
        summary = summarize_examples(result)
        end_usage = _read_vlm_usage(usage_path)
        payload = {
            "label": mode,
            "profile": mode,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_spec": run_spec,
            "summary": {
                "EM": summary["em"],
                "F1": summary["f1"],
                "vlm_rate": summary["vlm_rate"],
                "harm_rate": 0.0,
                "api_requests": end_usage["api_requests"] - start_usage["api_requests"],
                "prompt_tokens": end_usage["prompt_tokens"] - start_usage["prompt_tokens"],
                "completion_tokens": end_usage["completion_tokens"] - start_usage["completion_tokens"],
                "cost_usd": round(end_usage["cost_usd"] - start_usage["cost_usd"], 6),
                "runtime_sec": round(time.perf_counter() - started, 4),
            },
            "rows": result,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(payload, indent=2) + "\n")
    return payload


def _sharded_out_path(out: Path, shard_index: int | None, num_shards: int | None) -> Path:
    if num_shards is None:
        if shard_index is not None:
            raise SystemExit("--shard-index requires --num-shards.")
        return out
    if shard_index is None:
        raise SystemExit("--num-shards requires --shard-index.")
    try:
        select_shard([], shard_index, num_shards)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return out.with_name(f"{out.stem}_{shard_label(shard_index, num_shards)}{out.suffix}")


def _text_run_kwargs(resume: bool, shard_index: int | None, num_shards: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if resume:
        kwargs["resume"] = True
    if num_shards is not None:
        kwargs["shard_index"] = shard_index
        kwargs["num_shards"] = num_shards
    return kwargs


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
    parser.add_argument("--baseline", type=Path, help="Matching B0 result used to compute harm_rate.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reuse per-example checkpoints whose run fingerprint matches.")
    parser.add_argument("--shard-index", type=int, default=None, help="Zero-based shard to run; requires --num-shards.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shards; requires --shard-index.")
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
        "vlm_model": (
            settings.recovery.openai_model
            if settings.recovery.vlm_backend == "openai"
            else settings.recovery.anthropic_model
        ),
        "vlm_cost_rates": openai_cost_rates() if settings.recovery.vlm_backend == "openai" else None,
        "model_provenance": settings.model_provenance(),
    }

    if args.mode in {"colpali", "visrag"}:
        if args.resume or args.shard_index is not None or args.num_shards is not None:
            raise SystemExit("--resume and shard flags are not supported for visual modes yet.")
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_visual_baseline_to_result(
            settings,
            args.mode,
            args.out,
            args.max_examples,
            run_spec,
            args.dataset,
            args.split,
        )
        payload = _apply_baseline_harm(
            payload,
            args.out,
            args.baseline,
            dataset=args.dataset,
            split=args.split,
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    text_out = _sharded_out_path(args.out, args.shard_index, args.num_shards)
    text_kwargs = _text_run_kwargs(args.resume, args.shard_index, args.num_shards)

    if args.mode == "faar":
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_profile_to_result(
            settings, "faar_full", text_out, f"FAAR {args.dataset} {args.split}", args.max_examples, run_spec, args.dataset, args.split, **text_kwargs
        )
        payload = _apply_baseline_harm(
            payload,
            text_out,
            args.baseline,
            dataset=args.dataset,
            split=args.split,
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    if args.ablate:
        ablation_profile = {
            "no_gate": "faar_no_gate",
            "no_diagnosis": "faar_no_diagnosis",
            "no_wordlevel_llm": "faar_symspell",
            "no_semantic_retry": "faar_no_backtrack",
        }[args.ablate]
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
        payload = _run_profile_to_result(
            settings, ablation_profile, text_out, args.ablate, args.max_examples, run_spec, args.dataset, args.split, **text_kwargs
        )
        payload = _apply_baseline_harm(
            payload,
            text_out,
            args.baseline,
            dataset=args.dataset,
            split=args.split,
        )
        print(json.dumps(payload["summary"], indent=2))
        return

    key = (args.gate or "on", args.recovery or "off")
    if key not in BASELINE_MAP:
        raise SystemExit(f"Unsupported baseline combination: gate={args.gate}, recovery={args.recovery}")
    baseline_id, profile, label = BASELINE_MAP[key]
    if baseline_id == "B1":
        _require_key_for_paid_vlm(settings.recovery.vlm_backend)
    payload = _run_profile_to_result(settings, profile, text_out, label, args.max_examples, run_spec, args.dataset, args.split, **text_kwargs)
    default_baseline = Path("results/b0.json") if baseline_id == "B2" and Path("results/b0.json").exists() else None
    payload = _apply_baseline_harm(
        payload,
        text_out,
        args.baseline or default_baseline,
        dataset=args.dataset,
        split=args.split,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
