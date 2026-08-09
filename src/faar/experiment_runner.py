from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .answering import answer_from_hits
from .data import Phase0Repository
from .experiment_profiles import apply_profile
from .graph import build_graph
from .metrics import exact_match, ndcg_at_k, recall_at_k, token_f1
from .settings import AppSettings


def _effect_label(actual: float, counterfactual: float) -> str:
    if actual > counterfactual:
        return "improved"
    if actual < counterfactual:
        return "worsened"
    return "equal"


def run_profile(
    settings: AppSettings,
    profile_name: str,
    max_examples: int | None = None,
    output_dir: Path | None = None,
    example_ids: list[str] | None = None,
    selection: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    settings = apply_profile(settings, profile_name)
    repo = Phase0Repository(settings)
    graph = build_graph(settings)
    selected_ids = list(example_ids) if example_ids is not None else repo.list_example_ids()
    if max_examples is not None and example_ids is None:
        selected_ids = selected_ids[: max(0, max_examples)]
    base_output = output_dir or (settings.project_root / "logs/phase3" / profile_name)
    base_output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    for example_id in selected_ids:
        try:
            result = graph.invoke({"example_id": example_id})
        except Exception as exc:
            row = _error_row(
                profile_name,
                example_id,
                exc,
                run_id,
                seed,
                settings,
                selection,
                max_examples,
                len(selected_ids),
            )
            rows.append(row)
            (base_output / f"{example_id}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            continue
        question = result.get("question", "")
        hit_texts = [hit.chunk.text for hit in result.get("corrected_hits") or result.get("retrieved_hits", [])]
        gold = result["example"].correct_answer
        prediction = result.get("answer", "")
        # Direct-answer counterfactual: what the extractor would answer from the
        # pre-recovery retrieval hits alone. Comparing it with the recovered answer
        # makes the effect of typed recovery measurable per example.
        retrieved_hits = result.get("retrieved_hits", [])
        counterfactual_answer = answer_from_hits(question, retrieved_hits).get("answer", "")
        em = exact_match(prediction, gold)
        f1 = token_f1(prediction, gold)
        counterfactual_em = exact_match(counterfactual_answer, gold)
        counterfactual_f1 = token_f1(counterfactual_answer, gold)
        row = {
            "profile": profile_name,
            "example_id": example_id,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": prediction,
            "failure_type": result.get("failure_type", "pass"),
            "policy_action": result.get("policy_action", "answer_direct"),
            "action_outcome": result.get("action_outcome", {}),
            "metrics": {
                "ndcg@5": ndcg_at_k(hit_texts, gold, k=5),
                "recall@5": recall_at_k(hit_texts, gold, k=5),
                "em": em,
                "f1": f1,
            },
            "recovery_metrics": {
                "counterfactual_answer": counterfactual_answer,
                "recovery_changed_answer": counterfactual_answer != prediction,
                "em": {
                    "actual": round(em, 4),
                    "counterfactual": round(counterfactual_em, 4),
                    "delta": round(em - counterfactual_em, 4),
                    "effect": _effect_label(em, counterfactual_em),
                },
                "f1": {
                    "actual": round(f1, 4),
                    "counterfactual": round(counterfactual_f1, 4),
                    "delta": round(f1 - counterfactual_f1, 4),
                    "effect": _effect_label(f1, counterfactual_f1),
                },
            },
            "top_hit_texts": hit_texts[:5],
            "run_metadata": {
                "profile": profile_name,
                "run_id": run_id,
                "api_enabled": settings.recovery.api_enabled,
                "vlm_backend": settings.recovery.vlm_backend,
                "openai_model": settings.recovery.openai_model,
                "enable_byt5": settings.recovery.enable_byt5,
                "byt5_model": settings.recovery.byt5_model,
                "seed": seed,
                "evaluation_size": len(selected_ids),
                "selection": selection or {"max_examples": max_examples},
            },
        }
        rows.append(row)
        destination = base_output / f"{example_id}.json"
        destination.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return rows


def _error_row(
    profile_name: str,
    example_id: str,
    exc: Exception,
    run_id: str,
    seed: int,
    settings: AppSettings,
    selection: dict[str, Any] | None,
    max_examples: int | None,
    evaluation_size: int,
) -> dict[str, Any]:
    return {
        "profile": profile_name,
        "example_id": example_id,
        "question": "",
        "gold_answer": "",
        "predicted_answer": "",
        "failure_type": "error",
        "policy_action": "error",
        "action_outcome": {
            "action": "failed",
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        },
        "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0},
        "recovery_metrics": {
            "counterfactual_answer": "",
            "recovery_changed_answer": False,
            "em": {"actual": 0.0, "counterfactual": 0.0, "delta": 0.0, "effect": "equal"},
            "f1": {"actual": 0.0, "counterfactual": 0.0, "delta": 0.0, "effect": "equal"},
        },
        "top_hit_texts": [],
        "run_metadata": {
            "profile": profile_name,
            "run_id": run_id,
            "api_enabled": settings.recovery.api_enabled,
            "vlm_backend": settings.recovery.vlm_backend,
            "openai_model": settings.recovery.openai_model,
            "enable_byt5": settings.recovery.enable_byt5,
            "byt5_model": settings.recovery.byt5_model,
            "seed": seed,
            "evaluation_size": evaluation_size,
            "selection": selection or {"max_examples": max_examples},
        },
    }
