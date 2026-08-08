from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def _empty_recovery_outcomes() -> dict[str, Any]:
    return {
        "reviewed_count": 0,
        "changed_answer_count": 0,
        "changed_answer_rate": 0.0,
        "em_effect_distribution": {},
        "f1_effect_distribution": {},
    }


def summarize_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "ndcg@5": 0.0,
            "recall@5": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "visual_fallback_rate": 0.0,
            "recovery_outcomes": _empty_recovery_outcomes(),
        }
    ndcg = mean(row["metrics"]["ndcg@5"] for row in rows)
    recall = mean(row["metrics"]["recall@5"] for row in rows)
    em = mean(row["metrics"]["em"] for row in rows)
    f1 = mean(row["metrics"]["f1"] for row in rows)
    # Measure fallback usage by executed action (action_outcome), not planned route
    # (policy_action). This avoids over-counting in profiles that reroute to direct answer.
    visual_rate = mean(1.0 if (row.get("action_outcome") or {}).get("action") == "invoke_vlm" else 0.0 for row in rows)
    recovery_rows = [
        row
        for row in rows
        if row.get("recovery_metrics")
        and (row.get("action_outcome") or {}).get("action") in {"correct_text", "retry_retrieval", "invoke_vlm"}
    ]
    changed = sum(
        1 for row in recovery_rows if bool((row.get("recovery_metrics") or {}).get("recovery_changed_answer", False))
    )
    em_effects = Counter(
        ((row.get("recovery_metrics") or {}).get("em") or {}).get("effect", "equal") for row in recovery_rows
    )
    f1_effects = Counter(
        ((row.get("recovery_metrics") or {}).get("f1") or {}).get("effect", "equal") for row in recovery_rows
    )
    recovery_outcomes = {
        "reviewed_count": len(recovery_rows),
        "changed_answer_count": changed,
        "changed_answer_rate": round(changed / len(recovery_rows), 4) if recovery_rows else 0.0,
        "em_effect_distribution": dict(em_effects),
        "f1_effect_distribution": dict(f1_effects),
    }
    return {
        "count": len(rows),
        "ndcg@5": round(ndcg, 4),
        "recall@5": round(recall, 4),
        "em": round(em, 4),
        "f1": round(f1, 4),
        "visual_fallback_rate": round(visual_rate, 4),
        "recovery_outcomes": recovery_outcomes,
    }


def summarize_by_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["profile"], []).append(row)
    return {profile: summarize_examples(rows) for profile, rows in grouped.items()}
