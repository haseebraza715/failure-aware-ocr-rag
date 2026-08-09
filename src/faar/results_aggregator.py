from __future__ import annotations

from statistics import mean
from typing import Any

from .api_logging import is_valid_api_usage


def summarize_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "ndcg@5": 0.0,
            "recall@5": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "visual_fallback_rate": 0.0,
            "vlm_rate": 0.0,
        }
    ndcg = mean(row["metrics"]["ndcg@5"] for row in rows)
    recall = mean(row["metrics"]["recall@5"] for row in rows)
    em = mean(row["metrics"]["em"] for row in rows)
    f1 = mean(row["metrics"]["f1"] for row in rows)
    # Measure fallback usage by executed action (action_outcome), not planned route
    # (policy_action). This avoids over-counting in profiles that reroute to direct answer.
    visual_rate = mean(
        1.0 if (row.get("action_outcome") or {}).get("action") == "invoke_vlm" else 0.0 for row in rows
    )
    return {
        "count": len(rows),
        "ndcg@5": round(ndcg, 4),
        "recall@5": round(recall, 4),
        "em": round(em, 4),
        "f1": round(f1, 4),
        "visual_fallback_rate": round(visual_rate, 4),
        "vlm_rate": round(visual_rate, 4),
    }


def summarize_api_usage(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "api_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }
    for index, row in enumerate(rows):
        usage = row.get("api_usage") if isinstance(row, dict) else None
        if not is_valid_api_usage(usage):
            raise ValueError(f"row {index} has missing or invalid api_usage")
        totals["api_requests"] += int(usage["api_requests"])
        totals["prompt_tokens"] += int(usage["prompt_tokens"])
        totals["completion_tokens"] += int(usage["completion_tokens"])
        totals["cost_usd"] += float(usage["cost_usd"])
    totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
    return totals


def summarize_by_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["profile"], []).append(row)
    return {profile: summarize_examples(rows) for profile, rows in grouped.items()}
