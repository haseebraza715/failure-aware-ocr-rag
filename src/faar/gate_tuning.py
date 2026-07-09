from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .metrics import exact_match


THETA_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass(frozen=True)
class GateExample:
    example_id: str
    top_reranker_score: float
    needs_recovery: bool


def load_gate_examples(path: Path) -> list[GateExample]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected result rows in {path}.")

    examples: list[GateExample] = []
    for row in rows:
        gate = row.get("gate") or {}
        score = row.get("top_reranker_score", gate.get("top_reranker_score"))
        if score is None:
            raise ValueError(f"Missing top_reranker_score for example {row.get('example_id', '<unknown>')}.")
        metrics = row.get("metrics") or {}
        if "em" in metrics:
            needs_recovery = float(metrics["em"]) < 1.0
        else:
            needs_recovery = exact_match(
                str(row.get("predicted_answer", row.get("answer", ""))),
                str(row.get("gold_answer", row.get("correct_answer", ""))),
            ) < 1.0
        examples.append(
            GateExample(str(row.get("example_id", "")), float(score), needs_recovery)
        )
    if not examples:
        raise ValueError("No rows available for gate threshold tuning.")
    return examples


def metrics_at_theta(examples: Iterable[GateExample], theta: float) -> dict[str, float | int]:
    values = list(examples)
    # Low top-1 reranker confidence means the controller should enter recovery.
    tp = sum(example.needs_recovery and example.top_reranker_score < theta for example in values)
    fp = sum(not example.needs_recovery and example.top_reranker_score < theta for example in values)
    fn = sum(example.needs_recovery and example.top_reranker_score >= theta for example in values)
    tn = sum(not example.needs_recovery and example.top_reranker_score >= theta for example in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "theta": theta,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def search_threshold(examples: Iterable[GateExample], theta_grid: Iterable[float] = THETA_GRID) -> dict[str, Any]:
    values = list(examples)
    grid = [metrics_at_theta(values, theta) for theta in theta_grid]
    winner = max(grid, key=lambda item: (float(item["f1"]), float(item["precision"]), float(item["recall"]), -float(item["theta"])))
    return {
        "signal": "BGE-reranker-v2-m3 top-1 score",
        "positive_class": "B0 exact-match error; route to recovery when score < theta",
        "validation_examples": len(values),
        "grid": grid,
        "winner": winner,
        "pass_criteria": {
            "precision_at_least": 0.75,
            "recall_at_least": 0.70,
            "passed": winner["precision"] >= 0.75 and winner["recall"] >= 0.70,
        },
    }


def require_validation_payload(path: Path) -> None:
    payload = json.loads(path.read_text())
    run_spec = payload.get("run_spec") if isinstance(payload, dict) else None
    if not isinstance(run_spec, dict) or run_spec.get("split") != "val":
        raise ValueError(
            "Gate threshold tuning accepts only a result JSON produced with run_spec.split='val'; test results are forbidden."
        )


def write_locked_threshold(path: Path, search: dict[str, Any]) -> None:
    winner = search["winner"]
    payload = {
        "source_split": "val",
        "signal": search["signal"],
        "threshold": winner["theta"],
        "precision": winner["precision"],
        "recall": winner["recall"],
        "f1": winner["f1"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_locked_threshold(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("source_split") != "val":
        raise ValueError(f"Locked threshold at {path} does not declare a validation-only source.")
    return float(payload["threshold"])
