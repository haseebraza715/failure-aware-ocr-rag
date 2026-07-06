from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.metrics import exact_match, token_f1


REQUIRED_KEYS = ("EM", "F1", "vlm_rate", "harm_rate")


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if "rows" in payload:
        return list(payload["rows"])
    if "examples" in payload:
        return list(payload["examples"])
    if "summary" in payload and all(key in payload["summary"] for key in REQUIRED_KEYS):
        return []
    return [payload]


def evaluate_results(results_path: Path, baseline_path: Path | None = None) -> dict[str, float]:
    payload = json.loads(results_path.read_text())
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict) and all(key in summary for key in REQUIRED_KEYS):
        return {key: float(summary[key]) for key in REQUIRED_KEYS}

    rows = load_rows(results_path)
    baseline_by_id = {}
    if baseline_path and baseline_path.exists():
        baseline_by_id = {row.get("example_id"): row for row in load_rows(baseline_path)}

    if not rows:
        metrics = {"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0}
    else:
        em_values = []
        f1_values = []
        vlm_values = []
        harm_values = []
        for row in rows:
            row_metrics = row.get("metrics") or {}
            prediction = row.get("predicted_answer", row.get("answer", ""))
            gold = row.get("gold_answer", row.get("correct_answer", ""))
            em = float(row_metrics.get("em", exact_match(prediction, gold)))
            f1 = float(row_metrics.get("f1", token_f1(prediction, gold)))
            action = row.get("action_outcome") or {}
            visual = row.get("visual_result") or {}
            used_vlm = action.get("action") == "invoke_vlm" or visual.get("status") == "succeeded"
            baseline = baseline_by_id.get(row.get("example_id"))
            baseline_f1 = None
            if baseline:
                baseline_f1 = float((baseline.get("metrics") or {}).get("f1", 0.0))
            em_values.append(em)
            f1_values.append(f1)
            vlm_values.append(1.0 if used_vlm else 0.0)
            harm_values.append(1.0 if baseline_f1 is not None and f1 < baseline_f1 else 0.0)
        metrics = {
            "EM": mean(em_values),
            "F1": mean(f1_values),
            "vlm_rate": mean(vlm_values),
            "harm_rate": mean(harm_values),
        }
    return {key: round(float(metrics[key]), 4) for key in REQUIRED_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FAAR result JSON and emit the four required AAAI metrics.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    metrics = evaluate_results(args.results, baseline_path=args.baseline)
    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
