import json
from pathlib import Path

from faar.final_analysis import bootstrap_ci, pareto_frontier, summarize_analysis


def _result(path: Path, label: str, f1: float, cost: float, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "label": label,
                "summary": {"EM": f1, "F1": f1, "vlm_rate": 0.2, "harm_rate": 0.0, "cost_usd": cost},
                "rows": rows,
            }
        )
    )


def _row(example_id: str, f1: float, failure_type: str = "semantic") -> dict:
    return {
        "example_id": example_id,
        "failure_type": failure_type,
        "action_outcome": {"action": "invoke_vlm"},
        "metrics": {"em": f1, "f1": f1},
    }


def test_analysis_uses_rows_for_bootstrap_harm_and_breakdown(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0), _row("e2", 0.5)])
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0), _row("e2", 1.0, "structural")])
    analysis = summarize_analysis([faar], baseline)
    result = analysis["results"]["FAAR"]
    assert result["harm_rate"] == 0.5
    assert result["failure_type_breakdown"]["semantic"]["count"] == 1
    assert result["bootstrap_95_ci"]["F1"]["mean"] == 0.5


def test_bootstrap_ci_is_seeded_and_pareto_discards_dominated_points() -> None:
    assert bootstrap_ci([0.0, 1.0], samples=100, seed=42) == bootstrap_ci([0.0, 1.0], samples=100, seed=42)
    frontier = pareto_frontier(
        [
            {"label": "cheap", "cost_usd": 0.0, "F1": 0.4},
            {"label": "better", "cost_usd": 0.1, "F1": 0.8},
            {"label": "dominated", "cost_usd": 0.2, "F1": 0.7},
        ]
    )
    assert [point["label"] for point in frontier] == ["cheap", "better"]
