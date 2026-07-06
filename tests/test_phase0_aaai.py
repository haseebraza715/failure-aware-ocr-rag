import json
from pathlib import Path

from evaluate import evaluate_results
from split_dataset import build_split


def test_split_dataset_is_deterministic(tmp_path: Path) -> None:
    data_path = tmp_path / "qas.json"
    data_path.write_text(json.dumps([{"ID": f"ex{i}"} for i in range(20)]))
    first = build_split(data_path, seed=42)
    second = build_split(data_path, seed=42)
    assert first["splits"] == second["splits"]
    assert first["counts"] == {"train": 14, "val": 3, "test": 3}


def test_evaluate_emits_required_metrics(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "example_id": "ex1",
                        "predicted_answer": "certified engineers",
                        "gold_answer": "certified engineers",
                        "action_outcome": {"action": "answer_direct"},
                    }
                ]
            }
        )
    )
    metrics = evaluate_results(results)
    assert set(metrics) == {"EM", "F1", "vlm_rate", "harm_rate"}
    assert metrics["EM"] == 1.0
    assert metrics["F1"] == 1.0
