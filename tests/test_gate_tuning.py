import json
from pathlib import Path

import pytest

from faar.gate_tuning import (
    GateExample,
    load_gate_examples,
    require_paper_gate_threshold,
    require_validation_payload,
    search_threshold,
    write_locked_threshold,
)


def test_searches_exact_required_grid_and_selects_best_f1() -> None:
    examples = [
        GateExample("bad-1", 0.2, True),
        GateExample("bad-2", 0.35, True),
        GateExample("good-1", 0.8, False),
        GateExample("good-2", 0.9, False),
    ]
    result = search_threshold(examples)
    assert [item["theta"] for item in result["grid"]] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert result["winner"]["theta"] == 0.4
    assert result["pass_criteria"]["passed"] is True


def test_requires_validation_provenance(tmp_path: Path) -> None:
    result = tmp_path / "b0.json"
    result.write_text(json.dumps({"run_spec": {"split": "test"}, "rows": []}))
    with pytest.raises(ValueError, match="test results are forbidden"):
        require_validation_payload(result)


def test_loads_gate_score_and_exact_match_target(tmp_path: Path) -> None:
    result = tmp_path / "b0_val.json"
    result.write_text(
        json.dumps(
            {
                "run_spec": {"split": "val"},
                "rows": [
                    {
                        "example_id": "e1",
                        "top_reranker_score": 0.4,
                        "metrics": {"em": 0.0},
                    }
                ],
            }
        )
    )
    assert load_gate_examples(result) == [GateExample("e1", 0.4, True)]


def _locked(payload: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config/gate_threshold.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_require_paper_gate_threshold_accepts_passing_val_locked_threshold(tmp_path: Path) -> None:
    path = _locked(
        {"source_split": "val", "threshold": 0.4, "precision": 0.8, "recall": 0.75},
        tmp_path,
    )
    assert require_paper_gate_threshold(path)["threshold"] == 0.4


def test_require_paper_gate_threshold_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked gate threshold"):
        require_paper_gate_threshold(tmp_path / "config/gate_threshold.json")


def test_require_paper_gate_threshold_rejects_non_validation_source(tmp_path: Path) -> None:
    path = _locked(
        {"source_split": "test", "threshold": 0.4, "precision": 0.8, "recall": 0.75},
        tmp_path,
    )
    with pytest.raises(ValueError, match="source_split='val'"):
        require_paper_gate_threshold(path)


def test_require_paper_gate_threshold_rejects_sub_bar_precision(tmp_path: Path) -> None:
    path = _locked(
        {"source_split": "val", "threshold": 0.4, "precision": 0.7, "recall": 0.75},
        tmp_path,
    )
    with pytest.raises(ValueError, match="precision=0.7"):
        require_paper_gate_threshold(path)


def test_require_paper_gate_threshold_rejects_sub_bar_recall(tmp_path: Path) -> None:
    path = _locked(
        {"source_split": "val", "threshold": 0.4, "precision": 0.8, "recall": 0.6},
        tmp_path,
    )
    with pytest.raises(ValueError, match="recall=0.6"):
        require_paper_gate_threshold(path)


def test_require_paper_gate_threshold_rejects_invalid_numeric_fields(tmp_path: Path) -> None:
    path = _locked(
        {"source_split": "val", "threshold": "nan", "precision": 0.8, "recall": 0.75},
        tmp_path,
    )
    with pytest.raises(ValueError, match="finite values"):
        require_paper_gate_threshold(path)


def test_write_locked_threshold_rejects_failed_search(tmp_path: Path) -> None:
    search = search_threshold(
        [GateExample("bad", 0.9, True), GateExample("good", 0.1, False)]
    )
    assert search["pass_criteria"]["passed"] is False
    path = tmp_path / "config/gate_threshold.json"
    with pytest.raises(ValueError, match="did not pass"):
        write_locked_threshold(path, search)
    assert not path.exists()
