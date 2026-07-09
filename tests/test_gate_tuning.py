import json
from pathlib import Path

import pytest

from faar.gate_tuning import GateExample, load_gate_examples, require_validation_payload, search_threshold


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
