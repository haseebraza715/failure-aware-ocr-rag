import json
from pathlib import Path

from faar.phase4_analysis import (
    baseline_comparison,
    build_case_studies,
    load_phase3_rows,
    summarize_profile,
)


def _row(example_id: str, profile: str) -> dict:
    return {
        "example_id": example_id,
        "profile": profile,
        "question": f"Q {example_id}",
        "gold_answer": "gold",
        "predicted_answer": "pred",
        "failure_type": "pass",
        "policy_action": "answer_direct",
        "action_outcome": {"action": "answer_direct", "status": "succeeded"},
        "metrics": {"ndcg@5": 0.5, "recall@5": 0.5, "em": 0.5, "f1": 0.5},
    }


def test_load_phase3_rows_skips_non_dict_json_payloads(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs/phase3"
    (logs_root / "faar_full").mkdir(parents=True)
    (logs_root / "faar_full/good.json").write_text(json.dumps(_row("ex1", "faar_full")))
    (logs_root / "faar_full/list.json").write_text("[1, 2, 3]")
    (logs_root / "faar_full/string.json").write_text('"just a string"')
    rows = load_phase3_rows(logs_root, profiles=["faar_full"])
    assert len(rows["faar_full"]) == 1


def test_baseline_comparison_missing_baseline_returns_empty() -> None:
    assert baseline_comparison({"faar_full": {"count": 1, "em": 0.5}}, baseline_profile="naive_rag") == []


def test_baseline_comparison_reports_deltas() -> None:
    rows = baseline_comparison(
        {
            "naive_rag": {"count": 2, "em": 0.4, "f1": 0.3, "ndcg@5": 0.2, "recall@5": 0.1, "visual_fallback_rate": 0.0},
            "faar_full": {"count": 2, "em": 0.6, "f1": 0.5, "ndcg@5": 0.4, "recall@5": 0.3, "visual_fallback_rate": 0.5},
        }
    )
    by_profile = {row["profile"]: row for row in rows}
    assert by_profile["faar_full"]["delta_em_vs_baseline"] == 0.2
    assert by_profile["faar_full"]["delta_f1_vs_baseline"] == 0.2


def test_summarize_profile_empty_rows() -> None:
    summary = summarize_profile([])
    assert summary["count"] == 0
    assert summary["non_pass_rate"] == 0.0
    assert summary["action_distribution"] == {}
    assert summary["failure_distribution"] == {}


def test_build_case_studies_without_shared_ids_is_empty() -> None:
    rows_by_profile = {
        "faar_full": [_row("only-focus", "faar_full")],
        "naive_rag": [_row("only-baseline", "naive_rag")],
    }
    case_studies = build_case_studies(rows_by_profile)
    assert case_studies["improvements"] == []
    assert case_studies["regressions"] == []
    assert case_studies["ties"] == []


def test_load_phase3_rows_skips_missing_profile_dirs(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs/phase3"
    rows = load_phase3_rows(logs_root, profiles=["never_existed"])
    assert rows["never_existed"] == []
