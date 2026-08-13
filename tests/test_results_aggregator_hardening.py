from faar.results_aggregator import summarize_by_profile, summarize_examples


def test_summarize_examples_tolerates_missing_metrics() -> None:
    out = summarize_examples(
        [
            {"profile": "faar_full", "action_outcome": {"action": "answer_direct"}},
            {"profile": "faar_full", "metrics": None},
        ]
    )
    assert out["count"] == 2
    assert out["em"] == 0.0
    assert out["ndcg@5"] == 0.0
    assert out["f1"] == 0.0
    assert out["visual_fallback_rate"] == 0.0


def test_summarize_examples_tolerates_missing_action_outcome() -> None:
    out = summarize_examples(
        [
            {
                "profile": "faar_full",
                "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
            }
        ]
    )
    assert out["visual_fallback_rate"] == 0.0
    assert out["recovery_outcomes"]["reviewed_count"] == 0


def test_summarize_examples_coerces_non_numeric_metrics_to_zero() -> None:
    out = summarize_examples(
        [
            {
                "profile": "faar_full",
                "metrics": {"ndcg@5": "junk", "recall@5": None, "em": "1.0", "f1": 0.5},
            }
        ]
    )
    assert out["em"] == 1.0
    assert out["ndcg@5"] == 0.0
    assert out["f1"] == 0.5


def test_summarize_by_profile_buckets_rows_without_profile() -> None:
    summary = summarize_by_profile(
        [
            {"metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0}},
            {"metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0}},
        ]
    )
    assert summary["unknown"]["count"] == 2
    assert summary["unknown"]["em"] == 0.5


def test_summarize_by_profile_empty_input() -> None:
    assert summarize_by_profile([]) == {}


def test_summarize_examples_ignores_empty_metrics_values() -> None:
    out = summarize_examples(
        [
            {
                "profile": "faar_full",
                "action_outcome": {"action": "answer_direct"},
                "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0},
            }
        ]
    )
    assert out["count"] == 1
    assert out["recovery_outcomes"] == {
        "reviewed_count": 0,
        "changed_answer_count": 0,
        "changed_answer_rate": 0.0,
        "em_effect_distribution": {},
        "f1_effect_distribution": {},
    }
