import pytest

from faar.api_logging import is_valid_api_usage
from faar.results_aggregator import summarize_api_usage, summarize_by_profile, summarize_examples


def test_summarize_examples_empty() -> None:
    out = summarize_examples([])
    assert out["count"] == 0
    assert out["em"] == 0.0
    assert out["vlm_rate"] == 0.0


def test_summarize_by_profile() -> None:
    rows = [
        {
            "profile": "faar_full",
            "policy_action": "answer_direct",
            "action_outcome": {"action": "answer_direct"},
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
        },
        {
            "profile": "faar_full",
            "policy_action": "invoke_vlm",
            "action_outcome": {"action": "answer_direct"},
            "metrics": {"ndcg@5": 0.0, "recall@5": 0.0, "em": 0.0, "f1": 0.0},
        },
        {
            "profile": "faar_full",
            "policy_action": "invoke_vlm",
            "action_outcome": {"action": "invoke_vlm"},
            "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 0.0, "f1": 0.0},
        },
    ]
    summary = summarize_by_profile(rows)
    assert "faar_full" in summary
    assert summary["faar_full"]["count"] == 3
    assert summary["faar_full"]["visual_fallback_rate"] == 0.3333


def test_summarize_api_usage_totals_all_rows() -> None:
    rows = [
        {"api_usage": {"api_requests": 1, "prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.00045}},
        {"api_usage": {"api_requests": 2, "prompt_tokens": 50, "completion_tokens": 10, "cost_usd": 0.0002}},
        {"api_usage": {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}},
    ]
    totals = summarize_api_usage(rows)
    assert totals == {
        "api_requests": 3,
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "cost_usd": 0.00065,
    }


@pytest.mark.parametrize("usage", [None, {}, {"api_requests": 1.5, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}])
def test_summarize_api_usage_rejects_invalid_rows(usage) -> None:
    with pytest.raises(ValueError, match="invalid api_usage"):
        summarize_api_usage([{"api_usage": usage}])


def test_api_usage_requires_integral_counts() -> None:
    assert not is_valid_api_usage(
        {"api_requests": 1.5, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    )
