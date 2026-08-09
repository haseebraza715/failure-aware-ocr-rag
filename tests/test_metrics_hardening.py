from faar.metrics import exact_match, ndcg_at_k, recall_at_k, token_f1


def test_recall_at_k_zero_and_negative_k() -> None:
    hits = ["the answer is 42", "more text"]
    assert recall_at_k(hits, "42", k=0) == 0.0
    assert recall_at_k(hits, "42", k=-1) == 0.0
    assert recall_at_k(hits, "42", k=1) == 1.0


def test_recall_at_k_empty_gold() -> None:
    assert recall_at_k(["whatever"], "", k=5) == 0.0


def test_ndcg_at_k_zero_and_negative_k() -> None:
    hits = ["the answer is 42"]
    assert ndcg_at_k(hits, "42", k=0) == 0.0
    assert ndcg_at_k(hits, "42", k=-2) == 0.0


def test_ndcg_at_k_empty_gold_and_empty_hits() -> None:
    assert ndcg_at_k([], "42", k=5) == 0.0
    assert ndcg_at_k(["anything"], "", k=5) == 0.0


def test_ndcg_at_k_prefers_relevant_first() -> None:
    relevant_first = ndcg_at_k(["answer is 42", "irrelevant"], "42", k=2)
    relevant_second = ndcg_at_k(["irrelevant", "answer is 42"], "42", k=2)
    assert relevant_first > relevant_second


def test_token_f1_edge_cases() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("", "gold") == 0.0
    assert token_f1("pred", "") == 0.0
    assert token_f1("exact match", "exact match") == 1.0
    assert token_f1("unrelated", "completely different") == 0.0


def test_exact_match_edge_cases() -> None:
    assert exact_match("", "") == 1.0
    assert exact_match("  Hello,   WORLD! ", "hello world") == 1.0
    assert exact_match("hello", "world") == 0.0
