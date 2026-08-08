import json
from pathlib import Path

import pytest

from faar.final_analysis import _harm_rate, bootstrap_ci, pareto_frontier, summarize_analysis


def _run_spec(**overrides) -> dict:
    spec = {
        "dataset": "ohrbench",
        "split": "test",
        "seed": 42,
        "max_examples": None,
        "shard_index": None,
        "num_shards": None,
        "embedding_model": "nvidia/NV-Embed-v2",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "ocr_engine": "got-ocr-2",
        "model_provenance": {
            "embedding": {"repository": "nvidia/NV-Embed-v2", "revision": None},
            "reranker": {"repository": "BAAI/bge-reranker-v2-m3", "revision": None},
        },
    }
    spec.update(overrides)
    return spec


def _result(
    path: Path,
    label: str,
    f1: float,
    cost: float,
    rows: list[dict],
    *,
    profile: str = "faar_full",
    run_spec: dict | None = None,
) -> None:
    payload = {
        "label": label,
        "summary": {"EM": f1, "F1": f1, "vlm_rate": 0.2, "harm_rate": 0.0, "cost_usd": cost},
        "rows": rows,
    }
    if profile is not None:
        payload["profile"] = profile
    if run_spec is not None:
        payload["run_spec"] = run_spec
    path.write_text(json.dumps(payload))


def _row(example_id: str, f1: float, failure_type: str = "semantic") -> dict:
    return {
        "example_id": example_id,
        "failure_type": failure_type,
        "action_outcome": {"action": "invoke_vlm"},
        "metrics": {"em": f1, "f1": f1},
    }


def test_analysis_uses_rows_for_bootstrap_harm_and_breakdown(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0), _row("e2", 0.5)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0), _row("e2", 1.0, "structural")], run_spec=spec)
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


def test_analysis_rejects_baseline_that_is_not_naive_rag(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="faar_full", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=spec)
    with pytest.raises(ValueError, match="naive_rag"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_missing_baseline_run_spec(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=None)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=_run_spec())
    with pytest.raises(ValueError, match="run_spec"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_missing_result_run_spec(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=_run_spec())
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=None)
    with pytest.raises(ValueError, match="run_spec"):
        summarize_analysis([faar], baseline)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dataset", "paperQA"),
        ("split", "val"),
        ("seed", 7),
        ("max_examples", 50),
        ("shard_index", 1),
        ("num_shards", 2),
        ("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        ("reranker", "BAAI/bge-reranker-base"),
        ("ocr_engine", "tesseract"),
        ("model_provenance", {"embedding": {"repository": "other", "revision": None}}),
    ],
)
def test_analysis_rejects_run_spec_key_mismatch(tmp_path: Path, key: str, value: object) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=_run_spec(**{key: value}))
    with pytest.raises(ValueError, match=f"run_spec.{key}"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_run_spec_missing_match_key(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    partial = dict(spec)
    del partial["shard_index"]
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=partial)
    with pytest.raises(ValueError, match="shard_index"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_duplicate_result_example_ids(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0), _row("e1", 0.3)], run_spec=spec)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_duplicate_baseline_example_ids(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0), _row("e1", 0.8)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=spec)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_missing_example_id(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0), _row("", 0.3)], run_spec=spec)
    with pytest.raises(ValueError, match="missing example_id"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_row_not_covered_by_baseline(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0), _row("e9", 0.5)], run_spec=spec)
    with pytest.raises(ValueError, match="not covered"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_result_that_omits_baseline_row(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(
        baseline,
        "B0",
        0.5,
        0.0,
        [_row("e1", 1.0), _row("e2", 0.5)],
        profile="naive_rag",
        run_spec=spec,
    )
    _result(faar, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=spec)
    with pytest.raises(ValueError, match="omits baseline B0 rows"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_empty_result_rows(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", 0.0, 0.0, [], run_spec=spec)
    with pytest.raises(ValueError, match="no result rows"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_duplicate_result_labels(tmp_path: Path) -> None:
    baseline = tmp_path / "b0.json"
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    spec = _run_spec()
    _result(baseline, "B0", 0.5, 0.0, [_row("e1", 1.0)], profile="naive_rag", run_spec=spec)
    _result(first, "FAAR", 0.5, 0.2, [_row("e1", 0.0)], run_spec=spec)
    _result(second, "FAAR", 0.6, 0.3, [_row("e1", 1.0)], run_spec=spec)
    with pytest.raises(ValueError, match="duplicate result label"):
        summarize_analysis([first, second], baseline)


def test_load_result_rejects_malformed_rows(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"summary": {}, "rows": {"e1": _row("e1", 1.0)}}))
    with pytest.raises(ValueError, match="complete FAAR result"):
        summarize_analysis([malformed], malformed)


def test_harm_rate_rejects_uncovered_row_directly() -> None:
    baseline_rows = [_row("e1", 1.0)]
    with pytest.raises(ValueError, match="not covered"):
        _harm_rate([_row("e1", 0.0), _row("e9", 0.5)], baseline_rows)
