from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
SPEC = spec_from_file_location("cluster_merge_shards", CLUSTER / "merge_shards.py")
assert SPEC and SPEC.loader
merge_shards = module_from_spec(SPEC)
sys.modules["cluster_merge_shards"] = merge_shards
SPEC.loader.exec_module(merge_shards)


def _row(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "question": f"Q {example_id}",
        "gold_answer": "gold",
        "predicted_answer": "gold",
        "failure_type": "pass",
        "policy_action": "answer_direct",
        "action_outcome": {"action": "answer_direct", "status": "succeeded"},
        "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
        "api_usage": {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        "run_metadata": {"run_id": "r1"},
    }


def _shard(path: Path, rows: list[dict], *, shard_index: int = 0, num_shards: int = 2) -> dict:
    payload = {
        "label": "B2 ohrbench test",
        "profile": "faar_no_diagnosis",
        "run_spec": {
            "profile": "faar_no_diagnosis",
            "dataset": "ohrbench",
            "split": "test",
            "seed": 42,
            "max_examples": None,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "embedding_model": "nvidia/NV-Embed-v2",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "ocr_engine": "got-ocr-2",
            "model_provenance": {"embedding": {"repository": "nvidia/NV-Embed-v2"}},
            "manifest_sha256": "a" * 64,
        },
        "summary": {
            "EM": 1.0,
            "F1": 1.0,
            "vlm_rate": 0.0,
            "harm_rate": 0.0,
            "api_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "runtime_sec": 10.0,
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload))
    return payload


def test_merge_combines_disjoint_shards_in_order(tmp_path: Path) -> None:
    first_path = tmp_path / "b2_shard1of2.json"
    second_path = tmp_path / "b2_shard2of2.json"
    _shard(first_path, [_row("e1")], shard_index=0)
    _shard(second_path, [_row("e2")], shard_index=1)
    merged = merge_shards.merge_shards([first_path, second_path])
    assert [row["example_id"] for row in merged["rows"]] == ["e1", "e2"]
    assert merged["run_spec"]["shard_index"] is None
    assert merged["run_spec"]["num_shards"] is None
    assert merged["summary"]["EM"] == 1.0
    assert merged["summary"]["api_requests"] == 0


def test_merge_rejects_duplicate_example_ids(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _shard(first, [_row("e1")], shard_index=0)
    _shard(second, [_row("e1")], shard_index=1)
    with pytest.raises(SystemExit, match="duplicate example_id"):
        merge_shards.merge_shards([first, second])


def test_merge_rejects_mismatched_run_specs(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _shard(first, [_row("e1")], shard_index=0)
    payload = _shard(second, [_row("e2")], shard_index=1)
    payload["run_spec"]["seed"] = 7
    second.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="does not match"):
        merge_shards.merge_shards([first, second])


def test_merge_rejects_inconsistent_num_shards(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _shard(first, [_row("e1")], shard_index=0, num_shards=2)
    _shard(second, [_row("e2")], shard_index=1, num_shards=4)
    with pytest.raises(SystemExit, match="num_shards"):
        merge_shards.merge_shards([first, second])


def test_merge_cli_computes_harm_against_baseline(tmp_path: Path) -> None:
    first = tmp_path / "b2_shard1of2.json"
    second = tmp_path / "b2_shard2of2.json"
    baseline = tmp_path / "b0.json"
    _shard(first, [_row("e1")], shard_index=0)
    _shard(second, [_row("e2")], shard_index=1)
    _shard(
        baseline,
        [
            dict(_row("e1"), metrics={"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0}),
            dict(_row("e2"), metrics={"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0}),
        ],
        shard_index=0,
    )
    out = tmp_path / "b2_merged.json"
    assert merge_shards.main(["--out", str(out), "--baseline", str(baseline), str(first), str(second)]) == 0
    payload = json.loads(out.read_text())
    assert payload["summary"]["harm_rate"] == 0.0
    assert payload["rows"][0]["example_id"] == "e1"
