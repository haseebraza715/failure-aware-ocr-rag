#!/usr/bin/env python3
"""Merge per-shard baseline outputs into one unsharded result file.

Multi-GPU usage: launch one run.py process per GPU with
--shard-index i --num-shards N, then merge the shard outputs:

    python cluster/merge_shards.py --out results/b2.json --baseline results/b0.json \
        results/b2_shard1of8.json ... results/b2_shard8of8.json

The merged file is validated against the baseline run_spec, gets an
unsharded run_spec (shard_index/num_shards reset to null), recomputed
summary metrics and harm_rate, and is written atomically so it can be
validated by cluster/run_baselines.py on later resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.final_analysis import _harm_rate
from faar.results_aggregator import summarize_api_usage, summarize_examples
from faar.run_io import atomic_write_text


RUN_SPEC_MATCH_KEYS = (
    "profile",
    "dataset",
    "split",
    "seed",
    "max_examples",
    "embedding_model",
    "reranker",
    "ocr_engine",
    "model_provenance",
    "manifest_sha256",
)


def _load_shard(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"shard is not readable JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise SystemExit(f"shard must be a result payload with a rows list: {path}")
    run_spec = payload.get("run_spec")
    if not isinstance(run_spec, dict):
        raise SystemExit(f"shard has no run_spec provenance: {path}")
    return payload


def merge_shards(shard_paths: list[Path]) -> dict[str, Any]:
    if not shard_paths:
        raise SystemExit("at least one shard file is required")
    payloads = [_load_shard(path) for path in shard_paths]
    reference_spec = payloads[0]["run_spec"]
    for index, (path, payload) in enumerate(zip(shard_paths, payloads)):
        spec = payload["run_spec"]
        for key in RUN_SPEC_MATCH_KEYS:
            if key not in spec:
                raise SystemExit(f"shard {path} run_spec is missing {key}.")
            if index > 0 and spec[key] != reference_spec[key]:
                raise SystemExit(
                    f"shard {path} run_spec.{key}={spec[key]!r} does not match "
                    f"{shard_paths[0]} run_spec.{key}={reference_spec[key]!r}."
                )
        for key in ("shard_index", "num_shards"):
            if key not in spec:
                raise SystemExit(f"shard {path} run_spec is missing {key}; not a sharded output.")
        if spec["num_shards"] != reference_spec["num_shards"]:
            raise SystemExit(f"shard {path} declares num_shards={spec['num_shards']}; expected {reference_spec['num_shards']}.")
    seen: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for path, payload in zip(shard_paths, payloads):
        for row in payload["rows"]:
            example_id = str(row.get("example_id", ""))
            if not example_id:
                raise SystemExit(f"shard {path} contains a row without example_id.")
            if example_id in seen:
                raise SystemExit(
                    f"duplicate example_id {example_id!r} in shards {seen[example_id]} and {path}."
                )
            seen[example_id] = path
        rows.extend(payload["rows"])
    if not rows:
        raise SystemExit("merged shards contain no rows")
    examples_summary = summarize_examples(rows)
    api = summarize_api_usage(rows)
    merged = {
        "label": payloads[0].get("label", "merged"),
        "profile": payloads[0].get("profile"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_spec": {**reference_spec, "shard_index": None, "num_shards": None},
        "summary": {
            "EM": examples_summary["em"],
            "F1": examples_summary["f1"],
            "vlm_rate": examples_summary["vlm_rate"],
            "harm_rate": None,
            "api_requests": api["api_requests"],
            "prompt_tokens": api["prompt_tokens"],
            "completion_tokens": api["completion_tokens"],
            "cost_usd": api["cost_usd"],
            "runtime_sec": round(
                sum(float((payload.get("summary") or {}).get("runtime_sec") or 0.0) for payload in payloads),
                4,
            ),
        },
        "rows": sorted(rows, key=lambda row: str(row.get("example_id", ""))),
        "merged_from": [str(path) for path in shard_paths],
    }
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Merged unsharded output path")
    parser.add_argument("--baseline", type=Path, default=None, help="Matching B0 result used to compute harm_rate")
    parser.add_argument("shards", nargs="+", type=Path, help="Shard result JSON files")
    args = parser.parse_args(argv)
    payload = merge_shards(args.shards)
    if args.baseline is not None:
        if not args.baseline.is_file():
            raise SystemExit(f"baseline result does not exist: {args.baseline}")
        baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_rows = list(baseline_payload.get("rows", []))
        try:
            payload["summary"]["harm_rate"] = _harm_rate(payload["rows"], baseline_rows)
        except ValueError as exc:
            raise SystemExit(f"cannot compute harm_rate against baseline {args.baseline}: {exc}") from exc
    atomic_write_text(args.out, json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
