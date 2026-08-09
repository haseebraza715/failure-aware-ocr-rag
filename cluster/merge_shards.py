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

The full shard set must be supplied: every index in range(num_shards)
must be declared exactly once, and the merged example ids must match the
baseline when one is given. A partial set is rejected before any shard
metadata is cleared or the merged file is published.
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


def _validate_shard_declaration(path: Path, spec: dict[str, Any], reference_spec: dict[str, Any]) -> int:
    """Validate one shard's index/count declaration; return its shard_index."""
    for key in ("shard_index", "num_shards"):
        if key not in spec:
            raise SystemExit(f"shard {path} run_spec is missing {key}; not a sharded output.")
    index = spec["shard_index"]
    num_shards = spec["num_shards"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise SystemExit(
            f"shard {path} run_spec.shard_index must be a non-boolean integer; received {index!r}."
        )
    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise SystemExit(
            f"shard {path} run_spec.num_shards must be a non-boolean integer; received {num_shards!r}."
        )
    if num_shards < 1:
        raise SystemExit(
            f"shard {path} declares num_shards={num_shards}; expected a positive integer."
        )
    if num_shards != reference_spec["num_shards"]:
        raise SystemExit(
            f"shard {path} declares num_shards={num_shards}; expected {reference_spec['num_shards']}."
        )
    if not 0 <= index < num_shards:
        raise SystemExit(
            f"shard {path} declares shard_index={index}; expected an index in [0, {num_shards})."
        )
    return index


def _validate_shard_set(
    shard_paths: list[Path], payloads: list[dict[str, Any]], num_shards: int
) -> None:
    """Require exactly one shard per index in range(num_shards), no duplicates."""
    declared: dict[int, Path] = {}
    duplicates: list[str] = []
    for path, payload in zip(shard_paths, payloads):
        index = int(payload["run_spec"]["shard_index"])
        if index in declared:
            duplicates.append(f"shard index {index} declared by {declared[index]} and {path}")
        else:
            declared[index] = path
    missing = [index for index in range(num_shards) if index not in declared]
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing shard indices {missing}")
        details.extend(duplicates)
        raise SystemExit(
            f"incomplete shard set for num_shards={num_shards}: " + "; ".join(details)
        )


def _validate_expected_ids(
    merged_rows: list[dict[str, Any]], expected_example_ids: list[str]
) -> None:
    merged_ids = [str(row["example_id"]) for row in merged_rows]
    expected_set = set(expected_example_ids)
    merged_set = set(merged_ids)
    missing = [item for item in expected_example_ids if item not in merged_set]
    extra = [item for item in merged_ids if item not in expected_set]
    if missing or extra:
        raise SystemExit(
            f"merged rows do not match the expected example ids from the baseline "
            f"(missing {len(missing)}, extra {len(extra)})."
        )


def merge_shards(
    shard_paths: list[Path],
    *,
    expected_example_ids: list[str] | None = None,
) -> dict[str, Any]:
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
        _validate_shard_declaration(path, spec, reference_spec)
    # Completeness must succeed before any shard metadata is cleared or the
    # merged file is published: a partial shard set is never a valid unsharded run.
    _validate_shard_set(shard_paths, payloads, reference_spec["num_shards"])
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
    if expected_example_ids is not None:
        _validate_expected_ids(rows, expected_example_ids)
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
    parser.add_argument("--baseline", type=Path, default=None, help="Matching B0 result used to compute harm_rate and validate merged example ids")
    parser.add_argument("shards", nargs="+", type=Path, help="Shard result JSON files")
    args = parser.parse_args(argv)
    expected_example_ids = None
    baseline_payload = None
    if args.baseline is not None:
        if not args.baseline.is_file():
            raise SystemExit(f"baseline result does not exist: {args.baseline}")
        baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_rows = list(baseline_payload.get("rows", []))
        expected_example_ids = [str(row.get("example_id", "")) for row in baseline_rows]
    payload = merge_shards(args.shards, expected_example_ids=expected_example_ids)
    if baseline_payload is not None:
        try:
            payload["summary"]["harm_rate"] = _harm_rate(payload["rows"], baseline_rows)
        except ValueError as exc:
            raise SystemExit(f"cannot compute harm_rate against baseline {args.baseline}: {exc}") from exc
    atomic_write_text(args.out, json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
