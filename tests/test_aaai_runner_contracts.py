"""Contracts for the AAAI-facing experiment runner.

These tests intentionally describe the required runner behavior without loading
models or calling external services.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import run
from faar.settings import AppSettings


def _row(example_id: str, f1: float) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "predicted_answer": "prediction",
        "gold_answer": "gold",
        "metrics": {"em": 0.0, "f1": f1},
        "action_outcome": {"action": "answer_direct"},
    }


def _result(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "summary": {
            "EM": 0.0,
            "F1": 0.0,
            "vlm_rate": 0.0,
            "harm_rate": 0.0,
            "cost_usd": 0.0,
            "runtime_sec": 0.0,
        },
        "rows": rows,
    }


def _isolate_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    monkeypatch.setattr(run, "_settings_from_args", lambda args: settings)
    monkeypatch.setattr(AppSettings, "validate_runtime_paths", lambda self: None)
    monkeypatch.setattr(run, "_require_key_for_paid_vlm", lambda backend: None)
    monkeypatch.setattr(run, "load_benchmark_repository", lambda *args, **kwargs: object())


def test_b1_maps_to_a_dedicated_always_vlm_profile() -> None:
    baseline_id, profile, label = run.BASELINE_MAP[("off", "always_vlm")]

    assert baseline_id == "B1"
    assert profile == "faar_always_vlm"
    assert label == "Always-VLM"


@pytest.mark.parametrize("mode", ["colpali", "visrag"])
def test_visual_modes_dispatch_to_visual_baseline_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_visual_baseline(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, "kwargs": kwargs})
        return _result(mode, [])

    monkeypatch.setattr(run, "run_visual_baseline", fake_visual_baseline, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--mode", mode, "--out", str(tmp_path / f"{mode}.json")],
    )

    run.main()

    assert len(calls) == 1
    call = calls[0]
    assert mode in call["args"] or call["kwargs"].get("mode") == mode


@pytest.mark.parametrize(
    "mode_args, expected_profile",
    [
        (["--mode", "faar"], "faar_full"),
        (["--ablate", "no_gate"], "faar_no_gate"),
        (["--ablate", "no_diagnosis"], "faar_no_diagnosis"),
        (["--ablate", "no_wordlevel_llm"], "faar_symspell"),
        (["--ablate", "no_semantic_retry"], "faar_no_backtrack"),
    ],
)
def test_faar_and_ablations_compute_harm_against_supplied_matching_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode_args: list[str],
    expected_profile: str,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    baseline_path = tmp_path / "matching-baseline.json"
    baseline_path.write_text(json.dumps(_result("baseline", [_row("ex1", 0.75)])))
    out_path = tmp_path / f"{expected_profile}.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert profile == expected_profile
        payload = _result(profile, [_row("ex1", 0.25)])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            *mode_args,
            "--baseline",
            str(baseline_path),
            "--out",
            str(out_path),
        ],
    )

    run.main()

    payload = json.loads(out_path.read_text())
    assert payload["summary"]["harm_rate"] == 1.0


def test_saved_run_provenance_records_openai_cost_rates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_cli(monkeypatch, tmp_path)
    output_path = tmp_path / "b0.json"

    def fake_profile_run(
        settings: AppSettings,
        profile: str,
        out: Path,
        label: str,
        max_examples: int | None,
        run_spec: dict[str, Any],
        dataset: str,
        split: str,
    ) -> dict[str, Any]:
        payload = _result(profile, [])
        payload["run_spec"] = run_spec
        out.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run, "_run_profile_to_result", fake_profile_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--gate", "off", "--recovery", "off", "--out", str(output_path)],
    )

    run.main()

    saved = json.loads(output_path.read_text())
    rates = saved["run_spec"]["vlm_cost_rates"]
    assert rates["input_usd_per_million_tokens"] == 2.5
    assert rates["output_usd_per_million_tokens"] == 10.0
