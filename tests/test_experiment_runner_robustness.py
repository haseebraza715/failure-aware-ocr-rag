import json
from pathlib import Path

from faar.experiment_runner import run_profile
from faar.settings import AppSettings


class FlakyGraph:
    def __init__(self) -> None:
        self.questions = {}

    def invoke(self, state):
        example_id = state["example_id"]
        if example_id == "ex_bad":
            raise FileNotFoundError(f"OCR text artifact missing for example {example_id}")
        return {
            "question": f"Q for {example_id}",
            "example": type("Ex", (), {"correct_answer": "certified engineers"})(),
            "answer": "certified engineers",
            "failure_type": "semantic",
            "policy_action": "retry_retrieval",
            "action_outcome": {"action": "retry_retrieval", "status": "succeeded"},
            "retrieved_hits": [],
            "corrected_hits": [],
        }


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
        "ex2,manual/doc,What is required?,certified engineers,0\n"
        "ex3,manual/doc,What is required?,certified engineers,0\n"
    )
    for example_id in ("ex1", "ex2", "ex3"):
        (tmp_path / "artifacts/phase0/ocr_text" / f"{example_id}.txt").write_text(
            "===== PAGE 0 =====\ncertified engineers"
        )


def test_run_profile_continues_past_failing_example(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FlakyGraph())
    settings = AppSettings(project_root=tmp_path)
    rows = run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex_bad", "ex2"])
    assert len(rows) == 3
    good = [row for row in rows if row["example_id"] != "ex_bad"]
    bad = next(row for row in rows if row["example_id"] == "ex_bad")
    assert len(good) == 2
    assert all(row["metrics"]["em"] == 1.0 for row in good)
    assert bad["failure_type"] == "error"
    assert bad["action_outcome"]["action"] == "failed"
    assert "missing for example ex_bad" in bad["action_outcome"]["reason"]
    assert bad["metrics"]["em"] == 0.0


def test_run_profile_writes_json_for_failed_examples_too(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FlakyGraph())
    settings = AppSettings(project_root=tmp_path)
    run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex_bad"])
    out_dir = tmp_path / "logs/phase3" / "faar_full"
    assert (out_dir / "ex1.json").exists()
    assert (out_dir / "ex_bad.json").exists()
    payload = json.loads((out_dir / "ex_bad.json").read_text(encoding="utf-8"))
    assert payload["action_outcome"]["action"] == "failed"


def test_run_profile_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.experiment_runner.build_graph", lambda settings: FlakyGraph())
    settings = AppSettings(project_root=tmp_path)

    def _strip_run_id(rows):
        for row in rows:
            row["run_metadata"]["run_id"] = "<run_id>"
        return rows

    first = _strip_run_id(run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"]))
    second = _strip_run_id(run_profile(settings, profile_name="faar_full", example_ids=["ex1", "ex2"]))
    assert first == second
    run_ids = {row["run_metadata"]["run_id"] for row in run_profile(
        settings, profile_name="faar_full", example_ids=["ex1", "ex2"]
    )}
    assert len(run_ids) == 1
