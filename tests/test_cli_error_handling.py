from pathlib import Path

import pytest
import typer

from faar.cli import run_example


class ExplodingGraph:
    def invoke(self, state):
        raise ValueError("No retrieval chunks could be built for example 'ex1': the OCR text contains no words")


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text("example_id,doc_name,question,correct_answer,page_no\n")
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text('{"results":[]}')


def test_run_example_reports_pipeline_failure_clearly(monkeypatch, tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    monkeypatch.setattr("faar.cli.build_graph", lambda settings: ExplodingGraph())
    with pytest.raises(typer.BadParameter, match="Pipeline failed for example 'ex1'"):
        run_example(
            example_id="ex1",
            question=None,
            output=tmp_path / "logs/phase1/out.json",
            project_root=tmp_path,
            vlm_backend="mock",
            seed=42,
        )
