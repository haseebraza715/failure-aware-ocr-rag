from pathlib import Path

import pytest

from faar.graph import build_graph
from faar.settings import AppSettings


def _prepare_phase0(tmp_path: Path, ocr_text: str) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "logs/phase1").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        "ex1,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text(ocr_text)


def test_graph_rejects_empty_ocr_artifact(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, "")
    settings = AppSettings(project_root=tmp_path)
    graph = build_graph(settings)
    with pytest.raises(ValueError, match="empty"):
        graph.invoke({"example_id": "ex1"})


def test_graph_rejects_whitespace_only_ocr_artifact(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, "   \n\t  \n  ")
    settings = AppSettings(project_root=tmp_path)
    graph = build_graph(settings)
    with pytest.raises(ValueError, match="empty"):
        graph.invoke({"example_id": "ex1"})


def test_graph_rejects_page_marker_without_words_with_clear_message(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path, "===== PAGE 0 =====\n\n\n===== PAGE 1 =====\n")
    settings = AppSettings(project_root=tmp_path)
    graph = build_graph(settings)
    with pytest.raises(ValueError, match="No retrieval chunks could be built for example 'ex1'"):
        graph.invoke({"example_id": "ex1"})
