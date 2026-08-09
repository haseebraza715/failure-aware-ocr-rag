import json
from pathlib import Path

import pytest

from faar.data import Phase0Repository, _parse_ocr_pages, _parse_pages
from faar.settings import AppSettings


def _prepare_phase0(tmp_path: Path) -> None:
    (tmp_path / "data/phase0").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/ocr_text").mkdir(parents=True)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text(
        "example_id,doc_name,question,correct_answer,page_no\n"
        'ex1,manual/doc,What is required?,certified engineers,"[0, 2]"\n'
        "ex2,manual/doc,What is required?,certified engineers,0\n"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").write_text(
        "===== PAGE 0 =====\nfirst page\n===== PAGE 2 =====\nsecond page"
    )
    (tmp_path / "artifacts/phase0/ocr_text/ex2.txt").write_text("===== PAGE 0 =====\nno page marker here")


def test_relative_summary_paths_resolve_against_project_root(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "example_id": "ex1",
                        "ocr_text_path": "artifacts/phase0/ocr_text/ex1.txt",
                        "gt_text_path": "artifacts/phase0/gt/ex1.txt",
                        "image_paths": ["artifacts/phase0/images/ex1.png", "artifacts/phase0/images/missing.png"],
                    }
                ]
            }
        )
    )
    (tmp_path / "artifacts/phase0/gt").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/gt/ex1.txt").write_text("ground truth")
    (tmp_path / "artifacts/phase0/images").mkdir(parents=True)
    (tmp_path / "artifacts/phase0/images/ex1.png").write_bytes(b"png")
    repo = Phase0Repository(AppSettings(project_root=tmp_path))
    example = repo.get_example("ex1")
    assert example.ocr_text_path == (tmp_path / "artifacts/phase0/ocr_text/ex1.txt").resolve()
    assert example.gt_text_path == (tmp_path / "artifacts/phase0/gt/ex1.txt").resolve()
    assert example.image_paths == [(tmp_path / "artifacts/phase0/images/ex1.png").resolve()]


def test_malformed_page_no_raises_clear_errors() -> None:
    with pytest.raises(ValueError, match=r"Malformed page_no.*expected a list of integers"):
        _parse_pages("[1, x]")
    with pytest.raises(ValueError, match=r"Malformed page_no.*expected an integer"):
        _parse_pages("twelve")
    with pytest.raises(ValueError, match="page_no is empty"):
        _parse_pages("   ")


def test_manifest_without_example_id_column_raises(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="example_id"):
        Phase0Repository(AppSettings(project_root=tmp_path))


def test_manifest_with_zero_rows_raises(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "data/phase0/sample_manifest.csv").write_text("example_id,doc_name\n")
    with pytest.raises(ValueError, match="example_id"):
        Phase0Repository(AppSettings(project_root=tmp_path))


def test_summary_with_non_dict_payload_raises(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        Phase0Repository(AppSettings(project_root=tmp_path))


def test_summary_with_non_list_results_raises(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "data/phase0/phase0_asset_summary.json").write_text('{"results": {"ex1": {}}}')
    with pytest.raises(ValueError, match="results"):
        Phase0Repository(AppSettings(project_root=tmp_path))


def test_multi_page_parsing_and_single_page_fallback() -> None:
    pages = _parse_ocr_pages("===== PAGE 0 =====\nfirst\n===== PAGE 2 =====\nsecond")
    assert pages == {0: "first", 2: "second"}
    single = _parse_ocr_pages("no markers at all")
    assert single == {0: "no markers at all"}


def test_manual_labels_are_merged_into_record(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    (tmp_path / "data/phase0/manual_labels.csv").write_text(
        "example_id,question,correct_answer,ocr_output_snippet,failure_type,notes\n"
        "ex2,What is required?,certified engineers,snippet,text_corruption,noisy\n"
    )
    repo = Phase0Repository(AppSettings(project_root=tmp_path))
    record = repo.get_example_record("ex2")
    assert record["manual_failure_type"] == "text_corruption"
    assert record["manual_failure_notes"] == "noisy"


def test_page_no_parsing_accepts_list_and_scalar(tmp_path: Path) -> None:
    _prepare_phase0(tmp_path)
    repo = Phase0Repository(AppSettings(project_root=tmp_path))
    assert repo.get_example("ex1").page_ids == [0, 2]
    assert repo.get_example("ex2").page_ids == [0]
