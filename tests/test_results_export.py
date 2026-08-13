import csv
import json

from faar.results_export import write_json, write_metrics_csv


def test_write_json_creates_parent_dirs(tmp_path) -> None:
    target = tmp_path / "deep/nested/dir/out.json"
    write_json(target, {"a": 1, "b": [2, 3]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}


def test_write_metrics_csv_header_and_values(tmp_path) -> None:
    target = tmp_path / "qa_metrics.csv"
    rows = [
        {"profile": "faar_full", "count": 3, "em": 0.5, "f1": 0.25, "ndcg@5": 0.4, "recall@5": 0.6, "visual_fallback_rate": 0.1},
        {"profile": "naive_rag", "count": 3, "em": 0.5, "f1": 0.25, "ndcg@5": 0.4, "recall@5": 0.6, "visual_fallback_rate": 0.0},
    ]
    write_metrics_csv(target, rows)
    with target.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["profile"] == "faar_full"
    assert parsed[0]["em"] == "0.5"
    assert parsed[1]["visual_fallback_rate"] == "0.0"


def test_write_metrics_csv_tolerates_missing_columns(tmp_path) -> None:
    target = tmp_path / "sparse.csv"
    write_metrics_csv(target, [{"profile": "faar_full", "count": 1}])
    with target.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["em"] == ""
    assert parsed[0]["visual_fallback_rate"] == ""


def test_write_metrics_csv_round_trips_unicode(tmp_path) -> None:
    target = tmp_path / "unicode.csv"
    rows = [{"profile": "naïve_rag", "count": 1, "em": 0.0, "f1": 0.0, "ndcg@5": 0.0, "recall@5": 0.0, "visual_fallback_rate": 0.0}]
    write_metrics_csv(target, rows)
    with target.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["profile"] == "naïve_rag"
