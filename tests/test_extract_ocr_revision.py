from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import extract_ocr
from faar.settings import AppSettings


@pytest.mark.parametrize(
    ("revision", "message"),
    [
        ("", "must not be empty"),
        ("main", "not branch"),
        ("latest", "not branch"),
        ("abc123", "40-character hexadecimal"),
        ("g" * 40, "40-character hexadecimal"),
        ("a" * 40, "does not match config/model_revisions.json"),
    ],
)
def test_rejects_unlocked_got_ocr_revisions(
    tmp_path: Path,
    revision: str,
    message: str,
) -> None:
    settings = AppSettings(project_root=tmp_path)

    with pytest.raises(ValueError, match=message):
        extract_ocr.validate_locked_got_ocr(
            settings.recovery.got_ocr_model,
            revision,
            settings,
        )


def test_extract_ocr_cli_threads_locked_revision(monkeypatch, tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "example_id": "e1",
                        "question": "What is shown?",
                        "gold_answer": "answer",
                        "baseline_answer": "wrong",
                        "image_paths": [str(image)],
                    }
                ]
            }
        )
    )
    out = tmp_path / "ocr_texts"
    calls: list[dict[str, object]] = []

    def fake_extract(image_path: Path, model_name: str = "", revision: str | None = None) -> str:
        calls.append({"image_path": image_path, "model_name": model_name, "revision": revision})
        return "OCR TEXT"

    monkeypatch.setattr(extract_ocr, "extract_got_ocr", fake_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_ocr.py",
            "--engine",
            "got-ocr-2",
            "--samples",
            str(samples),
            "--out",
            str(out),
            "--model",
            settings.recovery.got_ocr_model,
            "--revision",
            settings.recovery.got_ocr_revision or "",
        ],
    )
    extract_ocr.main()

    assert calls[0]["revision"] == settings.recovery.got_ocr_revision
    manifest = json.loads((tmp_path / "label_studio/manifest.json").read_text())
    assert manifest["revision"] == settings.recovery.got_ocr_revision
