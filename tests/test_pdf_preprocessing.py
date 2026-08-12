from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from faar.pdf_preprocessing import export_docling_markdown, reset_docling_converter

ROOT = Path(__file__).resolve().parents[1]


def _install_fakes(monkeypatch, converter_cls, snapshot_dir: Path, downloads: dict | None = None) -> None:
    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Converted"

    class FakeResult(SimpleNamespace):
        pass

    class FakeConverter(converter_cls):
        def convert(self, conversion_input):
            return [
                FakeResult(
                    status=SimpleNamespace(name="SUCCESS"),
                    pages=[],
                    errors=[],
                    document=FakeDocument(),
                    render_as_markdown=lambda: "# Converted",
                )
            ]

    class FakeDocumentConversionInput:
        @classmethod
        def from_paths(cls, paths):
            return SimpleNamespace(paths=list(paths))

    def fake_hub_snapshot(repo_id: str, revision: str):
        if downloads is not None:
            downloads["repo_id"] = repo_id
            downloads["revision"] = revision
        return str(snapshot_dir)

    fake_converter_module = SimpleNamespace(DocumentConverter=FakeConverter)
    fake_document_module = SimpleNamespace(DocumentConversionInput=FakeDocumentConversionInput)
    fake_hub = SimpleNamespace(snapshot_download=fake_hub_snapshot)

    def fake_import(name: str):
        if name == "docling.document_converter":
            return fake_converter_module
        if name == "docling.datamodel.document":
            return fake_document_module
        if name == "huggingface_hub":
            return fake_hub
        raise ImportError(name)

    monkeypatch.setattr("faar.pdf_preprocessing.import_module", fake_import)
    monkeypatch.chdir(ROOT)
    reset_docling_converter()


def _write_pdf(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF")
    return path


def test_docling_adapter_exports_structured_markdown(monkeypatch, tmp_path) -> None:
    pdf_path = _write_pdf(tmp_path, "input.pdf")
    output_path = tmp_path / "output.md"

    class FakeConverter:
        def __init__(self, artifacts_path=None) -> None:
            pass

    _install_fakes(monkeypatch, FakeConverter, tmp_path / "snapshot")

    result = export_docling_markdown(pdf_path, output_path)

    assert result == output_path
    assert "# Converted" in output_path.read_text()
    sidecar = json.loads(output_path.with_suffix(".json").read_text())
    assert sidecar["status"] == "SUCCESS"
    assert sidecar["page_count"] == 0


def test_docling_converter_is_constructed_once_per_process(monkeypatch, tmp_path) -> None:
    constructions = {"count": 0}

    class FakeConverter:
        def __init__(self, artifacts_path=None) -> None:
            constructions["count"] += 1

    _install_fakes(monkeypatch, FakeConverter, tmp_path / "snapshot")
    export_docling_markdown(_write_pdf(tmp_path, "first.pdf"), tmp_path / "first.md")
    export_docling_markdown(_write_pdf(tmp_path, "second.pdf"), tmp_path / "second.md")
    assert constructions["count"] == 1
    reset_docling_converter()


def test_docling_artifacts_are_pinned_to_immutable_lock_revision(monkeypatch, tmp_path) -> None:
    downloads: dict = {}

    class FakeConverter:
        def __init__(self, artifacts_path=None) -> None:
            downloads["artifacts_path"] = str(artifacts_path)

    _install_fakes(monkeypatch, FakeConverter, tmp_path / "snapshot", downloads)
    export_docling_markdown(_write_pdf(tmp_path, "doc.pdf"), tmp_path / "doc.md")
    assert downloads["repo_id"] == "docling-project/docling-models"
    assert downloads["revision"] == "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4"
    assert downloads["artifacts_path"] == str(tmp_path / "snapshot")
