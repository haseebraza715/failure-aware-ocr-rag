from __future__ import annotations

import json
from types import SimpleNamespace

from faar.pdf_preprocessing import export_docling_markdown


def test_docling_adapter_exports_structured_markdown(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF")
    output_path = tmp_path / "output.md"

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Converted"

    class FakeResult(SimpleNamespace):
        pass

    class FakeConverter:
        def convert(self, conversion_input):
            assert conversion_input is not None
            return [
                FakeResult(
                    status=SimpleNamespace(name="SUCCESS"),
                    pages=[SimpleNamespace(page_no=0, page_hash="abc", size=SimpleNamespace(width=1, height=1), cells=[])],
                    errors=[],
                    document=FakeDocument(),
                    render_as_markdown=lambda: "# Converted",
                )
            ]

    class FakeDocumentConversionInput:
        @classmethod
        def from_paths(cls, paths):
            return SimpleNamespace(paths=list(paths))

    fake_converter_module = SimpleNamespace(DocumentConverter=FakeConverter)
    fake_document_module = SimpleNamespace(DocumentConversionInput=FakeDocumentConversionInput)

    def fake_import(name: str):
        if name == "docling.document_converter":
            return fake_converter_module
        if name == "docling.datamodel.document":
            return fake_document_module
        raise ImportError(name)

    monkeypatch.setattr("faar.pdf_preprocessing.import_module", fake_import)

    result = export_docling_markdown(pdf_path, output_path)

    assert result == output_path
    assert "# Converted" in output_path.read_text()
    sidecar = json.loads(output_path.with_suffix(".json").read_text())
    assert sidecar["status"] == "SUCCESS"
    assert sidecar["page_count"] == 1
