from __future__ import annotations

from types import SimpleNamespace

from faar.pdf_preprocessing import export_docling_markdown


def test_docling_adapter_exports_structured_markdown(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF")
    output_path = tmp_path / "output.md"

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Converted"

    class FakeConverter:
        def convert(self, path):
            assert path == pdf_path
            return SimpleNamespace(document=FakeDocument())

    fake_module = SimpleNamespace(DocumentConverter=FakeConverter)
    monkeypatch.setattr("faar.pdf_preprocessing.import_module", lambda name: fake_module)

    result = export_docling_markdown(pdf_path, output_path)

    assert result == output_path
    assert output_path.read_text() == "# Converted"
