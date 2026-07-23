from __future__ import annotations

from types import SimpleNamespace

from faar import ocr, recovery, retrieval, visual_baselines
from faar.settings import AppSettings


REVISION = "b" * 40


def test_text_model_loaders_receive_pinned_revisions(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        retrieval,
        "SentenceTransformer",
        lambda model, **kwargs: calls.append(("embedding", model, kwargs)) or object(),
    )
    monkeypatch.setattr(
        retrieval,
        "CrossEncoder",
        lambda model, **kwargs: calls.append(("reranker", model, kwargs)) or object(),
    )
    retrieval._load_embedding_model.cache_clear()
    retrieval._load_reranker.cache_clear()

    retrieval._load_embedding_model("NV-Embed-v2", REVISION)
    retrieval._load_reranker("bge-reranker-v2-m3", REVISION)

    assert calls[0] == (
        "embedding",
        "nvidia/NV-Embed-v2",
        {"revision": REVISION, "trust_remote_code": True},
    )
    assert calls[1] == (
        "reranker",
        "BAAI/bge-reranker-v2-m3",
        {"revision": REVISION, "trust_remote_code": True},
    )


def test_got_ocr_and_byt5_loaders_receive_pinned_revisions(monkeypatch) -> None:
    calls = []

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return SimpleNamespace(eval=lambda: "model")

    monkeypatch.setattr(ocr, "AutoProcessor", FakeLoader)
    monkeypatch.setattr(ocr, "AutoModelForImageTextToText", FakeLoader)
    monkeypatch.setattr(recovery, "AutoTokenizer", FakeLoader)
    monkeypatch.setattr(recovery, "AutoModelForSeq2SeqLM", FakeLoader)
    ocr._load_got_ocr.cache_clear()
    recovery._load_byt5.cache_clear()

    ocr._load_got_ocr("stepfun-ai/GOT-OCR-2.0-hf", REVISION)
    recovery._load_byt5("google/byt5-small", REVISION)

    assert all(call[1]["revision"] == REVISION for call in calls)


def test_visual_loader_receives_pinned_revision(monkeypatch, tmp_path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    calls = []

    class FakeRepository:
        def corpus_image_paths(self):
            return [image]

    monkeypatch.setattr(
        visual_baselines,
        "ColPaliRetriever",
        lambda *args: calls.append(args) or object(),
    )
    settings = AppSettings(project_root=tmp_path)
    settings.retrieval.colpali_revision = REVISION

    visual_baselines.build_visual_retriever("colpali", FakeRepository(), settings)

    assert calls[0][2] == REVISION
