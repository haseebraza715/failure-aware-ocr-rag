from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from faar import visual_baselines
from faar.settings import AppSettings


def _image_paths(tmp_path: Path, count: int = 6) -> list[Path]:
    from PIL import Image

    paths = []
    for index in range(count):
        path = tmp_path / f"page_{index}.png"
        Image.new("RGB", (8, 8), color=(index * 40, 0, 0)).save(path)
        paths.append(path)
    return paths


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeOutput:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings


class _CountingModel:
    def __init__(self, seq: int = 4, dim: int = 8, forward=None) -> None:
        self.seq = seq
        self.dim = dim
        self.forward_calls = 0
        self._forward = forward

    def __call__(self, **inputs):
        self.forward_calls += 1
        if self._forward is not None:
            return self._forward(self, inputs)
        n = inputs.get("_n", 1)
        base = torch.arange(n * self.seq * self.dim, dtype=torch.float32)
        return _FakeOutput(base.reshape(n, self.seq, self.dim))

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    def __call__(self, **inputs):
        images = inputs.get("images")
        return _FakeBatch({"_n": len(images) if images is not None else 1})

    def to(self, device):
        return self

    def score_retrieval(self, query_embeddings, passage_embeddings):
        count = passage_embeddings.shape[0]
        query = query_embeddings[0]
        dots = torch.einsum("qd,nsd->nqs", query, passage_embeddings)
        scores = dots.max(dim=2).values.sum(dim=1)
        return scores.unsqueeze(0)


def _install_colpali_fakes(monkeypatch: pytest.MonkeyPatch, model) -> _FakeProcessor:
    processor = _FakeProcessor()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return processor

    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: None)
    monkeypatch.setattr("transformers.ColPaliForRetrieval", FakeModelClass)
    monkeypatch.setattr("transformers.ColPaliProcessor", FakeProcessorClass)
    return processor


def _cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "visual_embeddings"


def test_cache_hit_skips_reencoding_and_loads_identical_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    first = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    assert model.forward_calls == 2
    key = visual_baselines._visual_cache_key(
        visual_baselines.VISUAL_CACHE_SCHEMA_VERSION,
        "mock/colpali",
        None,
        "torch.float32",
        paths,
    )
    assert (cache_dir / f"{key}.npz").is_file()
    assert (cache_dir / f"{key}.meta.json").is_file()

    calls_before = model.forward_calls
    second = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)

    assert model.forward_calls == calls_before
    assert torch.equal(first.image_embeddings, second.image_embeddings)
    assert second.image_embeddings.shape == (4, model.seq, model.dim)


@pytest.mark.parametrize("mutation", ["path", "revision"])
def test_cache_misses_on_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    from PIL import Image

    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    calls_after_first = model.forward_calls

    if mutation == "path":
        extra = tmp_path / "extra.png"
        Image.new("RGB", (8, 8)).save(extra)
        changed_paths = paths + [extra]
        revision = None
    else:
        changed_paths = paths
        revision = "abc123"

    second = visual_baselines.ColPaliRetriever(
        changed_paths, "mock/colpali", revision, batch_size=2, cache_dir=cache_dir
    )

    assert model.forward_calls > calls_after_first
    assert second.image_embeddings.shape[0] == len(changed_paths)


def test_cuda_oom_halves_batch_size_and_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _image_paths(tmp_path, count=6)
    batch_sizes: list[int] = []
    forward_calls = 0

    def flaky_forward(model: _CountingModel, inputs) -> _FakeOutput:
        nonlocal forward_calls
        forward_calls += 1
        n = inputs.get("_n", 1)
        batch_sizes.append(n)
        if forward_calls == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate ...")
        base = torch.arange(n * model.seq * model.dim, dtype=torch.float32)
        return _FakeOutput(base.reshape(n, model.seq, model.dim))

    model = _CountingModel(forward=flaky_forward)
    _install_colpali_fakes(monkeypatch, model)

    retriever = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=4)

    assert batch_sizes == [4, 2, 2, 2]
    assert retriever.image_embeddings.shape == (6, model.seq, model.dim)
    assert "CUDA OOM during ColPali embedding" in capsys.readouterr().out


def test_build_visual_retriever_points_cache_under_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def corpus_image_paths(self):
            return [tmp_path / "a.png"]

    class FakeColPali:
        def __init__(
            self,
            image_paths: list[Path],
            model_name: str,
            revision: str | None,
            batch_size: int,
            *,
            score_batch_size: int,
            cache_dir: Path | None,
        ) -> None:
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(visual_baselines, "ColPaliRetriever", FakeColPali)
    settings = AppSettings(project_root=tmp_path)

    visual_baselines.build_visual_retriever("colpali", FakeRepo(), settings)

    assert captured["cache_dir"] == tmp_path / "cache" / "visual_embeddings"


def test_build_visual_retriever_no_cache_when_settings_lacks_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def corpus_image_paths(self):
            return [tmp_path / "a.png"]

    class FakeVisRAG:
        def __init__(
            self,
            image_paths: list[Path],
            model_name: str,
            revision: str | None,
            batch_size: int,
            *,
            cache_dir: Path | None,
        ) -> None:
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(visual_baselines, "VisRAGRetriever", FakeVisRAG)
    settings = SimpleNamespace(
        retrieval=SimpleNamespace(visrag_model="mock/visrag", visrag_revision=None, visual_batch_size=1)
    )

    visual_baselines.build_visual_retriever("visrag", FakeRepo(), settings)

    assert captured["cache_dir"] is None
