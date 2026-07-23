from __future__ import annotations

import pytest

from faar.settings import AppSettings


def test_openai_requires_dated_snapshot(tmp_path) -> None:
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.openai_model = "gpt-4o"

    with pytest.raises(ValueError, match="gpt-4o-2024-11-20"):
        settings.validate_openai_snapshot()


def test_hugging_face_revisions_must_be_immutable_commit_shas(tmp_path) -> None:
    settings = AppSettings(project_root=tmp_path)
    locked = AppSettings(project_root=tmp_path)
    settings.retrieval.embedding_revision = None
    settings.retrieval.reranker_revision = None
    settings.recovery.got_ocr_revision = None
    settings.recovery.byt5_revision = None
    settings.retrieval.colpali_revision = None

    with pytest.raises(ValueError, match="embedding"):
        settings.validate_model_revisions(include_visual="colpali")

    settings.retrieval.embedding_revision = locked.retrieval.embedding_revision
    settings.retrieval.reranker_revision = locked.retrieval.reranker_revision
    settings.retrieval.colpali_revision = locked.retrieval.colpali_revision
    settings.recovery.got_ocr_revision = locked.recovery.got_ocr_revision
    settings.recovery.byt5_revision = locked.recovery.byt5_revision
    settings.validate_model_revisions(include_visual="colpali")

    provenance = settings.model_provenance()
    assert provenance["embedding"]["revision"] == locked.retrieval.embedding_revision
    assert provenance["vlm"]["repository"] == "gpt-4o-2024-11-20"

    settings.retrieval.embedding_revision = "a" * 40
    with pytest.raises(ValueError, match="model_revisions.json"):
        settings.validate_model_revisions(include_visual="colpali")
