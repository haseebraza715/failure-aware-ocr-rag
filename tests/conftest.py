import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EMBED_MODEL",
        "EMBED_MODEL_REPO",
        "EMBED_MODEL_REVISION",
        "RERANKER",
        "RERANKER_MODEL_REPO",
        "RERANKER_MODEL_REVISION",
        "GOT_OCR_MODEL",
        "GOT_OCR_MODEL_REPO",
        "GOT_OCR_MODEL_REVISION",
        "COLPALI_MODEL",
        "COLPALI_MODEL_REPO",
        "COLPALI_MODEL_REVISION",
        "VISRAG_MODEL",
        "VISRAG_MODEL_REPO",
        "VISRAG_MODEL_REVISION",
        "BYT5_MODEL_REPO",
        "BYT5_MODEL_REVISION",
        "OPENAI_MODEL",
        "OPENAI_INPUT_USD_PER_MTOK",
        "OPENAI_OUTPUT_USD_PER_MTOK",
        "VLM_BACKEND",
        "FAAR_EMBED_BATCH_SIZE",
        "FAAR_VISUAL_SCORE_BATCH_SIZE",
        "FAAR_MAX_RSS_GB",
        "FAAR_MIN_GPU_FREE_GB",
    ):
        monkeypatch.delenv(name, raising=False)
