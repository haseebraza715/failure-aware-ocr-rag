import pytest
from pydantic import ValidationError

from faar.settings import CorrectionSettings, GateSettings, RetrievalSettings


def test_chunk_size_words_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalSettings(chunk_size_words=0)
    with pytest.raises(ValidationError):
        RetrievalSettings(chunk_size_words=-5)


def test_chunk_overlap_must_be_below_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_words"):
        RetrievalSettings(chunk_size_words=180, chunk_overlap_words=180)
    with pytest.raises(ValidationError, match="chunk_overlap_words"):
        RetrievalSettings(chunk_size_words=10, chunk_overlap_words=15)
    RetrievalSettings(chunk_size_words=180, chunk_overlap_words=179)
    RetrievalSettings(chunk_size_words=180, chunk_overlap_words=0)


def test_top_k_and_backtrack_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalSettings(top_k=0)
    with pytest.raises(ValidationError):
        RetrievalSettings(semantic_backtrack_top_k=0)


def test_gate_thresholds_are_bounded() -> None:
    with pytest.raises(ValidationError):
        GateSettings(quality_threshold=1.5)
    with pytest.raises(ValidationError):
        GateSettings(weird_char_threshold=-0.1)
    with pytest.raises(ValidationError):
        GateSettings(lexical_floor=2.0)
    with pytest.raises(ValidationError):
        GateSettings(dense_floor=1.1)
    GateSettings(quality_threshold=1.0, weird_char_threshold=0.0)


def test_correction_length_ratios_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="min_length_ratio"):
        CorrectionSettings(min_length_ratio=2.0, max_length_ratio=1.0)
    CorrectionSettings(min_length_ratio=0.5, max_length_ratio=0.5)


def test_correction_ratio_bounds() -> None:
    with pytest.raises(ValidationError):
        CorrectionSettings(min_token_overlap=1.5)
    with pytest.raises(ValidationError):
        CorrectionSettings(min_weird_char_ratio=-1.0)
