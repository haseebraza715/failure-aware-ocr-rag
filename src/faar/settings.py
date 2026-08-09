from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class RetrievalSettings(BaseModel):
    chunk_size_words: int = Field(default=180, gt=0)
    chunk_overlap_words: int = Field(default=40, ge=0)
    top_k: int = Field(default=5, ge=1)
    semantic_backtrack_top_k: int = Field(default=8, ge=1)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
    embedding_backend: str = "local-hash-v1"
    embedding_batch_size: int = Field(default=64, gt=0, le=1024)
    max_chunks: int = Field(default=10_000, gt=0)

    @model_validator(mode="after")
    def _validate_chunk_geometry(self) -> RetrievalSettings:
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError(
                f"chunk_overlap_words ({self.chunk_overlap_words}) must be smaller than "
                f"chunk_size_words ({self.chunk_size_words})"
            )
        return self


class GateSettings(BaseModel):
    quality_threshold: float = Field(default=0.52, ge=0.0, le=1.0)
    structural_threshold: int = Field(default=1, ge=0)
    weird_char_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    lexical_floor: float = Field(default=0.10, ge=0.0, le=1.0)
    dense_floor: float = Field(default=0.20, ge=0.0, le=1.0)


class CorrectionSettings(BaseModel):
    """Experimental parameters of the ByT5 correction gate.

    The defaults reproduce the Phase 3 batch-run behavior; runs that vary them
    should record the values alongside their artifacts.
    """

    min_weird_char_ratio: float = Field(default=0.08, ge=0.0)
    min_length_ratio: float = Field(default=0.6, ge=0.0)
    max_length_ratio: float = Field(default=1.4, ge=0.0)
    min_token_overlap: float = Field(default=0.5, ge=0.0, le=1.0)
    max_noise_increase: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _validate_length_ratio_bounds(self) -> CorrectionSettings:
        if self.min_length_ratio > self.max_length_ratio:
            raise ValueError(
                f"min_length_ratio ({self.min_length_ratio}) must not exceed "
                f"max_length_ratio ({self.max_length_ratio})"
            )
        return self


class RecoverySettings(BaseModel):
    correction: CorrectionSettings = Field(default_factory=CorrectionSettings)
    byt5_model: str = "google/byt5-small"
    enable_backtracking: bool = True
    # Word-level correction uses a local ByT5 model. Offline reproducibility runs
    # disable it so results do not depend on whether the model is cached locally;
    # the committed evidence therefore reflects guarded-skip word-level recovery.
    enable_byt5: bool = True
    vlm_backend: str = "mock"
    openai_model: str = "gpt-4o"
    enable_vlm: bool = True
    api_enabled: bool = True
    request_timeout_seconds: int = 60


class ExperimentSettings(BaseModel):
    profile_name: str = "faar_full"
    disable_diagnosis: bool = False
    disable_backtracking: bool = False
    disable_vlm: bool = False
    force_direct_answer: bool = False


class AppSettings(BaseModel):
    project_root: Path = Field(default_factory=lambda: _default_project_root())
    phase0_manifest: Path | None = None
    phase0_summary: Path | None = None
    phase0_manual_labels: Path | None = None
    phase0_ocr_dir: Path | None = None
    logs_dir: Path | None = None
    artifacts_dir: Path | None = None
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)

    def model_post_init(self, __context: object) -> None:
        self.project_root = self.project_root.resolve()
        self.phase0_manifest = (self.phase0_manifest or self.project_root / "data/phase0/sample_manifest.csv").resolve()
        self.phase0_summary = (
            self.phase0_summary or self.project_root / "data/phase0/phase0_asset_summary.json"
        ).resolve()
        self.phase0_manual_labels = (
            self.phase0_manual_labels or self.project_root / "data/phase0/manual_labels.csv"
        ).resolve()
        self.phase0_ocr_dir = (self.phase0_ocr_dir or self.project_root / "artifacts/phase0/ocr_text").resolve()
        self.logs_dir = (self.logs_dir or self.project_root / "logs/phase1").resolve()
        self.artifacts_dir = (self.artifacts_dir or self.project_root / "artifacts/phase1").resolve()

    def validate_runtime_paths(self) -> None:
        required = {
            "project_root": self.project_root,
            "phase0_manifest": self.phase0_manifest,
            "phase0_ocr_dir": self.phase0_ocr_dir,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if path is None or not path.exists()]
        if missing:
            details = "\n".join(missing)
            raise FileNotFoundError(
                "FAAR settings validation failed.\n"
                "Set FAAR_PROJECT_ROOT or pass --project-root to point at the repository root.\n"
                f"Missing paths:\n{details}"
            )


def _default_project_root() -> Path:
    env_root = os.getenv("FAAR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.cwd()
