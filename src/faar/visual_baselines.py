from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .api_logging import openai_cost_rates
from .benchmarks import BenchmarkRepository
from .metrics import exact_match, token_f1
from .recovery import VisualFallback
from .resource_limits import (
    enforce_gpu_memory_fraction,
    enforce_memory_budget,
    release_cuda_cache,
    select_dtype,
    torch_device,
)
from .run_io import atomic_write_json, run_fingerprint, safe_checkpoint_stem
from .settings import AppSettings


class VisualRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]: ...


class ColPaliRetriever:
    def __init__(
        self,
        image_paths: list[Path],
        model_name: str,
        revision: str | None,
        batch_size: int = 4,
        score_batch_size: int = 32,
    ) -> None:
        torch = import_module("torch")
        image_module = import_module("PIL.Image")
        transformers = import_module("transformers")
        model_class = transformers.ColPaliForRetrieval
        processor_class = transformers.ColPaliProcessor

        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("ColPali requires a non-empty visual corpus; found no image paths.")
        batch_size = max(1, batch_size)
        score_batch_size = max(1, score_batch_size)
        self.score_batch_size = score_batch_size
        self.device = torch_device(torch)
        dtype = select_dtype(self.device, torch)
        enforce_gpu_memory_fraction(torch)
        self.model = (
            model_class.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        enforce_memory_budget("ColPali model load", torch)
        self.processor = processor_class.from_pretrained(model_name, revision=revision)
        embeddings = []
        try:
            for start in range(0, len(self.image_paths), batch_size):
                images = []
                inputs = None
                outputs = None
                try:
                    for path in self.image_paths[start : start + batch_size]:
                        with image_module.open(path) as source:
                            images.append(source.convert("RGB"))
                    enforce_memory_budget("ColPali batch", torch)
                    inputs = self.processor(images=images).to(self.device)
                    with torch.inference_mode():
                        outputs = self.model(**inputs)
                    embeddings.append(outputs.embeddings.detach().cpu())
                finally:
                    for image in images:
                        image.close()
                    del images, inputs, outputs
            self.image_embeddings = torch.cat(embeddings, dim=0)
        finally:
            del embeddings
            release_cuda_cache(torch)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        torch = import_module("torch")

        inputs = self.processor(text=[query]).to(self.device)
        try:
            with torch.no_grad():
                query_embeddings = self.model(**inputs).embeddings.detach().cpu()
            score_slices = []
            for start in range(0, len(self.image_paths), self.score_batch_size):
                stop = min(start + self.score_batch_size, len(self.image_paths))
                enforce_memory_budget("ColPali score batch", torch)
                score_slices.append(
                    self.processor.score_retrieval(
                        query_embeddings,
                        self.image_embeddings[start:stop],
                    )[0]
                )
            scores = torch.cat(score_slices, dim=0)
        finally:
            del inputs
            release_cuda_cache(torch)
        indices = torch.topk(scores, k=min(top_k, len(self.image_paths))).indices.tolist()
        return [(self.image_paths[index], float(scores[index])) for index in indices]


class VisRAGRetriever:
    INSTRUCTION = "Represent this query for retrieving relevant documents: "

    def __init__(
        self,
        image_paths: list[Path],
        model_name: str,
        revision: str | None,
        batch_size: int = 4,
    ) -> None:
        torch = import_module("torch")
        image_module = import_module("PIL.Image")
        transformers = import_module("transformers")

        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("VisRAG requires a non-empty visual corpus; found no image paths.")
        self.batch_size = max(1, batch_size)
        self.device = torch_device(torch)
        dtype = select_dtype(self.device, torch)
        enforce_gpu_memory_fraction(torch)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        self.model = (
            transformers.AutoModel.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        enforce_memory_budget("VisRAG model load", torch)
        chunks = []
        try:
            for start in range(0, len(self.image_paths), self.batch_size):
                images = []
                try:
                    for path in image_paths[start : start + self.batch_size]:
                        with image_module.open(path) as source:
                            images.append(source.convert("RGB"))
                    enforce_memory_budget("VisRAG batch", torch)
                    chunks.append(self._encode(images))
                finally:
                    for image in images:
                        image.close()
            self.image_embeddings = np.concatenate(chunks, axis=0)
        finally:
            del chunks
            release_cuda_cache(torch)

    def _encode(self, values: list[Any]) -> np.ndarray:
        torch = import_module("torch")
        functional = import_module("torch.nn.functional")

        is_text = isinstance(values[0], str)
        inputs = {
            "text": values if is_text else [""] * len(values),
            "image": [None] * len(values) if is_text else values,
            "tokenizer": self.tokenizer,
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
            mask = outputs.attention_mask
            weighted_mask = mask * mask.cumsum(dim=1)
            summed = torch.sum(outputs.last_hidden_state * weighted_mask.unsqueeze(-1).float(), dim=1)
            pooled = summed / weighted_mask.sum(dim=1, keepdim=True).float()
            return functional.normalize(pooled, p=2, dim=1).detach().cpu().numpy()

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        query_embedding = self._encode([self.INSTRUCTION + query])
        scores = (query_embedding @ self.image_embeddings.T)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.image_paths[int(index)], float(scores[int(index)])) for index in indices]


def build_visual_retriever(mode: str, repo: BenchmarkRepository, settings: AppSettings) -> VisualRetriever:
    images = repo.corpus_image_paths()
    if not images:
        raise ValueError(f"{mode} requires a non-empty visual corpus; found no image paths.")
    if mode == "colpali":
        return ColPaliRetriever(
            images,
            settings.retrieval.colpali_model,
            settings.retrieval.colpali_revision,
            settings.retrieval.visual_batch_size,
            score_batch_size=settings.retrieval.visual_score_batch_size,
        )
    if mode == "visrag":
        return VisRAGRetriever(
            images,
            settings.retrieval.visrag_model,
            settings.retrieval.visrag_revision,
            settings.retrieval.visual_batch_size,
        )
    raise ValueError(f"Unsupported visual baseline: {mode}")


def run_visual_baseline(
    settings: AppSettings,
    repo: BenchmarkRepository,
    mode: str,
    max_examples: int | None = None,
    *,
    resume: bool = False,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    dataset = getattr(repo, "dataset", None)
    split = getattr(repo, "split", None)
    fingerprint = run_fingerprint(
        settings,
        profile=mode,
        dataset=dataset,
        split=split,
        manifest_sha256=getattr(repo, "manifest_sha256", None),
    )
    example_ids = repo.list_example_ids()
    if max_examples is not None:
        example_ids = example_ids[: max(0, max_examples)]
    base_output = output_dir or (settings.project_root / "logs/phase3" / mode)
    base_output.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    if resume:
        for example_id in example_ids:
            cached = _load_visual_checkpoint(_visual_checkpoint_path(base_output, example_id))
            if cached is not None and _is_valid_visual_checkpoint(cached, example_id, mode, fingerprint):
                rows_by_id[example_id] = cached
            else:
                pending.append(example_id)
    else:
        pending = list(example_ids)
    retriever = None
    fallback = None
    if pending:
        retriever = build_visual_retriever(mode, repo, settings)
        fallback = VisualFallback(settings)
    for example_id in pending:
        assert retriever is not None
        assert fallback is not None
        example = repo.get_example(example_id)
        retrieved = retriever.retrieve(example.question, settings.retrieval.top_k)
        image_paths = [path for path, _ in retrieved]
        answer_result = fallback.answer(example.question, image_paths, "")
        prediction = str(answer_result.get("answer", ""))
        row = {
            "profile": mode,
            "example_id": example_id,
            "question": example.question,
            "gold_answer": example.correct_answer,
            "predicted_answer": prediction,
            "failure_type": "visual_retrieval",
            "policy_action": "invoke_vlm",
            "request_model": answer_result.get("request_model"),
            "response_model": answer_result.get("response_model"),
            "completed_at_utc": answer_result.get("completed_at_utc"),
            "cost_rates": (
                answer_result.get("cost_rates")
                or (openai_cost_rates() if settings.recovery.vlm_backend == "openai" else None)
            ),
            "action_outcome": {
                "action": "invoke_vlm",
                "status": answer_result.get("status", "unknown"),
                "reason": f"{mode}_retrieval_then_vlm",
            },
            "metrics": {
                "ndcg@5": 0.0,
                "recall@5": 0.0,
                "em": exact_match(prediction, example.correct_answer),
                "f1": token_f1(prediction, example.correct_answer),
            },
            "visual_hits": [
                {"image_path": str(path), "score": round(score, 6)} for path, score in retrieved
            ],
            "source_assets": {"ocr_text_path": "", "image_paths": [str(path) for path in image_paths]},
            "run_metadata": {
                "profile": mode,
                "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                "run_fingerprint": fingerprint,
                "api_enabled": settings.recovery.api_enabled,
                "vlm_backend": settings.recovery.vlm_backend,
                "openai_model": settings.recovery.openai_model,
                "cost_rates": openai_cost_rates() if settings.recovery.vlm_backend == "openai" else None,
                "evaluation_size": len(example_ids),
                "selection": {"max_examples": max_examples},
                "dataset": dataset,
                "split": split,
            },
        }
        rows_by_id[example_id] = row
        atomic_write_json(_visual_checkpoint_path(base_output, example_id), row)
    return [rows_by_id[example_id] for example_id in example_ids if example_id in rows_by_id]


def _visual_checkpoint_path(base_output: Path, example_id: str) -> Path:
    return base_output / f"{safe_checkpoint_stem(example_id)}.json"


def _load_visual_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) else None


def _is_valid_visual_checkpoint(
    row: dict[str, Any],
    example_id: str,
    mode: str,
    fingerprint: str,
) -> bool:
    metadata = row.get("run_metadata")
    metrics = row.get("metrics")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        return False
    metric_values = [metrics.get(name) for name in ("ndcg@5", "recall@5", "em", "f1")]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in metric_values
    ):
        return False
    return (
        row.get("example_id") == example_id
        and row.get("profile") == mode
        and metadata.get("run_fingerprint") == fingerprint
        and isinstance(row.get("question"), str)
        and isinstance(row.get("gold_answer"), str)
        and isinstance(row.get("predicted_answer"), str)
        and isinstance(row.get("action_outcome"), dict)
        and isinstance(row.get("visual_hits"), list)
        and isinstance(row.get("source_assets"), dict)
    )
