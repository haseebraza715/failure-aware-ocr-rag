from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .benchmarks import BenchmarkRepository
from .metrics import exact_match, token_f1
from .recovery import VisualFallback
from .settings import AppSettings


class VisualRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]: ...


def _torch_device():
    torch = import_module("torch")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ColPaliRetriever:
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
        model_class = transformers.ColPaliForRetrieval
        processor_class = transformers.ColPaliProcessor

        self.image_paths = image_paths
        self.device = _torch_device()
        dtype = torch.bfloat16 if self.device.type in {"cuda", "mps"} else torch.float32
        self.model = (
            model_class.from_pretrained(model_name, revision=revision, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.processor = processor_class.from_pretrained(model_name, revision=revision)
        embeddings = []
        for start in range(0, len(image_paths), batch_size):
            images = [image_module.open(path).convert("RGB") for path in image_paths[start : start + batch_size]]
            inputs = self.processor(images=images).to(self.device)
            with torch.no_grad():
                embeddings.append(self.model(**inputs).embeddings.detach().cpu())
        self.image_embeddings = torch.cat(embeddings, dim=0)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        torch = import_module("torch")

        inputs = self.processor(text=[query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**inputs).embeddings.detach().cpu()
            scores = self.processor.score_retrieval(query_embeddings, self.image_embeddings)[0]
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

        del batch_size
        self.image_paths = image_paths
        self.device = _torch_device()
        dtype = torch.bfloat16 if self.device.type in {"cuda", "mps"} else torch.float32
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
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.image_embeddings = self._encode([image_module.open(path).convert("RGB") for path in image_paths])

    def _encode(self, values: list[Any]) -> np.ndarray:
        torch = import_module("torch")
        functional = import_module("torch.nn.functional")

        is_text = isinstance(values[0], str)
        inputs = {
            "text": values if is_text else [""] * len(values),
            "image": [None] * len(values) if is_text else values,
            "tokenizer": self.tokenizer,
        }
        with torch.no_grad():
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
    if mode == "colpali":
        return ColPaliRetriever(
            images,
            settings.retrieval.colpali_model,
            settings.retrieval.colpali_revision,
            settings.retrieval.visual_batch_size,
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
) -> list[dict[str, Any]]:
    retriever = build_visual_retriever(mode, repo, settings)
    fallback = VisualFallback(settings)
    example_ids = repo.list_example_ids()
    if max_examples is not None:
        example_ids = example_ids[: max(0, max_examples)]
    rows = []
    for example_id in example_ids:
        example = repo.get_example(example_id)
        retrieved = retriever.retrieve(example.question, settings.retrieval.top_k)
        image_paths = [path for path, _ in retrieved]
        answer_result = fallback.answer(example.question, image_paths, "")
        prediction = str(answer_result.get("answer", ""))
        rows.append(
            {
                "profile": mode,
                "example_id": example_id,
                "question": example.question,
                "gold_answer": example.correct_answer,
                "predicted_answer": prediction,
                "failure_type": "visual_retrieval",
                "policy_action": "invoke_vlm",
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
            }
        )
    return rows
