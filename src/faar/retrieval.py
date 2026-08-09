from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .resource_limits import (
    enforce_gpu_memory_fraction,
    enforce_memory_budget,
    release_cuda_cache,
    select_dtype,
    torch_device,
)
from .settings import RetrievalSettings
from .types import Chunk, RetrievalHit


MODEL_ALIASES = {
    "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

CORPUS_CACHE_SCHEMA_VERSION = 1


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values)
    return (values - low) / (high - low)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%$]+", text.lower())


def _corpus_text_digest(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _is_cuda_oom(exc: BaseException, torch_module: Any | None = None) -> bool:
    if torch_module is not None and isinstance(exc, getattr(torch_module.cuda, "OutOfMemoryError", ())):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _encode_with_oom_retry(
    embedder: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    torch = import_module("torch")
    while True:
        try:
            return (
                embedder.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                ).astype("float32")
            )
        except Exception as exc:
            if not _is_cuda_oom(exc, torch):
                raise
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            release_cuda_cache(torch)
            print(
                f"[faar] CUDA OOM during corpus encode; halving batch size -> {batch_size}",
                flush=True,
            )


def _encode_corpus_embeddings(
    embedder: SentenceTransformer,
    chunks: list[Chunk],
    settings: RetrievalSettings,
    cache_dir: Path | None,
) -> np.ndarray:
    if cache_dir is None or not chunks:
        return _encode_with_oom_retry(
            embedder,
            [chunk.text for chunk in chunks],
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
        )
    torch = import_module("torch")
    dtype_name = str(select_dtype(torch_device(torch), torch))
    text_digest = _corpus_text_digest(chunks)
    key = hashlib.sha256(
        f"{CORPUS_CACHE_SCHEMA_VERSION}|{settings.embedding_model}|"
        f"{settings.embedding_revision}|{dtype_name}|{text_digest}".encode("utf-8")
    ).hexdigest()
    cache_dir = cache_dir.expanduser().resolve()
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.meta.json"
    if npz_path.is_file() and meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            valid = (
                stored.get("schema_version") == CORPUS_CACHE_SCHEMA_VERSION
                and stored.get("model") == settings.embedding_model
                and stored.get("revision") == settings.embedding_revision
                and stored.get("dtype") == dtype_name
                and stored.get("text_digest") == text_digest
                and stored.get("count") == len(chunks)
            )
            if valid:
                embeddings = np.load(npz_path)["embeddings"]
                if embeddings.shape == (len(chunks), int(stored["dim"])) and embeddings.dtype == np.float32:
                    print(
                        f"[faar] corpus embeddings cache hit: {npz_path.name} ({len(chunks)} chunks)",
                        flush=True,
                    )
                    return embeddings
        except (OSError, ValueError, TypeError, KeyError):
            pass
    embeddings = _encode_with_oom_retry(
        embedder,
        [chunk.text for chunk in chunks],
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_npz = npz_path.with_name(f".{npz_path.stem}.tmp")
        tmp_meta = meta_path.with_name(f".{meta_path.name}.tmp")
        np.savez(tmp_npz, embeddings=embeddings)
        os.replace(Path(f"{tmp_npz}.npz"), npz_path)
        meta = {
            "schema_version": CORPUS_CACHE_SCHEMA_VERSION,
            "model": settings.embedding_model,
            "revision": settings.embedding_revision,
            "dtype": dtype_name,
            "text_digest": text_digest,
            "count": len(chunks),
            "dim": int(embeddings.shape[1]),
        }
        tmp_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_meta, meta_path)
    except OSError as exc:
        print(f"[faar] could not persist corpus embedding cache: {exc}", flush=True)
    return embeddings


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str, revision: str | None) -> SentenceTransformer:
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    torch = import_module("torch")
    device = torch_device(torch)
    dtype = select_dtype(device, torch)
    enforce_gpu_memory_fraction(torch)
    return SentenceTransformer(
        resolved_name,
        revision=revision,
        trust_remote_code=True,
        device=str(device),
        model_kwargs={"torch_dtype": dtype},
    )


@lru_cache(maxsize=2)
def _load_reranker(model_name: str, revision: str | None) -> CrossEncoder:
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    torch = import_module("torch")
    device = torch_device(torch)
    dtype = select_dtype(device, torch)
    enforce_gpu_memory_fraction(torch)
    return CrossEncoder(
        resolved_name,
        revision=revision,
        trust_remote_code=True,
        device=str(device),
        automodel_args={"torch_dtype": dtype},
    )


def _to_probability(score: float) -> float:
    """Map a cross-encoder logit to the [0, 1] gate score range."""
    return float(1.0 / (1.0 + np.exp(-score)))


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        settings: RetrievalSettings,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.chunks = chunks
        self.settings = settings
        self._bm25_tokens = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)
        self._embedder = _load_embedding_model(settings.embedding_model, settings.embedding_revision)
        self._reranker = _load_reranker(settings.reranker, settings.reranker_revision)
        torch = import_module("torch")
        enforce_memory_budget("text retrieval model load", torch)
        corpus_embeddings = _encode_corpus_embeddings(self._embedder, chunks, settings, cache_dir)
        self._dense_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self._dense_index.add(corpus_embeddings)
        del corpus_embeddings
        enforce_memory_budget("text retrieval corpus encode", torch)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = top_k or self.settings.top_k
        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        query_embedding = self._embedder.encode(
            [query],
            batch_size=self.settings.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        dense_scores, dense_indices = self._dense_index.search(query_embedding, k=min(k, len(self.chunks)))
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        dense_lookup = np.zeros(len(self.chunks), dtype=np.float32)
        for idx, score in zip(dense_indices, dense_scores):
            dense_lookup[int(idx)] = float(score)

        norm_bm25 = _normalize_scores(bm25_scores)
        norm_dense = _normalize_scores(dense_lookup)
        fused = 0.45 * norm_dense + 0.35 * norm_bm25 + 0.20 * _rrf_component(bm25_scores, dense_lookup)

        candidate_indices = np.argsort(fused)[::-1][:k]
        candidate_pairs = [(query, self.chunks[int(idx)].text) for idx in candidate_indices]
        reranker_logits = self._reranker.predict(candidate_pairs, show_progress_bar=False)
        reranker_scores = [_to_probability(float(score)) for score in reranker_logits]
        ranked_candidates = sorted(
            zip(candidate_indices, reranker_scores), key=lambda pair: pair[1], reverse=True
        )
        hits: list[RetrievalHit] = []
        for idx, reranker_score in ranked_candidates:
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[int(idx)],
                    bm25_score=float(norm_bm25[idx]),
                    dense_score=float(norm_dense[idx]),
                    fused_score=float(fused[idx]),
                    reranker_score=round(reranker_score, 6),
                )
            )
        return hits


def _rrf_component(bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = 60) -> np.ndarray:
    bm25_rank = np.argsort(np.argsort(-bm25_scores))
    dense_rank = np.argsort(np.argsort(-dense_scores))
    return (1.0 / (k + bm25_rank + 1)) + (1.0 / (k + dense_rank + 1))
