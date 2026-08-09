from __future__ import annotations

import math
import re
from functools import lru_cache
from hashlib import blake2b
from typing import Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .settings import RetrievalSettings
from .types import Chunk, RetrievalHit


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if not math.isfinite(low) or not math.isfinite(high) or math.isclose(low, high):
        return np.ones_like(values)
    return (values - low) / (high - low)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%$]+", text.lower())


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> "_Embedder":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'sentence-transformers' embedding backend needs the ml extra: "
            "pip install 'faar[ml]'. The default 'local-hash-v1' backend does not."
        ) from exc
    return SentenceTransformer(model_name)


class _Embedder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        batch_size: int | None = None,
    ) -> np.ndarray: ...


class LocalHashEmbedder:
    """Deterministic dependency-free feature hashing for offline demos/tests."""

    dimensions = 256

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        batch_size: int | None = None,
    ) -> np.ndarray:
        del convert_to_numpy, batch_size
        matrix = np.zeros((len(sentences), self.dimensions), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            for token in _tokenize(sentence):
                digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1.0, norms)
        return matrix


def _build_embedder(settings: RetrievalSettings) -> _Embedder:
    if settings.embedding_backend == "local-hash-v1":
        return LocalHashEmbedder()
    if settings.embedding_backend == "sentence-transformers":
        return _load_embedding_model(settings.embedding_model)
    raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: RetrievalSettings) -> None:
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        if len(chunks) > settings.max_chunks:
            raise ValueError(
                f"chunk count {len(chunks)} exceeds configured max_chunks={settings.max_chunks}"
            )
        self.chunks = chunks
        self.settings = settings
        self._bm25_tokens = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)
        self._embedder = _build_embedder(settings)
        corpus_embeddings = self._embedder.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=settings.embedding_batch_size,
        ).astype("float32")
        self._dense_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self._dense_index.add(corpus_embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = top_k if top_k is not None else self.settings.top_k
        if k <= 0:
            return []
        k = min(k, len(self.chunks))
        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        query_embedding = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.settings.embedding_batch_size,
        ).astype("float32")
        dense_scores, dense_indices = self._dense_index.search(query_embedding, k=k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        dense_lookup = np.zeros(len(self.chunks), dtype=np.float32)
        for idx, score in zip(dense_indices, dense_scores):
            if idx >= 0:
                dense_lookup[int(idx)] = float(score)

        norm_bm25 = _normalize_scores(bm25_scores)
        norm_dense = _normalize_scores(dense_lookup)
        fused = 0.45 * norm_dense + 0.35 * norm_bm25 + 0.20 * _rrf_component(bm25_scores, dense_lookup)

        candidate_indices = np.argsort(-fused, kind="stable")[:k]
        hits: list[RetrievalHit] = []
        for idx in candidate_indices:
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[int(idx)],
                    bm25_score=float(norm_bm25[idx]),
                    dense_score=float(norm_dense[idx]),
                    fused_score=float(fused[idx]),
                )
            )
        return hits


def _rrf_component(bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = 60) -> np.ndarray:
    bm25_rank = np.argsort(np.argsort(-bm25_scores, kind="stable"), kind="stable")
    dense_rank = np.argsort(np.argsort(-dense_scores, kind="stable"), kind="stable")
    return (1.0 / (k + bm25_rank + 1)) + (1.0 / (k + dense_rank + 1))
