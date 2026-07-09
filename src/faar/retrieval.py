from __future__ import annotations

import math
import re
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .settings import RetrievalSettings
from .types import Chunk, RetrievalHit


MODEL_ALIASES = {
    "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


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


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    return SentenceTransformer(resolved_name, trust_remote_code=True)


@lru_cache(maxsize=2)
def _load_reranker(model_name: str) -> CrossEncoder:
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    return CrossEncoder(resolved_name, trust_remote_code=True)


def _to_probability(score: float) -> float:
    """Map a cross-encoder logit to the [0, 1] gate score range."""
    return float(1.0 / (1.0 + np.exp(-score)))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], settings: RetrievalSettings) -> None:
        self.chunks = chunks
        self.settings = settings
        self._bm25_tokens = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens)
        self._embedder = _load_embedding_model(settings.embedding_model)
        self._reranker = _load_reranker(settings.reranker)
        corpus_embeddings = self._embedder.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        self._dense_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self._dense_index.add(corpus_embeddings)
        self._corpus_embeddings = corpus_embeddings

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = top_k or self.settings.top_k
        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        query_embedding = self._embedder.encode(
            [query],
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
