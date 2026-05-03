"""
src/fatat_al_arab/retrievers/dense.py
======================================
Why this file exists: Dense semantic retriever leg of the triple-hybrid
pipeline. It computes cosine similarity between the query embedding and
pre-built GATE-AraBERT chunk embeddings stored in memory.
When triggered: Inside retrieve_node (Stage 4) — uses HyDE vector if available, else query vector.
Purpose: AraBERT 768-dim cosine semantic search — strongest on thematic / paraphrase matches

At demo scale (~4,500 chunks × 768-d vectors ≈ 13 MB float32) a brute-force
numpy dot-product scan is fast enough: ~5ms on a laptop. We only switch to
Qdrant HNSW for larger corpora (§2.4.2 of the architecture doc).

Architecture decisions:
  - Embeddings are pre-computed during index build and stored in a numpy array.
  - retrieve() encodes the query on the fly, then dot-product against the matrix.
  - All vectors are L2-normalised so cosine sim = dot product.
  - Falls back gracefully if sentence-transformers is unavailable (CI mode).
"""

from __future__ import annotations

import functools

import numpy as np

from fatat_al_arab.embed import get_dense_embedding
from fatat_al_arab.rrf import ScoredChunk


# M5b embed cache — repeated identical queries (e.g. CRAG re-query with the same
# HyDE passage, or back-to-back Streamlit calls) skip the model forward-pass.
# maxsize=256 covers any realistic session without unbounded growth.
@functools.lru_cache(maxsize=256)
def _cached_encode(query: str) -> list:
    """Return the dense embedding for *query*, caching by exact string."""
    return get_dense_embedding(query)


class DenseRetriever:
    """
    Dense cosine-similarity retriever over pre-built AraBERT embeddings.

    Usage:
        retriever = DenseRetriever(chunks, embeddings)
        results   = retriever.retrieve("من هو ناصر الهزاني", n=10)
    """

    def __init__(
        self,
        chunks: list[ScoredChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Args:
            chunks:     List of ScoredChunk — the full chunk corpus.
            embeddings: float32 ndarray of shape (len(chunks), DENSE_DIM).
                        Must be L2-normalised (so dot = cosine similarity).
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) != embeddings rows ({embeddings.shape[0]})"
            )
        self._chunks = chunks
        # Ensure float32 and row-normalised defensively
        self._embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings /= norms

    def retrieve(self, query: str, n: int = 10) -> list[ScoredChunk]:
        """
        Return top-n chunks by cosine similarity to *query*, descending.
        rrf_score is set to the raw cosine similarity so rrf.fuse() can rank.
        """
        qvec = np.array(_cached_encode(query), dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec /= norm

        sims = self._embeddings @ qvec          # shape (N,)
        top_idx = np.argsort(sims)[::-1][:n]

        results: list[ScoredChunk] = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0.0:
                break
            c = self._chunks[idx]
            results.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    rrf_score=score,
                    text=c.text,
                    level=c.level,
                    anchor_id=c.anchor_id,
                    poet_name=c.poet_name,
                    source_volume=c.source_volume,
                    source_page=c.source_page,
                    source_image_path=c.source_image_path,
                    manuscript_short_key=c.manuscript_short_key,
                    extra=c.extra,
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)
