"""
src/fatat_al_arab/retrievers/bm25.py
=====================================
Why this file exists: BM25 is the sparse retriever leg of the triple-hybrid
pipeline. It is the backstop for exact-token queries — poet names, volume
identifiers, rare Nabati words — where dense vector similarity degrades.

Architecture:
  - BM25Retriever wraps fatat_al_arab.embed.BM25Index.
  - At construction it receives the same chunk list that index.py built, so
    the corpus positions align perfectly with Qdrant payload lookups.
  - retrieve(query, n) returns List[ScoredChunk] sorted by BM25 score.

Why rank-bm25 (not Whoosh or Elasticsearch): zero-dependency, pure-Python,
sufficient for 1,502 × 3 levels = ~4,500 chunks at demo scale.
"""

from __future__ import annotations

import numpy as np

from fatat_al_arab.embed import BM25Index
from fatat_al_arab.rrf import ScoredChunk


class BM25Retriever:
    """
    BM25 sparse retriever over a pre-built chunk corpus.

    Usage:
        retriever = BM25Retriever(chunks)  # chunks: list[ScoredChunk proto]
        results   = retriever.retrieve("شعر ابن هزاني", n=10)
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        """
        Args:
            chunks: List of ScoredChunk objects — the full chunk corpus.
                    Positions must be stable (BM25Index uses position-based lookup).
        """
        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._index = BM25Index(texts)

    def retrieve(self, query: str, n: int = 10) -> list[ScoredChunk]:
        """
        Return top-n chunks ranked by BM25 score, descending.
        The returned ScoredChunk objects have rrf_score set to the raw BM25
        score so they can be passed directly to rrf.fuse().
        """
        top = self._index.get_top_n(query, n=n)
        results: list[ScoredChunk] = []
        for idx, score in top:
            if score <= 0.0:
                continue
            c = self._chunks[idx]
            results.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    rrf_score=score,        # raw BM25 score carried as rrf_score
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
