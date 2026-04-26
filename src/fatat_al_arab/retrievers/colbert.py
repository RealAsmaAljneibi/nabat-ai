"""
src/fatat_al_arab/retrievers/colbert.py
========================================
Why this file exists: ColBERT late-interaction retrieval is the third leg of
the triple-hybrid pipeline. It computes MaxSim between query token embeddings
and document token embeddings, giving finer-grained token-level matching than
bi-encoder dense retrieval.

Current status: STUB (MVP scope).
Why stubbed: installing pylate or RAGatouille adds ~2 GB of dependencies and
requires GPU memory for efficient MaxSim computation. At demo scale (1,502
anchors) the BM25 + dense combination already provides strong retrieval.
ColBERT is gated behind a feature flag and returns an empty list when called.

Architecture contract (for M5 when it's wired into the graph):
  - retrieve(query, n) -> list[ScoredChunk]
  - same interface as BM25Retriever and DenseRetriever
  - enabled when COLBERT_ENABLED=true in .env

Why keep the stub now: rrf.fuse() takes a variable-length list of ranked
lists, so passing an empty list from this stub has zero overhead and the
fusion logic is identical once ColBERT is turned on.
"""

from __future__ import annotations

import os

from fatat_al_arab.rrf import ScoredChunk

COLBERT_ENABLED = os.getenv("COLBERT_ENABLED", "false").lower() == "true"


class ColBERTRetriever:
    """
    ColBERT late-interaction retriever — stub for MVP.
    Returns an empty list unless COLBERT_ENABLED=true in environment.
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, n: int = 10) -> list[ScoredChunk]:
        """
        Return top-n ColBERT results.
        Currently returns [] — stub until post-MVP ColBERT integration.
        """
        if not COLBERT_ENABLED:
            return []
        # Post-MVP: implement MaxSim via pylate or RAGatouille here.
        raise NotImplementedError(
            "ColBERT retriever is enabled but not yet implemented. "
            "Unset COLBERT_ENABLED or implement MaxSim logic."
        )

    def is_enabled(self) -> bool:
        return COLBERT_ENABLED

    def __len__(self) -> int:
        return len(self._chunks)
