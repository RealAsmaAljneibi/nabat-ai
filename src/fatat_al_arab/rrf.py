"""
src/fatat_al_arab/rrf.py
========================
Why this file exists: M3 Stage 3 — Reciprocal Rank Fusion (RRF) merges
independent ranked lists from BM25, dense, and ColBERT retrievers into a
single ranked list without requiring score calibration between retrievers.

Formula: RRF(d) = Σ_r  1 / (k + rank_r(d))
  k=60 is the standard constant from Cormack et al. 2009. It dampens the
  advantage of the top position — a rank-1 result gets 1/61 ≈ 0.016, rank-2
  gets 1/62 ≈ 0.016, so a document that consistently places mid-range across
  all three retrievers can outscore a document that only tops one list.

Why k=60 (not 5 or 100): the original Cormack paper found k=60 robust across
TREC tasks. Arabic poetry retrieval needs the same damping because BM25 will
dominate when an exact poet name is in the query, and we want dense to still
contribute.

Public API:
  fuse(ranked_lists, k=60) -> list[ScoredChunk]

ScoredChunk is a simple dict-like container so callers don't need to import
a heavyweight type from qdrant-client just to do fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredChunk:
    """
    A retrieval result with enough metadata for citation injection.

    Fields mirror the Qdrant payload we write in index.py plus the fused score.

    M3 additions: genre and emotions come from the M2d silver-baseline enrichment
    (heuristic_v1 classifier). They are stored on the chunk so that:
      - retrieve.py can hard-filter on genre without touching the registry again
      - synthesise.py can badge the response with 🔸 when genre_source != "human"
      - the Streamlit facet sidebar can filter results client-side
    """
    chunk_id:    str              # globally unique: "{anchor_id}__{level}__{seq}"
    rrf_score:   float            # fused RRF score (higher = more relevant)
    text:        str              # the chunk text that was retrieved
    level:       str              # "verse" | "group" | "poem"
    anchor_id:   str              # joins back to anchor_registry_phase4.json
    poet_name:   str   = ""
    source_volume: str = ""
    source_page:   int = 0
    source_image_path: str = ""
    manuscript_short_key: str = ""
    # M3: genre + emotion fields (silver baseline — may be "غير_محدد" if classifier abstained)
    genre:             str        = "غير_محدد"   # e.g. "غزل", "رثاء", "غير_محدد"
    genre_confidence:  float      = 0.0          # 0.0–1.0 margin score from heuristic
    genre_source:      str        = ""           # "heuristic_v1" | "human" | ""
    emotions:          list[str]  = field(default_factory=list)  # e.g. ["longing", "grief"]
    extra:       dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id":           self.chunk_id,
            "rrf_score":          self.rrf_score,
            "text":               self.text,
            "level":              self.level,
            "anchor_id":          self.anchor_id,
            "poet_name":          self.poet_name,
            "source_volume":      self.source_volume,
            "source_page":        self.source_page,
            "source_image_path":  self.source_image_path,
            "manuscript_short_key": self.manuscript_short_key,
            "genre":              self.genre,
            "genre_confidence":   self.genre_confidence,
            "genre_source":       self.genre_source,
            "emotions":           self.emotions,
            **self.extra,
        }


def fuse(
    ranked_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_n: int | None = None,
) -> list[ScoredChunk]:
    """
    Fuse multiple independently ranked lists with RRF.

    Args:
        ranked_lists:  Each inner list is already sorted descending by its own
                       retriever score. The position in the list is the rank.
        k:             RRF damping constant (default 60).
        top_n:         If given, return only the top *top_n* results.

    Returns:
        Merged list of ScoredChunk, sorted descending by rrf_score.
        Each returned chunk has rrf_score set to the fused value.
        All other fields come from the first ranked_list that contained
        that chunk_id (metadata is assumed identical across retrievers).
    """
    if not ranked_lists:
        return []

    # Accumulate RRF scores and keep a reference chunk per id
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, ScoredChunk] = {}

    for ranked in ranked_lists:
        for rank_zero, chunk in enumerate(ranked):
            rank_one = rank_zero + 1          # 1-based
            score_contribution = 1.0 / (k + rank_one)
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution
            if cid not in chunk_store:
                chunk_store[cid] = chunk

    # Sort by fused score descending, then by chunk_id for deterministic ties
    fused = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))

    result: list[ScoredChunk] = []
    for cid in fused:
        c = chunk_store[cid]
        # Return a new ScoredChunk with rrf_score set (don't mutate the original)
        result.append(
            ScoredChunk(
                chunk_id=c.chunk_id,
                rrf_score=rrf_scores[cid],
                text=c.text,
                level=c.level,
                anchor_id=c.anchor_id,
                poet_name=c.poet_name,
                source_volume=c.source_volume,
                source_page=c.source_page,
                source_image_path=c.source_image_path,
                manuscript_short_key=c.manuscript_short_key,
                genre=c.genre,
                genre_confidence=c.genre_confidence,
                genre_source=c.genre_source,
                emotions=list(c.emotions),
                extra=c.extra,
            )
        )

    if top_n is not None:
        return result[:top_n]
    return result


def scores_only(fused: list[ScoredChunk]) -> list[float]:
    """Convenience: extract rrf_score list (useful in tests)."""
    return [c.rrf_score for c in fused]
