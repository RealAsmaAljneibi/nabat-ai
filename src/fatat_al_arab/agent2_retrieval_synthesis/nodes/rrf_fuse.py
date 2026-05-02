"""
agent2_retrieval_synthesis/nodes/rrf_fuse.py
=============================================
Why this node exists: §2.5 Stage 5 — merges the three independent ranked lists
from Stage 4 into one definitive top-N ranking via Reciprocal Rank Fusion.

Two things happen here that don't happen in rrf.fuse() itself:

  1. Soft filters (score boost) — Stage 3 self_query may extract low-confidence
     filters (e.g., "probably about Ibn Yahya but not certain"). These become
     a multiplicative boost on RRF score rather than a hard cut. A chunk whose
     poet_name matches the soft filter gets a 1.5× boost; mismatches are
     unchanged. Why boost-not-filter: soft filters should bias ranking, not
     eliminate candidates — the user might not have named the poet correctly.

  2. De-duplication across levels — the same anchor entry appears at verse,
     group, and poem level. After fusion, if the same anchor_id appears in
     multiple chunk levels, we keep only the top-scoring level. This prevents
     the LLM synthesis stage from seeing three copies of the same content.
     Exception: if intent == "interpretive" we keep all levels so the
     synthesiser can compare verse vs. stanza-level context.

Architecture refs: §2.5 Stage 5 (RRF Fusion), §2.4 Stage 3 (self-query filters).
"""

from __future__ import annotations

import logging
from typing import Any

from fatat_al_arab.rrf import ScoredChunk, fuse
from fatat_al_arab.state import AgentState

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RRF_K         = 60      # RRF damping constant (Cormack et al. 2009)
TOP_N_FUSED   = 20      # top results to carry forward to Stage 6
SOFT_BOOST    = 1.5     # multiplicative score boost when soft filter matches

# Source-tier weights applied after fusion.
# Why: reference corpus (scholarly PDFs) and MAAI7103 oral bayts score highly
# on BM25/dense for thematic queries but should not outrank primary manuscript
# verse content. These multipliers ensure primary manuscripts always rank first
# unless the corpus genuinely has no relevant content.
SOURCE_WEIGHT_REFERENCE  = 0.35  # scholarly PDFs — context only, not primary verse
SOURCE_WEIGHT_SECONDARY  = 0.65  # MAAI7103 oral / online secondary sources
SOURCE_WEIGHT_PRIMARY    = 1.0   # primary manuscripts — no change


def _dicts_to_chunks(raw: list[dict]) -> list[ScoredChunk]:
    """Reconstruct ScoredChunk objects from the serialised AgentState dicts."""
    chunks = []
    for d in raw:
        if not d.get("chunk_id"):
            continue
        chunks.append(ScoredChunk(
            chunk_id=d["chunk_id"],
            rrf_score=float(d.get("rrf_score", 0.0)),
            text=d.get("text", ""),
            level=d.get("level", "verse"),
            anchor_id=d.get("anchor_id", ""),
            poet_name=d.get("poet_name", ""),
            source_volume=d.get("source_volume", ""),
            source_page=str(d.get("source_page", "")),
            source_image_path=d.get("source_image_path", ""),
            manuscript_short_key=d.get("manuscript_short_key", ""),
            source_type=d.get("source_type", ""),
            data_tier=d.get("data_tier", ""),
            is_secondary_source=bool(d.get("is_secondary_source", False)),
            parent_poem_id=d.get("parent_poem_id", ""),
            poem_matla=d.get("poem_matla", ""),
        ))
    return chunks


def _apply_soft_boost(
    chunks: list[ScoredChunk],
    filters: dict,
) -> list[ScoredChunk]:
    """
    Apply multiplicative score boost for soft-filter matches.
    Returns a new list (doesn't mutate originals) sorted descending.
    """
    if not filters:
        return chunks

    poet_soft = (filters.get("poet_name") or "").lower()
    vol_soft  = filters.get("source_volume") or ""
    ms_soft   = filters.get("manuscript_short_key") or ""

    boosted = []
    for c in chunks:
        multiplier = 1.0
        if poet_soft and poet_soft in c.poet_name.lower():
            multiplier *= SOFT_BOOST
        if vol_soft and c.source_volume == vol_soft:
            multiplier *= SOFT_BOOST
        if ms_soft and c.manuscript_short_key == ms_soft:
            multiplier *= SOFT_BOOST
        boosted.append(
            ScoredChunk(
                chunk_id=c.chunk_id,
                rrf_score=c.rrf_score * multiplier,
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
    return sorted(boosted, key=lambda c: (-c.rrf_score, c.chunk_id))


def _apply_source_weights(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """
    Downweight reference corpus and secondary-source chunks after RRF fusion.

    Why: reference PDFs (reference_*) and MAAI7103 oral bayts achieve high BM25
    and dense scores on thematic queries because they share Nabati vocabulary, but
    they should not crowd out primary manuscript verse chunks which are what the
    scholar actually wants. The multipliers ensure primary content always ranks
    first; secondary/reference content only surfaces when primary corpus has
    no relevant match.

    Detected by chunk_id prefix because is_secondary_source and source_type
    are not reliably populated in the current index (index rebuild needed).
    This function is safe to run even after rebuild — prefix detection is
    redundant once the flags are correct, but still correct.
    """
    weighted = []
    for c in chunks:
        cid = c.chunk_id
        if cid.startswith("reference_") or c.level == "reference":
            w = SOURCE_WEIGHT_REFERENCE
        elif (
            cid.startswith("maai7103_")
            or cid.startswith("aldiwan_")
            or cid.startswith("4byt_")
            or cid.startswith("ecssr_")
            or c.is_secondary_source
            or c.source_type in ("oral", "online", "secondary", "online_digitized")
        ):
            w = SOURCE_WEIGHT_SECONDARY
        else:
            w = SOURCE_WEIGHT_PRIMARY

        if w == 1.0:
            weighted.append(c)
        else:
            weighted.append(ScoredChunk(
                chunk_id=c.chunk_id,
                rrf_score=c.rrf_score * w,
                text=c.text,
                level=c.level,
                anchor_id=c.anchor_id,
                poet_name=c.poet_name,
                source_volume=c.source_volume,
                source_page=c.source_page,
                source_image_path=c.source_image_path,
                manuscript_short_key=c.manuscript_short_key,
                source_type=c.source_type,
                data_tier=c.data_tier,
                is_secondary_source=c.is_secondary_source,
                parent_poem_id=c.parent_poem_id,
                poem_matla=c.poem_matla,
                extra=c.extra,
            ))
    return sorted(weighted, key=lambda c: (-c.rrf_score, c.chunk_id))


def _dedup_by_anchor(
    chunks: list[ScoredChunk],
    keep_all_levels: bool = False,
) -> list[ScoredChunk]:
    """
    Keep only the top-scoring chunk per anchor_id unless keep_all_levels=True.
    Why: multiple chunk levels for the same anchor would present duplicate
    content to the LLM synthesiser. Only interpretive queries need all levels.
    """
    if keep_all_levels:
        return chunks

    seen: dict[str, ScoredChunk] = {}
    for c in chunks:
        aid = c.anchor_id
        if aid not in seen or c.rrf_score > seen[aid].rrf_score:
            seen[aid] = c
    # Re-sort after dedup
    return sorted(seen.values(), key=lambda c: (-c.rrf_score, c.chunk_id))


def rrf_fuse_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 5.

    Reads:
      state["bm25_results"], ["dense_results"], ["colbert_results"]
      state["query_context"]  — for soft filters and intent
    Writes:
      state["rrf_top5"]   — top-N fused + deduped chunks as list[dict]
                            (named rrf_top5 in state schema for historical reasons
                             but now carries TOP_N_FUSED results)
    """
    qc = state.get("query_context") or {}
    filters_soft:   dict = qc.get("filters_soft") or {}
    detected_intent: str = qc.get("detected_intent") or "semantic"

    # Reconstruct ScoredChunk lists from state dicts
    bm25_chunks    = _dicts_to_chunks(state.get("bm25_results")    or [])
    dense_chunks   = _dicts_to_chunks(state.get("dense_results")   or [])
    colbert_chunks = _dicts_to_chunks(state.get("colbert_results") or [])

    if not bm25_chunks and not dense_chunks and not colbert_chunks:
        logger.warning("rrf_fuse_node: all retriever results are empty.")
        return {**state, "rrf_top5": []}

    # RRF fusion
    fused = fuse(
        [bm25_chunks, dense_chunks, colbert_chunks],
        k=RRF_K,
        top_n=TOP_N_FUSED * 2,   # 2× headroom before soft boost + dedup
    )

    # Soft filter boost
    if filters_soft:
        fused = _apply_soft_boost(fused, filters_soft)

    # Source-tier downweighting: reference PDFs and secondary oral/online sources
    # must not outrank primary manuscript verse content
    fused = _apply_source_weights(fused)

    # De-duplicate across chunk levels (keep all for interpretive queries)
    keep_all = (detected_intent == "interpretive")
    fused = _dedup_by_anchor(fused, keep_all_levels=keep_all)

    # Final top-N cut
    top = fused[:TOP_N_FUSED]

    logger.debug(
        "rrf_fuse_node: %d bm25 + %d dense + %d colbert → %d fused → %d after dedup/cut",
        len(bm25_chunks), len(dense_chunks), len(colbert_chunks), len(fused), len(top),
    )

    from fatat_al_arab.state import trace_append
    trace_summary = f"{len(top)} unique chunks ranked (k={RRF_K}, from {len(fused)} fused)"
    return {
        **state,
        "rrf_top5": [c.to_dict() for c in top],
        "agent_trace": trace_append(state, stage="5", icon="🔀", label="RRF Fusion", summary=trace_summary),
    }
