"""
agent2_retrieval_synthesis/nodes/retrieve.py
=============================================
Why this node exists: §2.5 Stage 4 — the first Agent 2 node. It takes
QueryContext from Agent 1 and dispatches three independent retrieval calls
(BM25, Dense, ColBERT stub), honouring any hard filters from Stage 3
self-query so irrelevant manuscript volumes are never surfaced.

Design decisions:
  1. Index singleton — the IndexBundle is loaded once per process and cached
     at module level. Re-loading the 13 MB embedding matrix per query would
     add ~200 ms on every call.

  2. Hard filters — BM25 uses post-hoc filtering on its top-50 results so the
     full-corpus BM25Retriever can be cached (see bm25.get_bm25_retriever).
     Dense still pre-filters because it needs a sub-matrix slice aligned to
     the filtered chunk list; rebuilding the dense retriever per query is cheap.

  3. §5 failure budget — each retriever call is wrapped in try/except. A timed-
     out or crashed retriever is logged as dropped; the other two still contribute
     to the RRF fusion step. State key `retrieval_dropped_retrievers` tracks this.

  4. Query text priority: if hyde_passage is present (Stage 2 HyDE) we use it
     for dense retrieval (it's a hypothetical verse that matches the style of
     what we're looking for). BM25 always uses the raw Arabic query terms.

  5. M5b parallel retrieval — BM25, Dense, and ColBERT are independent so they
     run concurrently via ThreadPoolExecutor(max_workers=3). Wall-clock latency
     drops from sum(t_bm25 + t_dense + t_colbert) to max(t_bm25, t_dense, t_colbert).
     Each callable returns (results, timing_ms) and catches its own exceptions so
     the §5 failure-budget semantics are preserved.

Architecture refs: §2.5 Stage 4 (Triple Hybrid Retrieval), §5 (failure budgets).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fatat_al_arab.index import IndexBundle, load_index, DEFAULT_QDRANT
from fatat_al_arab.rrf import ScoredChunk
from fatat_al_arab.state import AgentState

logger = logging.getLogger(__name__)

# ── Module-level index singleton ──────────────────────────────────────────────
# Why module-level: LangGraph re-enters this node for each call; re-loading the
# embedding matrix every time would dominate latency. The singleton is safe
# because IndexBundle is read-only after construction.

_INDEX: IndexBundle | None = None


def _get_index() -> IndexBundle:
    """Load the index once; return cached copy on subsequent calls."""
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index(DEFAULT_QDRANT)
    return _INDEX


def set_index(bundle: IndexBundle) -> None:
    """
    Inject a pre-built IndexBundle — used in tests to avoid loading from disk.
    Must be called before retrieve_node runs.
    """
    global _INDEX
    _INDEX = bundle


def _apply_hard_filters(
    chunks: list[ScoredChunk],
    filters: dict,
) -> list[ScoredChunk]:
    """
    Filter the chunk list to only those matching ALL hard filters.
    Hard filters come from self_query Stage 3; confidence ≥ 0.7.

    Supported filter keys:
      manuscript_short_key: str       — exact match
      source_volume:        str       — exact match
      poet_name:            str       — substring match (normalised)
      level:                str       — "verse" | "group" | "poem"
      genre:                str       — exact match on genre label (M3, silver baseline)
      emotions_any:         list[str] — at least one emotion must appear in chunk.emotions (M3)

    Why genre uses exact match: the taxonomy is a closed vocabulary (10 labels from
    nabati_taxonomy.py); partial matching would confuse غزل and غزو.

    Why emotions_any instead of emotions_all: the user says "poems with grief" —
    they want any poem that carries grief, not only poems tagged with grief AND nothing
    else. The silver baseline tags at most 3 emotions per verse so an AND would be
    very restrictive.

    Safety net: if the genre/emotion filter drops ALL results, we fall back to unfiltered
    (same §5 pattern used for manuscript/poet filters).
    """
    if not filters:
        return chunks

    result = chunks
    if ms_key := filters.get("manuscript_short_key"):
        result = [c for c in result if c.manuscript_short_key == ms_key]
    if vol := filters.get("source_volume"):
        result = [c for c in result if c.source_volume == vol]
    if poet := filters.get("poet_name"):
        poet_lower = poet.lower()
        result = [c for c in result if poet_lower in c.poet_name.lower()]
    if level := filters.get("level"):
        result = [c for c in result if c.level == level]

    # M3: genre hard filter
    if genre := filters.get("genre"):
        genre_filtered = [c for c in result if c.genre == genre]
        # §5 safety: if genre filter eliminates everything, skip it
        if genre_filtered:
            result = genre_filtered
        else:
            logger.warning(
                "_apply_hard_filters: genre filter '%s' matched 0 chunks — skipping.", genre
            )

    # M3: emotion any-match filter
    if emotions_any := filters.get("emotions_any"):
        emotions_set = set(emotions_any)
        emotion_filtered = [c for c in result if emotions_set.intersection(c.emotions)]
        if emotion_filtered:
            result = emotion_filtered
        else:
            logger.warning(
                "_apply_hard_filters: emotions_any filter %s matched 0 chunks — skipping.",
                emotions_any,
            )

    return result


def _chunks_to_dicts(chunks: list[ScoredChunk]) -> list[dict]:
    """Serialise ScoredChunk list to plain dicts for AgentState storage."""
    return [c.to_dict() for c in chunks]


def retrieve_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 4.

    Reads:
      state["query_context"]  — for query text, HyDE passage, hard/soft filters
    Writes:
      state["bm25_results"]          — list[dict] ranked by BM25 score
      state["dense_results"]         — list[dict] ranked by cosine sim
      state["colbert_results"]       — list[dict] (empty stub)
      state["retriever_timings"]     — {bm25_ms, dense_ms, colbert_ms}
      state["retrieval_dropped_retrievers"] — names of failed retrievers
    """
    qc = state.get("query_context") or {}
    query_ar: str = qc.get("query_ar") or state.get("raw_query", "")
    hyde_passage: str | None = qc.get("hyde_passage")
    filters_hard: dict = qc.get("filters_hard") or {}

    # On CRAG re-query, blend the LLM's strategy into the BM25 query so it
    # searches for the specific terms the grader identified as missing.
    requery_count = int(state.get("crag_requery_count") or 0)
    crag_strategy = (state.get("crag_requery_strategy") or "").strip()
    if requery_count > 0 and crag_strategy:
        query_ar = f"{query_ar} {crag_strategy}"
        logger.info("retrieve_node: re-query #%d — augmented with CRAG strategy.", requery_count)

    dropped: list[str] = list(state.get("retrieval_dropped_retrievers") or [])
    timings: dict = {}
    bm25_results:    list[dict] = []
    dense_results:   list[dict] = []
    colbert_results: list[dict] = []

    # Build a BM25 query that combines the raw Arabic query, any Arabic
    # paraphrases from bilingual_expand, and the HyDE hypothetical verse.
    # Why: BM25 is a bag-of-words model — more Arabic surface forms = better
    # token coverage over the Nabati lexicon without changing the retriever.
    ar_variants: list[str] = list(qc.get("query_variants_ar") or [])[:3]
    hyde_snippet = (hyde_passage or "")[:400]  # cap so BM25 tokeniser stays fast
    bm25_query = " ".join(filter(None, [query_ar] + ar_variants + [hyde_snippet]))

    try:
        bundle = _get_index()
        all_chunks = bundle.chunks

        # Apply hard filters to the full chunk corpus before scoring
        filtered_chunks = _apply_hard_filters(all_chunks, filters_hard)
        if not filtered_chunks:
            # Hard filter eliminated everything — fall back to unfiltered
            logger.warning(
                "retrieve_node: hard filters %s eliminated all chunks; "
                "falling back to unfiltered corpus.", filters_hard
            )
            filtered_chunks = all_chunks

        # Pre-compute the filtered embedding sub-matrix here (sequential; needs
        # bundle.embeddings which is not thread-safe to slice concurrently).
        import numpy as np
        chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(bundle.chunks)}
        filtered_indices = [
            chunk_id_to_idx[c.chunk_id]
            for c in filtered_chunks
            if c.chunk_id in chunk_id_to_idx
        ]
        filtered_embeddings = bundle.embeddings[filtered_indices]
        # Use HyDE passage for dense if available (richer semantic signal)
        dense_query = hyde_passage or query_ar

        # ── M5b: parallel retrieval callables ────────────────────────────────
        # The three retrieval legs are independent; running them concurrently
        # reduces wall-clock latency from sum to max of their individual times.
        # Each callable returns (results: list[dict], timing_ms: float) and
        # raises on hard failure so the caller can mark the retriever as dropped.

        def _run_bm25() -> tuple[list[dict], float]:
            from fatat_al_arab.retrievers.bm25 import get_bm25_retriever
            t0 = time.perf_counter()
            retriever = get_bm25_retriever(bundle.chunks)
            # Fetch 50 from the full corpus; apply hard filters post-hoc so
            # the full-corpus retriever can be reused across queries.
            raw = retriever.retrieve(bm25_query, n=50)
            chunks = _apply_hard_filters(raw, filters_hard)[:20]
            return _chunks_to_dicts(chunks), round((time.perf_counter() - t0) * 1000, 1)

        def _run_dense() -> tuple[list[dict], float]:
            from fatat_al_arab.retrievers.dense import DenseRetriever
            t0 = time.perf_counter()
            chunks = DenseRetriever(filtered_chunks, filtered_embeddings).retrieve(dense_query, n=20)
            return _chunks_to_dicts(chunks), round((time.perf_counter() - t0) * 1000, 1)

        def _run_colbert() -> tuple[list[dict], float]:
            from fatat_al_arab.retrievers.colbert import ColBERTRetriever
            t0 = time.perf_counter()
            chunks = ColBERTRetriever(filtered_chunks).retrieve(query_ar, n=20)
            return _chunks_to_dicts(chunks), round((time.perf_counter() - t0) * 1000, 1)

        _jobs = {"bm25": _run_bm25, "dense": _run_dense, "colbert": _run_colbert}
        with ThreadPoolExecutor(max_workers=3) as _pool:
            _futures = {_pool.submit(fn): name for name, fn in _jobs.items()}
            for fut in as_completed(_futures):
                name = _futures[fut]
                try:
                    results, elapsed = fut.result()
                    timings[f"{name}_ms"] = elapsed
                    if name == "bm25":
                        bm25_results = results
                    elif name == "dense":
                        dense_results = results
                    else:
                        colbert_results = results
                except Exception as exc:
                    logger.warning("retrieve_node: %s failed: %s", name, exc)
                    dropped.append(name)
                    timings[f"{name}_ms"] = -1

    except Exception as exc:
        logger.error("retrieve_node: index load failed: %s", exc)
        dropped.append("all")

    from fatat_al_arab.state import trace_append
    _n = lambda r: len(r)
    trace_summary = (
        f"BM25: {_n(bm25_results)} · Dense: {_n(dense_results)} · ColBERT: {_n(colbert_results)}"
        + (f" (dropped: {', '.join(dropped)})" if dropped else "")
    )
    return {
        **state,
        "bm25_results":    bm25_results,
        "dense_results":   dense_results,
        "colbert_results": colbert_results,
        "retriever_timings": timings,
        "retrieval_dropped_retrievers": dropped,
        "agent_trace": trace_append(state, stage="4", icon="📡", label="Triple Retrieval", summary=trace_summary),
    }
