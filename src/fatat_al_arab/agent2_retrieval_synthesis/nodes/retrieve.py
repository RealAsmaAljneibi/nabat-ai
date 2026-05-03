"""
agent2_retrieval_synthesis/nodes/retrieve.py
=============================================
Why this node exists: §2.5 Stage 4 — the first Agent 2 node. It takes
QueryContext from Agent 1 and dispatches three independent retrieval calls
(BM25, Dense, ColBERT stub), honouring any hard filters from Stage 3
self-query so irrelevant manuscript volumes are never surfaced.
When triggered: Stage 4 — first Agent-2 node when track = poetic_rag.
Purpose: Triple hybrid (BM25 + Dense + ColBERT) in parallel; honours preferred_source → source_type filter; demotes speculative LLM genre/emotion/theme filters from hard → soft when router pinned source.

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

    # Source-type filter (set by the router via preferred_source).
    # Accepts a single value or a list.
    #
    # Why STRICT (no §5 fallback like genre/emotions): when the router classifies
    # a query as "Sheikh Zayed poetry" the user explicitly wants the
    # online_digitized corpus — silently leaking manuscript chunks would defeat
    # the whole point of source-aware routing.
    #
    # Empty-source_type tolerance: about 1k chunks in the index have an empty
    # source_type field (data-quality gap from older ingest). When the user wants
    # online_digitized, accept empty too — those chunks are almost all post-1990
    # UAE leadership poems that pre-date the source_type tagging.
    if src_type := filters.get("source_type"):
        wanted = {src_type} if isinstance(src_type, str) else set(src_type)
        if "online_digitized" in wanted:
            wanted = wanted | {"", None}    # tolerate untagged online entries

        def _ok(c) -> bool:
            st = c.source_type
            if st in wanted:
                return True
            # Default empty to manuscript only when caller asked for manuscript
            return (not st) and ("manuscript" in wanted)

        result = [c for c in result if _ok(c)]
        if not result:
            logger.info(
                "_apply_hard_filters: source_type filter %s matched 0 chunks — "
                "this leg returns empty (strict).", wanted,
            )

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
    filters_hard: dict = dict(qc.get("filters_hard") or {})

    # Inject source_type filter from the router's preferred_source verdict.
    # any_corpus  → no source filter (search all); manuscript_corpus / online_corpus
    # → restrict to that subset of the index. Self-query may have already set a
    # source_type — respect that and don't overwrite it.
    preferred = qc.get("preferred_source")
    filters_soft: dict = dict(qc.get("filters_soft") or {})
    if preferred and "source_type" not in filters_hard:
        if preferred == "manuscript_corpus":
            filters_hard["source_type"] = "manuscript"
        elif preferred == "online_corpus":
            filters_hard["source_type"] = "online_digitized"
        # any_corpus / poet_bio / corpus_stats / general_knowledge → no filter
        if filters_hard.get("source_type"):
            logger.info(
                "retrieve_node: preferred_source=%s → source_type=%s",
                preferred, filters_hard["source_type"],
            )
            # When the router has confidently set the source bucket, demote
            # speculative LLM filters (genre/emotion) from hard to soft so they
            # don't over-prune the legitimate hits. Poet-name + source remain hard.
            for speculative in ("genre", "emotions_any", "theme"):
                if speculative in filters_hard:
                    filters_soft[speculative] = filters_hard.pop(speculative)
                    logger.info(
                        "retrieve_node: demoted %s=%r from hard→soft (router set preferred_source).",
                        speculative, filters_soft[speculative],
                    )

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
