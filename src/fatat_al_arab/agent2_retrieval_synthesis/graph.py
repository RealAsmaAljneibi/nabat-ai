"""
src/fatat_al_arab/agent2_retrieval_synthesis/graph.py
======================================================
Why this file exists: §2.5 Agent 2 — wires all ten stages into a LangGraph
StateGraph with three conditional loops.
When triggered: At process start (compiled once); invoked by orchestrator.run_agent2() per turn.
Purpose: LangGraph StateGraph for Stages 4-10 with conditional edges (CRAG re-query, CRAG verdict re-query, Self-RAG retry).

Stage sequence (§2.5):
  Stage 4  → retrieve
  Stage 5  → rrf_fuse
  Stage 6  → resolve_heritage
  Stage 7  → crag_grader
  Stage 8  → synthesise
  Stage 9  → reflect
  Stage 10 → format_variants → END

Three conditional loops:

  Loop A — CRAG re-query (after rrf_fuse, §5 max 1 retry):
    if rrf_top5 empty AND crag_requery_count == 0:
      → crag_requery_prep → retrieve (relax filters, try again)
    else:
      → resolve_heritage

  Loop B — CRAG verdict re-query (after crag_grader, §5 max 1 retry):
    if verdict == "Incorrect" AND crag_requery_count < 1:
      → crag_requery_prep → retrieve (expand query)
    else:
      → synthesise (proceeds, possibly as refusal)

  Loop C — Self-RAG retry (after reflect, §5 max 2 retries):
    if verdict == "retry" AND self_rag_retries < 2:
      → synthesise (with critique from reflect node)
    else:
      → format_variants

Architecture refs: §2.5 (Agent 2 stages 4-10), §5 (failure budgets and retry caps).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END

    from fatat_al_arab.state import AgentState
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import retrieve_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.resolve_heritage import resolve_heritage_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.crag_grader import crag_grader_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.synthesise import synthesise_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.reflect import reflect_node, SELF_RAG_MAX_RETRIES
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.format_variants import format_variants_node

    # ── Conditional edge helpers ──────────────────────────────────────────────

    def _after_rrf_fuse(state: AgentState) -> str:
        """
        Loop A: if fused results empty and haven't retried yet → re-query.
        Otherwise → resolve_heritage.
        """
        rrf_top      = state.get("rrf_top5") or []
        requery_count = int(state.get("crag_requery_count") or 0)
        if not rrf_top and requery_count < 1:
            logger.info(
                "agent2_graph: rrf_top5 empty, requery_count=%d — Loop A re-query.",
                requery_count,
            )
            return "crag_requery_prep"
        return "resolve_heritage"

    def _after_crag_grader(state: AgentState) -> str:
        """
        Loop B: if CRAG verdict is Incorrect and haven't retried → re-query.
        Otherwise → synthesise (may emit refusal internally).
        """
        verdict       = state.get("crag_verdict") or "Ambiguous"
        requery_count = int(state.get("crag_requery_count") or 0)
        if verdict == "Incorrect" and requery_count < 1:
            logger.info(
                "agent2_graph: crag_verdict=Incorrect, requery_count=%d — Loop B re-query.",
                requery_count,
            )
            return "crag_requery_prep"
        return "synthesise"

    def _after_reflect(state: AgentState) -> str:
        """
        Loop C: if Self-RAG verdict is 'retry' and under cap → synthesise again.
        Otherwise → format_variants.
        """
        verdict     = state.get("self_rag_verdict") or "pass"
        retry_count = int(state.get("self_rag_retries") or 0)
        if verdict == "retry" and retry_count < SELF_RAG_MAX_RETRIES:
            logger.info(
                "agent2_graph: self_rag_verdict=retry (%d/%d) — Loop C retry.",
                retry_count, SELF_RAG_MAX_RETRIES,
            )
            return "synthesise"
        return "format_variants"

    def _before_requery_retrieve(state: AgentState) -> AgentState:
        """
        Shared prep node for Loops A and B:
        increment crag_requery_count and relax hard filters.
        """
        qc = dict(state.get("query_context") or {})
        qc["filters_hard"] = {}   # relax hard filters
        count = int(state.get("crag_requery_count") or 0)
        return {**state, "query_context": qc, "crag_requery_count": count + 1}

    # ── Graph construction ────────────────────────────────────────────────────

    def build_agent2_graph() -> StateGraph:
        """
        Build and compile the full Agent 2 LangGraph graph for stages 4-10.
        Returns a compiled graph — call .invoke(state) to run it.
        """
        graph = StateGraph(AgentState)

        # ── Nodes ─────────────────────────────────────────────────────────────
        graph.add_node("retrieve",          retrieve_node)
        graph.add_node("rrf_fuse",          rrf_fuse_node)
        graph.add_node("crag_requery_prep", _before_requery_retrieve)
        graph.add_node("resolve_heritage",  resolve_heritage_node)
        graph.add_node("crag_grader",       crag_grader_node)
        graph.add_node("synthesise",        synthesise_node)
        graph.add_node("reflect",           reflect_node)
        graph.add_node("format_variants",   format_variants_node)

        # ── Entry ─────────────────────────────────────────────────────────────
        graph.set_entry_point("retrieve")

        # ── Linear edges ──────────────────────────────────────────────────────
        graph.add_edge("retrieve",         "rrf_fuse")
        graph.add_edge("crag_requery_prep","retrieve")   # shared re-entry for Loops A+B
        graph.add_edge("resolve_heritage", "crag_grader")
        graph.add_edge("synthesise",       "reflect")
        graph.add_edge("format_variants",  END)

        # ── Conditional edges ─────────────────────────────────────────────────

        # Loop A: rrf_fuse → resolve_heritage  OR  crag_requery_prep
        graph.add_conditional_edges(
            "rrf_fuse",
            _after_rrf_fuse,
            {
                "resolve_heritage": "resolve_heritage",
                "crag_requery_prep": "crag_requery_prep",
            },
        )

        # Loop B: crag_grader → synthesise  OR  crag_requery_prep
        graph.add_conditional_edges(
            "crag_grader",
            _after_crag_grader,
            {
                "synthesise":       "synthesise",
                "crag_requery_prep": "crag_requery_prep",
            },
        )

        # Loop C: reflect → synthesise  OR  format_variants
        graph.add_conditional_edges(
            "reflect",
            _after_reflect,
            {
                "synthesise":      "synthesise",
                "format_variants": "format_variants",
            },
        )

        return graph.compile()

    _AGENT2_GRAPH = None   # lazy singleton

    def get_agent2_graph():
        """Return the compiled Agent 2 graph (built once per process)."""
        global _AGENT2_GRAPH
        if _AGENT2_GRAPH is None:
            _AGENT2_GRAPH = build_agent2_graph()
        return _AGENT2_GRAPH

    LANGGRAPH_AVAILABLE = True

except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning(
        "agent2_graph: langgraph not installed — graph unavailable. "
        "Nodes can still be called directly for testing."
    )

    def get_agent2_graph():
        raise ImportError(
            "langgraph is not installed. Install with: pip install langgraph"
        )

    def build_agent2_graph():
        raise ImportError(
            "langgraph is not installed. Install with: pip install langgraph"
        )
