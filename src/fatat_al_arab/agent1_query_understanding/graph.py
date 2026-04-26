"""
agent1_query_understanding/graph.py
=====================================
Why this file exists: §2.4 commits to a LangGraph-wired Agent 1. The graph is the
executable implementation of the §2.4 stage sequence. Having it in one file means
a reviewer can open §2.4 and this file side-by-side and trace every node → edge →
condition to the architecture document.

Stage sequence (§2.4):
  Stage 1 → bilingual_analyzer
  Stage 2 → bilingual_expand  (paraphrases)
  Stage 2 → hyde              (hypothetical verse embedding)
  Stage 3 → self_query        (typed filter extraction)

Conditional edges:
  After bilingual_analyzer: if clarification needed → END (return clarification_question)
  After all stages: always → END

Agent 1 tool restriction (§2.9): only translate_query, extract_filters,
hyde_passage, expand_bilingual are available. The graph does not expose any
retrieval tool — the tool registry in tools.py enforces this.

Architecture refs: §2.4 (Agent 1 stages 1-3), §2.6 (state contract),
§2.9 (Agent 1 tool registry), §5 (failure budgets in each node).
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from ..state import AgentState
from .nodes.bilingual_analyzer import bilingual_analyzer_node
from .nodes.bilingual_expand import bilingual_expand_node
from .nodes.hyde import hyde_node
from .nodes.self_query import self_query_node
from .nodes.intent_router import intent_router_node
from .nodes.semantic_router import semantic_router_node

logger = logging.getLogger(__name__)

# ── Node names (used as string keys in StateGraph) ────────────────────────────
# Why constants: using string literals directly in add_edge/add_conditional_edges
# calls leads to silent bugs when a node is renamed. Constants make typos a
# NameError instead.

NODE_ROUTER   = "intent_router"       # Stage 0.5a — deterministic fast path
NODE_SEMANTIC = "semantic_router"     # Stage 0.5b — LLM-backed 4-track classifier
NODE_ANALYZE  = "bilingual_analyzer"
NODE_EXPAND   = "bilingual_expand"
NODE_HYDE     = "hyde"
NODE_QUERY    = "self_query"


# ── Conditional edge functions ────────────────────────────────────────────────

def _after_router(state: AgentState) -> Literal["semantic_router", "__end__"]:
    """
    Why a conditional here: Stage 0.5a (intent_router) may flag the query as a
    deterministic counting / metadata / unsupported-dim question, in which case
    the answer comes from deterministic_answer_node in Agent 2. When that
    happens we short-circuit — no LLM tokens needed. Otherwise we fall through
    to Stage 0.5b (semantic_router) for capabilities/instructor_debug detection.
    """
    qc = state.get("query_context") or {}
    if qc.get("answer_source") == "registry_lookup":
        logger.info("agent1 graph: intent_router matched — skipping Stages 0.5b+1-3.")
        return END  # type: ignore[return-value]
    return NODE_SEMANTIC


def _after_semantic(state: AgentState) -> Literal["bilingual_analyzer", "__end__"]:
    """
    Why a conditional here: Stage 0.5b (semantic_router) catches capabilities /
    instructor_debug / registry_lookup queries that regex can't detect. When it
    short-circuits (answer_source="registry_lookup") we skip the bilingual
    analyzer and HyDE — all of which cost LLM tokens we don't need.
    """
    qc = state.get("query_context") or {}
    if qc.get("answer_source") == "registry_lookup":
        logger.info(
            "agent1 graph: semantic_router short-circuited — track=%s, skipping Stages 1-3.",
            qc.get("track", "?"),
        )
        return END  # type: ignore[return-value]
    return NODE_ANALYZE


def _after_analyze(state: AgentState) -> Literal["bilingual_expand", "__end__"]:
    """
    Why a conditional here: if the query is genuinely ambiguous (intent_confidence
    < 0.5 → needs_clarification=True), we short-circuit to END and surface the
    clarification question to the UI. Continuing into Stage 2/3 on an unclear query
    wastes up to 3 additional LLM calls and produces noisy filters.

    The orchestrator reads state["query_context"]["needs_clarification"] and routes
    the response to the Streamlit chat input rather than Agent 2.
    """
    qc = state.get("query_context", {})
    if qc.get("needs_clarification"):
        logger.debug("agent1 graph: clarification needed — routing to END.")
        return END  # type: ignore[return-value]
    return NODE_EXPAND


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent1_graph() -> StateGraph:
    """
    Construct and compile the Agent 1 LangGraph.

    Why returning the compiled graph (not just the builder): callers import
    `agent1_graph` (the compiled object) and call `.invoke()` on it directly.
    Building at module load means the graph topology is validated at import time,
    not at first query — easier to catch wiring bugs.
    """
    builder = StateGraph(AgentState)

    # Add nodes in stage order
    builder.add_node(NODE_ROUTER,   intent_router_node)
    builder.add_node(NODE_SEMANTIC, semantic_router_node)
    builder.add_node(NODE_ANALYZE,  bilingual_analyzer_node)
    builder.add_node(NODE_EXPAND,   bilingual_expand_node)
    builder.add_node(NODE_HYDE,     hyde_node)
    builder.add_node(NODE_QUERY,    self_query_node)

    # Entry point — Stage 0.5a (regex router) runs first
    builder.set_entry_point(NODE_ROUTER)

    # Conditional: after Stage 0.5a check for deterministic intent
    builder.add_conditional_edges(
        NODE_ROUTER,
        _after_router,
        {
            NODE_SEMANTIC: NODE_SEMANTIC,
            END:           END,
        },
    )

    # Conditional: after Stage 0.5b check for LLM-detected non-poetic track
    builder.add_conditional_edges(
        NODE_SEMANTIC,
        _after_semantic,
        {
            NODE_ANALYZE: NODE_ANALYZE,
            END:          END,
        },
    )

    # Conditional: after Stage 1 check for clarification
    builder.add_conditional_edges(
        NODE_ANALYZE,
        _after_analyze,
        {
            NODE_EXPAND: NODE_EXPAND,
            END:         END,
        },
    )

    # Stage 2 (expand then hyde) — sequential, no condition
    builder.add_edge(NODE_EXPAND, NODE_HYDE)

    # Stage 2 → Stage 3
    builder.add_edge(NODE_HYDE, NODE_QUERY)

    # Stage 3 → END
    builder.add_edge(NODE_QUERY, END)

    return builder.compile()


# ── Module-level compiled graph ───────────────────────────────────────────────
# Why module-level: the orchestrator imports this object directly so it can call
# `agent1_graph.invoke(state)` without rebuilding the graph on every query.

agent1_graph = build_agent1_graph()


def run(raw_query: str, input_image_path: str | None = None) -> AgentState:
    """
    Convenience entry point for tests and the orchestrator.

    Args:
        raw_query:         User's question text (already OCR'd if from an image).
        input_image_path:  Path to the source image, stored in state for the UI.

    Returns:
        Final AgentState after all Agent 1 stages have run.
    """
    from ..state import make_agent_state

    initial_state = make_agent_state(raw_query)
    if input_image_path is not None:
        initial_state["input_image_path"] = input_image_path

    result: AgentState = agent1_graph.invoke(initial_state)
    return result
