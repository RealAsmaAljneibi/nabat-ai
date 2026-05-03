"""
creative_poet/graph.py
======================================
Why this file exists: wires the four creative agents into a LangGraph StateGraph
with genuine agentic loops — Observe → Reflect → Retry — following the same pattern
as Agent 2's CRAG re-query and Self-RAG retry loops.

Full topology (each path is a named sub-graph within the same StateGraph):

  ── scaffold path (Al-Mulhim) ───────────────────────────────────────────────
  mulhim_retrieve → mulhim_generate → mulhim_validate
                          ↑ (retry once if quality check fails)
                          mulhim_validate → END (quality_ok or retries exhausted)

  ── coauthor path (Al-Musharik) ─────────────────────────────────────────────
  musharik_retrieve → musharik_generate → musharik_quality
                             ↑ (retry once if < AJUZ_CANDIDATE_COUNT valid candidates)
                             musharik_quality → END (quality_ok or retries exhausted)

  ── preserve path (Al-Hafiz) ────────────────────────────────────────────────
  hafiz_nassikh_check
    ├─ in_corpus=True  → hafiz_retrieve → hafiz_generate → enforce_attribution → END
    └─ in_corpus=False → END  (informative "not in corpus" message)

  ── critique path (Al-Muqayyim) ─────────────────────────────────────────────
  muqayyim → END  (single LLM call; critique quality is inherently subjective)

Retry bounds are enforced by named constants in each node file (MAX_SCAFFOLD_RETRIES,
MAX_AJUZ_RETRIES) — the conditional edge functions here read the same counter.

Architecture refs: §2.5 (same StateGraph pattern as Agent 2), personas.py (4 personas),
                   external_tools.py (inter-agent bridges), tools.py (permission sets).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END

    from fatat_al_arab.state import CompositionState
    from creative_poet.nodes.mulhim import (
        mulhim_retrieve_node,
        mulhim_generate_node,
        mulhim_validate_node,
        MAX_SCAFFOLD_RETRIES,
    )
    from creative_poet.nodes.musharik import (
        musharik_retrieve_node,
        musharik_generate_node,
        musharik_quality_gate_node,
        MAX_AJUZ_RETRIES,
    )
    from creative_poet.nodes.hafiz import (
        hafiz_nassikh_check_node,
        hafiz_retrieve_node,
        hafiz_generate_node,
    )
    from creative_poet.nodes.muqayyim import muqayyim_node
    from creative_poet.nodes.composition_guardrails import enforce_attribution

    # ── Mode dispatch ─────────────────────────────────────────────────────────

    def _dispatch_mode(state: CompositionState) -> str:
        """Route to the correct creative sub-graph entry node based on mode."""
        mode = (state.get("composition_context") or {}).get("mode", "scaffold")
        return {
            "scaffold": "mulhim_retrieve",
            "coauthor": "musharik_retrieve",
            "preserve": "hafiz_nassikh_check",
            "critique": "muqayyim",
        }.get(mode, "mulhim_retrieve")

    # ── Conditional edge functions ────────────────────────────────────────────

    def _mulhim_route_after_validate(state: CompositionState) -> str:
        """
        Retry mulhim_generate if the scaffold failed quality and the budget allows.
        MAX_SCAFFOLD_RETRIES is imported from mulhim.py so the bounds live in one place.
        """
        if state.get("scaffold_quality_ok"):
            return END
        if (state.get("scaffold_retry_count") or 0) < MAX_SCAFFOLD_RETRIES:
            return "mulhim_generate"
        return END

    def _musharik_route_after_quality(state: CompositionState) -> str:
        """
        Retry musharik_generate if candidates failed quality and the budget allows.
        """
        if state.get("ajuz_quality_ok"):
            return END
        if (state.get("ajuz_retry_count") or 0) < MAX_AJUZ_RETRIES:
            return "musharik_generate"
        return END

    def _hafiz_route_after_nassikh(state: CompositionState) -> str:
        """
        Short-circuit to END if Al-Nassikh reports 0 poems for the poet.
        Otherwise continue to Fatat Al-Arab retrieval.
        """
        if (state.get("nassikh_poet_count") or 0) > 0:
            return "hafiz_retrieve"
        return END

    # ── Graph construction ────────────────────────────────────────────────────

    def build_creative_graph() -> "StateGraph":
        graph = StateGraph(CompositionState)

        # ── scaffold path nodes ───────────────────────────────────────────────
        graph.add_node("mulhim_retrieve", mulhim_retrieve_node)
        graph.add_node("mulhim_generate", mulhim_generate_node)
        graph.add_node("mulhim_validate", mulhim_validate_node)

        # ── coauthor path nodes ───────────────────────────────────────────────
        graph.add_node("musharik_retrieve", musharik_retrieve_node)
        graph.add_node("musharik_generate", musharik_generate_node)
        graph.add_node("musharik_quality",  musharik_quality_gate_node)

        # ── preserve path nodes ───────────────────────────────────────────────
        graph.add_node("hafiz_nassikh_check", hafiz_nassikh_check_node)
        graph.add_node("hafiz_retrieve",      hafiz_retrieve_node)
        graph.add_node("hafiz_generate",      hafiz_generate_node)
        graph.add_node("enforce_attribution", enforce_attribution)

        # ── critique path node ────────────────────────────────────────────────
        graph.add_node("muqayyim", muqayyim_node)

        # ── Entry: dispatch on mode ───────────────────────────────────────────
        graph.set_conditional_entry_point(
            _dispatch_mode,
            {
                "mulhim_retrieve":      "mulhim_retrieve",
                "musharik_retrieve":    "musharik_retrieve",
                "hafiz_nassikh_check":  "hafiz_nassikh_check",
                "muqayyim":             "muqayyim",
            },
        )

        # ── scaffold path edges ───────────────────────────────────────────────
        graph.add_edge("mulhim_retrieve", "mulhim_generate")
        graph.add_edge("mulhim_generate", "mulhim_validate")
        graph.add_conditional_edges(
            "mulhim_validate",
            _mulhim_route_after_validate,
            {
                "mulhim_generate": "mulhim_generate",
                END:               END,
            },
        )

        # ── coauthor path edges ───────────────────────────────────────────────
        graph.add_edge("musharik_retrieve", "musharik_generate")
        graph.add_edge("musharik_generate", "musharik_quality")
        graph.add_conditional_edges(
            "musharik_quality",
            _musharik_route_after_quality,
            {
                "musharik_generate": "musharik_generate",
                END:                 END,
            },
        )

        # ── preserve path edges ───────────────────────────────────────────────
        graph.add_conditional_edges(
            "hafiz_nassikh_check",
            _hafiz_route_after_nassikh,
            {
                "hafiz_retrieve": "hafiz_retrieve",
                END:              END,
            },
        )
        graph.add_edge("hafiz_retrieve",      "hafiz_generate")
        graph.add_edge("hafiz_generate",      "enforce_attribution")
        graph.add_edge("enforce_attribution", END)

        # ── critique path edge ────────────────────────────────────────────────
        graph.add_edge("muqayyim", END)

        return graph.compile()

    _CREATIVE_GRAPH = None  # lazy singleton — same pattern as Agent 2

    def get_creative_graph():
        """Return the compiled creative graph (built once per process)."""
        global _CREATIVE_GRAPH
        if _CREATIVE_GRAPH is None:
            _CREATIVE_GRAPH = build_creative_graph()
        return _CREATIVE_GRAPH

    LANGGRAPH_AVAILABLE = True

except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning(
        "creative_graph: langgraph not installed — "
        "nodes can still be called directly via orchestrator.run_creative()."
    )

    def get_creative_graph():
        raise ImportError("langgraph is not installed. Run: pip install langgraph")

    def build_creative_graph():
        raise ImportError("langgraph is not installed. Run: pip install langgraph")
