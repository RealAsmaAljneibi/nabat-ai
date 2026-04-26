"""
agent2_retrieval_synthesis/tools.py
=====================================
Why this file exists: §2.9 defines an explicit "Agent Capability Surface" — a
restricted set of tools that Agent 2 is allowed to call. The restriction is
symmetric with agent1/tools.py: Agent 2 owns retrieval and synthesis tools;
Agent 1 is forbidden from importing them.

Agent 2 tool registry (§2.9):
  retrieve_triple      — triple hybrid retrieval (BM25 + Dense + ColBERT)
  rrf_fuse             — Reciprocal Rank Fusion (k=60, top-5)
  resolve_heritage     — MSA→Khaleeji text swap via manuscript registry
  crag_grade           — passage-level CRAG relevance grading
  synthesise           — grounded generation with mandatory citations
  reflect              — Self-RAG faithfulness + relevance + completeness check
  format_multi_variant — al-Maktub + orthographic + al-Mantuq output variants

Why no query-understanding tools here: §2.9 says Agent 2 must not perform
query understanding. Those tools (translate_query, hyde_passage, etc.) belong
exclusively to agent1/tools.py. The guard below raises ToolNotPermittedError
if any Agent 2 node tries to import an Agent 1 tool by name.

Architecture refs: §2.9 (Agent Capability Surface), §0 constraint 3 (three-
worker split), §M4 spec (tool registry enforcement at import time).
"""

from __future__ import annotations

from typing import Callable


class ToolNotPermittedError(ImportError):
    """
    Why ImportError subclass: a module that tries to import a forbidden tool
    will fail at import time (not at runtime), which surfaces wiring bugs
    during development and CI — not in the demo.
    """
    pass


# ── Tool implementations imported from the node modules ──────────────────────
# Each import pulls the node function that constitutes the tool's entry point.
# Naming follows the §2.9 registry names exactly.

from .nodes.retrieve import retrieve_node as retrieve_triple          # noqa: E402
from .nodes.rrf_fuse import rrf_fuse_node as rrf_fuse                # noqa: E402
from .nodes.resolve_heritage import resolve_heritage_node as resolve_heritage  # noqa: E402
from .nodes.crag_grader import crag_grader_node as crag_grade         # noqa: E402
from .nodes.synthesise import synthesise_node as synthesise           # noqa: E402
from .nodes.reflect import reflect_node as reflect                    # noqa: E402
from .nodes.format_variants import format_variants_node as format_multi_variant  # noqa: E402


# ── Registry ──────────────────────────────────────────────────────────────────
# Why a plain dict: tool lookup in the orchestrator is a dict __getitem__, not a
# dynamic dispatch pattern. Keeping it simple means the registry is auditable in
# one glance.

AGENT2_TOOLS: dict[str, Callable] = {
    "retrieve_triple":    retrieve_triple,
    "rrf_fuse":           rrf_fuse,
    "resolve_heritage":   resolve_heritage,
    "crag_grade":         crag_grade,
    "synthesise":         synthesise,
    "reflect":            reflect,
    "format_multi_variant": format_multi_variant,
}

# Query-understanding tools that Agent 1 owns — listed here so the guard below
# can give a helpful error message when something in Agent 2 tries to use one.
_AGENT1_TOOLS = frozenset({
    "translate_query",
    "extract_filters",
    "hyde_passage",
    "expand_bilingual",
})


def get_tool(name: str) -> Callable:
    """
    Retrieve a tool by name. Raises ToolNotPermittedError if the tool is not in
    the Agent 2 registry.

    Why a function instead of direct dict access: the function can give a
    contextual error that names the correct agent to look in.
    """
    if name in AGENT2_TOOLS:
        return AGENT2_TOOLS[name]

    if name in _AGENT1_TOOLS:
        raise ToolNotPermittedError(
            f"Tool '{name}' belongs to Agent 1 (agent1_query_understanding/tools.py). "
            "Agent 2 is not permitted to call query-understanding tools (§2.9)."
        )

    raise ToolNotPermittedError(
        f"Tool '{name}' is not registered for Agent 2. "
        f"Available tools: {sorted(AGENT2_TOOLS)}."
    )


def list_tools() -> list[str]:
    """Return the names of all tools available to Agent 2 (for debug panel)."""
    return sorted(AGENT2_TOOLS)
