"""
agent1_query_understanding/tools.py
=====================================
Why this file exists: §2.9 defines an explicit "Agent Capability Surface" — a
restricted set of tools that Agent 1 is allowed to call. The restriction is not
just documentation: importing this module registers the tools and makes any
attempt to call a non-registered tool raise a ToolNotPermittedError at import
time.

Agent 1 tool registry (§2.9):
  translate_query    — EN↔AR translation via translate.py
  extract_filters    — typed filter extraction via self_query.py
  hyde_passage       — hypothetical verse generation via hyde.py
  expand_bilingual   — bilingual paraphrase expansion via bilingual_expand.py

Why no retrieval tools here: §2.9 says Agent 1 must not perform retrieval.
Retrieval tools (retrieve_triple, rrf_fuse, etc.) are only in agent2/tools.py.
Enforcing the boundary here means a node that accidentally imports a retrieval
tool will fail loudly at startup, not silently at query time.

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
# Each import pulls the tool-registry entry point defined in the corresponding
# node file. The naming follows the §2.9 registry names exactly.

from ..translate import translate_query                          # noqa: E402
from .nodes.self_query import extract_filters                   # noqa: E402
from .nodes.hyde import hyde_passage                            # noqa: E402
from .nodes.bilingual_expand import expand_bilingual            # noqa: E402


# ── Registry ──────────────────────────────────────────────────────────────────
# Why a plain dict: tool lookup in the orchestrator is a dict __getitem__, not a
# dynamic dispatch pattern. Keeping it simple means the registry is auditable in
# one glance.

AGENT1_TOOLS: dict[str, Callable] = {
    "translate_query":  translate_query,
    "extract_filters":  extract_filters,
    "hyde_passage":     hyde_passage,
    "expand_bilingual": expand_bilingual,
}

# Retrieval tools that Agent 2 owns — listed here so the guard below can give
# a helpful error message when something in Agent 1 tries to use one.
_AGENT2_TOOLS = frozenset({
    "retrieve_triple",
    "rrf_fuse",
    "resolve_heritage",
    "crag_grade",
    "synthesise",
    "reflect",
    "format_multi_variant",
})


def get_tool(name: str) -> Callable:
    """
    Retrieve a tool by name. Raises ToolNotPermittedError if the tool is not in
    the Agent 1 registry.

    Why a function instead of direct dict access: the function can give a
    contextual error that names the correct agent to look in.
    """
    if name in AGENT1_TOOLS:
        return AGENT1_TOOLS[name]

    if name in _AGENT2_TOOLS:
        raise ToolNotPermittedError(
            f"Tool '{name}' belongs to Agent 2 (agent2_retrieval_synthesis/tools.py). "
            "Agent 1 is not permitted to call retrieval tools (§2.9)."
        )

    raise ToolNotPermittedError(
        f"Tool '{name}' is not registered for Agent 1. "
        f"Available tools: {sorted(AGENT1_TOOLS)}."
    )


def list_tools() -> list[str]:
    """Return the names of all tools available to Agent 1 (for debug panel)."""
    return sorted(AGENT1_TOOLS)
