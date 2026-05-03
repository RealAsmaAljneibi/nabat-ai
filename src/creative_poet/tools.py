"""
creative_poet/tools.py
=====================================
Why this file exists: mirrors the tool-registry pattern from agent1 and agent2.
Making tool permissions explicit and enforced at the registry level means a code
reviewer can open this file and immediately see which agent is allowed to call what.
The ToolNotPermittedError fires when a node attempts a tool outside its declared set.

Tool boundary summary:
  Al-Mulhim   — retrieve (external) + fingerprint + scaffold (never raw composition)
  Al-Musharik — retrieve similar (external) + ajuz generation only
  Al-Hafiz    — nassikh check (external) + retrieve (external) + verse + badge (mandatory)
  Al-Muqayyim — score_verse only (no index access — critique works on submitted text)

External tool names (QUERY_NASSIKH_POET_STATS, QUERY_FATAT_RETRIEVE) map to functions
in external_tools.py and must be explicitly declared here to cross the agent boundary.
"""

from __future__ import annotations

# ── Tool name constants ────────────────────────────────────────────────────────

RETRIEVE_STYLE_EXEMPLARS    = "retrieve_style_exemplars"
EXTRACT_STYLE_FINGERPRINT   = "extract_style_fingerprint"
GENERATE_SCAFFOLD           = "generate_scaffold"
RETRIEVE_SIMILAR_STRUCTURES = "retrieve_similar_structures"
GENERATE_AJUZ_CANDIDATES    = "generate_ajuz_candidates"
GENERATE_PRESERVATION_VERSE = "generate_preservation_verse"
FORMAT_SYNTHETIC_BADGE      = "format_synthetic_badge"   # mandatory for Al-Hafiz
SCORE_VERSE_QUALITY         = "score_verse_quality"

# External agent bridge tools — declared here so permission checking is centralised.
QUERY_NASSIKH_POET_STATS    = "query_nassikh_poet_stats"  # → al_nassikh.corpus_stats
QUERY_FATAT_RETRIEVE        = "query_fatat_retrieve"      # → Fatat Al-Arab retrieve chain

# ── M9 inter-agent consultation tools ─────────────────────────────────────────
# Why: cross-talk between creative agents enhances quality (graded ajuz, voice
# fingerprints, intertextuality flags) but must be permission-gated like every
# other tool in the registry — auditable, bounded, and named.
QUERY_FATAT_VOICE_BLEND      = "query_fatat_voice_blend"     # Hafiz: 2-call retrieval (voice + topic)
CONSULT_MUQAYYIM_GRADE       = "consult_muqayyim_grade"      # Musharik → Muqayyim batched grading
CONSULT_HAFIZ_FINGERPRINT    = "consult_hafiz_fingerprint"   # Mulhim → Hafiz voice fingerprint
CONSULT_NASSIKH_INTERTEXT    = "consult_nassikh_intertext"   # Muqayyim → Nassikh boolean check

# ── Per-agent permission sets ─────────────────────────────────────────────────
# Why frozensets: immutable at runtime so a node cannot add permissions to itself.

MULHIM_TOOLS = frozenset({
    RETRIEVE_STYLE_EXEMPLARS,
    EXTRACT_STYLE_FINGERPRINT,
    GENERATE_SCAFFOLD,
    QUERY_FATAT_RETRIEVE,         # calls external_tools.fatat_retrieve_exemplars()
    CONSULT_HAFIZ_FINGERPRINT,    # M9: voice fingerprint when target_poet specified
})

MUSHARIK_TOOLS = frozenset({
    RETRIEVE_SIMILAR_STRUCTURES,
    GENERATE_AJUZ_CANDIDATES,
    QUERY_FATAT_RETRIEVE,         # calls external_tools.fatat_retrieve_similar_structures()
    CONSULT_MUQAYYIM_GRADE,       # M9: batched grading of generated candidates
})

HAFIZ_TOOLS = frozenset({
    RETRIEVE_STYLE_EXEMPLARS,
    GENERATE_PRESERVATION_VERSE,
    FORMAT_SYNTHETIC_BADGE,       # cannot be removed — part of the ethical contract
    QUERY_NASSIKH_POET_STATS,     # pre-flight check via external_tools.nassikh_check_poet()
    QUERY_FATAT_RETRIEVE,         # calls external_tools.fatat_retrieve_exemplars()
    QUERY_FATAT_VOICE_BLEND,      # M9: 2-call voice + topic blend for richer exemplars
})

MUQAYYIM_TOOLS = frozenset({
    SCORE_VERSE_QUALITY,
    # No corpus content access — Muqayyim evaluates submitted text. The new
    # CONSULT_NASSIKH_INTERTEXT tool returns ONLY booleans + attribution
    # metadata (never verse content) so the isolation invariant holds.
    CONSULT_NASSIKH_INTERTEXT,    # M9: boolean intertextuality check
})

_ALL_PERMISSIONS: dict[str, frozenset[str]] = {
    "mulhim":   MULHIM_TOOLS,
    "musharik": MUSHARIK_TOOLS,
    "hafiz":    HAFIZ_TOOLS,
    "muqayyim": MUQAYYIM_TOOLS,
}


class ToolNotPermittedError(PermissionError):
    """Raised when a creative agent calls a tool outside its permission set."""

    def __init__(self, agent: str, tool: str) -> None:
        super().__init__(
            f"Creative agent '{agent}' is not permitted to call '{tool}'. "
            f"Permitted tools: {sorted(_ALL_PERMISSIONS.get(agent.lower(), set()))}"
        )


def check_tool_permitted(agent: str, tool: str) -> None:
    """Raise ToolNotPermittedError if the tool is outside the agent's permission set."""
    permitted = _ALL_PERMISSIONS.get(agent.lower(), frozenset())
    if tool not in permitted:
        raise ToolNotPermittedError(agent, tool)


def list_tools(agent: str) -> list[str]:
    """Return sorted list of tools permitted for an agent. Used in debug panels."""
    return sorted(_ALL_PERMISSIONS.get(agent.lower(), frozenset()))
