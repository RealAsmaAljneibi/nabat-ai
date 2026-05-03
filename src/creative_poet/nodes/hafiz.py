"""
creative_poet/nodes/hafiz.py
==========================================
Why this node exists: Al-Hafiz (الحافظ — The Memory Keeper) preserves the voice of
deceased poets for new occasions. It generates one complete verse (bayt) that honors
their documented style — explicitly labeled as synthetic with a mandatory attribution badge.

Three sub-nodes (LangGraph graph registers these individually):
  hafiz_nassikh_check_node — queries Al-Nassikh: is this poet indexed at all?
  hafiz_retrieve_node      — calls Fatat Al-Arab for exemplar verses
  hafiz_generate_node      — LLM verse generation + mandatory badge wrapping

Then the graph routes to enforce_attribution() as the non-bypassable badge gate.

Why three stages instead of one: the Al-Nassikh pre-flight check (stage M3a) short-circuits
the pipeline before any LLM call when the poet is not in the corpus, avoiding both wasted
compute and the ethical risk of generating a verse from parametric memory alone.

Ethical constraints:
  1. hafiz_generate_node calls format_synthetic_badge() and wrap_with_badge() before
     writing to state["final_output"] — the badge is written here, not outside.
  2. The graph routes all output through enforce_attribution() as final verification.
  3. If nassikh says 0 poems, the graph routes directly to END with an informative message.

§5 failure budget: if the poet has no indexed verses, return "not in corpus" rather than
generating from parametric memory. If LLM generation fails, return honest error.

Tools declared in tools.py:
  QUERY_NASSIKH_POET_STATS, QUERY_FATAT_RETRIEVE, GENERATE_PRESERVATION_VERSE,
  FORMAT_SYNTHETIC_BADGE
"""

from __future__ import annotations

import json
import logging

from fatat_al_arab.llm import chat
from creative_poet.personas import HAFIZ_PERSONA
from fatat_al_arab.state import trace_append
from creative_poet.nodes.composition_guardrails import format_synthetic_badge, wrap_with_badge
from creative_poet.tools import (
    check_tool_permitted,
    QUERY_NASSIKH_POET_STATS,
    QUERY_FATAT_RETRIEVE,
    QUERY_FATAT_VOICE_BLEND,
    GENERATE_PRESERVATION_VERSE,
    FORMAT_SYNTHETIC_BADGE,
)

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_HAFIZ = (
    HAFIZ_PERSONA
    + """
Task — Voice Preservation Verse (Stage M3c):
Given indexed verses by a deceased poet and a new occasion, compose ONE complete
verse (bayt: one sadr + one ajuz) that honors their documented style.

STRICT RULES:
  - Use ONLY the imagery and vocabulary present in the indexed verses provided.
  - Do not invent imagery absent from the corpus.
  - source_anchors must be real anchor_ids from the provided exemplars.
  - The verse must respect the poet's documented meter and dialect register.

Return JSON:
{
  "verse":           "<full bayt in Khaleeji Arabic>",
  "sadr":            "<first hemistich>",
  "ajuz":            "<second hemistich>",
  "meter":           "<Arabic meter name used>",
  "source_anchors":  ["<anchor_id of each exemplar that influenced this verse>"],
  "imagery_sources": ["<note: 'image X drawn from anchor_id Y'>"]
}
Return ONLY valid JSON.
"""
)


# ── Sub-node 1: Al-Nassikh pre-flight check ───────────────────────────────────

def hafiz_nassikh_check_node(state: dict) -> dict:
    """
    LangGraph node — Stage M3a.
    Queries Al-Nassikh's corpus_stats for the target poet's indexed poem count.
    If the poet is not in the corpus, writes a clear "not in corpus" message and routes
    directly to END (bypassing retrieval and generation entirely).

    Permission enforced: QUERY_NASSIKH_POET_STATS.

    Reads:  composition_context (target_poet)
    Writes: nassikh_poet_count, final_output (if poet not in corpus)
    """
    check_tool_permitted("hafiz", QUERY_NASSIKH_POET_STATS)

    from creative_poet.external_tools import nassikh_check_poet

    ctx         = state.get("composition_context") or {}
    target_poet = ctx.get("target_poet") or ""

    if not target_poet.strip():
        msg = "يجب تحديد اسم الشاعر. (Poet name is required for voice preservation.)"
        return {
            **state,
            "nassikh_poet_count": 0,
            "preservation_verse": "",
            "attribution_badge":  "",
            "final_output":       msg,
            "guardrail_passed":   False,
            "guardrail_flags":    list(state.get("guardrail_flags") or []) + ["no_poet_specified"],
            "agent_trace": trace_append(
                state, stage="M3a", icon="📜", label="Al-Hafiz / Nassikh Check",
                summary="No poet specified",
            ),
        }

    nassikh_result = nassikh_check_poet(target_poet)
    poem_count     = nassikh_result["poem_count"]
    in_corpus      = nassikh_result["in_corpus"]

    logger.debug(
        "hafiz_nassikh_check: poet=%r in_corpus=%s poem_count=%d",
        target_poet, in_corpus, poem_count,
    )

    if not in_corpus:
        msg = (
            f"## الحافظ — Al-Hafiz\n\n"
            f"لا تتوفر أبيات مفهرسة لـ **{target_poet}** في الأرشيف "
            f"(Al-Nassikh reports 0 poems).\n\n"
            f"Voice preservation requires an indexed corpus. "
            f"Add their manuscripts via Tab B first, then rebuild the index."
        )
        return {
            **state,
            "nassikh_poet_count": 0,
            "preservation_verse": "",
            "attribution_badge":  "",
            "final_output":       msg,
            "guardrail_passed":   False,
            "guardrail_flags":    list(state.get("guardrail_flags") or []) + ["poet_not_in_corpus"],
            "agent_trace": trace_append(
                state, stage="M3a", icon="📜", label="Al-Hafiz / Nassikh Check",
                summary=f"Poet '{target_poet}' not in corpus (0 poems) — routing to END",
            ),
        }

    return {
        **state,
        "nassikh_poet_count": poem_count,
        "agent_trace": trace_append(
            state, stage="M3a", icon="📜", label="Al-Hafiz / Nassikh Check",
            summary=f"Al-Nassikh: {poem_count} poems for {target_poet} — proceeding",
        ),
    }


# ── Sub-node 2: retrieve exemplars ───────────────────────────────────────────

def hafiz_retrieve_node(state: dict) -> dict:
    """
    LangGraph node — Stage M3b.
    Calls Fatat Al-Arab's retrieval chain for the target poet's verses.
    Only reached when nassikh_check confirmed poet is in corpus.
    Permission enforced: QUERY_FATAT_RETRIEVE.

    Reads:  composition_context (target_poet, genre), nassikh_poet_count
    Writes: style_exemplars, fatat_retrieved_count
    """
    check_tool_permitted("hafiz", QUERY_FATAT_RETRIEVE)
    check_tool_permitted("hafiz", QUERY_FATAT_VOICE_BLEND)

    from creative_poet.external_tools import hafiz_retrieve_voice_blend

    ctx         = state.get("composition_context") or {}
    target_poet = ctx.get("target_poet") or ""
    genre       = ctx.get("genre") or None
    occasion    = ctx.get("occasion") or ctx.get("theme") or None

    # M9 Tier-1: 2-call retrieval blend — voice samples (any topic) + topical
    # samples. The voice pool teaches the LLM the poet's signature style; the
    # topical pool grounds the new occasion. Result: ~15-20 deduplicated exemplars.
    exemplars = hafiz_retrieve_voice_blend(
        target_poet=target_poet or "",
        occasion=occasion,
        genre=genre,
        voice_n=15,
        topic_n=5,
    )

    logger.debug("hafiz_retrieve_node: %d blended exemplars for poet=%r", len(exemplars), target_poet)
    return {
        **state,
        "style_exemplars":     exemplars,
        "fatat_retrieved_count": len(exemplars),
        "agent_trace": trace_append(
            state, stage="M3b", icon="📜", label="Al-Hafiz / Retrieve (voice+topic blend)",
            summary=f"{len(exemplars)} blended exemplars for {target_poet}",
        ),
    }


# ── Sub-node 3: generate + badge ─────────────────────────────────────────────

def hafiz_generate_node(state: dict) -> dict:
    """
    LangGraph node — Stage M3c.
    LLM generates one preservation verse, then wraps it with the mandatory attribution badge.
    After this node the graph routes to enforce_attribution() for badge verification.
    Permissions enforced: GENERATE_PRESERVATION_VERSE, FORMAT_SYNTHETIC_BADGE.

    Reads:  composition_context (target_poet, occasion/theme), style_exemplars
    Writes: preservation_verse, attribution_badge, final_output (badge-wrapped)
    """
    check_tool_permitted("hafiz", GENERATE_PRESERVATION_VERSE)
    check_tool_permitted("hafiz", FORMAT_SYNTHETIC_BADGE)

    ctx         = state.get("composition_context") or {}
    target_poet = ctx.get("target_poet") or ""
    occasion    = ctx.get("occasion") or ctx.get("theme") or ""
    exemplars   = state.get("style_exemplars") or []

    if not exemplars:
        err = (
            f"## الحافظ — Al-Hafiz\n\n"
            f"لا تتوفر أبيات مفهرسة لـ **{target_poet}** في الأرشيف.\n\n"
            f"No indexed verses for **{target_poet}** — cannot preserve their voice "
            f"without corpus grounding. Add their manuscripts via Tab B first."
        )
        return {
            **state,
            "preservation_verse": "",
            "attribution_badge":  "",
            "final_output":       err,
            "guardrail_passed":   False,
            "guardrail_flags":    list(state.get("guardrail_flags") or []) + ["no_indexed_verses_for_poet"],
            "agent_trace": trace_append(
                state, stage="M3c", icon="📜", label="Al-Hafiz / Generate",
                summary=f"No indexed verses for {target_poet}",
            ),
        }

    passage_texts = [
        {
            "anchor_id": e.get("anchor_id", ""),
            "text":      (e.get("matla_text") or e.get("text") or "")[:300],
        }
        for e in exemplars[:8]
    ]

    verse_result: dict = {}
    generation_failed  = False
    try:
        result = chat(
            prompt=(
                f"Poet: {target_poet}\n"
                f"New occasion: {occasion or 'not specified'}\n"
                f"Indexed verses to draw from:\n"
                f"{json.dumps(passage_texts, ensure_ascii=False)}"
            ),
            system=_SYSTEM_HAFIZ,
            json_schema={"type": "object"},
            max_tokens=500,
        )
        verse_result = result if isinstance(result, dict) else json.loads(str(result))
    except Exception as exc:
        logger.warning("hafiz_generate: verse generation failed (%s)", exc)
        generation_failed = True

    if generation_failed or not verse_result.get("verse"):
        err = "تعذّر توليد البيت. (Verse generation failed — please retry.)"
        return {
            **state,
            "preservation_verse": "",
            "attribution_badge":  "",
            "final_output":       err,
            "guardrail_passed":   False,
            "guardrail_flags":    list(state.get("guardrail_flags") or []) + ["verse_generation_failed"],
            "agent_trace": trace_append(
                state, stage="M3c", icon="📜", label="Al-Hafiz / Generate",
                summary="Verse generation failed",
            ),
        }

    verse_text     = verse_result.get("verse", "")
    source_anchors = verse_result.get("source_anchors") or [
        e.get("anchor_id", "") for e in exemplars[:3]
    ]

    # Mandatory badge — written here and then verified by enforce_attribution()
    badge = format_synthetic_badge(
        agent_name="Al-Hafiz / الحافظ",
        poet_name=target_poet,
        anchor_ids=[a for a in source_anchors if a],
    )
    final_output = wrap_with_badge(
        _render_verse(verse_result, target_poet, occasion),
        badge,
    )

    return {
        **state,
        "preservation_verse": verse_text,
        "attribution_badge":  badge,
        "final_output":       final_output,
        "guardrail_passed":   True,
        "guardrail_flags":    [],
        "agent_trace": trace_append(
            state, stage="M3c", icon="📜", label="Al-Hafiz / Generate",
            summary=f"Voice of {target_poet} preserved · badge applied",
            detail=f"Meter: {verse_result.get('meter','?')} · Sources: {source_anchors[:3]}",
        ),
    }


# ── Monolithic wrapper (orchestrator direct-node fallback) ────────────────────

def hafiz_node(state: dict) -> dict:
    """
    Sequential wrapper that chains sub-nodes.
    Used only by orchestrator.run_creative() when LangGraph is unavailable.
    The LangGraph graph registers the three sub-nodes + enforce_attribution directly.
    """
    state = hafiz_nassikh_check_node(state)
    if (state.get("nassikh_poet_count") or 0) == 0:
        return state  # poet not in corpus — return informative message
    state = hafiz_retrieve_node(state)
    state = hafiz_generate_node(state)
    return state


def _render_verse(verse_result: dict, poet: str, occasion: str) -> str:
    """Format the preservation verse as readable markdown for the UI."""
    lines = [
        f"## الحافظ — Al-Hafiz: Voice of {poet}",
        f"**Occasion:** {occasion or '—'}  ·  **Meter:** {verse_result.get('meter','—')}",
        "",
        f"> {verse_result.get('sadr','—')}",
        f"> {verse_result.get('ajuz','—')}",
        "",
    ]
    sources = verse_result.get("imagery_sources") or []
    if sources:
        lines.append("**Imagery sources:**")
        for s in sources:
            lines.append(f"  - {s}")
    return "\n".join(lines)
