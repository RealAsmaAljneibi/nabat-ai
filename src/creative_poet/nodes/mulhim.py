"""
creative_poet/nodes/mulhim.py
==========================================
Why this file exists: Al-Mulhim (الملهِم — The Inspirer) is the first creative agent.
It does NOT write a poem for the human. It reads the corpus via Fatat Al-Arab's
retrieval chain, extracts the target poet's distinctive voice fingerprint, and assembles
a compositional scaffold — the structured frame the human poet builds within.

Three sub-nodes (LangGraph graph registers these individually):
  mulhim_retrieve_node  — calls Fatat Al-Arab to retrieve poet exemplars
  mulhim_generate_node  — LLM: fingerprint extraction + scaffold generation
  mulhim_validate_node  — quality check; conditional edge → retry or END

Agentic loop (Observe → Reflect → Retry):
  mulhim_retrieve → mulhim_generate → mulhim_validate
                          ↑ (retry if quality fails, scaffold_retry_count < MAX_SCAFFOLD_RETRIES)

§5 failure budget:
  - If retrieval < MIN_EXEMPLARS_REQUIRED, fatat_retrieve_exemplars() relaxes to genre.
  - LLM failures produce a graceful fallback with whatever partial data exists.
  - If scaffold fails validation after MAX_SCAFFOLD_RETRIES, the best partial output is returned.

Tools declared in tools.py:
  QUERY_FATAT_RETRIEVE, EXTRACT_STYLE_FINGERPRINT, GENERATE_SCAFFOLD
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fatat_al_arab.llm import chat
from creative_poet.personas import MULHIM_PERSONA
from fatat_al_arab.state import trace_append
from creative_poet.tools import (
    check_tool_permitted,
    QUERY_FATAT_RETRIEVE,
    EXTRACT_STYLE_FINGERPRINT,
    GENERATE_SCAFFOLD,
)

logger = logging.getLogger(__name__)

# ── §5 failure budget constants ───────────────────────────────────────────────
MAX_SCAFFOLD_RETRIES = 1   # one retry on quality failure — same budget as CRAG re-query

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_FINGERPRINT = (
    MULHIM_PERSONA
    + """
Task — Style Fingerprint Extraction (Stage M1b):
Given a set of indexed verses by a Nabati poet, extract the distinctive patterns
that define their poetic voice.

Return JSON with this exact shape:
{
  "vocabulary_fingerprint": ["<word or phrase that recurs or is characteristic>"],
  "dominant_imagery":       ["<image or metaphor pattern with example anchor_id>"],
  "rhyme_sounds":           ["<dominant final vowel+consonant sounds, e.g. '-اني', '-ول'>"],
  "preferred_meter":        "<Arabic meter name, e.g. الكامل, البسيط, or 'mixed/unclear'>",
  "dialect_register":       "<Khaleeji | Najdi | MSA | mixed>",
  "opening_patterns":       ["<how this poet typically opens a verse>"],
  "thematic_preoccupations":["<recurring theme drawn from the verses>"]
}
Return ONLY valid JSON. No surrounding prose.
"""
)

_SYSTEM_SCAFFOLD = (
    MULHIM_PERSONA
    + """
Task — Compositional Scaffold Generation (Stage M1c):
Given a style fingerprint and the poet's occasion/theme, build a scaffold the human
poet can use as a frame. The scaffold is NOT a poem — it is a structured guide.

Return JSON with this exact shape:
{
  "meter_recommendation":       "<meter name + one sentence explaining why it suits this occasion>",
  "rhyme_scheme":               "<e.g., AABA — recommended based on the target poet's typical usage>",
  "opening_image_suggestions":  ["<3 distinct opening images in Khaleeji Arabic>"],
  "thematic_arc":               "<2 sentences: how this poet would move through this theme>",
  "dialect_note":               "<advice on Khaleeji vs MSA register for this occasion>",
  "exemplar_anchors":           ["<anchor_id of the most relevant reference verse — real IDs only>"]
}
Return ONLY valid JSON. No surrounding prose.
"""
)


# ── Sub-node 1: retrieve ──────────────────────────────────────────────────────

def mulhim_retrieve_node(state: dict) -> dict:
    """
    LangGraph node — Stage M1a.
    Calls Fatat Al-Arab's retrieval chain to fetch exemplar verses for the target poet.
    Permission enforced: QUERY_FATAT_RETRIEVE.

    Reads:  composition_context (target_poet, genre)
    Writes: style_exemplars, fatat_retrieved_count
    """
    check_tool_permitted("mulhim", QUERY_FATAT_RETRIEVE)

    from creative_poet.external_tools import fatat_retrieve_exemplars

    ctx         = state.get("composition_context") or {}
    target_poet = ctx.get("target_poet") or ""
    genre       = ctx.get("genre") or None

    exemplars = fatat_retrieve_exemplars(
        target_poet=target_poet or None,
        genre=genre,
        n=10,
    )

    # M9 Tier-2: when target_poet is specified, consult Al-Hafiz for a voice
    # fingerprint. This gives the scaffold prompt access to Hafiz's compact
    # style summary (opening phrases, signature imagery, dialect register) on
    # top of the raw exemplars. Bounded by depth/budget; degrades silently.
    fingerprint: dict = {}
    consult_summary = ""
    if target_poet.strip():
        try:
            from creative_poet.tools import check_tool_permitted as _check, CONSULT_HAFIZ_FINGERPRINT
            from creative_poet.external_tools import hafiz_voice_fingerprint, _record_consult
            _check("mulhim", CONSULT_HAFIZ_FINGERPRINT)
            t0 = __import__("time").perf_counter()
            fingerprint = hafiz_voice_fingerprint(target_poet, state=state)
            ms = (__import__("time").perf_counter() - t0) * 1000
            state = _record_consult(state, "mulhim", "hafiz", ms)
            if fingerprint.get("available"):
                consult_summary = " · Hafiz voice fingerprint cached"
        except Exception as exc:
            logger.info("mulhim_retrieve_node: hafiz consult skipped (%s)", exc)

    logger.debug("mulhim_retrieve_node: %d exemplars for poet=%r", len(exemplars), target_poet)
    return {
        **state,
        "style_exemplars":     exemplars,
        "fatat_retrieved_count": len(exemplars),
        "cached_voice_fingerprint": fingerprint,
        "agent_trace": trace_append(
            state, stage="M1a", icon="✨", label="Al-Mulhim / Retrieve",
            summary=f"{len(exemplars)} exemplars for {target_poet or '?'}{consult_summary}",
        ),
    }


# ── Sub-node 2: generate ──────────────────────────────────────────────────────

def mulhim_generate_node(state: dict) -> dict:
    """
    LangGraph node — Stages M1b + M1c.
    Two LLM calls: fingerprint extraction then scaffold generation.
    Permission enforced: EXTRACT_STYLE_FINGERPRINT, GENERATE_SCAFFOLD.

    Reads:  style_exemplars, composition_context (target_poet, occasion, theme)
    Writes: style_fingerprint, scaffold, scaffold_quality_ok (False — validate decides)
            Increments scaffold_retry_count if this is a retry.
    """
    check_tool_permitted("mulhim", EXTRACT_STYLE_FINGERPRINT)
    check_tool_permitted("mulhim", GENERATE_SCAFFOLD)

    ctx         = state.get("composition_context") or {}
    target_poet = ctx.get("target_poet") or ""
    occasion    = ctx.get("occasion") or ctx.get("theme") or ""
    exemplars   = state.get("style_exemplars") or []

    retry_count = (state.get("scaffold_retry_count") or 0) + (
        1 if state.get("scaffold_retry_count") is not None else 0
    )

    if not exemplars:
        fallback = (
            f"## الملهِم — Al-Mulhim\n\n"
            f"لم يتوفر في الأرشيف ما يكفي من الأبيات للشاعر **{target_poet or '(غير محدد)'}**.\n\n"
            f"No indexed verses found for this poet. "
            f"Try a different name or use a genre filter (e.g., غزل, رثاء, مديح)."
        )
        return {
            **state,
            "style_fingerprint":  {},
            "scaffold":           fallback,
            "scaffold_quality_ok": False,
            "scaffold_retry_count": retry_count,
            "guardrail_passed":   False,
            "guardrail_flags":    list(state.get("guardrail_flags") or []) + ["no_exemplars_found"],
            "agent_trace": trace_append(
                state, stage="M1b", icon="✨", label="Al-Mulhim / Generate",
                summary=f"No exemplars for {target_poet or '?'} — fallback scaffold",
            ),
        }

    # ── Stage M1b — extract style fingerprint ────────────────────────────────
    passage_texts = [
        {
            "anchor_id": e.get("anchor_id", ""),
            "text":      (e.get("matla_text") or e.get("text") or "")[:300],
        }
        for e in exemplars[:8]
    ]
    fingerprint: dict = {}
    try:
        result = chat(
            prompt=(
                f"Poet: {target_poet}\n"
                f"Verses:\n{json.dumps(passage_texts, ensure_ascii=False)}"
            ),
            system=_SYSTEM_FINGERPRINT,
            json_schema={"type": "object"},
            max_tokens=600,
        )
        fingerprint = result if isinstance(result, dict) else json.loads(str(result))
    except Exception as exc:
        logger.warning("mulhim_generate: fingerprint extraction failed (%s)", exc)

    # M9 Tier-2: merge Hafiz's voice fingerprint (collected in mulhim_retrieve_node)
    # into the local one. Hafiz's view is shorter and corpus-grounded; merging
    # gives the scaffold prompt both perspectives without an extra LLM round-trip.
    hafiz_fp = state.get("cached_voice_fingerprint") or {}
    if hafiz_fp.get("available"):
        fingerprint = {
            **fingerprint,
            "hafiz_opening_phrases":   hafiz_fp.get("opening_phrases", []),
            "hafiz_signature_imagery": hafiz_fp.get("signature_imagery", []),
            "hafiz_dialect_register":  hafiz_fp.get("dialect_register"),
            "hafiz_meter_hint":        hafiz_fp.get("meter_hint"),
        }

    # ── Stage M1c — generate scaffold ────────────────────────────────────────
    scaffold_str = ""
    try:
        scaffold_raw = chat(
            prompt=(
                f"Target poet: {target_poet}\n"
                f"Occasion/theme: {occasion}\n"
                f"Style fingerprint:\n{json.dumps(fingerprint, ensure_ascii=False)}\n"
                f"Exemplar anchor IDs: {[e.get('anchor_id','') for e in exemplars[:5]]}"
            ),
            system=_SYSTEM_SCAFFOLD,
            json_schema={"type": "object"},
            max_tokens=600,
        )
        scaffold_dict = scaffold_raw if isinstance(scaffold_raw, dict) else json.loads(str(scaffold_raw))
        scaffold_str = _render_scaffold(scaffold_dict, target_poet, occasion)
    except Exception as exc:
        logger.warning("mulhim_generate: scaffold generation failed (%s)", exc)
        scaffold_str = (
            f"تعذّر توليد الإطار التركيبي. "
            f"(Scaffold generation failed: {exc})\n\n"
            f"**Meter identified:** {fingerprint.get('preferred_meter','—')}\n"
            f"**Rhyme sounds:** {', '.join(fingerprint.get('rhyme_sounds',[]))}"
        )

    return {
        **state,
        "style_fingerprint":    fingerprint,
        "scaffold":             scaffold_str,
        "scaffold_quality_ok":  False,   # mulhim_validate_node makes the final call
        "scaffold_retry_count": retry_count,
        "agent_trace": trace_append(
            state, stage="M1b", icon="✨", label="Al-Mulhim / Generate",
            summary=(
                f"Fingerprint extracted · scaffold drafted "
                f"(retry {retry_count}/{MAX_SCAFFOLD_RETRIES})"
            ),
            detail=(
                f"Meter: {fingerprint.get('preferred_meter','?')} · "
                f"Dialect: {fingerprint.get('dialect_register','?')}"
            ),
        ),
    }


# ── Sub-node 3: validate ──────────────────────────────────────────────────────

def mulhim_validate_node(state: dict) -> dict:
    """
    LangGraph node — Observe → Reflect step for Al-Mulhim.
    Checks whether the scaffold meets minimum quality criteria. The graph's conditional
    edge routes back to mulhim_generate_node for one retry if quality fails.

    Reads:  scaffold, style_fingerprint, scaffold_retry_count
    Writes: scaffold_quality_ok, final_output (when quality passed or retries exhausted)
    """
    scaffold    = state.get("scaffold") or ""
    fingerprint = state.get("style_fingerprint") or {}
    quality_ok  = _scaffold_is_quality(scaffold, fingerprint)

    ctx         = state.get("composition_context") or {}
    target_poet = ctx.get("target_poet") or "?"
    retry_count = state.get("scaffold_retry_count") or 0

    if quality_ok or retry_count >= MAX_SCAFFOLD_RETRIES:
        # Accept: quality ok, or retries exhausted — return best available output
        return {
            **state,
            "scaffold_quality_ok": quality_ok,
            "final_output":        scaffold,
            "guardrail_passed":    quality_ok,
            "guardrail_flags":     (
                [] if quality_ok
                else list(state.get("guardrail_flags") or []) + ["scaffold_quality_low"]
            ),
            "agent_trace": trace_append(
                state, stage="M1c", icon="✨", label="Al-Mulhim / Validate",
                summary=(
                    f"Scaffold {'accepted' if quality_ok else 'accepted (retries exhausted)'} "
                    f"for {target_poet}"
                ),
            ),
        }

    # Quality check failed and budget remains — signal for retry
    logger.info(
        "mulhim_validate: quality check failed (retry %d/%d) — routing back to generate",
        retry_count, MAX_SCAFFOLD_RETRIES,
    )
    return {
        **state,
        "scaffold_quality_ok": False,
        "agent_trace": trace_append(
            state, stage="M1c", icon="✨", label="Al-Mulhim / Validate",
            summary=f"Quality insufficient — retry {retry_count + 1}/{MAX_SCAFFOLD_RETRIES}",
        ),
    }


def _scaffold_is_quality(scaffold_text: str, fingerprint: dict) -> bool:
    """
    A scaffold has sufficient quality when:
      - It has meaningful content (> 150 chars)
      - The fingerprint extracted at least a meter and rhyme_sounds
    """
    if not scaffold_text or len(scaffold_text) < 150:
        return False
    required_fields = {"preferred_meter", "rhyme_sounds", "vocabulary_fingerprint"}
    return bool(fingerprint) and bool(required_fields & fingerprint.keys())


# ── Monolithic wrapper (orchestrator direct-node fallback) ────────────────────

def mulhim_node(state: dict) -> dict:
    """
    Sequential wrapper that chains sub-nodes.
    Used only by orchestrator.run_creative() when LangGraph is unavailable.
    The LangGraph graph registers the three sub-nodes directly.
    """
    state = mulhim_retrieve_node(state)
    state = mulhim_generate_node(state)
    state = mulhim_validate_node(state)
    # one retry if quality failed
    if not state.get("scaffold_quality_ok") and (state.get("scaffold_retry_count") or 0) < MAX_SCAFFOLD_RETRIES:
        state = mulhim_generate_node(state)
        state = mulhim_validate_node(state)
    return state


def _render_scaffold(scaffold: dict, poet: str, occasion: str) -> str:
    """Format the scaffold JSON into readable markdown for the Streamlit UI."""
    lines = [
        "## الإطار التركيبي — Compositional Scaffold",
        f"**Poet style reference:** {poet}  ·  **Occasion:** {occasion}",
        "",
        f"**Recommended meter:** {scaffold.get('meter_recommendation', '—')}",
        f"**Rhyme scheme:** {scaffold.get('rhyme_scheme', '—')}",
        f"**Dialect register:** {scaffold.get('dialect_note', '—')}",
        "",
        "**Opening image suggestions:**",
    ]
    for img in scaffold.get("opening_image_suggestions") or []:
        lines.append(f"  - {img}")
    lines += [
        "",
        f"**Thematic arc:** {scaffold.get('thematic_arc', '—')}",
        "",
    ]
    anchors = scaffold.get("exemplar_anchors") or []
    if anchors:
        lines.append(f"**Reference verses:** `{'` · `'.join(anchors)}`")
    lines += [
        "",
        "> *هذا إطار للإلهام لا قصيدة — الفعل الإبداعي لك.*",
        "> *This scaffold is a guide, not a poem. The creative act is yours.*",
    ]
    return "\n".join(lines)
