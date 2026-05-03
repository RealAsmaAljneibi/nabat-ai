"""
agent1_query_understanding/nodes/bilingual_analyzer.py
=======================================================
Why this node exists: §2.4 Stage 1 is the entry gate for Agent 1. Before any
retrieval decision is made, we need three things the rest of the pipeline depends on:

  1. What language is the query in? (langdetect gives a fast answer; LLM confirms)
  2. What does the user actually want? (factual lookup vs. thematic search vs.
     literary interpretation — this drives CRAG thresholds downstream)
  3. Is the Khaleeji dialect present? (affects which text_* field in the Qdrant
     payload is most relevant: text_khaleeji vs. text_msa_summary)

This node populates: query_lang, query_ar, query_en, detected_intent,
detected_dialect, intent_confidence — all §2.4 Step 1 output fields.
When triggered: Stage 1 — first 
LLM-bearing Agent-1 node when track = poetic_rag.
Purpose: Language detection + intent confidence + Khaleeji dialect ID; routes to clarification if confidence < 0.5.
Architecture refs: §2.4 Stage 1 (Bilingual Query Analysis), §2.6 (state contract),
§5 (intent_confidence < 0.5 → clarification path rather than retrieval).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ...llm import chat
from ...personas import FATAT_PERSONA
from ...state import AgentState, QueryContext, trace_append
from ...translate import translate

logger = logging.getLogger(__name__)

# ── Intent detection prompt ───────────────────────────────────────────────────
# Why a structured JSON output: the downstream clarification check
# (intent_confidence < 0.5) needs a numeric score, not just a label. Asking the
# model for JSON forces it to commit to a confidence rather than hedging in prose.

_SYSTEM_ANALYZER = FATAT_PERSONA + """\
Task — Stage 1 (Bilingual Analysis): analyse the user's question and return a \
JSON object with the following fields:

{
  "detected_intent": "factual" | "semantic" | "interpretive",
  "detected_dialect": "khaleeji" | "najdi" | "msa" | "unknown",
  "intent_confidence": <float 0.0–1.0>,
  "dialect_features": [<list of dialect markers found, may be empty>],
  "needs_clarification": <bool>,
  "clarification_question": <string or null>
}

Intent definitions:
  "factual"       — the user asks for a specific fact: a poet's name, a poem title,
                    a manuscript reference, a page number, a verse count.
  "semantic"      — the user asks about a theme, image, emotion, or concept that
                    requires thematic retrieval (e.g., "poems about camels", "love
                    metaphors in Khaleeji verse").
  "interpretive"  — the user asks for literary analysis, comparison, or explication
                    of a specific verse or poem.

Dialect markers to look for (non-exhaustive): فدوة، بعد، يبه، هيه، ذا، چذا،
ابشر، ياهل، خبر، يوه، ما، زين، قبال.

If prior conversation turns are provided, use them to resolve pronouns and
implicit references (e.g., "what else did he write?" refers to the poet named
in the previous turn). Do NOT repeat information already given in prior answers.

Set needs_clarification=true only if the query is genuinely ambiguous in a way
that would produce useless retrieval (e.g., a single word with no context).
Return ONLY valid JSON, no explanation.
"""


def _format_history_for_prompt(history: list[dict], max_turns: int = 5) -> str:
    """
    Why this helper: the analyzer prompt needs a compact, readable summary of
    prior turns so the LLM can resolve pronouns and implicit references
    (e.g., "what else did he write?" → the poet named in the previous answer).
    We cap at max_turns and skip refusals — they carry no useful context.
    """
    relevant = [t for t in history if not t.get("is_refusal", False)][-max_turns:]
    if not relevant:
        return ""
    lines = ["Prior conversation turns (oldest first):"]
    for i, turn in enumerate(relevant, 1):
        q = turn.get("query", "").strip()
        r = turn.get("response", "").strip()
        # Truncate long responses so the context block stays token-efficient
        if len(r) > 300:
            r = r[:300] + "…"
        lines.append(f"  Turn {i} — Q: {q}")
        lines.append(f"          A: {r}")
    return "\n".join(lines)


def _detect_language(text: str) -> str:
    """
    Why langdetect first: it is deterministic and free — no API call needed for
    a task as simple as 'is this Arabic script?'. The LLM step then confirms the
    dialect. If langdetect is unavailable we fall back to a simple heuristic
    (presence of Arabic Unicode block characters).
    """
    try:
        from langdetect import detect
        lang = detect(text)
        # langdetect returns "ar" for Arabic; anything else we treat as "en"
        return "ar" if lang == "ar" else "en"
    except Exception:
        # Heuristic fallback: count Arabic Unicode characters
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        return "ar" if arabic_chars / max(len(text), 1) > 0.3 else "en"


def _call_analyzer(query_ar: str, history: list[dict] | None = None) -> dict:
    """
    Why we always analyse the Arabic version: intent and dialect detection work
    better on the native language. Even if the user typed in English, we translate
    first and analyse the Arabic.

    Why history is injected here (M4b): the LLM resolves implicit references
    (e.g., "what else did he write?") using the prior turns. Without this,
    every query is treated as a cold start and pronoun resolution fails.
    """
    # Build prompt: prepend history block when available so the model has context
    history_block = _format_history_for_prompt(history or [])
    if history_block:
        prompt = f"{history_block}\n\nCurrent query: {query_ar}"
    else:
        prompt = query_ar

    raw = chat(
        prompt=prompt,
        system=_SYSTEM_ANALYZER,
        json_schema={"type": "object"},
        max_tokens=300,
    )
    if isinstance(raw, dict):
        return raw
    # Defensive: chat() with json_schema should always parse, but guard anyway
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("bilingual_analyzer: could not parse LLM output as JSON: %r", raw)
        return {
            "detected_intent": "semantic",
            "detected_dialect": "unknown",
            "intent_confidence": 0.5,
            "dialect_features": [],
            "needs_clarification": False,
            "clarification_question": None,
        }


def bilingual_analyzer_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.4 Stage 1.

    Reads:   state["raw_query"]
    Writes:  state["query_context"] (partial — only Stage 1 fields)

    Why partial QueryContext here: LangGraph accumulates state across nodes.
    Writing only the Stage 1 fields here means Stage 2 and Stage 3 nodes can
    be tested independently with a frozen Stage-1 output.
    """
    raw_query: str = state["raw_query"]
    # M4b: read conversation history from state (empty list on first turn)
    history: list[dict] = state.get("conversation_history") or []
    logger.debug(
        "bilingual_analyzer: raw_query=%r, prior_turns=%d",
        raw_query[:80], len(history),
    )

    # Step 1 — language detection (fast, no API call)
    query_lang = _detect_language(raw_query)

    # Step 2 — get both language versions (translate.py handles the §5 fallback)
    both = translate(raw_query, source_lang=query_lang)
    query_ar = both["ar"]
    query_en = both["en"]

    # Step 3 — intent + dialect analysis on the Arabic version
    # Why pass history: allows the LLM to resolve "what else did he write?"
    # by referring to the poet named in a previous turn (M4b).
    analysis = _call_analyzer(query_ar, history=history)

    detected_intent:    str   = analysis.get("detected_intent", "semantic")
    detected_dialect:   str   = analysis.get("detected_dialect", "unknown")
    intent_confidence:  float = float(analysis.get("intent_confidence", 0.5))
    needs_clarification: bool = bool(analysis.get("needs_clarification", False))
    clarification_q: str | None = analysis.get("clarification_question")

    # §5 guard: intent_confidence < 0.5 → clarification path
    if intent_confidence < 0.5 and not needs_clarification:
        logger.info(
            "bilingual_analyzer: low intent_confidence (%.2f) — setting clarification flag.",
            intent_confidence,
        )
        needs_clarification = True
        if not clarification_q:
            clarification_q = (
                "هل يمكنك توضيح سؤالك أكثر؟ / Could you clarify your question further?"
            )

    # Build the partial QueryContext for Stage 1.
    # Why merge: semantic_router (Stage 0.5b) may have set track/router_source
    # before bilingual_analyzer runs. Replacing query_context wholesale would
    # lose those upstream routing fields.
    existing_qc = state.get("query_context") or {}
    qc: QueryContext = {
        **existing_qc,
        "query_lang":       query_lang,
        "query_ar":         query_ar,
        "query_en":         query_en,
        "detected_intent":  detected_intent,
        "detected_dialect": detected_dialect,
        "intent_confidence": intent_confidence,
    }
    if needs_clarification:
        qc["needs_clarification"]    = True
        qc["clarification_question"] = clarification_q

    trace_summary = (
        f"lang={query_lang} · intent={detected_intent} ({intent_confidence:.0%}) · "
        f"dialect={detected_dialect}"
        + (" · ⚠️ needs clarification" if needs_clarification else "")
    )
    return {
        **state,
        "query_context": qc,
        "agent_trace": trace_append(state, stage="1", icon="🌐", label="Bilingual Analysis", summary=trace_summary),
    }
