"""
agent1_query_understanding/nodes/bilingual_expand.py
=====================================================
Why this node exists: §2.4 Stage 2 (Bilingual Multi-Query) improves retrieval
recall by generating 3-5 Arabic paraphrases and 3-5 English paraphrases of the
user's query. These variants are merged by the RRF fuser downstream, which means
a thematically-phrased query in English can still surface a verse that only matches
a more classical Arabic phrasing.

This node is the first half of Stage 2; hyde.py handles the second half
(HyDE — generating a hypothetical verse to use as a dense query seed).

Design decision: variants are generated in a single LLM call (both languages at
once) to minimise API calls — important given the §5 note that a single query
can already trigger 5-8 LLM calls and the Together.ai free tier caps at ~60/min.

Architecture refs: §2.4 Stage 2 (Bilingual Multi-Query), §2.9 tool registry
('expand_bilingual'), §5 (no explicit retry budget for expansion — if the call
fails we degrade gracefully to a single-variant list).
"""

from __future__ import annotations

import json
import logging

from ...llm import chat
from ...state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_EXPAND = """\
You are a specialist in Nabati (Khaleeji Gulf) poetry and Arabic literature.
Given a user query, generate paraphrases that will help retrieve relevant poems
from a manuscript collection.

Return a JSON object with exactly two keys:
{
  "variants_ar": [<3 to 5 Arabic paraphrases, including dialectal Khaleeji forms>],
  "variants_en": [<3 to 5 English paraphrases, covering thematic and literal angles>]
}

Rules:
- Each variant should use different vocabulary / framing from the original.
- Arabic variants must include at least one in Khaleeji dialect (Gulf Arabic).
- English variants may use synonyms, broader themes, or poetic terminology.
- Do NOT repeat the original query verbatim as a variant.
- Return ONLY the JSON object, no explanation.
"""


def _expand(query_ar: str, query_en: str) -> tuple[list[str], list[str]]:
    """
    Call the LLM once with both the Arabic and English query to get parallel
    variant lists. Returns (variants_ar, variants_en).

    Why pass both languages in the prompt: the model can cross-reference meaning
    across both and produce higher-quality variants than if given only one side.
    """
    prompt = (
        f"Arabic query: {query_ar}\n"
        f"English query: {query_en}\n\n"
        "Generate bilingual paraphrases as specified."
    )
    raw = chat(
        prompt=prompt,
        system=_SYSTEM_EXPAND,
        json_schema={"type": "object"},
        max_tokens=512,
    )
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except Exception:
            logger.warning("bilingual_expand: could not parse LLM output: %r", raw)
            data = {}

    variants_ar: list[str] = data.get("variants_ar", [])
    variants_en: list[str] = data.get("variants_en", [])

    # Guard: ensure we have at least one variant per language (use original as fallback)
    if not variants_ar:
        variants_ar = [query_ar]
    if not variants_en:
        variants_en = [query_en]

    return variants_ar, variants_en


def bilingual_expand_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.4 Stage 2 (bilingual expansion half).

    Reads:   state["query_context"]["query_ar"], ["query_en"]
    Writes:  state["query_context"]["query_variants_ar"], ["query_variants_en"]

    Why we skip this node if clarification is needed: there's no point generating
    variants for a query we're going to ask the user to rephrase anyway.
    """
    qc = state.get("query_context", {})

    # Short-circuit if clarification is flagged
    if qc.get("needs_clarification"):
        logger.debug("bilingual_expand: skipping — clarification path active.")
        return state

    query_ar: str = qc.get("query_ar", state.get("raw_query", ""))
    query_en: str = qc.get("query_en", state.get("raw_query", ""))

    try:
        variants_ar, variants_en = _expand(query_ar, query_en)
    except Exception as exc:
        # §5 degraded gracefully: single-item variant lists from the original query
        logger.warning("bilingual_expand: expansion failed (%s), using originals.", exc)
        variants_ar = [query_ar]
        variants_en = [query_en]

    updated_qc = {**qc, "query_variants_ar": variants_ar, "query_variants_en": variants_en}
    return {**state, "query_context": updated_qc}


def expand_bilingual(query_ar: str, query_en: str) -> dict:
    """
    Tool-registry entry point (§2.9 'expand_bilingual').
    Named and typed to match the tool registry in tools.py.
    Returns {"variants_ar": [...], "variants_en": [...]}
    """
    ar, en = _expand(query_ar, query_en)
    return {"variants_ar": ar, "variants_en": en}
