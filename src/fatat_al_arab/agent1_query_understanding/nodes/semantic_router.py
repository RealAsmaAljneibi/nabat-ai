"""
agent1_query_understanding/nodes/semantic_router.py
=====================================================
Why this file exists: Stage 0.5b — after the deterministic regex router fails
to classify a query, we need to distinguish four user intention families before
spending 5-10 seconds on the full RAG pipeline:

  capabilities    — "how can you help me?" (never needs retrieval)
  registry_lookup — factual corpus stats (LLM-detected backup for regex misses)
  instructor_debug — pipeline-introspection questions from instructors/devs
  poetic_rag      — everything else (poetry search, thematic, literary)

Design decisions:
  - JSON-mode LLM call (same pattern as bilingual_analyzer.py:130–135)
  - Threading timeout of 1.5 s (same pattern as hyde.py:47–86)
  - lru_cache(256) keyed on aggressively normalised Arabic/English query
  - Default: "poetic_rag" on timeout, error, or low-confidence result

Architecture ref: §2.4 Stage 0.5b (Semantic Router), plan §Decisions §1–8.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from functools import lru_cache
from typing import Optional

from ...llm import chat
from ...state import AgentState

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
ROUTER_TIMEOUT_S = 1.5    # §Decisions §7: wall-clock cap on the semantic router call
CONFIDENCE_THRESHOLD = 0.6  # §Decisions §1 risk mitigation: require ≥0.6 to accept non-poetic_rag

_VALID_TRACKS = frozenset({"capabilities", "registry_lookup", "instructor_debug", "poetic_rag"})

# ── Bilingual cache-key builder ───────────────────────────────────────────────
# Why NOT reusing normalise_arabic from embed.py: that function strips all
# non-Arabic characters (step 6: _NON_ARABIC_SEP), leaving empty strings for
# purely English queries. Here we only normalise Arabic *within* the text so
# that English characters (and the lru_cache keys they form) are preserved.
_ALEF_RE     = re.compile(r"[آأإٱ]")
_HARAKAT_RE  = re.compile(r"[ً-ٰٟ]")
_TATWEEL_RE  = re.compile(r"ـ")


def _make_cache_key(text: str) -> str:
    """Stable lru_cache key for both Arabic and English queries."""
    t = text.lower()
    t = _ALEF_RE.sub("ا", t)
    t = t.replace("ة", "ه")
    t = t.replace("ى", "ي")
    t = _TATWEEL_RE.sub("", t)
    t = _HARAKAT_RE.sub("", t)
    return " ".join(t.split())

# ── Classification prompt ─────────────────────────────────────────────────────
_SYSTEM_ROUTER = """\
You are an intent classifier for NABAT-AI, a bilingual Arabic/English research
assistant for Khaleeji Nabati poetry manuscripts.

Classify the user query into exactly ONE track. Return ONLY valid JSON:

{
  "track": "capabilities" | "registry_lookup" | "instructor_debug" | "poetic_rag",
  "subintent": <string or null>,
  "confidence": <float 0.0-1.0>,
  "alt_family": <second most likely track, or null>,
  "reasoning": <one short sentence>
}

Track definitions:
  "capabilities"      — user asks what the system can do, what it covers, how to use it.
                        Examples: "how can you help me?", "what can you search for?",
                                  "what are your features?", "ماذا تستطيع أن تفعل؟"
  "registry_lookup"   — factual lookup answerable from the manuscript registry without
                        vector search (counting, dates, regions, collector names).
                        Examples: "how many manuscripts?", "who collected them?",
                                  "what region are they from?"
  "instructor_debug"  — instructor / developer asks to inspect pipeline internals.
                        Examples: "what was your CRAG verdict?", "explain RRF",
                                  "show the last turn debug info", "explain your fallback",
                                  "what self-query filters did you extract?"
  "poetic_rag"        — semantic poetry search, thematic query, literary question,
                        anything that requires retrieving verses from the corpus.
                        Examples: "poems about longing", "verses on falconry",
                                  "what did Al-Hazani write about the desert?"

IMPORTANT: Default to "poetic_rag" when uncertain. Only classify as "capabilities" or
"instructor_debug" when very confident. Never classify a genuine poetry question as
"registry_lookup" — that track is for corpus-level statistics only.
Return ONLY the JSON object, no surrounding text.
"""


# ── Cached classifier ─────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _classify_cached(normalised_query: str) -> dict:
    """
    Why cached: demo queries repeat frequently. lru_cache(256) keeps the most
    recent 256 normalised queries free of cloud API calls. Keyed on normalised
    text (harakat stripped, ya/alef-maksura unified) so vocalised and unvocalised
    versions of the same query hit the same cache slot — see §Decision §8.
    """
    result: list[Optional[dict]] = [None]
    error:  list[Optional[Exception]] = [None]

    def _target() -> None:
        try:
            result[0] = chat(
                prompt=f"intent_router_classify:: {normalised_query}",
                system=_SYSTEM_ROUTER,
                json_schema={"type": "object"},
                max_tokens=120,
            )
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=ROUTER_TIMEOUT_S)

    if t.is_alive():
        logger.warning(
            "semantic_router: LLM call exceeded %.1f s timeout — defaulting to poetic_rag.",
            ROUTER_TIMEOUT_S,
        )
        return _default_result("timeout")

    if error[0]:
        logger.warning(
            "semantic_router: LLM call failed (%s) — defaulting to poetic_rag.",
            error[0],
        )
        return _default_result("error")

    raw = result[0]
    if isinstance(raw, dict) and _is_valid(raw):
        return raw

    logger.warning(
        "semantic_router: unexpected LLM output %r — defaulting to poetic_rag.", raw
    )
    return _default_result("invalid_response")


def _default_result(reason: str) -> dict:
    return {
        "track":      "poetic_rag",
        "subintent":  None,
        "confidence": 0.5,
        "alt_family": None,
        "reasoning":  f"default — {reason}",
    }


def _is_valid(d: dict) -> bool:
    """Basic schema check on the LLM response dict."""
    return (
        isinstance(d.get("track"), str)
        and d["track"] in _VALID_TRACKS
        and isinstance(d.get("confidence"), (int, float))
    )


# ── Node callable ─────────────────────────────────────────────────────────────

def semantic_router_node(state: AgentState) -> AgentState:
    """
    Stage 0.5b — LLM-backed intent classification.
    Called only when Stage 0.5a (intent_router) did NOT short-circuit.

    Reads:  state["raw_query"]
    Writes: state["query_context"] —
              track, intent_subintent, intent_confidence_router,
              intent_alt_family, intent_router_reasoning, router_source,
              answer_source (set to "registry_lookup" for non-poetic_rag),
              query_lang, query_ar, query_en (minimal set for deterministic_answer_node)

    Why we set minimal Stage-1 fields here: deterministic_answer_node needs
    query_lang to choose the response language, but bilingual_analyzer is skipped
    for non-poetic_rag tracks to save latency. We supply safe heuristic values.
    """
    raw_query = state.get("raw_query", "")
    if not raw_query:
        return state

    norm_key = _make_cache_key(raw_query)
    result = _classify_cached(norm_key)

    track      = result.get("track", "poetic_rag")
    confidence = float(result.get("confidence", 0.5))

    if track != "poetic_rag" and confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "semantic_router: track=%s confidence=%.2f below threshold %.2f — falling back to poetic_rag.",
            track, confidence, CONFIDENCE_THRESHOLD,
        )
        track  = "poetic_rag"
        result = _default_result("low_confidence")

    logger.info(
        "semantic_router: track=%s confidence=%.2f subintent=%r reasoning=%r",
        track, confidence,
        result.get("subintent"),
        result.get("reasoning", ""),
    )

    has_arabic = any("؀" <= c <= "ۿ" for c in raw_query)
    qc = state.get("query_context") or {}

    new_qc: dict = {
        **qc,
        "track":                   track,
        "intent_subintent":        result.get("subintent"),
        "intent_confidence_router": confidence,
        "intent_alt_family":       result.get("alt_family"),
        "intent_router_reasoning": result.get("reasoning"),
        "router_source":           "llm",
    }

    if track != "poetic_rag":
        # All non-poetic_rag tracks must pair with answer_source="registry_lookup"
        # so the orchestrator routes to deterministic_answer_node (§validate_query_context).
        new_qc["answer_source"] = "registry_lookup"
        if "query_lang" not in new_qc:
            new_qc["query_lang"] = "ar" if has_arabic else "en"
        if "query_ar" not in new_qc:
            new_qc["query_ar"] = raw_query if has_arabic else ""
        if "query_en" not in new_qc:
            new_qc["query_en"] = "" if has_arabic else raw_query
    else:
        new_qc["answer_source"] = new_qc.get("answer_source", "rag_pipeline")

    state["query_context"] = new_qc  # type: ignore[assignment]
    return state
