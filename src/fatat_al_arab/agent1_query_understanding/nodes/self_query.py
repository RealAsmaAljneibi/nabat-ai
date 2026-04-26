"""
agent1_query_understanding/nodes/self_query.py
================================================
Why this node exists: §2.4 Stage 3 (Self-Query Extraction) converts natural-
language intent into typed Qdrant filters. Without this, every query does a full-
collection scan regardless of whether the user said "show me what Ibn Yahya wrote
about the sea" — a constraint that could reduce the search space from 1,500 to
~80 candidates instantly.

Two filter tiers (§M4 spec):
  - hard filters  (confidence ≥ 0.7): applied as Qdrant payload pre-filters
                  (fast, deterministic, may return zero results if wrong)
  - soft filters  (confidence < 0.7): applied as score boosts inside RRF
                  (looser, acts like a preference rather than a gate)

Manuscript detection is registry-aware (§M4 spec): when the user mentions a
manuscript by any of its names ("Huber Manuscript", "مخطوطة ابن يحي", "Ibn Yahya
vol 3"), we resolve it to the canonical short_key before the filter is stored.
This prevents the Qdrant filter from failing on name variations.

Architecture refs: §2.4 Stage 3 (Self-Query Extraction), §2.9 tool registry
('extract_filters'), §5 (1 retry, 3 s timeout — fallback: unfiltered retrieval).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

from ...llm import chat
from ...state import AgentState

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SELF_QUERY_TIMEOUT_S      = 3.0   # §5 hard timeout
HARD_FILTER_CONFIDENCE    = 0.7   # threshold for hard vs. soft filter

_SYSTEM_SELF_QUERY = """\
You are a specialist in Nabati (Khaleeji Gulf) poetry manuscripts. Your task is to
extract structured search filters from a poetry query.

Return a JSON object with these fields (all optional except "confidence"):
{
  "poet":                 <string | null>   — poet's Arabic name as written, or null,
  "manuscript_id":        <string | null>   — manuscript identifier mentioned, or null,
  "manuscript_name_hint": <string | null>   — raw name as user typed it (for fuzzy match),
  "theme":                <string | null>   — primary theme keyword (Arabic preferred),
  "genre":                <string | null>   — one of the Nabati genre labels below, or null,
  "emotions":             <list[str] | null> — subset of the emotion labels below, or null/[],
  "verse_min":            <int | null>      — minimum verse number if specified,
  "verse_max":            <int | null>      — maximum verse number if specified,
  "page_min":             <int | null>      — minimum page number if specified,
  "page_max":             <int | null>      — maximum page number if specified,
  "confidence":           <float 0.0–1.0>   — your confidence in the overall extraction
}

Valid genre labels (use EXACTLY one of these strings, or null):
  غزل (love/romantic), رثاء (elegy/lamentation), مديح (praise/panegyric),
  هجاء (satire/invective), فخر (boasting/self-praise), حكمة (wisdom/aphorism),
  وصف (description/nature), دينية (religious/devotional), غزو (raid/war narrative),
  غير_محدد (unclassified — use this if the genre is ambiguous)

Valid emotion labels (use one or more, or null):
  longing, grief, nostalgia, love, joy, awe, pride, fear, anger, hope

Rules:
- Set a field to null if not mentioned or not clearly implied.
- Use the Arabic poet name as it would appear in classical sources.
- For manuscripts, return the name exactly as the user typed it in "manuscript_name_hint".
- For genre: set to null if the user did not specify a genre — do not guess.
- For emotions: set to null or [] if the user did not mention emotional tone.
- "confidence" reflects how certain you are about the overall extraction, not per-field.
- Return ONLY the JSON object, no explanation.

Examples:
  "show me love poems" → genre: "غزل"
  "أريد قصائد الغزل" → genre: "غزل"
  "poems about grief and loss" → genre: "رثاء", emotions: ["grief"]
  "praise poems for ibn rashid" → genre: "مديح"
  "wisdom sayings about patience" → genre: "حكمة"
  "war poems, poems about raids" → genre: "غزو"
"""

# ── Registry helpers ──────────────────────────────────────────────────────────

def _resolve_manuscript(name_hint: Optional[str]) -> Optional[str]:
    """
    Why registry resolution here: the Qdrant filter uses short_key values
    (e.g. 'ibn_yahya_401_600'), not display names. If we store the raw user string
    as the filter value, the filter will never match anything.

    Tries exact short_key match first, then english_name, then arabic_name,
    then a RapidFuzz partial-match fallback. Returns None if nothing matches at
    score ≥ 60 (better to run unfiltered than to filter on the wrong manuscript).
    """
    if not name_hint:
        return None

    try:
        from al_nassikh.registry import by_short_key, list_all
    except ImportError:
        logger.warning("self_query: could not import registry — skipping manuscript resolution.")
        return None

    all_entries = list_all()

    # Exact short_key match (e.g. if something upstream already normalised it)
    if by_short_key(name_hint):
        return name_hint

    hint_lower = name_hint.lower().strip()

    # Try substring match on english_name and arabic_name
    for entry in all_entries:
        en = entry.get("english_name", "").lower()
        ar = entry.get("arabic_name", "")
        sk = entry.get("short_key", "")
        if hint_lower in en or hint_lower in ar.lower() or hint_lower in sk:
            logger.debug(
                "self_query: resolved '%s' → '%s' (substring match)", name_hint, sk
            )
            return sk

    # Fuzzy fallback (RapidFuzz)
    try:
        from rapidfuzz import process, fuzz  # noqa: PLC0415

        candidates = {
            e["short_key"]: e["english_name"] + " " + e["arabic_name"]
            for e in all_entries
        }
        match = process.extractOne(
            hint_lower,
            candidates,
            scorer=fuzz.partial_ratio,
        )
        if match and match[1] >= 60:
            logger.debug(
                "self_query: fuzzy resolved '%s' → '%s' (score=%d)",
                name_hint, match[2], match[1],
            )
            return match[2]  # match[2] is the key (short_key)
    except ImportError:
        pass

    logger.info(
        "self_query: could not resolve manuscript '%s' — filter will be skipped.", name_hint
    )
    return None


# ── Extraction with timeout ───────────────────────────────────────────────────

def _extract_with_timeout(query_ar: str, query_en: str) -> Optional[dict]:
    """
    Why threading timeout: same reason as hyde.py — we need a wall-clock cap that
    works regardless of the provider's built-in timeout. §5 says 3 s.
    """
    result: list[Optional[dict]] = [None]
    error:  list[Optional[Exception]] = [None]

    def _target() -> None:
        try:
            prompt = (
                f"Arabic query: {query_ar}\n"
                f"English query: {query_en}\n\n"
                "Extract structured filters."
            )
            raw = chat(
                prompt=prompt,
                system=_SYSTEM_SELF_QUERY,
                json_schema={"type": "object"},
                max_tokens=300,
            )
            result[0] = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=SELF_QUERY_TIMEOUT_S)

    if t.is_alive():
        logger.warning(
            "self_query: extraction exceeded %.1f s — proceeding with no filters (§5).",
            SELF_QUERY_TIMEOUT_S,
        )
        return None

    if error[0]:
        logger.warning("self_query: extraction failed — %s — proceeding with no filters.", error[0])
        return None

    return result[0]


def _build_filters(raw: dict) -> tuple[dict, dict]:
    """
    Convert the raw LLM extraction into hard_filters and soft_filters dicts
    suitable for the Qdrant retriever.

    Why two dicts: §M4 says "confidence ≥ 0.7 → hard Qdrant pre-filter;
    < 0.7 → soft boost". The retriever node reads both and applies them
    differently (pre-filter vs. RRF score bonus).
    """
    confidence: float = float(raw.get("confidence", 0.0))

    # Resolve manuscript name to short_key
    ms_short_key = _resolve_manuscript(raw.get("manuscript_name_hint"))
    if ms_short_key is None and raw.get("manuscript_id"):
        ms_short_key = _resolve_manuscript(raw.get("manuscript_id"))

    candidate_filters: dict[str, Any] = {}

    if raw.get("poet"):
        candidate_filters["poet_name"] = raw["poet"]
    if ms_short_key:
        candidate_filters["manuscript_short_key"] = ms_short_key
    if raw.get("theme"):
        candidate_filters["theme"] = raw["theme"]

    # M3: genre filter — only add if explicitly stated and not the abstention label
    genre = raw.get("genre")
    if genre and genre != "غير_محدد":
        candidate_filters["genre"] = genre

    # M3: emotion filter — a list; stored as "emotions_any" (match any element)
    # Why "emotions_any" not "emotions": the payload stores a list; we want "at least
    # one of the requested emotions is present in the chunk's emotion list".
    emotions = raw.get("emotions") or []
    if emotions:
        candidate_filters["emotions_any"] = list(emotions)

    # Numeric range filters (page / verse)
    if raw.get("page_min") is not None:
        candidate_filters["page_min"] = raw["page_min"]
    if raw.get("page_max") is not None:
        candidate_filters["page_max"] = raw["page_max"]
    if raw.get("verse_min") is not None:
        candidate_filters["verse_min"] = raw["verse_min"]
    if raw.get("verse_max") is not None:
        candidate_filters["verse_max"] = raw["verse_max"]

    if confidence >= HARD_FILTER_CONFIDENCE:
        return candidate_filters, {}
    else:
        return {}, candidate_filters


def self_query_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.4 Stage 3 (Self-Query Extraction).

    Reads:   state["query_context"]["query_ar"], ["query_en"]
    Writes:  state["query_context"]["filters_hard"], ["filters_soft"],
             ["self_query_raw"]

    Failure behaviour: if extraction times out or fails, filters are left empty
    and retrieval proceeds without any pre-filtering (§5 "fallback: unfiltered
    retrieval").
    """
    qc = state.get("query_context", {})

    if qc.get("needs_clarification"):
        logger.debug("self_query: skipping — clarification path active.")
        return state

    query_ar: str = qc.get("query_ar", state.get("raw_query", ""))
    query_en: str = qc.get("query_en", state.get("raw_query", ""))

    raw = _extract_with_timeout(query_ar, query_en)

    if raw is None:
        # §5 fallback — empty filters means retrieval scans the full index
        updated_qc = {
            **qc,
            "filters_hard": {},
            "filters_soft": {},
            "self_query_raw": {},
        }
        return {**state, "query_context": updated_qc}

    hard_filters, soft_filters = _build_filters(raw)

    logger.debug(
        "self_query: hard=%r soft=%r (confidence=%.2f)",
        hard_filters, soft_filters, raw.get("confidence", 0.0),
    )

    updated_qc = {
        **qc,
        "filters_hard": hard_filters,
        "filters_soft": soft_filters,
        "self_query_raw": raw,
    }
    return {**state, "query_context": updated_qc}


def extract_filters(query_ar: str, query_en: str) -> dict:
    """
    Tool-registry entry point (§2.9 'extract_filters').
    Returns {"filters_hard": {...}, "filters_soft": {...}, "raw": {...}}
    """
    raw = _extract_with_timeout(query_ar, query_en) or {}
    hard, soft = _build_filters(raw) if raw else ({}, {})
    return {"filters_hard": hard, "filters_soft": soft, "raw": raw}
