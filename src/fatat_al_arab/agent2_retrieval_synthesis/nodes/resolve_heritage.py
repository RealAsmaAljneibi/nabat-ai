"""
agent2_retrieval_synthesis/nodes/resolve_heritage.py
=====================================================
Why this node exists: §2.5 Stage 6 — every answer must carry a full citation
(poet | volume | page | image path). The fused chunks from Stage 5 carry only
the fields baked into the Qdrant payload. This node joins back to the full
anchor registry and the poets_bio.json to attach:
  - The exact matla text (opening verse) as ground-truth text
  - Poet biography (bilingual, ≤120 words)
  - Volume and page for the "turn to page X" citation
  - Source image path so the Streamlit viewer can open the folio

Why join here (not in Stage 4): Stage 4 retrieves at speed; carrying full
registry entries through the RRF step would double memory usage per query.
Better to resolve citations only for the top-N chunks that survive fusion.

Unresolvable citation handling: §2.9 guardrails say "if citation cannot be
resolved, abstain rather than hallucinate". If anchor_id is not in the
registry we mark the chunk as `citation_resolvable: false` and it will be
filtered out by the guardrail node in Stage 10.

Architecture refs: §2.5 Stage 6 (Heritage Resolution), §2.9 (Guardrail 1).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from fatat_al_arab.state import AgentState, trace_append

logger = logging.getLogger(__name__)

# ── Data paths ────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent          # …/handwritten-poems
_REGISTRY_PATH = _REPO_ROOT / "data" / "ground_truth" / "anchor_registry_phase4.json"
_BIOS_PATH     = _REPO_ROOT / "data" / "ground_truth" / "poets_bio.json"


# ── Cached data loaders ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_registry() -> dict[str, dict]:
    """
    Load anchor_registry_phase4.json keyed by source_row_id.
    Cached — the 1,502-entry registry is ~800 KB; loading it once is fine.
    """
    try:
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            entries = json.load(f)
        return {e["source_row_id"]: e for e in entries if e.get("source_row_id")}
    except FileNotFoundError:
        logger.warning("resolve_heritage: registry not found at %s", _REGISTRY_PATH)
        return {}


@lru_cache(maxsize=1)
def _load_bios() -> dict[str, dict]:
    """
    Load poets_bio.json keyed by normalised_name.
    Falls back to poet_name if normalised_name is absent.
    """
    try:
        with open(_BIOS_PATH, encoding="utf-8") as f:
            bios = json.load(f)
        index = {}
        for bio in bios:
            key = bio.get("normalised_name") or bio.get("poet_name") or ""
            if key:
                index[key] = bio
            # Also index by poet_name directly for looser matching
            pn = bio.get("poet_name") or ""
            if pn and pn != key:
                index[pn] = bio
        return index
    except FileNotFoundError:
        logger.warning("resolve_heritage: bios not found at %s", _BIOS_PATH)
        return {}


def _find_bio(poet_name: str, bios: dict[str, dict]) -> dict | None:
    """
    Look up a poet biography. Tries exact match, then substring scan.
    Why substring: poet names in the registry may include honorifics not
    present in the bios index (e.g., "الشاعر ناصر الهزاني" vs "ناصر الهزاني").
    """
    if not poet_name:
        return None
    # Exact match
    if poet_name in bios:
        return bios[poet_name]
    # Substring scan (expensive but only over ~30 bios)
    pn_lower = poet_name.lower()
    for key, bio in bios.items():
        if key.lower() in pn_lower or pn_lower in key.lower():
            return bio
    return None


# ── Resolution logic ──────────────────────────────────────────────────────────

def _resolve_chunk(chunk_dict: dict, registry: dict, bios: dict) -> dict:
    """
    Enrich a single chunk dict with full citation data.
    Returns a new dict with added fields; never mutates the input.
    """
    anchor_id = chunk_dict.get("anchor_id") or ""
    level     = (chunk_dict.get("level") or "").lower()
    registry_entry = registry.get(anchor_id)

    resolved = dict(chunk_dict)   # shallow copy

    # Reference-level chunks (scholarly PDFs) carry their own citation payload —
    # bilingual book title + page — in `extra`. They never need a manuscript
    # registry row; treat them as resolvable so the synthesiser is allowed to
    # cite them.
    if level == "reference":
        extra = chunk_dict.get("extra") or {}
        resolved["citation_resolvable"]  = True
        resolved["matla_text"]           = chunk_dict.get("text") or ""
        resolved["poet_name"]            = ""
        resolved["source_volume"]        = extra.get("book_title_ar") or chunk_dict.get("source_volume", "")
        resolved["source_page"]          = int(chunk_dict.get("source_page") or 0)
        resolved["source_image_path"]    = extra.get("source_pdf") or chunk_dict.get("source_image_path", "")
        resolved["manuscript_short_key"] = ""
        resolved["manuscript_arabic_name"]  = extra.get("book_title_ar", "")
        resolved["manuscript_english_name"] = extra.get("book_title_en", "")
        resolved["bio_ar"] = ""
        resolved["bio_en"] = ""
        resolved["bio_sources"] = []
        # The two extra fields the UI uses for the 📚 bilingual badge:
        resolved["reference_topic_ar"] = extra.get("topic_ar", "")
        resolved["reference_topic_en"] = extra.get("topic_en", "")
        return resolved

    if registry_entry is None:
        resolved["citation_resolvable"] = False
        resolved["citation_note"] = f"anchor_id '{anchor_id}' not found in registry"
        logger.debug("resolve_heritage: unresolvable anchor_id=%r", anchor_id)
        return resolved

    # ── Citation fields from registry ──────────────────────────────────
    resolved["citation_resolvable"]  = True
    resolved["matla_text"]           = registry_entry.get("matla_text") or ""
    resolved["poet_name"]            = registry_entry.get("poet_name") or chunk_dict.get("poet_name") or ""
    resolved["source_volume"]        = registry_entry.get("source_volume") or chunk_dict.get("source_volume") or ""
    resolved["source_page"]          = int(registry_entry.get("page_number") or chunk_dict.get("source_page") or 0)
    resolved["source_image_path"]    = registry_entry.get("source_image_path") or chunk_dict.get("source_image_path") or ""
    resolved["manuscript_short_key"] = registry_entry.get("manuscript_short_key") or chunk_dict.get("manuscript_short_key") or ""
    resolved["manuscript_arabic_name"]  = registry_entry.get("manuscript_arabic_name") or ""
    resolved["manuscript_english_name"] = registry_entry.get("manuscript_english_name") or ""
    resolved["layout_type"]          = registry_entry.get("layout_type") or ""
    resolved["occasion"]             = registry_entry.get("occasion")

    # ── Poet biography ─────────────────────────────────────────────────
    bio = _find_bio(resolved["poet_name"], bios)
    if bio:
        resolved["bio_ar"] = bio.get("bio_ar") or ""
        resolved["bio_en"] = bio.get("bio_en") or ""
        resolved["bio_sources"] = bio.get("sources") or []
    else:
        resolved["bio_ar"] = ""
        resolved["bio_en"] = ""
        resolved["bio_sources"] = []

    return resolved


# ── LangGraph node ─────────────────────────────────────────────────────────────

def resolve_heritage_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 6.

    Reads:
      state["rrf_top5"]       — fused chunks from Stage 5 (list[dict])
    Writes:
      state["resolved_passages"]  — enriched chunks with full citation payload
                                    (list[dict]); unresolvable ones are tagged
                                    citation_resolvable=False for the guardrail.
    """
    rrf_top = state.get("rrf_top5") or []
    if not rrf_top:
        logger.warning("resolve_heritage_node: rrf_top5 is empty — nothing to resolve.")
        return {**state, "resolved_passages": []}

    registry = _load_registry()
    bios     = _load_bios()

    resolved = [_resolve_chunk(chunk, registry, bios) for chunk in rrf_top]

    resolvable_count   = sum(1 for r in resolved if r.get("citation_resolvable"))
    unresolvable_count = len(resolved) - resolvable_count
    # Why split row-backed vs roll-up: aggregate levels (manuscript/poet/era/genre/
    # emotion) and reference (PDF) chunks have no registry row by design — that's
    # not an error. Only WARN when a row-backed chunk (verse/group/poem) genuinely
    # fails to resolve, which indicates a registry/index drift bug.
    _NON_ROW_LEVELS = {"manuscript", "poet", "era", "genre", "emotion", "reference"}
    real_misses = [
        r for r in resolved
        if not r.get("citation_resolvable")
        and (r.get("level") or "").lower() not in _NON_ROW_LEVELS
    ]
    if real_misses:
        logger.warning(
            "resolve_heritage_node: %d/%d row-backed chunks have unresolvable citations "
            "(possible registry drift).",
            len(real_misses), len(resolved),
        )
    elif unresolvable_count:
        logger.debug(
            "resolve_heritage_node: %d/%d chunks unresolved by design (aggregate/reference levels).",
            unresolvable_count, len(resolved),
        )

    logger.debug(
        "resolve_heritage_node: resolved %d/%d passages.",
        resolvable_count, len(resolved),
    )
    trace_summary = f"resolved {resolvable_count}/{len(resolved)} passages — citations attached"
    return {
        **state,
        "resolved_passages": resolved,
        "agent_trace": trace_append(state, stage="6", icon="📜", label="Heritage Resolution", summary=trace_summary),
    }


# ── Utility for tests / evaluation ────────────────────────────────────────────

def clear_caches() -> None:
    """Clear the LRU caches — used in tests that inject custom data paths."""
    _load_registry.cache_clear()
    _load_bios.cache_clear()
