"""
agent2_retrieval_synthesis/nodes/format_variants.py
====================================================
Why this node exists: §2.5 Stage 10 — the final output stage. It takes the
synthesised Arabic draft and produces the four-variant output dict the Streamlit
UI displays, then runs the three guardrails. If any guardrail fires, the refusal
template replaces the draft before it reaches the UI.

Four output variants (§2.5 Stage 10, §3.2):
  al_maktub     — manuscript-script rendering (Arabic, right-to-left, original
                  orthography from matla_text). This is the "as written" version.
  orthographic  — Arabic Modern Standard lightly normalised (alef variants unified,
                  harakat stripped). Findable by a non-specialist.
  al_mantuq     — dialectal/spoken form of the query answer in Khaleeji Arabic.
                  Used by the Streamlit audio tab (TTS) in M8/M9.
  citations     — structured citation list: [{anchor_id, poet, volume, page, image}]

Why four variants: the target users span native Arabic speakers who want the
authentic orthography, researchers who use MSA search tools, and the demo panel
judge who wants to see the system handles dialectal normalisation.

Guardrails fire here, not in a separate node, because they gate the final_response
field. If we had a separate node the graph would need an extra edge; embedding
the check in Stage 10 keeps the graph simpler and makes the failure mode obvious.

Architecture refs: §2.5 Stage 10 (Multi-Variant Formatter), §2.9 (guardrails),
§3.2 (output schema).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fatat_al_arab.guardrails import (
    run_all,
    REFUSAL_TEMPLATE_AR,
    REFUSAL_TEMPLATE_EN,
)
from fatat_al_arab.state import AgentState

logger = logging.getLogger(__name__)


# ── Arabic normalisation helpers ──────────────────────────────────────────────

_ALEF_VARIANTS = re.compile(r"[آأإٱ]")
_HARAKAT       = re.compile(r"[\u064B-\u065F\u0670]")
_TATWEEL       = re.compile(r"ـ")


def _to_orthographic(text: str) -> str:
    """Light MSA normalisation: unify alef variants, strip harakat/tatweel."""
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _HARAKAT.sub("", text)
    text = _TATWEEL.sub("", text)
    return " ".join(text.split())


def _build_citation_list(passages: list[dict], citations_used: list[str]) -> list[dict]:
    """
    Build the structured citation list from resolved passages.
    Only include passages whose anchor_id appears in citations_used
    (or all resolvable passages if citations_used is empty — belt-and-suspenders).
    """
    resolvable = [p for p in passages if p.get("citation_resolvable", True)]

    if citations_used:
        # Keep only passages referenced in the draft
        cited_set = set(citations_used)
        resolvable = [
            p for p in resolvable
            if p.get("anchor_id") in cited_set or p.get("chunk_id") in cited_set
        ] or resolvable   # fallback: all if filter is too strict

    out: list[dict] = []
    for p in resolvable:
        cit: dict = {
            "anchor_id":     p.get("anchor_id", ""),
            "chunk_id":      p.get("chunk_id", ""),
            "level":         p.get("level", ""),
            "poet_name":     p.get("poet_name", ""),
            "source_volume": p.get("source_volume", ""),
            "source_page":   p.get("source_page", 0),
            "source_image_path": p.get("source_image_path", ""),
            "manuscript_arabic_name":  p.get("manuscript_arabic_name", ""),
            "manuscript_english_name": p.get("manuscript_english_name", ""),
        }
        # Pass through bilingual reference-book fields when present so the
        # Streamlit 📚 badge can render the EN + AR title side by side without
        # re-querying the chunk store.
        extra = p.get("extra") or {}
        is_reference = (p.get("level") or "").lower() == "reference" or extra.get("book_title_en")
        if is_reference:
            cit["book_title_ar"] = extra.get("book_title_ar") or p.get("manuscript_arabic_name", "")
            cit["book_title_en"] = extra.get("book_title_en") or p.get("manuscript_english_name", "")
            cit["topic_ar"]      = extra.get("topic_ar", p.get("reference_topic_ar", ""))
            cit["topic_en"]      = extra.get("topic_en", p.get("reference_topic_en", ""))
        out.append(cit)
    return out


def _make_al_mantuq(draft: str, passages: list[dict]) -> str:
    """
    Dialectal Khaleeji rendering — use bio_ar snippets and matla texts as-is.
    For the MVP this is the same as the draft with a dialect marker header.
    A proper dialectal transformer would be a separate M8+ component.
    """
    matlas = [
        p.get("matla_text") or "" for p in passages
        if p.get("citation_resolvable") and p.get("matla_text")
    ]
    if matlas:
        matla_block = "\n".join(f"• {m}" for m in matlas[:3])
        return f"[الصيغة الخليجية المنطوقة]\n{matla_block}\n\n{draft}"
    return draft


def format_variants_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 10.

    Reads:
      state["draft_response"]     — Arabic draft from Stage 8
      state["resolved_passages"]  — enriched chunks from Stage 6
      state["citations_used"]     — anchor_ids cited in draft
      state["passage_ids_used"]   — chunk_ids that contributed
      state["is_refusal"]         — True → emit refusal template
      state["crag_verdict"]       — forwarded to guardrail (c)
    Writes:
      state["formatted_response"] — {al_maktub, orthographic, al_mantuq, citations}
      state["guardrail_passed"]   — bool
      state["guardrail_flags"]    — list of failure reasons
      state["final_response"]     — plain text for the UI (Arabic)
    """
    draft:        str        = state.get("draft_response") or ""
    passages:     list[dict] = state.get("resolved_passages") or []
    citations:    list[str]  = state.get("citations_used") or []
    passage_ids:  list[str]  = state.get("passage_ids_used") or []
    is_refusal:   bool       = bool(state.get("is_refusal"))
    crag_verdict: str | None = state.get("crag_verdict")

    # ── Run guardrails ─────────────────────────────────────────────────────────
    # Load registry entries for guardrail (a) — use resolved_passages as proxy
    # (the guardrail checks citation tags in the draft match real anchor_ids)
    registry_proxy = [
        {"source_row_id": p.get("anchor_id"), "source_volume": p.get("source_volume"),
         "page_number": p.get("source_page")}
        for p in passages if p.get("anchor_id")
    ]

    guardrail_result = run_all(
        response_text=draft,
        anchor_registry=registry_proxy,
        approved_passages=passages,
        passage_ids_used=passage_ids,
        is_refusal=is_refusal,
        crag_verdict=crag_verdict,
    )

    final_text = draft
    if not guardrail_result.passed and not is_refusal:
        logger.warning(
            "format_variants_node: guardrail FAILED: %s — substituting refusal.",
            guardrail_result.flags,
        )
        final_text = f"{REFUSAL_TEMPLATE_AR}\n\n{REFUSAL_TEMPLATE_EN}"
        is_refusal = True

    # ── Build four-variant output ──────────────────────────────────────────────
    # al_maktub: original manuscript script — use matla_texts verbatim
    al_maktub_lines = [
        p.get("matla_text") or p.get("text") or ""
        for p in passages
        if p.get("citation_resolvable") and (p.get("matla_text") or p.get("text"))
    ]
    al_maktub = "\n".join(al_maktub_lines) if al_maktub_lines else final_text

    formatted_response = {
        "al_maktub":    al_maktub,
        "orthographic": _to_orthographic(final_text),
        "al_mantuq":    _make_al_mantuq(final_text, passages),
        "citations":    _build_citation_list(passages, citations),
    }

    logger.debug(
        "format_variants_node: guardrail=%s, citations=%d, is_refusal=%s",
        "PASS" if guardrail_result.passed else "FAIL",
        len(formatted_response["citations"]),
        is_refusal,
    )

    return {
        **state,
        "formatted_response": formatted_response,
        "guardrail_passed":   guardrail_result.passed or is_refusal,
        "guardrail_flags":    guardrail_result.flags,
        "final_response":     final_text,
        "is_refusal":         is_refusal,
    }
