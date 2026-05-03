"""
creative_poet/nodes/musharik.py
=============================================
Why this node exists: Al-Musharik (المشارك — The Co-Author) is the interactive partner.
The human writes the first hemistich (sadr — صدر); Al-Musharik offers exactly three
candidate second hemistichs (ajuz — عجز), each annotated with meter conformity, rhyme
match, and thematic consistency. The poet selects, rejects, or ignores all three.

Three sub-nodes (LangGraph graph registers these individually):
  musharik_retrieve_node     — calls Fatat Al-Arab to find similar verse structures
  musharik_generate_node     — LLM: generate 3 ajuz candidates
  musharik_quality_gate_node — validates candidate count/quality; conditional retry

Agentic loop (Observe → Reflect → Retry):
  musharik_retrieve → musharik_generate → musharik_quality_gate
                             ↑ (retry once if < AJUZ_CANDIDATE_COUNT valid candidates)

§5 failure budget: if retrieval yields no similar structures, generate from sadr alone —
the LLM's parametric Nabati knowledge is sufficient for stylistic plausibility when the
goal is suggestion, not verbatim citation.

Tools declared in tools.py:
  QUERY_FATAT_RETRIEVE, GENERATE_AJUZ_CANDIDATES
"""

from __future__ import annotations

import json
import logging

from fatat_al_arab.llm import chat
from creative_poet.personas import MUSHARIK_PERSONA
from fatat_al_arab.state import trace_append
from creative_poet.tools import (
    check_tool_permitted,
    QUERY_FATAT_RETRIEVE,
    GENERATE_AJUZ_CANDIDATES,
)

logger = logging.getLogger(__name__)

# ── §5 failure budget constants ───────────────────────────────────────────────
AJUZ_CANDIDATE_COUNT = 3    # always offer exactly 3 options
MAX_AJUZ_RETRIES     = 1    # one retry on quality failure

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_MUSHARIK = (
    MUSHARIK_PERSONA
    + """
Task — Ajuz Candidate Generation (Stage M2):
Given a human poet's sadr (first hemistich) and examples of how Nabati poets complete
similar verse structures from the corpus, generate exactly 3 candidate ajuz (second
hemistich) options. The candidates must differ meaningfully:
  Candidate 1 — literal / direct continuation of the sadr's explicit imagery
  Candidate 2 — metaphorical / extended image from the same thematic root
  Candidate 3 — emotional / turning point — shifts register or introduces contrast

Return JSON:
{
  "candidates": [
    {
      "ajuz":             "<the candidate second hemistich in Khaleeji Arabic>",
      "type":             "literal|metaphorical|emotional",
      "meter_ok":         true|false,
      "rhyme_ok":         true|false,
      "thematic_score":   <1-5>,
      "annotation":       "<one sentence: how this ajuz relates to the sadr's imagery>"
    }
  ]
}
Return ONLY valid JSON. Exactly 3 candidates in the array.
"""
)


# ── Sub-node 1: retrieve ──────────────────────────────────────────────────────

def musharik_retrieve_node(state: dict) -> dict:
    """
    LangGraph node — Stage M2a.
    Calls Fatat Al-Arab to find corpus verses with structural patterns similar to the sadr.
    Permission enforced: QUERY_FATAT_RETRIEVE.

    Reads:  composition_context (input_sadr)
    Writes: style_exemplars (similar verse structures), fatat_retrieved_count
    """
    check_tool_permitted("musharik", QUERY_FATAT_RETRIEVE)

    from creative_poet.external_tools import fatat_retrieve_similar_structures

    ctx  = state.get("composition_context") or {}
    sadr = ctx.get("input_sadr") or ""

    similar: list[dict] = []
    if sadr.strip():
        similar = fatat_retrieve_similar_structures(sadr, n=5)

    logger.debug("musharik_retrieve_node: %d similar structures for sadr=%r", len(similar), sadr[:40])
    return {
        **state,
        "style_exemplars":     similar,
        "fatat_retrieved_count": len(similar),
        "agent_trace": trace_append(
            state, stage="M2a", icon="🤝", label="Al-Musharik / Retrieve",
            summary=f"{len(similar)} similar structures from Fatat Al-Arab",
        ),
    }


# ── Sub-node 2: generate ──────────────────────────────────────────────────────

def musharik_generate_node(state: dict) -> dict:
    """
    LangGraph node — Stage M2b.
    LLM call: generate exactly AJUZ_CANDIDATE_COUNT ajuz candidates.
    Permission enforced: GENERATE_AJUZ_CANDIDATES.

    Reads:  composition_context (input_sadr, session_verses), style_exemplars
    Writes: ajuz_candidates, ajuz_quality_ok (False — quality_gate_node decides)
            Increments ajuz_retry_count if this is a retry.
    """
    check_tool_permitted("musharik", GENERATE_AJUZ_CANDIDATES)

    ctx            = state.get("composition_context") or {}
    sadr: str      = ctx.get("input_sadr") or ""
    session_verses = ctx.get("session_verses") or []
    similar        = state.get("style_exemplars") or []

    retry_count = (state.get("ajuz_retry_count") or 0) + (
        1 if state.get("ajuz_retry_count") is not None else 0
    )

    if not sadr.strip():
        empty_msg = "لم يُقدَّم الصدر. أدخل الشطر الأول من البيت. (No sadr provided.)"
        return {
            **state,
            "ajuz_candidates":  [],
            "ajuz_quality_ok":  False,
            "ajuz_retry_count": retry_count,
            "final_output":     empty_msg,
            "guardrail_passed": False,
            "guardrail_flags":  list(state.get("guardrail_flags") or []) + ["empty_sadr"],
            "agent_trace": trace_append(
                state, stage="M2b", icon="🤝", label="Al-Musharik / Generate",
                summary="No sadr provided",
            ),
        }

    examples = [
        {
            "text":      (p.get("matla_text") or p.get("text") or "")[:200],
            "anchor_id": p.get("anchor_id", ""),
            "poet":      p.get("poet_name", ""),
        }
        for p in similar[:3]
    ]

    session_ctx = ""
    if session_verses:
        recent = session_verses[-4:]
        session_ctx = "\n\nEstablished session verses (for rhyme continuity):\n" + "\n".join(recent)

    prompt = (
        f"Human poet's sadr: {sadr}\n\n"
        f"Similar corpus verses (structural reference):\n"
        f"{json.dumps(examples, ensure_ascii=False)}"
        f"{session_ctx}"
    )

    candidates: list[dict] = []
    try:
        result = chat(
            prompt=prompt,
            system=_SYSTEM_MUSHARIK,
            json_schema={"type": "object"},
            max_tokens=700,
        )
        parsed     = result if isinstance(result, dict) else json.loads(str(result))
        candidates = (parsed.get("candidates") or [])[:AJUZ_CANDIDATE_COUNT]
    except Exception as exc:
        logger.warning("musharik_generate: ajuz generation failed (%s)", exc)

    return {
        **state,
        "ajuz_candidates":  candidates,
        "ajuz_quality_ok":  False,   # musharik_quality_gate_node decides
        "ajuz_retry_count": retry_count,
        "agent_trace": trace_append(
            state, stage="M2b", icon="🤝", label="Al-Musharik / Generate",
            summary=(
                f"{len(candidates)} candidates generated "
                f"(retry {retry_count}/{MAX_AJUZ_RETRIES})"
            ),
        ),
    }


# ── Sub-node 3: quality gate ──────────────────────────────────────────────────

def musharik_quality_gate_node(state: dict) -> dict:
    """
    LangGraph node — Observe → Reflect step for Al-Musharik.
    Checks whether all AJUZ_CANDIDATE_COUNT candidates have valid ajuz text.
    The graph's conditional edge routes back to musharik_generate_node for one retry.

    Reads:  ajuz_candidates, ajuz_retry_count
    Writes: ajuz_quality_ok, final_output (when quality passed or retries exhausted)
    """
    candidates  = state.get("ajuz_candidates") or []
    quality_ok  = _ajuz_candidates_are_quality(candidates)
    retry_count = state.get("ajuz_retry_count") or 0
    ctx         = state.get("composition_context") or {}
    sadr        = ctx.get("input_sadr") or ""

    if quality_ok or retry_count >= MAX_AJUZ_RETRIES:
        # M9 Tier-1: consult Al-Muqayyim to grade and re-rank candidates.
        # Respects Muqayyim's isolation — graded text IS the user-submitted
        # work-in-progress, not corpus content. Returns top-ranked candidates
        # with muqayyim_score injected. On any failure (depth cap, LLM down)
        # the candidates pass through unchanged so the pipeline degrades gracefully.
        consult_trace = ""
        if candidates:
            try:
                from creative_poet.tools import check_tool_permitted as _check, CONSULT_MUQAYYIM_GRADE
                from creative_poet.external_tools import muqayyim_grade_ajuz_candidates, _record_consult
                _check("musharik", CONSULT_MUQAYYIM_GRADE)
                t0 = __import__("time").perf_counter()
                graded = muqayyim_grade_ajuz_candidates(sadr, candidates, state=state)
                ms = (__import__("time").perf_counter() - t0) * 1000
                state = _record_consult(state, "musharik", "muqayyim", ms)
                if graded:
                    candidates = graded[:3]   # take top-3 after re-ranking
                    consult_trace = f" · graded by Muqayyim → top-{len(candidates)}"
            except Exception as exc:
                logger.info("musharik_quality_gate: muqayyim consult skipped (%s)", exc)

        final_output = (
            _render_candidates(sadr, candidates)
            if candidates
            else "تعذّر توليد مقترحات العجز. (Ajuz generation failed — please retry.)"
        )
        candidate_summary = " · ".join(
            f"[{i+1}:{c.get('type','?')} {c.get('thematic_score','?')}/5"
            + (f" m={c['muqayyim_score']:.1f}" if c.get("muqayyim_score") else "")
            + "]"
            for i, c in enumerate(candidates)
        ) if candidates else "none generated"

        return {
            **state,
            "ajuz_candidates":  candidates,    # may be re-ranked
            "ajuz_quality_ok":  quality_ok,
            "final_output":     final_output,
            "guardrail_passed": bool(candidates),
            "guardrail_flags":  (
                [] if candidates
                else list(state.get("guardrail_flags") or []) + ["no_candidates_generated"]
            ),
            "agent_trace": trace_append(
                state, stage="M2c", icon="🤝", label="Al-Musharik / Quality Gate",
                summary=f"{len(candidates)} ajuz · {candidate_summary}{consult_trace}",
            ),
        }

    # Quality check failed and budget remains — signal for retry
    logger.info(
        "musharik_quality_gate: quality failed (retry %d/%d) — routing back to generate",
        retry_count, MAX_AJUZ_RETRIES,
    )
    return {
        **state,
        "ajuz_quality_ok": False,
        "agent_trace": trace_append(
            state, stage="M2c", icon="🤝", label="Al-Musharik / Quality Gate",
            summary=f"Quality insufficient — retry {retry_count + 1}/{MAX_AJUZ_RETRIES}",
        ),
    }


def _ajuz_candidates_are_quality(candidates: list) -> bool:
    """All AJUZ_CANDIDATE_COUNT candidates must have non-empty ajuz text."""
    if len(candidates) < AJUZ_CANDIDATE_COUNT:
        return False
    return all(c.get("ajuz", "").strip() for c in candidates)


# ── Monolithic wrapper (orchestrator direct-node fallback) ────────────────────

def musharik_node(state: dict) -> dict:
    """
    Sequential wrapper that chains sub-nodes.
    Used only by orchestrator.run_creative() when LangGraph is unavailable.
    """
    state = musharik_retrieve_node(state)
    state = musharik_generate_node(state)
    state = musharik_quality_gate_node(state)
    # one retry if quality failed
    if not state.get("ajuz_quality_ok") and (state.get("ajuz_retry_count") or 0) < MAX_AJUZ_RETRIES:
        state = musharik_generate_node(state)
        state = musharik_quality_gate_node(state)
    return state


def _render_candidates(sadr: str, candidates: list[dict]) -> str:
    """Format the three ajuz candidates as a readable markdown panel."""
    lines = [
        "## المشارك — Al-Musharik: Ajuz Candidates",
        f"**Your sadr:** {sadr}",
        "",
        "---",
    ]
    labels = {
        "literal":      "📌 Literal",
        "metaphorical": "🌙 Metaphorical",
        "emotional":    "💧 Emotional",
    }
    for i, c in enumerate(candidates):
        kind       = labels.get(c.get("type", ""), f"Option {i+1}")
        meter_icon = "✅" if c.get("meter_ok") else "⚠️"
        rhyme_icon = "✅" if c.get("rhyme_ok")  else "⚠️"
        lines += [
            f"### {kind} — Candidate {i+1}",
            f"> **{c.get('ajuz', '—')}**",
            f"",
            f"Meter {meter_icon} · Rhyme {rhyme_icon} · "
            f"Thematic score **{c.get('thematic_score','?')}/5**",
            f"_{c.get('annotation','')}_",
            "",
            "---",
        ]
    lines += [
        "",
        "> *اختر أو ارفض أو تجاهل — القرار لك دائماً.*",
        "> *Select, reject, or ignore — the choice is always yours.*",
    ]
    return "\n".join(lines)
