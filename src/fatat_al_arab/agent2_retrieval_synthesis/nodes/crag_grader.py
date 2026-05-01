"""
agent2_retrieval_synthesis/nodes/crag_grader.py
================================================
Why this node exists: §2.5 Stage 7 — Corrective RAG grading. Before the LLM
synthesises an answer it needs to know whether the retrieved passages actually
answer the question. If all passages are graded Incorrect, we either re-query
(first occurrence, §5 max 1 CRAG re-query) or fire the scoped-refusal path.

CRAG grading is an LLM call over the top resolved passages. The prompt asks:
  "Does this passage provide information relevant to [query]?"
  Grade: Correct | Ambiguous | Incorrect

Why LLM-graded (not a keyword heuristic): Nabati poetry questions are often
thematic ("poems about longing for the homeland"). A BM25-heavy retriever may
return passages with overlapping vocabulary that do not actually answer the
theme. Only the LLM can judge relevance at that level of nuance.

§5 failure budget: if the LLM grading call fails, default to Ambiguous so
synthesis still proceeds rather than blocking the whole request.

Verdict logic:
  - Any Correct  → verdict "Correct"  (synthesis proceeds)
  - All Ambiguous → verdict "Ambiguous" (synthesis proceeds with caveats)
  - All Incorrect → verdict "Incorrect" (re-query once, then refusal)

Architecture refs: §2.5 Stage 7 (CRAG), §5 (failure budgets, re-query cap 1).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fatat_al_arab.llm import chat
from fatat_al_arab.state import AgentState

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_GRADER = """\
You are an expert in Nabati Khaleeji poetry and classical Arabic manuscripts.
Grade each retrieved passage for relevance to the user query, then decide what to search for next.

Return a JSON object with this exact shape:
{
  "grades": [
    {
      "chunk_id":   "<id from input>",
      "label":      "Correct" | "Ambiguous" | "Incorrect",
      "confidence": <float 0.0–1.0>,
      "rationale":  "<one sentence: why this passage does or does not answer the query>"
    }
  ],
  "requery_strategy": "<if any grade is Incorrect or Ambiguous: 1–2 sentences naming the specific themes, Arabic terms, or manuscript sections that would yield better results. If all grades are Correct, return an empty string.>"
}

Grade "Correct" if the passage directly answers or provides strong evidence for the query.
Grade "Ambiguous" if topically related but does not directly answer.
Grade "Incorrect" if unrelated or contradicts the query.

Return ONLY valid JSON, no prose.
"""


def _grade_passages(query_ar: str, passages: list[dict]) -> tuple[list[dict], str]:
    """
    Call the LLM to grade each passage and generate a re-query strategy.
    Returns (grades, requery_strategy). Falls back to Ambiguous on LLM failure (§5 budget).
    """
    if not passages:
        return [], ""

    passage_summaries = [
        {"chunk_id": p.get("chunk_id", ""), "text": (p.get("matla_text") or p.get("text") or "")[:200]}
        for p in passages
    ]

    prompt = (
        f"Query: {query_ar}\n\n"
        f"Passages:\n{json.dumps(passage_summaries, ensure_ascii=False)}"
    )

    try:
        result = chat(
            prompt=prompt,
            system=_SYSTEM_GRADER,
            json_schema={"type": "object"},
            max_tokens=700,
        )
        if isinstance(result, str):
            result = json.loads(result)
        if isinstance(result, dict) and "grades" in result:
            return result["grades"], result.get("requery_strategy", "")
        # Old-format fallback: bare array (LLM ignored schema)
        if isinstance(result, list):
            return result, ""
    except Exception as exc:
        logger.warning("crag_grader: LLM grading failed (%s) — defaulting to Ambiguous.", exc)

    # §5 fallback: proceed with Ambiguous so synthesis is not blocked
    return (
        [
            {
                "chunk_id":   p.get("chunk_id", ""),
                "label":      "Ambiguous",
                "confidence": 0.5,
                "rationale":  "Grading unavailable — defaulting to Ambiguous (§5 fallback).",
            }
            for p in passages
        ],
        "",
    )


def _compute_verdict(grades: list[dict]) -> str:
    """
    Aggregate per-passage grades into an overall verdict.
    Any Correct → "Correct"; all Ambiguous → "Ambiguous"; all Incorrect → "Incorrect".
    """
    if not grades:
        return "Incorrect"
    labels = [g.get("label", "Incorrect") for g in grades]
    if "Correct" in labels:
        return "Correct"
    if all(l == "Incorrect" for l in labels):
        return "Incorrect"
    return "Ambiguous"


def crag_grader_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 7.

    Reads:
      state["resolved_passages"]  — enriched chunks from Stage 6
      state["query_context"]      — for query_ar
    Writes:
      state["crag_grades"]   — per-passage grade list
      state["crag_verdict"]  — "Correct" | "Ambiguous" | "Incorrect"
    """
    qc = state.get("query_context") or {}
    query_ar: str = qc.get("query_ar") or state.get("raw_query", "")
    passages: list[dict] = state.get("resolved_passages") or []

    # Only grade resolvable passages — unresolvable ones are already flagged
    gradeable = [p for p in passages if p.get("citation_resolvable", True)]

    from fatat_al_arab.state import trace_append

    if not gradeable:
        logger.warning("crag_grader_node: no resolvable passages to grade — verdict Incorrect.")
        return {
            **state,
            "crag_grades":  [],
            "crag_verdict": "Incorrect",
            "agent_trace": trace_append(state, stage="7", icon="⚖️", label="CRAG Grader", summary="No gradeable passages → Incorrect — re-querying"),
        }

    grades, requery_strategy = _grade_passages(query_ar, gradeable)
    verdict = _compute_verdict(grades)

    logger.debug(
        "crag_grader_node: graded %d passages → verdict=%s requery_strategy=%r",
        len(grades), verdict, requery_strategy[:80] if requery_strategy else "",
    )

    from collections import Counter
    grade_counts = Counter(g.get("label", "?") for g in grades)
    grade_str = " · ".join(f"{cnt} {lbl}" for lbl, cnt in sorted(grade_counts.items()))

    # Find the most critical grade's LLM-generated rationale to surface in the trace
    problem_grade = next(
        (g for g in grades if g.get("label") in ("Incorrect", "Ambiguous") and g.get("rationale")),
        None,
    )

    if verdict in ("Incorrect", "Ambiguous") and problem_grade:
        rationale = problem_grade["rationale"]
        rationale_preview = rationale[:150] + "…" if len(rationale) > 150 else rationale
        if requery_strategy:
            strat_preview = requery_strategy[:100] + "…" if len(requery_strategy) > 100 else requery_strategy
            trace_detail  = f'"{rationale_preview}"\n→ Searching instead for: {strat_preview}'
            trace_summary = f"{grade_str} — re-querying"
        else:
            trace_detail  = f'"{rationale_preview}"'
            trace_summary = f"{grade_str} → {verdict}"
    elif requery_strategy:
        strat_preview = requery_strategy[:120] + "…" if len(requery_strategy) > 120 else requery_strategy
        trace_detail  = f"Searching instead for: {strat_preview}"
        trace_summary = f"{grade_str} — re-querying"
    else:
        trace_detail  = ""
        trace_summary = f"{grade_str} → {verdict}"

    return {
        **state,
        "crag_grades":           grades,
        "crag_verdict":          verdict,
        "crag_requery_strategy": requery_strategy,
        "agent_trace": trace_append(state, stage="7", icon="⚖️", label="CRAG Grader", summary=trace_summary, detail=trace_detail),
    }
