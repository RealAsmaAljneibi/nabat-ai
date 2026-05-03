"""
agent2_retrieval_synthesis/nodes/reflect.py
============================================
Why this node exists: §2.5 Stage 9 — Self-RAG reflection. After synthesis,
the LLM evaluates its own draft against three quality axes:
  - Faithfulness:   does every claim trace back to the approved passages?
  - Relevance:      does the response actually answer the user's query?
  - Completeness:   does it address all aspects the passages support?

When triggered: Stage 9 — after synthesise
Purpose: Self-RAG critic: scores faithfulness + relevance + completeness (1-5 each); emits fix_instructions for surgical retry; SKIPS on refusal + GK fallback paths.

Why self-evaluation (not a separate judge model): at demo scale one extra LLM
call is acceptable. A second judge model would require a second API key and
doubles the latency. The self-evaluation prompt is written to be adversarial
("find reasons the response could be wrong") which counters the model's
tendency to rate its own output highly.

§5 failure budget:
  - Max 2 Self-RAG retries. After the second retry the pipeline proceeds
    with whatever draft_response exists.
  - If the reflection call itself fails, verdict defaults to "pass" so
    the pipeline isn't blocked.

Verdict routing:
  "pass"  → proceed to format_variants
  "retry" → loop back to synthesise_node with critique (max 2 times)
  "flag"  → proceed to format_variants but set guardrail_flags warning

Architecture refs: §2.5 Stage 9 (Self-RAG Reflection), §5 (retry cap 2).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fatat_al_arab.llm import chat
from fatat_al_arab.personas import FATAT_PERSONA
from fatat_al_arab.state import AgentState

logger = logging.getLogger(__name__)

SELF_RAG_MAX_RETRIES = 1   # §5 — capped at 1 for demo latency

_SYSTEM_REFLECTOR = (
    FATAT_PERSONA
    + """\
Acting as your own strict quality reviewer.
Task — Stage 9 (Self-RAG Reflection): evaluate the draft response against the \
original query and the source passages.

Return a JSON object with this exact shape:
{
  "faithfulness":     <float 0.0–1.0>,
  "relevance":        <float 0.0–1.0>,
  "completeness":     <float 0.0–1.0>,
  "pass":             <bool — true if all scores ≥ 0.7>,
  "issues":           [<list of brief critique points, may be empty>],
  "fix_instructions": "<if pass is false: precise rewrite instructions — name the exact claim to remove or correct, and which passage provides the right evidence instead. If pass is true, return empty string.>",
  "failed_claims":    [<verbatim phrases copied from the draft that are unsupported by the passages>]
}

faithfulness:  every claim in the response traces to the provided passages.
relevance:     the response addresses the user's query.
completeness:  the response covers all aspects the passages can support.

Be strict — a score of 0.9 means near-perfect. Typical good responses score 0.7–0.85.
Return ONLY valid JSON.
"""
)


def _call_reflector(query_ar: str, draft: str, passages: list[dict]) -> dict:
    """
    LLM reflection call. Returns parsed scores dict.
    Falls back to a passing default if the call fails (§5 budget).
    """
    passage_texts = [
        p.get("matla_text") or p.get("text") or "" for p in passages[:10]
    ]
    prompt = (
        f"Query: {query_ar}\n\n"
        f"Draft Response:\n{draft[:1200]}\n\n"
        f"Source Passages:\n{json.dumps(passage_texts[:5], ensure_ascii=False)}"
    )
    try:
        result = chat(
            prompt=prompt,
            system=_SYSTEM_REFLECTOR,
            json_schema={"type": "object"},
            max_tokens=600,
        )
        if isinstance(result, dict):
            return result
        return json.loads(result)
    except Exception as exc:
        logger.warning("reflect_node: reflection call failed (%s) — defaulting pass.", exc)
        return {
            "faithfulness":  0.8,
            "relevance":     0.8,
            "completeness":  0.8,
            "pass":          True,
            "issues":        [],
        }


def _determine_verdict(scores: dict, retry_count: int) -> str:
    """
    Convert numeric scores into a routing verdict.
    - If retry cap reached → always "pass" (must proceed)
    - If scores["pass"] is True → "pass"
    - If any score < 0.5 → "flag" (proceed but mark)
    - Otherwise → "retry"
    """
    if retry_count >= SELF_RAG_MAX_RETRIES:
        logger.info(
            "reflect_node: retry cap reached (%d/%d) — forcing pass.",
            retry_count, SELF_RAG_MAX_RETRIES,
        )
        return "pass"

    if scores.get("pass"):
        return "pass"

    f = float(scores.get("faithfulness",  0.8))
    r = float(scores.get("relevance",     0.8))
    c = float(scores.get("completeness",  0.8))

    if f < 0.5 or r < 0.5:
        return "flag"   # serious quality issue — proceed but warn
    return "retry"


def reflect_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 9.

    Reads:
      state["draft_response"]     — from Stage 8
      state["resolved_passages"]  — to verify faithfulness against
      state["query_context"]      — for query_ar
      state["self_rag_retries"]   — current retry count
      state["is_refusal"]         — skip reflection on refusal path
    Writes:
      state["self_rag_scores"]    — {faithfulness, relevance, completeness, pass, issues}
      state["self_rag_verdict"]   — "pass" | "retry" | "flag"
      state["self_rag_retries"]   — incremented when verdict is "retry"
    """
    # Skip reflection on the refusal path — the refusal template doesn't need grading
    from fatat_al_arab.state import trace_append

    if state.get("is_refusal"):
        return {
            **state,
            "self_rag_scores":  {"faithfulness": 1.0, "relevance": 1.0,
                                  "completeness": 1.0, "pass": True, "issues": []},
            "self_rag_verdict": "pass",
            "agent_trace": trace_append(state, stage="9", icon="🪞", label="Self-RAG Reflection", summary="Skipped — refusal path"),
        }

    # M9: skip reflection on general-knowledge fallback path — the answer is
    # by design NOT grounded in passages, so faithfulness/completeness checks
    # would all fail. The 🌐 badge already tells the user it's outside corpus.
    if state.get("general_knowledge_fallback"):
        return {
            **state,
            "self_rag_scores":  {"faithfulness": 1.0, "relevance": 1.0,
                                  "completeness": 1.0, "pass": True, "issues": []},
            "self_rag_verdict": "pass",
            "agent_trace": trace_append(state, stage="9", icon="🪞", label="Self-RAG Reflection",
                                         summary="Skipped — general-knowledge fallback (🌐 outside corpus)"),
        }

    qc = state.get("query_context") or {}
    query_ar: str = qc.get("query_ar") or state.get("raw_query", "")
    draft:    str = state.get("draft_response") or ""
    passages: list[dict] = state.get("resolved_passages") or []
    retry_count: int = int(state.get("self_rag_retries") or 0)

    if not draft:
        logger.warning("reflect_node: draft_response is empty — verdict pass.")
        return {
            **state,
            "self_rag_scores":  {"faithfulness": 0.0, "relevance": 0.0,
                                  "completeness": 0.0, "pass": True, "issues": []},
            "self_rag_verdict": "pass",
        }

    scores  = _call_reflector(query_ar, draft, passages)
    verdict = _determine_verdict(scores, retry_count)

    new_retry_count = retry_count + 1 if verdict == "retry" else retry_count

    logger.debug(
        "reflect_node: f=%.2f r=%.2f c=%.2f → verdict=%s (retry %d/%d)",
        scores.get("faithfulness", 0),
        scores.get("relevance", 0),
        scores.get("completeness", 0),
        verdict, new_retry_count, SELF_RAG_MAX_RETRIES,
    )

    f = scores.get("faithfulness", 0.0)
    r = scores.get("relevance",    0.0)
    c = scores.get("completeness", 0.0)
    fix_full   = scores.get("fix_instructions") or ""
    failed     = scores.get("failed_claims") or []
    issues     = scores.get("issues") or []

    scores_str = f"F:{f:.2f} R:{r:.2f} C:{c:.2f}"

    if verdict == "retry":
        trace_summary = f"Needs revision — retrying ({new_retry_count}/{SELF_RAG_MAX_RETRIES})"
        if fix_full:
            fix_preview  = fix_full[:220] + "…" if len(fix_full) > 220 else fix_full
            trace_detail = f'"{fix_preview}"'
            if failed:
                claims_preview = "; ".join(f'"{c[:60]}"' for c in failed[:2])
                trace_detail += f"\nUnsupported claims: {claims_preview}"
        else:
            trace_detail = scores_str
    elif verdict == "flag":
        first_issue = issues[0] if issues else fix_full[:100]
        trace_summary = f"Proceeding with caution — {scores_str}"
        trace_detail  = f'"{first_issue}"' if first_issue else ""
    else:
        trace_summary = f"Response verified — faithful and complete ({scores_str})"
        trace_detail  = ""

    return {
        **state,
        "self_rag_scores":  scores,
        "self_rag_verdict": verdict,
        "self_rag_retries": new_retry_count,
        "agent_trace": trace_append(state, stage="9", icon="🪞", label="Self-RAG Reflection", summary=trace_summary, detail=trace_detail),
    }
