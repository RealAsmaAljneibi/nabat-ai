"""
agent2_retrieval_synthesis/nodes/synthesise.py
===============================================
Why this node exists: §2.5 Stage 8 — the LLM composes the final draft answer
from the graded, resolved passages. This is the only place in the pipeline where
the LLM is allowed to produce free-form Arabic prose. Every other stage (grader,
reflector, formatter) works on structured JSON.

Mandatory citation injection (§2.9 Guardrail a):
  Every factual sentence must carry an [anchor_id:…] tag. The synthesis prompt
  instructs the model to tag each claim with the chunk_id it came from. The
  guardrail node in Stage 10 then verifies these tags resolve to real registry
  entries. Without this gate, the model can silently hallucinate a manuscript
  reference that sounds plausible.

Why the response is Arabic-first: the query is in Arabic (or has been translated
to Arabic by Agent 1). The synthesis prompt always uses the Arabic query and
Arabic passage text. The bilingual format_variants node in Stage 10 adds the
English translation.

§5 failure budget: if synthesis fails, the refusal template fires — we never
return a half-composed answer.

Architecture refs: §2.5 Stage 8 (Synthesis), §2.9 (Guardrail a, citation tags).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fatat_al_arab.llm import chat
from fatat_al_arab.state import AgentState, trace_append

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_SYNTHESISER = """\
أنتِ فتاة العرب (الباحثة العربية)، خبيرة الشعر النبطي الخليجي في نظام NABAT-AI.
المهمة — المرحلة 8 (التوليف): صياغة إجابة دقيقة ومستندة إلى المقاطع المسترجعة \
من المخطوطات المرقَّمة.

قواعد صارمة:
1. استند فقط إلى المقاطع المرفقة — لا تضف معلومات من خارجها.
2. لكل ادعاء أو اقتباس، ضع وسم التوثيق مباشرةً بعده بالشكل: [anchor_id:المعرّف]
3. اقتبس أبيات الشعر حرفيًّا كما وردت في المقاطع — لا تُعدِّل نصَّ البيت.
4. إذا لم تكفِ المقاطع للإجابة، أجب بـ: INSUFFICIENT_PASSAGES
5. أجب بالعربية في المقام الأول.
6. إذا وُجدت أدوار سابقة في المحادثة، لا تُعِد ذكر المعلومات التي سبق تقديمها.
   بل أضِف إليها أو وسِّعها أو أجِب من زاوية مختلفة.

You are a specialist in Nabati Khaleeji poetry and manuscript scholarship.
Your task: write an accurate answer grounded exclusively in the retrieved passages.
Strict rules (same as above in English): cite every claim with [anchor_id:ID],
quote verses verbatim, say INSUFFICIENT_PASSAGES if the passages don't suffice.
If prior conversation turns are provided, do NOT repeat information already given —
build on it, extend it, or answer from a different angle.
"""

_SYSTEM_SYNTHESISER_RETRY = """\
أنتِ فتاة العرب (الباحثة العربية)، خبيرة الشعر النبطي الخليجي في نظام NABAT-AI.
المهمة — المرحلة 8 إعادة كتابة (Self-RAG): أعيدي صياغة الإجابة مع مراعاة \
الملاحظات أدناه. نفس القواعد: توثيق كل ادعاء بـ [anchor_id:…]، اقتباس حرفي، \
لا إضافات خارج المقاطع.

You are Fatat Al-Arab (فتاة العرب — The Arabian Scholar), NABAT-AI's bilingual \
Khaleeji Nabati poetry expert.
Task — Stage 8 retry (Self-RAG rewrite): rewrite the answer addressing the \
critique below. Same rules apply: cite every claim, quote verbatim, no content \
outside the passages.
"""


def _format_history_for_synthesis(history: list[dict], max_turns: int = 5) -> str:
    """
    Why a dedicated formatter for synthesis: the synthesiser needs to know what
    was already told to the user so it doesn't repeat it. We strip refusal turns
    (they contain no useful information to avoid repeating) and cap at max_turns
    to keep the prompt within the token budget (≈1200 tokens reserved for history).
    """
    relevant = [t for t in history if not t.get("is_refusal", False)][-max_turns:]
    if not relevant:
        return ""
    lines = ["--- Prior conversation (do NOT repeat this information) ---"]
    for i, turn in enumerate(relevant, 1):
        q = turn.get("query", "").strip()
        r = turn.get("response", "").strip()
        # Cap each response at 400 chars — we need context, not the full prior answer
        if len(r) > 400:
            r = r[:400] + "…"
        lines.append(f"Turn {i} Q: {q}")
        lines.append(f"Turn {i} A: {r}")
    lines.append("--- End of prior conversation ---")
    return "\n".join(lines)


def _build_prompt(
    query_ar: str,
    passages: list[dict],
    critique: str | None = None,
    history: list[dict] | None = None,
) -> str:
    """
    Build the synthesis prompt from query + resolved passages.
    Why history is injected here (M6b): gives the synthesiser the last-5-turns
    context so it extends prior answers rather than repeating them verbatim.
    """
    passage_blocks = []
    for p in passages:
        block = {
            "chunk_id":       p.get("chunk_id", ""),
            "anchor_id":      p.get("anchor_id", ""),
            "text":           p.get("matla_text") or p.get("text") or "",
            "poet_name":      p.get("poet_name", ""),
            "source_volume":  p.get("source_volume", ""),
            "source_page":    p.get("source_page", 0),
        }
        passage_blocks.append(block)

    prompt_parts = []

    # Prepend history block when we are in a multi-turn session
    history_block = _format_history_for_synthesis(history or [])
    if history_block:
        prompt_parts += [history_block, ""]

    prompt_parts += [
        f"السؤال / Query: {query_ar}",
        "",
        f"المقاطع / Passages:\n{json.dumps(passage_blocks, ensure_ascii=False, indent=2)}",
    ]
    if critique:
        prompt_parts += ["", f"ملاحظات على الإجابة السابقة / Critique:\n{critique}"]

    return "\n".join(prompt_parts)


def _extract_citations(text: str) -> list[str]:
    """Extract anchor_id values from [anchor_id:…] tags in the response."""
    import re
    return re.findall(r"\[anchor_id:([^\]]+)\]", text)


def synthesise_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.5 Stage 8.

    Reads:
      state["resolved_passages"]  — enriched chunks from Stage 6
      state["query_context"]      — for query_ar
      state["self_rag_retries"]   — to detect retry path
      state["crag_verdict"]       — if Incorrect, skip synthesis → refusal
    Writes:
      state["draft_response"]     — Arabic-first synthesised answer
      state["citations_used"]     — anchor_id list extracted from draft
      state["passage_ids_used"]   — chunk_ids that contributed
    """
    from fatat_al_arab.guardrails import REFUSAL_TEMPLATE_AR, REFUSAL_TEMPLATE_EN

    qc = state.get("query_context") or {}
    query_ar: str = qc.get("query_ar") or state.get("raw_query", "")
    passages: list[dict] = state.get("resolved_passages") or []
    crag_verdict: str = state.get("crag_verdict") or "Ambiguous"
    retry_count: int = int(state.get("self_rag_retries") or 0)
    # M6b: read conversation history so the synthesiser avoids repeating prior answers
    history: list[dict] = state.get("conversation_history") or []

    # If CRAG verdict is Incorrect and no passages survive → refusal path
    resolvable = [p for p in passages if p.get("citation_resolvable", True)]
    if crag_verdict == "Incorrect" or not resolvable:
        logger.info("synthesise_node: CRAG verdict=%s — firing refusal.", crag_verdict)
        refusal = f"{REFUSAL_TEMPLATE_AR}\n\n{REFUSAL_TEMPLATE_EN}"
        return {
            **state,
            "draft_response":   refusal,
            "citations_used":   [],
            "passage_ids_used": [],
            "is_refusal":       True,
        }

    # Decide prompt: first attempt or Self-RAG retry with critique
    critique: str | None = None
    system = _SYSTEM_SYNTHESISER
    if retry_count > 0:
        scores = state.get("self_rag_scores") or {}
        # Prefer fix_instructions (precise rewrite directive) over generic issues list
        fix_instructions = (scores.get("fix_instructions") or "").strip()
        issues = scores.get("issues") or []
        critique = fix_instructions or ("; ".join(issues) if issues else "Improve faithfulness and completeness.")
        system = _SYSTEM_SYNTHESISER_RETRY

    prompt = _build_prompt(query_ar, resolvable, critique=critique, history=history)

    try:
        draft = chat(
            prompt=prompt,
            system=system,
            max_tokens=800,
        )
        if isinstance(draft, dict):
            draft = json.dumps(draft, ensure_ascii=False)

        # If the model signals insufficient passages, fire refusal
        if "INSUFFICIENT_PASSAGES" in (draft or ""):
            logger.info("synthesise_node: model signalled INSUFFICIENT_PASSAGES.")
            refusal = f"{REFUSAL_TEMPLATE_AR}\n\n{REFUSAL_TEMPLATE_EN}"
            return {
                **state,
                "draft_response":   refusal,
                "citations_used":   [],
                "passage_ids_used": [],
                "is_refusal":       True,
            }

    except Exception as exc:
        logger.error("synthesise_node: LLM call failed (%s) — firing refusal.", exc)
        refusal = f"{REFUSAL_TEMPLATE_AR}\n\n{REFUSAL_TEMPLATE_EN}"
        return {
            **state,
            "draft_response":   refusal,
            "citations_used":   [],
            "passage_ids_used": [],
            "is_refusal":       True,
        }

    citations     = _extract_citations(draft)
    passage_ids   = [p.get("chunk_id", "") for p in resolvable]

    logger.debug(
        "synthesise_node: draft length=%d chars, citations=%d.",
        len(draft), len(citations),
    )
    retry_count = state.get("self_rag_retries") or 0
    trace_summary = (
        f"draft {len(draft)} chars · {len(citations)} citation(s)"
        + (f" · retry #{retry_count}" if retry_count else "")
    )
    return {
        **state,
        "draft_response":   draft,
        "citations_used":   citations,
        "passage_ids_used": passage_ids,
        "is_refusal":       False,
        "agent_trace": trace_append(state, stage="8", icon="✍️", label="Synthesis", summary=trace_summary),
    }
