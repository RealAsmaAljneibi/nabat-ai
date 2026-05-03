"""
creative_poet/external_tools.py
=================================
Why this file exists: the creative pipeline must communicate with two external agents —
Al-Nassikh (corpus statistics) and Fatat Al-Arab (retrieval chain). This module is the
formal, named bridge: each function corresponds to a declared tool in tools.py and is
gated by check_tool_permitted() at call sites, so a reviewer can trace which creative
agent calls which external system.

Before this file existed, retrieval was buried in private helpers (_retrieve_poet_exemplars)
with no declared permission boundary. Moving it here makes the inter-agent communication
explicit and auditable.

External contracts:
  nassikh_check_poet()                  → al_nassikh.corpus_stats.count_poems_by_poet()
  fatat_retrieve_exemplars()            → retrieve→rrf_fuse→resolve_heritage chain
  fatat_retrieve_similar_structures()   → same chain, no poet filter (for Al-Musharik)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum corpus hits before falling back to genre-level retrieval.
MIN_EXEMPLARS_REQUIRED = 3

# ── Consultation safeguards ─────────────────────────────────────────────────
# Why: cross-agent calls can recurse indefinitely (Mulhim→Hafiz→Mulhim) and
# cost N× the LLM tokens of a single-agent flow. The bridge enforces a hard
# depth cap and shared wall-clock budget so any consultation chain has a
# bounded cost even when an agent makes a wrong choice to consult.
MAX_CONSULTATION_DEPTH = 2
DEFAULT_CONSULTATION_BUDGET_MS = 8000


class ConsultationBudgetExhausted(RuntimeError):
    """Raised when a cross-agent call exceeds depth or budget."""


def _check_budget(state: dict, from_agent: str, to_agent: str) -> None:
    """
    Inspect the CompositionState's consultation guards and raise if exceeded.
    Called at the top of every cross-agent consult function.
    """
    depth  = int(state.get("consultation_depth") or 0)
    # Why explicit None check: budget=0 is a valid (exhausted) state and must
    # NOT coalesce to the default. `state.get("k") or default` would do that.
    raw_budget = state.get("consultation_budget_ms")
    budget = int(raw_budget if raw_budget is not None else DEFAULT_CONSULTATION_BUDGET_MS)
    if depth >= MAX_CONSULTATION_DEPTH:
        raise ConsultationBudgetExhausted(
            f"depth cap reached ({depth}/{MAX_CONSULTATION_DEPTH}) — "
            f"refusing {from_agent}→{to_agent}"
        )
    if budget <= 0:
        raise ConsultationBudgetExhausted(
            f"budget exhausted — refusing {from_agent}→{to_agent}"
        )


def _record_consult(
    state: dict, from_agent: str, to_agent: str, ms: float, status: str = "ok"
) -> dict:
    """
    Mutate state to record a consultation entry for trace + budget bookkeeping.
    Returns the updated state dict (callers can choose to merge or assign).
    """
    new_state = dict(state)
    new_state["consultation_depth"]    = int(state.get("consultation_depth") or 0) + 1
    new_state["consultation_budget_ms"] = max(
        0, int(state.get("consultation_budget_ms") or DEFAULT_CONSULTATION_BUDGET_MS) - int(ms)
    )
    trace = list(state.get("consultation_trace") or [])
    trace.append({"from": from_agent, "to": to_agent, "ms": round(ms, 1), "status": status})
    new_state["consultation_trace"] = trace
    return new_state


def nassikh_check_poet(poet_name: str) -> dict:
    """
    Query Al-Nassikh for how many poems are indexed for this poet.
    Called by hafiz_nassikh_check_node before any voice-preservation generation.
    Returns zero silently if Al-Nassikh is unavailable (graceful degradation).

    Returns:
        {"poet_name": str, "poem_count": int, "in_corpus": bool}
    """
    try:
        from al_nassikh.corpus_stats import count_poems_by_poet
        count = count_poems_by_poet(poet_name)
    except Exception as exc:
        logger.warning(
            "external_tools: nassikh_check_poet('%s') failed (%s) — treating count as 0",
            poet_name, exc,
        )
        count = 0
    return {
        "poet_name":  poet_name,
        "poem_count": count,
        "in_corpus":  count > 0,
    }


def fatat_retrieve_exemplars(
    target_poet: Optional[str],
    genre: Optional[str],
    n: int = 10,
) -> list[dict]:
    """
    Call Fatat Al-Arab's retrieve→rrf_fuse→resolve_heritage chain for poet exemplars.

    Primary filter: target_poet name (hard filter).
    Fallback: genre-level retrieval when poet-specific results < MIN_EXEMPLARS_REQUIRED.
    This mirrors the original _retrieve_poet_exemplars() helper in mulhim.py but is now
    a named tool callable by any permitted creative agent.

    Used by: mulhim_retrieve_node (permission: QUERY_FATAT_RETRIEVE)
             hafiz_retrieve_node  (permission: QUERY_FATAT_RETRIEVE)
    """
    from fatat_al_arab.state import make_agent_state, make_query_context
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import retrieve_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.resolve_heritage import (
        resolve_heritage_node,
    )

    def _run(q_ar: str, hard: dict, soft: dict) -> list[dict]:
        qc = make_query_context(
            query_lang="ar",
            query_ar=q_ar,
            query_en=q_ar,
            detected_intent="semantic",
            detected_dialect="khaleeji",
            intent_confidence=0.9,
        )
        qc["filters_hard"]       = hard
        qc["filters_soft"]       = soft
        qc["query_variants_ar"]  = [q_ar] + (
            [f"شعر {target_poet}", f"قصائد {target_poet}"] if target_poet else []
        )
        qc["query_variants_en"]  = [f"poetry {target_poet}"] if target_poet else []
        s = make_agent_state(q_ar)
        s["query_context"] = qc
        s = retrieve_node(s)
        s = rrf_fuse_node(s)
        s = resolve_heritage_node(s)
        return s.get("resolved_passages") or []

    base_query = f"أبيات {target_poet}" if target_poet else "أبيات نبطية"
    exemplars = _run(
        q_ar=base_query,
        hard={"poet_name": target_poet} if target_poet else {},
        soft={"genre": genre} if genre else {},
    )

    if len(exemplars) < MIN_EXEMPLARS_REQUIRED and genre:
        logger.info(
            "external_tools: poet='%s' yielded %d exemplars — relaxing to genre='%s'",
            target_poet, len(exemplars), genre,
        )
        exemplars = _run(
            q_ar=f"أبيات في {genre}",
            hard={"genre": genre},
            soft={},
        )

    return exemplars[:n]


# ── NEW: Tier-1 consultation — Al-Hafiz double retrieval (voice + topic) ────
def hafiz_retrieve_voice_blend(
    target_poet: str,
    occasion: Optional[str] = None,
    genre: Optional[str] = None,
    voice_n: int = 15,
    topic_n: int = 5,
) -> list[dict]:
    """
    Two-call retrieval blend for voice preservation: pure voice samples + topical
    relevance. Why: a single retrieval with both poet + topic filters can shrink
    to <5 hits when the poet has lots of indexed verse but few about the chosen
    topic. The LLM then has too few exemplars to capture voice. The blend gives
    the model strong VOICE prior (15 verses, any topic) plus topical grounding
    (5 verses on the requested topic).

    Returns up to (voice_n + topic_n) deduplicated exemplars, voice-first.
    Used by: hafiz_retrieve_node (replaces single fatat_retrieve_exemplars call).
    """
    voice_pool = fatat_retrieve_exemplars(
        target_poet=target_poet, genre=None, n=voice_n,
    )
    if occasion or genre:
        topic_query = occasion or genre or ""
        topic_pool = fatat_retrieve_exemplars(
            target_poet=target_poet, genre=genre, n=topic_n,
        )
    else:
        topic_pool = []

    # Deduplicate by chunk_id while preserving order (voice first)
    seen: set[str] = set()
    blended: list[dict] = []
    for source in (voice_pool, topic_pool):
        for ex in source:
            cid = ex.get("chunk_id") or ex.get("anchor_id") or id(ex)
            if cid not in seen:
                seen.add(cid)
                blended.append(ex)
    logger.info(
        "hafiz_retrieve_voice_blend: poet=%r voice=%d topic=%d → blended=%d",
        target_poet, len(voice_pool), len(topic_pool), len(blended),
    )
    return blended


# ── NEW: Tier-1 consultation — Musharik → Muqayyim (grade ajuz candidates) ──
def muqayyim_grade_ajuz_candidates(
    sadr: str, candidates: list[dict], state: Optional[dict] = None,
) -> list[dict]:
    """
    Batched critique of ajuz candidates. Asks Al-Muqayyim for a single LLM call
    that scores all candidates on meter / rhyme / authenticity, then returns the
    candidates re-ordered by overall score with `muqayyim_score` injected.

    Respects Muqayyim's isolation invariant: it grades SUBMITTED text only —
    the candidates ARE submitted text. No corpus content is fed in.

    Args:
      sadr:       the human's first hemistich, used as context for the critique
      candidates: list of {ajuz, ...} dicts to grade
      state:      CompositionState for budget/depth check (optional in stand-alone use)
    """
    import json as _json
    if state is not None:
        _check_budget(state, "musharik", "muqayyim")
    if not candidates:
        return candidates
    try:
        from fatat_al_arab.llm import chat
        from creative_poet.personas import MUQAYYIM_PERSONA
    except Exception as exc:
        logger.warning("muqayyim grader unavailable (%s) — returning candidates unranked.", exc)
        return candidates
    payload = [{"id": i, "ajuz": (c.get("ajuz") or "")} for i, c in enumerate(candidates)]
    system = (
        MUQAYYIM_PERSONA
        + "\nGrade each ajuz candidate completing the given sadr on three "
        "dimensions (meter, rhyme, authenticity) on a 1–5 scale. Return JSON: "
        '{"grades":[{"id":<int>,"meter":<int>,"rhyme":<int>,"authenticity":<int>,'
        '"overall":<float>,"note":"<one-line critique>"}]}'
    )
    prompt = f"Sadr: {sadr}\n\nCandidates: {_json.dumps(payload, ensure_ascii=False)}"
    t0 = time.perf_counter()
    try:
        result = chat(prompt=prompt, system=system, json_schema={"type": "object"}, max_tokens=400)
        if isinstance(result, str):
            result = _json.loads(result)
        grades = (result or {}).get("grades") or []
    except Exception as exc:
        logger.warning("muqayyim_grade_ajuz_candidates: LLM failed (%s) — unranked.", exc)
        return candidates
    ms = (time.perf_counter() - t0) * 1000
    logger.info("muqayyim_grade_ajuz_candidates: graded %d in %.0fms", len(grades), ms)
    # Merge scores back into candidates
    by_id = {int(g.get("id", -1)): g for g in grades}
    enriched: list[dict] = []
    for i, cand in enumerate(candidates):
        g = by_id.get(i, {})
        c = dict(cand)
        c["muqayyim_score"] = float(g.get("overall") or 0.0)
        c["muqayyim_note"]  = g.get("note") or ""
        c["muqayyim_meter"] = g.get("meter")
        c["muqayyim_rhyme"] = g.get("rhyme")
        c["muqayyim_authenticity"] = g.get("authenticity")
        enriched.append(c)
    enriched.sort(key=lambda c: c.get("muqayyim_score", 0.0), reverse=True)
    return enriched


# ── NEW: Tier-2 consultation — Mulhim → Hafiz (voice fingerprint) ───────────
def hafiz_voice_fingerprint(
    target_poet: str, state: Optional[dict] = None,
) -> dict:
    """
    Quick voice fingerprint for a target poet: opens, closes, signature imagery,
    rhyme letters, dialect register. Used by Mulhim when scaffolding "in poet X's
    voice" so the resulting scaffold reflects the poet's actual style.

    Three steps:
      1. fatat_retrieve_exemplars (poet-only, n=10)
      2. LLM extracts fingerprint from those 10 verses
      3. Returns a small dict the caller can drop into a scaffold prompt
    """
    import json as _json
    if state is not None:
        _check_budget(state, "mulhim", "hafiz")
    exemplars = fatat_retrieve_exemplars(target_poet=target_poet, genre=None, n=10)
    if not exemplars:
        return {"poet": target_poet, "available": False, "reason": "no exemplars"}
    try:
        from fatat_al_arab.llm import chat
        from creative_poet.personas import HAFIZ_PERSONA
    except Exception as exc:
        logger.warning("hafiz_voice_fingerprint: llm unavailable (%s)", exc)
        return {"poet": target_poet, "available": False, "reason": "llm offline"}
    sample = [
        {"text": (e.get("matla_text") or e.get("text") or "")[:150]}
        for e in exemplars[:8]
    ]
    system = (
        HAFIZ_PERSONA
        + "\nExtract a compact voice fingerprint from the given verses by this "
        "poet. Return JSON only: "
        '{"opening_phrases":[str],"signature_imagery":[str],"key_vocabulary":[str],'
        '"rhyme_letters":[str],"dialect_register":"<msa|khaleeji|najdi|mixed>",'
        '"meter_hint":"<short prosodic note>"}'
    )
    prompt = (
        f"Poet: {target_poet}\nSample verses: "
        f"{_json.dumps(sample, ensure_ascii=False)}"
    )
    t0 = time.perf_counter()
    try:
        result = chat(prompt=prompt, system=system, json_schema={"type": "object"}, max_tokens=400)
        if isinstance(result, str):
            result = _json.loads(result)
    except Exception as exc:
        logger.warning("hafiz_voice_fingerprint: llm failed (%s)", exc)
        return {"poet": target_poet, "available": False, "reason": str(exc)}
    ms = (time.perf_counter() - t0) * 1000
    logger.info("hafiz_voice_fingerprint: %s in %.0fms", target_poet, ms)
    fingerprint = result if isinstance(result, dict) else {}
    fingerprint["poet"]      = target_poet
    fingerprint["available"] = True
    fingerprint["sample_count"] = len(exemplars)
    return fingerprint


# ── NEW: Tier-2 consultation — Muqayyim → Nassikh (intertextuality boolean) ─
def nassikh_check_intertextuality(
    verse_line: str, state: Optional[dict] = None,
) -> dict:
    """
    Boolean-only intertextuality check. Asks the index whether a hemistich
    appears verbatim in any indexed corpus text. Returns ONLY {in_corpus,
    attributed_poet, manuscript_short_key} — NEVER the corpus verse content.

    This is critical: Al-Muqayyim's contract is to evaluate SUBMITTED text
    without being biased by corpus content. Returning only metadata (boolean +
    attribution) preserves that invariant while letting the critic flag legitimate
    intertextuality vs. plagiarism.

    Implementation: BM25 lookup with high score threshold (≥ 0.9 normalised
    overlap). Below threshold → in_corpus=False.
    """
    if state is not None:
        _check_budget(state, "muqayyim", "nassikh")
    line = (verse_line or "").strip()
    if len(line) < 8:
        return {"in_corpus": False, "reason": "line too short for verbatim match"}
    try:
        from fatat_al_arab.index import load_index, DEFAULT_QDRANT
        from fatat_al_arab.retrievers.bm25 import get_bm25_retriever
        bundle = load_index(DEFAULT_QDRANT)
        retriever = get_bm25_retriever(bundle.chunks)
        hits = retriever.retrieve(line, n=3)
    except Exception as exc:
        logger.warning("nassikh_check_intertextuality: index unavailable (%s)", exc)
        return {"in_corpus": False, "reason": str(exc)}
    if not hits:
        return {"in_corpus": False}
    # Normalised character-level overlap as the verbatim signal — robust to
    # diacritic / spacing differences. We never return the matched text itself.
    import re
    def _norm(s: str) -> str:
        s = re.sub(r"[ً-ٟـ]", "", s or "")  # strip harakat + tatweel
        s = re.sub(r"\s+", " ", s).strip()
        return s
    norm_line = _norm(line)
    for h in hits:
        text = h.text or h.matla_text or ""
        norm_text = _norm(text)
        # Substring containment (line inside corpus text or vice versa) is the
        # cleanest verbatim signal for short hemistiches.
        if norm_line and norm_text and (norm_line in norm_text or norm_text in norm_line):
            return {
                "in_corpus":            True,
                "attributed_poet":      (h.poet_name or "").strip() or "unknown",
                "manuscript_short_key": (h.manuscript_short_key or "").strip(),
                # NO verse content returned — by design.
            }
    return {"in_corpus": False}


# ── NEW: General-knowledge fallback (when corpus has nothing) ───────────────
def llm_general_knowledge_fallback(
    query: str, hint_poet: Optional[str] = None,
) -> dict:
    """
    Last-resort answer when the corpus + online retrieval both come up empty
    AND the user asked about a known poet. Calls the LLM in "what do you know
    about X" mode. The result is BADGED with a clear "🌐 Outside corpus —
    LLM general knowledge" warning so the user knows it isn't a cited answer.

    Returns: {answer, source: "general_knowledge", badge: "..."}
    """
    try:
        from fatat_al_arab.llm import chat
    except Exception as exc:
        return {
            "answer": "",
            "source": "unavailable",
            "badge":  "❌ LLM unavailable",
            "error":  str(exc),
        }
    # Disambiguation hint goes in BOTH the system prompt and the user prompt so
    # the model can't ignore it. Critical for poet names with similar surface
    # forms (Hazza bin Zayed vs Hamdan bin Mohammed, etc.).
    disambig_clause = (
        f"\n\nIMPORTANT: the query is specifically about **{hint_poet}**. "
        f"Do NOT confuse them with any similarly-named person. If you do not "
        f"know specific details about {hint_poet}, say so honestly rather than "
        f"substituting another poet."
    ) if hint_poet else ""
    system = (
        "You are answering a query that is NOT in the indexed Khaleeji Nabati "
        "manuscript corpus. Answer from your general knowledge briefly (4-8 "
        "sentences) in Arabic first then English. Be honest about uncertainty. "
        "Do NOT fabricate verse text — describe themes/style rather than quoting."
        + disambig_clause
    )
    try:
        prompt_text = f"Query: {query}"
        if hint_poet:
            prompt_text += f"\n\n(Disambiguation: this is about **{hint_poet}** specifically.)"
        answer = chat(
            prompt=prompt_text,
            system=system,
            max_tokens=600,
        )
    except Exception as exc:
        return {"answer": "", "source": "unavailable", "badge": "❌ LLM call failed", "error": str(exc)}
    return {
        "answer": answer if isinstance(answer, str) else str(answer),
        "source": "general_knowledge",
        "badge":  "🌐 Outside corpus — LLM general knowledge (not cited)",
    }


def fatat_retrieve_similar_structures(sadr: str, n: int = 5) -> list[dict]:
    """
    Find corpus verses with structural/thematic patterns similar to the given sadr.
    No poet or genre filter — cast wide to find the best structural matches.

    Used by: musharik_retrieve_node (permission: QUERY_FATAT_RETRIEVE)
    """
    from fatat_al_arab.state import make_agent_state, make_query_context
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import retrieve_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
    from fatat_al_arab.agent2_retrieval_synthesis.nodes.resolve_heritage import (
        resolve_heritage_node,
    )

    qc = make_query_context(
        query_lang="ar",
        query_ar=sadr,
        query_en=sadr,
        detected_intent="semantic",
        detected_dialect="khaleeji",
        intent_confidence=0.85,
    )
    qc["filters_hard"]      = {}
    qc["filters_soft"]      = {}
    qc["query_variants_ar"] = [sadr]

    s = make_agent_state(sadr)
    s["query_context"] = qc
    s = retrieve_node(s)
    s = rrf_fuse_node(s)
    s = resolve_heritage_node(s)
    return (s.get("resolved_passages") or [])[:n]
