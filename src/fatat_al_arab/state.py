"""
src/fatat_al_arab/state.py
==========================
Why this file exists: §2.6 of the architecture defines the LangGraph state
contract as the handshake between Agent 1 (Query & Understanding) and
Agent 2 (Retrieval & Synthesis). Both agents read from and write to these
TypedDicts — if the schema drifts, the orchestrator breaks at the seam.
Having one canonical definition here means any schema change is a one-line
diff, visible in review.
Where it's called: agent1/graph.py, agent2/graph.py, creative/graph.py
Purpose: Defines the state contract between Agent 1 and Agent 2

When triggered: At every node entry (read) and exit (write) — touched on every turn
Purpose: Single source of truth for inter-agent contracts (QueryContext, AgentState, CompositionState TypedDicts + factories)
────────────────────────────────────────────────────────────────────────────────
Two state types:
  QueryContext  — Agent 1's output / Agent 2's input (§2.4 Step 6 output)
  AgentState    — full graph state that flows through both agents end-to-end

Architecture refs: §2.4 (Agent 1 stages 1-3), §2.5 (Agent 2 stages 4-10),
§2.6 (state contract), §2.8 (failure budgets encoded as Optional fields).
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict, NotRequired


# ── QueryContext ──────────────────────────────────────────────────────────────
# Why a separate type: Agent 1 serialises this dict and passes it to Agent 2.
# Keeping it separate from AgentState means Agent 2 can be tested with a
# frozen QueryContext fixture without running Agent 1 at all (M5 acceptance gate).

class QueryContext(TypedDict):
    """
    Agent 1 output — everything Agent 2 needs to know about the user's intent.
    Populated field-by-field as Agent 1 nodes execute (§2.4 stages 1-3).

    Fields are Optional where the stage that fills them might be skipped due
    to a §5 failure budget (e.g., HyDE times out → hyde_embedding stays None,
    retrieval proceeds without it).
    """

    # ── Stage 0.5: Routing (set by intent_router / semantic_router) ──────
    # Why separate from Stage 1: routing fields are written before any LLM call
    # so the orchestrator can short-circuit without running bilingual_analyzer.
    answer_source:        NotRequired[str]         # "registry_lookup" | "rag_pipeline"
    deterministic_intent: NotRequired[Optional[str]]   # registry-lookup intent key
    track:                NotRequired[str]         # "poetic_rag" | "capabilities" | "registry_lookup" | "pipeline_debug" | "image_grounded_provenance"
    intent_genre:         NotRequired[Optional[str]] # canonical Arabic genre label set by intent_router for genre-aware counting
    intent_subintent:     NotRequired[Optional[str]]   # semantic router sub-intent
    intent_confidence_router: NotRequired[float]   # semantic router confidence (0-1)
    intent_alt_family:    NotRequired[Optional[str]]   # second-ranked track family
    intent_router_reasoning: NotRequired[Optional[str]] # router's reasoning text
    router_source:        NotRequired[str]         # "regex" | "prototype_arabert" | "prototype_tfidf" | "llm" | "default"
    router_cues:          NotRequired[list]        # regex cue IDs that fired
    # Source bucket the router believes will best answer this query.
    # Read by retrieve.py to filter the Qdrant payload by `source_type`, and by
    # the answer-display layer to pick the right badge.
    # Values: "manuscript_corpus" | "online_corpus" | "any_corpus"
    #         | "poet_bio" | "corpus_stats" | "general_knowledge"
    preferred_source:     NotRequired[Optional[str]]

    # ── Stage 1: Bilingual Analyzer ──────────────────────────────────
    query_lang:        str                  # "ar" | "en"
    query_ar:          str                  # Arabic version of the query
    query_en:          str                  # English version (translated if input was AR)
    detected_intent:   str                  # "factual" | "semantic" | "interpretive"
    detected_dialect:  str                  # "khaleeji" | "najdi" | "msa" | "unknown"
    intent_confidence: float                # 0.0–1.0; < 0.5 → clarification path

    # ── Stage 2: HyDE + Bilingual Expand ─────────────────────────────
    hyde_passage:      NotRequired[Optional[str]]    # hypothetical Nabati verse (AR)
    hyde_embedding:    NotRequired[Optional[list]]   # dense vector of hyde_passage
    query_variants_ar: NotRequired[list[str]]        # 3-5 Arabic paraphrases
    query_variants_en: NotRequired[list[str]]        # 3-5 English paraphrases

    # ── Stage 3: Self-Query ───────────────────────────────────────────
    # Why split into hard/soft: §M4 says confidence ≥ 0.7 → Qdrant pre-filter
    # (hard); < 0.7 → score boost only (soft).
    filters_hard:      NotRequired[dict]    # applied as Qdrant payload filter
    filters_soft:      NotRequired[dict]    # applied as score boost in RRF
    self_query_raw:    NotRequired[dict]    # raw LLM extraction (for debug panel)

    # ── Clarification path ────────────────────────────────────────────
    needs_clarification:     NotRequired[bool]
    clarification_question:  NotRequired[Optional[str]]


# ── AgentState ────────────────────────────────────────────────────────────────
# Why a superset: the LangGraph graph threads a single mutable dict through
# all nodes. Keeping AgentState as one flat TypedDict (rather than nested
# objects) keeps LangGraph's merge semantics simple — every key is a scalar
# or list that can be updated independently.

class AgentState(TypedDict):
    """
    Full graph state — flows through Agent 1 → QueryContext handoff → Agent 2.
    Agent 1 nodes write to the top section; Agent 2 nodes write to the bottom.
    The orchestrator reads both.
    """

    # ── Input ─────────────────────────────────────────────────────────
    raw_query:          str                 # original user input (text or OCR'd from image)
    input_image_path:   NotRequired[Optional[str]]   # if question arrived as image

    # ── Agent 1 output (mirrored from QueryContext) ───────────────────
    query_context:      NotRequired[Optional[QueryContext]]

    # ── Agent 2 — Stage 4: Triple Hybrid Retrieval ────────────────────
    # Why three separate lists: they feed the RRF fuser independently;
    # a slow retriever is dropped without touching the other two (§5 budget).
    bm25_results:       NotRequired[list[dict]]   # [{chunk_id, score, payload}]
    dense_results:      NotRequired[list[dict]]
    colbert_results:    NotRequired[list[dict]]
    retriever_timings:  NotRequired[dict]          # {bm25_ms, dense_ms, colbert_ms}

    # ── Agent 2 — Stage 5: RRF Fusion ────────────────────────────────
    rrf_top5:           NotRequired[list[dict]]   # top-5 fused results

    # ── Agent 2 — Stage 6: Heritage Resolution ───────────────────────
    resolved_passages:  NotRequired[list[dict]]   # Khaleeji text swapped in

    # ── Agent 2 — Stage 7: CRAG Grading ──────────────────────────────
    crag_grades:            NotRequired[list[dict]]   # [{chunk_id, label, confidence, rationale}]
    crag_verdict:           NotRequired[str]          # "Correct" | "Ambiguous" | "Incorrect"
    crag_requery_count:     NotRequired[int]          # 0 or 1 (max 1 re-query per §5)
    crag_requery_strategy:  NotRequired[str]          # LLM-generated hint: what to search for on re-query

    # ── Agent 2 — Stage 8: Synthesis ─────────────────────────────────
    draft_response:     NotRequired[str]
    citations_used:     NotRequired[list[str]]    # anchor_ids referenced in draft
    passage_ids_used:   NotRequired[list[str]]    # chunk_ids that were quoted

    # ── Agent 2 — Stage 9: Self-RAG Reflection ───────────────────────
    self_rag_scores:    NotRequired[dict]         # {faithfulness, relevance, completeness, pass, issues, fix_instructions, failed_claims}
    self_rag_verdict:   NotRequired[str]          # "pass" | "retry" | "flag"
    self_rag_retries:   NotRequired[int]          # 0–2 (max 2 per §5)

    # ── Agent 2 — Stage 10: Multi-Variant Formatter ──────────────────
    formatted_response: NotRequired[dict]         # {al_maktub, orthographic, al_mantuq, citations}

    # ── Guardrails (post-Stage 10) ────────────────────────────────────
    guardrail_passed:   NotRequired[bool]
    guardrail_flags:    NotRequired[list[str]]    # which guardrail fired and why

    # ── Final output ─────────────────────────────────────────────────
    final_response:     NotRequired[str]          # emitted to the UI
    is_refusal:         NotRequired[bool]         # True → "not in corpus" template fired
    # M9: synthesise set this when it routed to the LLM general-knowledge fallback
    # because the corpus didn't have the requested entity. format_variants reads
    # it to skip guardrails (the 🌐 badge IS the safety mechanism on this path).
    general_knowledge_fallback: NotRequired[bool]

    # ── Failure tracking (§5 failure-handling budget) ─────────────────
    # Why in state: the orchestrator reads these to decide whether to fire
    # the fallback-LLM edge or the clarification path.
    llm_fallback_active:        NotRequired[bool]
    agent1_error:               NotRequired[Optional[str]]
    agent2_error:               NotRequired[Optional[str]]
    retrieval_dropped_retrievers: NotRequired[list[str]]  # names of timed-out retrievers

    # ── Multi-turn conversation history (M4b / M6b) ───────────────────
    # Why here and not in QueryContext: history is consumed by both Agent 1
    # (bilingual_analyzer uses it for intent continuity) and Agent 2
    # (synthesise uses it to avoid repeating prior answers). Keeping it in
    # AgentState — which spans both agents — avoids duplicating the field.
    # Each entry: {"query": str, "response": str, "is_refusal": bool}
    conversation_history: NotRequired[list[dict]]   # last ≤5 prior turns (oldest first)

    # ── Multimodal input telemetry ───────────────────────────────────────
    input_modality:       NotRequired[str]   # "text" | "voice" | "image"

    # ── Pipeline debug snapshot (prior turn) ────────────────────────────
    # Why here: the pipeline_debug answer node reads the previous turn's
    # pipeline internals. The UI copies last_result into this field before
    # calling run_agent1 when the query looks like a debug/inspector request.
    debug_snapshot:       NotRequired[dict]  # copied from prior turn AgentState

    # ── Agent Reasoning Trace (EXT-4) ────────────────────────────────
    # Chronological log built up as nodes execute. Each entry is a dict:
    #   {"stage": str, "icon": str, "label": str, "summary": str, "detail": str}
    # The Streamlit UI renders this as a single collapsible timeline panel so
    # observers can watch the system reason without opening 5 separate expanders.
    agent_trace:        NotRequired[list[dict]]

    # ── Debug / evaluation metadata ──────────────────────────────────
    stage_timings:      NotRequired[dict]   # {stage_name: ms} for M10 efficiency axis
    conversation_id:    NotRequired[str]    # multi-turn session ID
    turn_index:         NotRequired[int]


# ── Factory helpers ───────────────────────────────────────────────────────────
# Why factories: constructing TypedDicts with 20+ Optional fields inline is
# error-prone. These give every caller a known-good starting point.

def make_query_context(
    query_lang: str,
    query_ar: str,
    query_en: str,
    detected_intent: str = "semantic",
    detected_dialect: str = "unknown",
    intent_confidence: float = 0.5,
) -> QueryContext:
    """Minimal valid QueryContext — Optional fields absent until stages fill them."""
    return QueryContext(
        query_lang=query_lang,
        query_ar=query_ar,
        query_en=query_en,
        detected_intent=detected_intent,
        detected_dialect=detected_dialect,
        intent_confidence=intent_confidence,
    )


def make_agent_state(raw_query: str) -> AgentState:
    """Minimal valid AgentState at the start of a turn."""
    return AgentState(
        raw_query=raw_query,
        crag_requery_count=0,
        self_rag_retries=0,
        llm_fallback_active=False,
        is_refusal=False,
        guardrail_passed=False,
        guardrail_flags=[],
        retrieval_dropped_retrievers=[],
        stage_timings={},
        agent_trace=[],
    )


def trace_append(state: dict, stage: str, icon: str, label: str, summary: str, detail: str = "") -> list[dict]:
    """
    Read the current agent_trace from state, append a new entry, return the updated list.
    Each node calls this and writes the result back as {"agent_trace": trace_append(...)}.

    Why here: centralising the schema means the UI renderer has one contract to depend on.
    """
    current = list(state.get("agent_trace") or [])
    current.append({"stage": stage, "icon": icon, "label": label, "summary": summary, "detail": detail})
    return current


# ── CompositionContext ────────────────────────────────────────────────────────
# Why a separate type from QueryContext: the creative pipeline has a fundamentally
# different input contract — it takes composition parameters (mode, poet, occasion)
# not a natural-language user query. Keeping it separate means the creative graph
# can be tested in isolation from the RAG pipeline, exactly like QueryContext
# lets Agent 2 be tested without running Agent 1.

class CompositionContext(TypedDict):
    """
    Input contract for the creative composition pipeline (Agent 3 / Worker 4).
    The orchestrator's run_creative() populates this and passes it to the graph.

    mode selects which creative agent fires:
      "scaffold"  → Al-Mulhim  — compositional scaffold for a living poet
      "coauthor"  → Al-Musharik — ajuz candidates for an in-progress verse
      "preserve"  → Al-Hafiz   — voice preservation for a deceased poet
      "critique"  → Al-Muqayyim — quality critique of a submitted poem
    """
    mode:            str                        # required — one of the four values above
    target_poet:     NotRequired[Optional[str]] # poet name used by Mulhim, Hafiz
    genre:           NotRequired[Optional[str]] # Nabati genre from nabati_taxonomy.py
    theme:           NotRequired[Optional[str]] # free-text topic / theme
    occasion:        NotRequired[Optional[str]] # رثاء | مديح | غزل | وصف | فخر
    input_sadr:      NotRequired[Optional[str]] # mode=coauthor: human's first hemistich
    input_poem:      NotRequired[Optional[str]] # mode=critique: full poem text to evaluate
    session_verses:  NotRequired[list[str]]     # co-author session: accumulated verse pairs


class CompositionState(TypedDict):
    """
    Full graph state for the creative composition pipeline.
    Flows through Agent 3 nodes in the same way AgentState flows through Agents 1+2.
    Kept flat (no nested objects) so LangGraph merge semantics stay simple.
    """
    composition_context:   CompositionContext

    # ── Al-Mulhim (mode=scaffold) outputs ────────────────────────────────
    style_exemplars:    NotRequired[list[dict]]  # retrieved verses from target poet
    style_fingerprint:  NotRequired[dict]        # {vocabulary, imagery, rhyme_sounds, meter, dialect}
    scaffold:           NotRequired[str]         # markdown compositional guide for the poet

    # ── Al-Musharik (mode=coauthor) outputs ──────────────────────────────
    ajuz_candidates:    NotRequired[list[dict]]  # [{ajuz, meter_ok, rhyme_ok, thematic_score, annotation}]

    # ── Al-Hafiz (mode=preserve) outputs ─────────────────────────────────
    preservation_verse: NotRequired[str]         # the composed verse in poet's manner
    attribution_badge:  NotRequired[str]         # mandatory synthetic attribution badge text

    # ── Al-Muqayyim (mode=critique) outputs ──────────────────────────────
    critique_scores:       NotRequired[dict]      # {meter, rhyme, authenticity, occasion, overall}
    critique_annotations:  NotRequired[list[dict]] # [{line, dimension, verdict, annotation, suggestion}]

    # ── External agent handshake (inter-agent communication) ─────────────
    # Written by external_tools.py bridge calls so the graph can make routing
    # decisions based on what Al-Nassikh and Fatat Al-Arab actually returned.
    nassikh_poet_count:    NotRequired[int]  # poems Al-Nassikh found for target_poet
    fatat_retrieved_count: NotRequired[int]  # exemplars Fatat Al-Arab returned

    # ── Inter-agent consultation safeguards ──────────────────────────────
    # Why these exist: cross-agent calls (Mulhim→Hafiz, Musharik→Muqayyim,
    # Muqayyim→Nassikh) can recurse and explode token cost. consultation_depth
    # is incremented on every cross-call and the bridge rejects calls when it
    # reaches MAX_CONSULTATION_DEPTH. consultation_budget_ms is the wall-clock
    # cap shared across nested calls — once exhausted, calls return empty.
    consultation_depth:       NotRequired[int]   # default 0; cap at 2
    consultation_budget_ms:   NotRequired[int]   # default 8000ms; decremented per call
    consultation_trace:       NotRequired[list]  # [{from, to, ms, status}]
    # Cross-turn memory threaded through from the search session (M9 inter-agent)
    conversation_history:     NotRequired[list]  # last-5 turn dicts from search mode
    # Voice fingerprint cached when one agent consults another for it
    cached_voice_fingerprint: NotRequired[dict]  # {poet, vocab, openings, imagery, dialect}

    # ── Agentic loop state (observe → reflect → retry) ───────────────────
    # Mirrors CRAG's requery_count and Self-RAG's reflect_count in AgentState.
    # Max retries enforced as named constants in mulhim.py / musharik.py.
    scaffold_quality_ok:  NotRequired[bool]  # mulhim_validate: scaffold has required fields
    scaffold_retry_count: NotRequired[int]   # mulhim_generate retries so far (max 1)
    ajuz_quality_ok:      NotRequired[bool]  # musharik_quality: ≥ AJUZ_CANDIDATE_COUNT valid
    ajuz_retry_count:     NotRequired[int]   # musharik_generate retries so far (max 1)

    # ── Common (mirrors AgentState pattern) ──────────────────────────────
    final_output:    NotRequired[str]        # UI-ready response
    guardrail_passed: NotRequired[bool]
    guardrail_flags:  NotRequired[list[str]]
    agent_trace:     NotRequired[list[dict]]
    stage_timings:   NotRequired[dict]


def make_composition_state(ctx: "CompositionContext") -> "CompositionState":
    """Minimal valid CompositionState — analogous to make_agent_state()."""
    return CompositionState(
        composition_context=ctx,
        guardrail_passed=False,
        guardrail_flags=[],
        agent_trace=[],
        stage_timings={},
        consultation_depth=0,
        consultation_budget_ms=8000,
        consultation_trace=[],
    )


def validate_query_context(qc: dict) -> None:
    """
    Why: a track/answer_source mismatch would silently send a capabilities or
    pipeline_debug query into the 10-second RAG pipeline. Fail loudly at the
    seam so routing bugs surface immediately rather than producing a misleading
    'no results' answer.

    Invariant: any non-poetic_rag track must pair with answer_source='registry_lookup'
    so the orchestrator routes to deterministic_answer_node instead of retrieval.
    """
    track  = qc.get("track")
    source = qc.get("answer_source")
    if track and track != "poetic_rag" and source != "registry_lookup":
        raise ValueError(
            f"routing invariant violated: track={track!r} requires "
            f"answer_source='registry_lookup', got {source!r}"
        )
