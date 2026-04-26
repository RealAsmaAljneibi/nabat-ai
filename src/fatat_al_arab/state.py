"""
src/fatat_al_arab/state.py
==========================
Why this file exists: §2.6 of the architecture defines the LangGraph state
contract as the handshake between Agent 1 (Query & Understanding) and
Agent 2 (Retrieval & Synthesis). Both agents read from and write to these
TypedDicts — if the schema drifts, the orchestrator breaks at the seam.
Having one canonical definition here means any schema change is a one-line
diff, visible in review.

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
    track:                NotRequired[str]         # "poetic_rag" | "capabilities" | "registry_lookup" | "instructor_debug" | "image_grounded_provenance"
    intent_genre:         NotRequired[Optional[str]] # canonical Arabic genre label set by intent_router for genre-aware counting
    intent_subintent:     NotRequired[Optional[str]]   # semantic router sub-intent
    intent_confidence_router: NotRequired[float]   # semantic router confidence (0-1)
    intent_alt_family:    NotRequired[Optional[str]]   # second-ranked track family
    intent_router_reasoning: NotRequired[Optional[str]] # router's reasoning text
    router_source:        NotRequired[str]         # "regex" | "llm" | "default"
    router_cues:          NotRequired[list]        # regex cue IDs that fired

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
    self_rag_scores:    NotRequired[dict]         # {faithfulness, relevance, completeness}
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

    # ── Instructor debug snapshot (prior turn) ───────────────────────────
    # Why here: the instructor_debug answer node reads the previous turn's
    # pipeline internals. The UI copies last_result into this field before
    # calling run_agent1 when the query looks like a debug/inspector request.
    debug_snapshot:       NotRequired[dict]  # copied from prior turn AgentState

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
    )


def validate_query_context(qc: dict) -> None:
    """
    Why: a track/answer_source mismatch would silently send a capabilities or
    instructor_debug query into the 10-second RAG pipeline. Fail loudly at the
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
