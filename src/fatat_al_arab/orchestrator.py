"""
src/fatat_al_arab/orchestrator.py
==================================
Why this file exists: M7 — the single public entry point that stitches Agent 1
(Query Understanding) and Agent 2 (Retrieval & Synthesis) into one callable
function. Callers — the Streamlit UI, the CLI, and the test suite — all go
through `run()`. Nothing else imports or wires the two agents together.

Design principle: LangGraph-first, direct-node fallback.
  - If langgraph is installed, we use the compiled graphs from agent1/graph.py
    and agent2/graph.py for the full conditional-edge behaviour.
  - If langgraph is NOT installed (CI environment), we call nodes directly in
    the §2.4 / §2.5 stage order. This keeps every test green without a GPU or
    a full LangGraph install.

Failure handling (§5):
  - Any exception inside Agent 1 or Agent 2 is caught here.
  - On catastrophic failure we return a state whose `final_response` is the
    bilingual refusal template and whose `agent2_error` records the exception.
  - We never raise from `run()` — the UI must always receive a displayable dict.

Architecture refs: §2.4 (Agent 1), §2.5 (Agent 2), §2.6 (state contract),
§2.9 (guardrail refusal template), §5 (failure budgets).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .state import AgentState, make_agent_state, validate_query_context
from .guardrails import REFUSAL_TEMPLATE_AR, REFUSAL_TEMPLATE_EN

logger = logging.getLogger(__name__)

# ── LangGraph availability probe ──────────────────────────────────────────────
# Why probe at import time: the CI environment does not have langgraph installed.
# Probing here avoids repeated try/except inside `run()` and makes the decision
# visible at module load — easier to debug than a runtime ImportError deep in
# the call stack.

try:
    import langgraph  # noqa: F401
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.info(
        "orchestrator: langgraph not installed — will call nodes directly."
    )


# ── Agent 1 node imports (always safe — no langgraph dependency in nodes) ─────

def _import_agent1_nodes() -> dict:
    """
    Return a dict of available Agent 1 node callables.
    All nodes are imported unconditionally because the node .py files
    themselves do not import langgraph.
    """
    from .agent1_query_understanding.nodes.bilingual_analyzer import bilingual_analyzer_node

    nodes: dict = {"bilingual_analyzer": bilingual_analyzer_node}

    # Stage 0.5 — intent router (deterministic fast path). Imported first so
    # the direct-node runner below can call it before the LLM-based analyzer.
    try:
        from .agent1_query_understanding.nodes.intent_router import intent_router_node
        nodes["intent_router"] = intent_router_node
    except ImportError:
        pass

    # Optional nodes — import if present (they always are in the current codebase,
    # but the docstring says to be defensive so tests can monkey-patch selectively).
    try:
        from .agent1_query_understanding.nodes.semantic_router import semantic_router_node
        nodes["semantic_router"] = semantic_router_node
    except ImportError:
        pass

    try:
        from .agent1_query_understanding.nodes.bilingual_expand import bilingual_expand_node
        nodes["bilingual_expand"] = bilingual_expand_node
    except ImportError:
        pass

    try:
        from .agent1_query_understanding.nodes.hyde import hyde_node
        nodes["hyde"] = hyde_node
    except ImportError:
        pass

    try:
        from .agent1_query_understanding.nodes.self_query import self_query_node
        nodes["self_query"] = self_query_node
    except ImportError:
        pass

    return nodes


def _import_agent2_nodes() -> dict:
    """Return Agent 2 node callables in stage order (stages 4-10)."""
    from .agent2_retrieval_synthesis.nodes.retrieve       import retrieve_node
    from .agent2_retrieval_synthesis.nodes.rrf_fuse       import rrf_fuse_node
    from .agent2_retrieval_synthesis.nodes.resolve_heritage import resolve_heritage_node
    from .agent2_retrieval_synthesis.nodes.crag_grader    import crag_grader_node
    from .agent2_retrieval_synthesis.nodes.synthesise     import synthesise_node
    from .agent2_retrieval_synthesis.nodes.reflect        import reflect_node
    from .agent2_retrieval_synthesis.nodes.format_variants import format_variants_node
    from .agent2_retrieval_synthesis.nodes.deterministic_answer import deterministic_answer_node

    return {
        "deterministic_answer": deterministic_answer_node,   # registry-lookup fast path
        "retrieve":          retrieve_node,
        "rrf_fuse":          rrf_fuse_node,
        "resolve_heritage":  resolve_heritage_node,
        "crag_grader":       crag_grader_node,
        "synthesise":        synthesise_node,
        "reflect":           reflect_node,
        "format_variants":   format_variants_node,
    }


# ── Agent 1 runner ────────────────────────────────────────────────────────────

def _run_agent1(state: AgentState) -> AgentState:
    """
    Execute Agent 1 stages 1-3 and return the updated state.

    Why the clarification short-circuit: the same logic as the LangGraph
    conditional edge in agent1/graph.py. If bilingual_analyzer sets
    needs_clarification=True, we skip stages 2-3 — running HyDE and
    self-query on an unclear query wastes up to 3 LLM calls and pollutes
    the QueryContext with noisy filters.
    """
    if _LANGGRAPH_AVAILABLE:
        try:
            from .agent1_query_understanding.graph import get_agent1_graph  # type: ignore
        except (ImportError, AttributeError):
            # graph.py exports agent1_graph directly; no get_agent1_graph wrapper
            from .agent1_query_understanding.graph import agent1_graph  # type: ignore
            return agent1_graph.invoke(state)
        else:
            return get_agent1_graph().invoke(state)

    # ── Direct-node fallback (no LangGraph) ──────────────────────────────────
    nodes = _import_agent1_nodes()

    # Stage 0.5a — intent_router (deterministic fast path). Runs first so that
    # counting / metadata / unsupported-dim questions never hit the LLM.
    # If it sets answer_source="registry_lookup" we skip Stages 0.5b + 1-3.
    if "intent_router" in nodes:
        state = nodes["intent_router"](state)
        qc_after_router = state.get("query_context") or {}
        if qc_after_router.get("answer_source") == "registry_lookup":
            logger.info("orchestrator: intent_router (Stage 0.5a) short-circuited Agent 1.")
            return state

    # Stage 0.5b — semantic router (LLM-backed 4-track classifier).
    # Only runs when Stage 0.5a did not short-circuit. Catches capabilities,
    # instructor_debug, and other non-poetic queries the regex can't detect.
    if "semantic_router" in nodes:
        state = nodes["semantic_router"](state)
        qc_after_semantic = state.get("query_context") or {}
        # Fail-fast invariant: track/answer_source mismatch is a routing bug.
        try:
            validate_query_context(qc_after_semantic)
        except ValueError as exc:
            logger.error("orchestrator: routing invariant violated — %s", exc)
            state["agent1_error"] = str(exc)
            raise
        if qc_after_semantic.get("answer_source") == "registry_lookup":
            logger.info(
                "orchestrator: semantic_router (Stage 0.5b) short-circuited Agent 1 — track=%s.",
                qc_after_semantic.get("track", "?"),
            )
            return state

    # Stage 1 — always run for poetic_rag
    state = nodes["bilingual_analyzer"](state)

    # Short-circuit if clarification needed (mirrors LangGraph conditional edge)
    qc = state.get("query_context") or {}
    if qc.get("needs_clarification"):
        logger.debug("orchestrator: Agent 1 clarification path — skipping stages 2-3.")
        return state

    # Stage 2a — bilingual expand (optional)
    if "bilingual_expand" in nodes:
        state = nodes["bilingual_expand"](state)

    # Stage 2b — HyDE (optional)
    if "hyde" in nodes:
        state = nodes["hyde"](state)

    # Stage 3 — self-query (optional)
    if "self_query" in nodes:
        state = nodes["self_query"](state)

    return state


# ── Agent 2 runner ────────────────────────────────────────────────────────────

def _run_agent2(state: AgentState) -> AgentState:
    """
    Execute Agent 2 stages 4-10 and return the final state.

    Why no conditional loops in direct-node mode: the LangGraph graph handles
    the three retry loops (CRAG re-query A/B, Self-RAG loop C). When calling
    nodes directly we run each stage once — the nodes' own §5 failure budgets
    still apply (they return graceful degradation, not exceptions).

    Fast-path bypass: when the intent_router (Agent 1 Stage 0.5) flagged the
    query as a deterministic lookup, we skip retrieval entirely and call the
    deterministic_answer_node, which reads corpus_stats and formats a
    bilingual answer with a registry citation. This is the fix for the
    "how many poems do I have?" refusal bug observed on 2026-04-23.
    """
    nodes_all = _import_agent2_nodes()

    qc = state.get("query_context") or {}
    if qc.get("answer_source") == "registry_lookup":
        logger.info("orchestrator: deterministic answer path — bypassing retrieval.")
        return nodes_all["deterministic_answer"](state)

    if _LANGGRAPH_AVAILABLE:
        try:
            from .agent2_retrieval_synthesis.graph import get_agent2_graph
            return get_agent2_graph().invoke(state)
        except ImportError:
            pass  # fall through to direct-node path

    # ── Direct-node fallback (no LangGraph) ──────────────────────────────────
    for stage in (
        "retrieve",
        "rrf_fuse",
        "resolve_heritage",
        "crag_grader",
        "synthesise",
        "reflect",
        "format_variants",
    ):
        state = nodes_all[stage](state)

    return state


# ── Public split entry points ────────────────────────────────────────────────
# Why split: the Streamlit UI wraps Agent 1 and Agent 2 in separate spinners
# with track-aware copy. run() is kept for backward compatibility (CLI, tests,
# eval harness) and simply calls both in sequence.

def run_agent1(
    raw_query: str,
    input_image_path: Optional[str] = None,
    conversation_id: Optional[str] = None,
    turn_index: int = 0,
    conversation_history: Optional[list] = None,
    input_modality: str = "text",
    debug_snapshot: Optional[dict] = None,
) -> dict:
    """
    Run Agent 1 (query understanding + routing) and return the intermediate
    AgentState. The UI uses this to read the resolved track before choosing
    Phase-2 spinner copy, then calls run_agent2(state) to complete the turn.

    Returns an AgentState dict that is always safe to pass to run_agent2().
    On Agent-1 failure, builds a minimal fallback QueryContext so Agent 2 can
    still attempt retrieval.
    """
    try:
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError(f"raw_query must be a non-empty string, got {raw_query!r}")
    except Exception as exc:
        return _catastrophic_failure(exc, raw_query="", conversation_id=conversation_id,
                                     turn_index=turn_index)

    state: AgentState = make_agent_state(raw_query)
    state["stage_timings"] = {}
    state["input_modality"] = input_modality

    if input_image_path is not None:
        state["input_image_path"] = input_image_path
    if conversation_id is not None:
        state["conversation_id"] = conversation_id
    state["turn_index"] = turn_index
    if conversation_history:
        state["conversation_history"] = list(conversation_history)[-5:]
    if debug_snapshot is not None:
        state["debug_snapshot"] = debug_snapshot

    t0 = time.perf_counter()
    agent1_ms = 0.0
    try:
        state = _run_agent1(state)
    except Exception as exc:
        logger.exception("orchestrator: Agent 1 raised — recording error and continuing.")
        state["agent1_error"] = str(exc)
        if not state.get("query_context"):
            from .state import make_query_context
            state["query_context"] = make_query_context(
                query_lang="ar" if any("\u0600" <= c <= "\u06ff" for c in raw_query) else "en",
                query_ar=raw_query,
                query_en=raw_query,
            )
    finally:
        agent1_ms = round((time.perf_counter() - t0) * 1000, 1)
        timings: dict = state.get("stage_timings") or {}
        timings["agent1_ms"] = agent1_ms
        state["stage_timings"] = timings

    return dict(state)


def run_agent2(state: dict) -> dict:
    """
    Run Agent 2 (retrieval + synthesis) on an AgentState produced by run_agent1().
    Returns the final AgentState. On catastrophic failure returns a refusal dict.
    """
    agent1_ms: float = (state.get("stage_timings") or {}).get("agent1_ms", 0.0)
    raw_query: str = state.get("raw_query", "")
    conversation_id = state.get("conversation_id")
    turn_index: int = state.get("turn_index", 0)

    t1 = time.perf_counter()
    try:
        result = _run_agent2(state)  # type: ignore[arg-type]
    except Exception as exc:
        logger.exception("orchestrator: Agent 2 raised — returning refusal.")
        return _catastrophic_failure(
            exc,
            raw_query=raw_query,
            base_state=state,  # type: ignore[arg-type]
            conversation_id=conversation_id,
            turn_index=turn_index,
            agent1_ms=agent1_ms,
        )
    finally:
        agent2_ms = round((time.perf_counter() - t1) * 1000, 1)
        timings: dict = state.get("stage_timings") or {}
        timings["agent2_ms"] = agent2_ms
        state["stage_timings"] = timings

    return dict(result)


# ── Public entry point (backward-compatible wrapper) ──────────────────────────

def run(
    raw_query: str,
    input_image_path: Optional[str] = None,
    conversation_id: Optional[str] = None,
    turn_index: int = 0,
    conversation_history: Optional[list] = None,
    input_modality: str = "text",
) -> dict:
    """
    Run the full NABAT-AI RAG pipeline for one user turn.

    Why this is the entry point: the CLI and evaluation harness call
    `orchestrator.run()` — they don't need the two-phase split. The Streamlit
    UI calls run_agent1() + run_agent2() separately for the two-phase spinner.

    Returns:
        Final AgentState dict. Always returns a dict — never raises.
    """
    state = run_agent1(
        raw_query,
        input_image_path=input_image_path,
        conversation_id=conversation_id,
        turn_index=turn_index,
        conversation_history=conversation_history,
        input_modality=input_modality,
    )
    if state.get("agent2_error") or (state.get("guardrail_flags") or []) == ["catastrophic_failure"]:
        return state
    return run_agent2(state)

# ── Catastrophic failure helper ───────────────────────────────────────────────

def _catastrophic_failure(
    exc: Exception,
    raw_query: str = "",
    base_state: Optional[AgentState] = None,
    conversation_id: Optional[str] = None,
    turn_index: int = 0,
    agent1_ms: float = 0.0,
) -> dict:
    """
    Build a minimal refusal state when an unhandled exception escapes a pipeline stage.

    Why we don't re-raise: the Streamlit UI has no try/except around `run()` — a
    bare exception would crash the demo. The refusal template is the safe fallback.
    """
    logger.error("orchestrator: catastrophic failure — %s: %s", type(exc).__name__, exc)

    state: dict = dict(base_state) if base_state else {}
    state["raw_query"]       = raw_query
    state["agent2_error"]    = f"{type(exc).__name__}: {exc}"
    state["final_response"]  = f"{REFUSAL_TEMPLATE_AR}\n\n{REFUSAL_TEMPLATE_EN}"
    state["is_refusal"]      = True
    state["guardrail_passed"] = False
    state["guardrail_flags"] = ["catastrophic_failure"]
    state["crag_verdict"]    = state.get("crag_verdict") or "Incorrect"
    state["self_rag_verdict"] = state.get("self_rag_verdict") or "flag"
    state["formatted_response"] = state.get("formatted_response") or {
        "al_maktub":    state["final_response"],
        "orthographic": state["final_response"],
        "al_mantuq":    "",
        "citations":    [],
    }
    timings: dict = state.get("stage_timings") or {}
    if agent1_ms:
        timings["agent1_ms"] = agent1_ms
    state["stage_timings"] = timings
    if conversation_id is not None:
        state["conversation_id"] = conversation_id
    state["turn_index"] = turn_index
    return state


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "من هو ناصر الهزاني؟"
    result = run(query)
    print(result.get("final_response", "No response"))
