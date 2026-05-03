"""
tests/test_semantic_intent_router.py
=====================================
Why this file exists: unit tests for the Stage 0.5b semantic router in stub
mode (no network, no API key). Verifies that:
  - All 4 tracks are returned correctly by the stub LLM
  - lru_cache reuses results (no duplicate LLM calls)
  - Timeout + garbage-JSON fallback both default to poetic_rag
  - Low-confidence non-poetic results fall back to poetic_rag
  - validate_query_context raises on track/answer_source mismatch
  - The full routing pipeline (regex → LLM) works end-to-end for all tracks

Run with: LLM_PROVIDER=stub PYTHONPATH=src pytest tests/test_semantic_intent_router.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── Ensure src is on path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("LLM_PROVIDER", "stub")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_state(query: str) -> dict:
    from fatat_al_arab.state import make_agent_state
    return make_agent_state(query)


def _router_classify(query: str) -> dict:
    """Run the full Stage 0.5a → 0.5b pipeline and return QueryContext."""
    from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
    from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import semantic_router_node

    state = _make_state(query)
    state = intent_router_node(state)
    qc = state.get("query_context") or {}
    if qc.get("answer_source") != "registry_lookup":
        state = semantic_router_node(state)
    return state.get("query_context") or {}


# ── Normaliser tests ─────────────────────────────────────────────────────────

class TestNormaliserCollapsesvocalised:
    def test_harakat_stripped(self):
        from fatat_al_arab.embed import normalise_arabic
        assert normalise_arabic("كَمْ") == normalise_arabic("كم")

    def test_ta_marbuta_unified(self):
        from fatat_al_arab.embed import normalise_arabic
        assert normalise_arabic("قصيدة") == normalise_arabic("قصيده")

    def test_vocalised_query_matches_plain(self):
        from fatat_al_arab.embed import normalise_arabic
        # The plan's canonical test: vocalised vs plain qasida form
        vocalised = "كَمْ قَصِيدَةً"
        plain     = "كم قصيده"
        assert normalise_arabic(vocalised) == normalise_arabic(plain)

    def test_alef_variants_unified(self):
        from fatat_al_arab.embed import normalise_arabic
        assert normalise_arabic("آدم") == normalise_arabic("ادم")
        assert normalise_arabic("أحمد") == normalise_arabic("احمد")
        assert normalise_arabic("إبراهيم") == normalise_arabic("ابراهيم")

    def test_ya_alef_maksura_unified(self):
        from fatat_al_arab.embed import normalise_arabic
        assert normalise_arabic("مكى") == normalise_arabic("مكي")
        assert normalise_arabic("علي") == normalise_arabic("علي")

    def test_tatweel_stripped(self):
        from fatat_al_arab.embed import normalise_arabic
        assert normalise_arabic("حـبيب") == normalise_arabic("حبيب")


# ── Validate query context ────────────────────────────────────────────────────

class TestValidateQueryContext:
    def test_poetic_rag_no_constraint(self):
        from fatat_al_arab.state import validate_query_context
        # poetic_rag can have any answer_source
        validate_query_context({"track": "poetic_rag", "answer_source": "rag_pipeline"})

    def test_capabilities_requires_registry_lookup(self):
        from fatat_al_arab.state import validate_query_context
        with pytest.raises(ValueError, match="routing invariant"):
            validate_query_context({"track": "capabilities", "answer_source": "rag_pipeline"})

    def test_capabilities_valid(self):
        from fatat_al_arab.state import validate_query_context
        validate_query_context({"track": "capabilities", "answer_source": "registry_lookup"})

    def test_pipeline_debug_requires_registry_lookup(self):
        from fatat_al_arab.state import validate_query_context
        with pytest.raises(ValueError):
            validate_query_context({"track": "pipeline_debug", "answer_source": "rag_pipeline"})

    def test_no_track_no_error(self):
        from fatat_al_arab.state import validate_query_context
        # No track set — invariant doesn't apply
        validate_query_context({"answer_source": "rag_pipeline"})


# ── Regex router short-circuit ────────────────────────────────────────────────

class TestRegexRouterShortCircuit:
    def test_counting_query_short_circuits(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("how many poems are in this corpus?")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert qc.get("track") == "registry_lookup"
        assert qc.get("router_source") == "regex"

    def test_list_poets_query_short_circuits(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("who are the poets in this corpus?")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert qc.get("deterministic_intent") == "list_poets"
        assert qc.get("router_source") == "regex"

    def test_child_grief_count_query_short_circuits(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("how many poems are about grieving a son or daughter?")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert qc.get("deterministic_intent") == "count_child_grief_poems"
        assert qc.get("router_source") == "regex"

    def test_age_query_short_circuits(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("how old are these manuscripts?")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert qc.get("deterministic_intent") == "age"

    def test_unsupported_dim_wasm(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("find my family's wasm in this manuscript")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert qc.get("deterministic_intent", "").startswith("unsupported_dim_wasm")

    def test_unsupported_dim_marginalia(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("can you find the marginalia annotations?")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert "unsupported_dim_marginalia" in qc.get("deterministic_intent", "")

    def test_poetry_query_falls_through(self):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("poems about camels at dusk")
        result = intent_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("answer_source") == "rag_pipeline"


# ── Semantic router (stub LLM) ────────────────────────────────────────────────

class TestSemanticRouterStub:
    def test_capabilities_query(self):
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached,
        )
        _classify_cached.cache_clear()
        state = _make_state("how can you help me?")
        state["query_context"] = {"answer_source": "rag_pipeline"}
        result = semantic_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("track") == "capabilities"
        assert qc.get("answer_source") == "registry_lookup"

    def test_pipeline_debug_query(self):
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached,
        )
        _classify_cached.cache_clear()
        state = _make_state("what was your crag verdict?")
        state["query_context"] = {"answer_source": "rag_pipeline"}
        result = semantic_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("track") == "pipeline_debug"

    def test_poetic_rag_default(self):
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached,
        )
        _classify_cached.cache_clear()
        state = _make_state("poems about longing and the desert")
        state["query_context"] = {"answer_source": "rag_pipeline"}
        result = semantic_router_node(state)
        qc = result.get("query_context") or {}
        assert qc.get("track") == "poetic_rag"
        assert qc.get("answer_source") == "rag_pipeline"

    def test_cache_hit_no_second_call(self):
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached,
        )
        _classify_cached.cache_clear()
        state1 = _make_state("how can you help me?")
        state1["query_context"] = {"answer_source": "rag_pipeline"}
        semantic_router_node(state1)
        info1 = _classify_cached.cache_info()

        state2 = _make_state("how can you help me?")
        state2["query_context"] = {"answer_source": "rag_pipeline"}
        semantic_router_node(state2)
        info2 = _classify_cached.cache_info()

        # Second call must be a cache hit
        assert info2.hits > info1.hits

    def test_garbage_json_falls_back_to_poetic_rag(self):
        """When the LLM returns invalid JSON, default to poetic_rag."""
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached, _default_result,
        )
        _classify_cached.cache_clear()

        with patch("fatat_al_arab.agent1_query_understanding.nodes.semantic_router.chat",
                   return_value="NOT JSON AT ALL"):
            state = _make_state("some unusual query xyz123")
            state["query_context"] = {"answer_source": "rag_pipeline"}
            result = semantic_router_node(state)
            qc = result.get("query_context") or {}
            assert qc.get("track") == "poetic_rag"

    def test_low_confidence_falls_back(self):
        """Low confidence non-poetic track → poetic_rag."""
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached,
        )
        _classify_cached.cache_clear()

        with patch("fatat_al_arab.agent1_query_understanding.nodes.semantic_router.chat",
                   return_value=json.dumps({
                       "track": "capabilities",
                       "subintent": None,
                       "confidence": 0.3,  # below threshold
                       "alt_family": None,
                       "reasoning": "low confidence test",
                   })):
            state = _make_state("some ambiguous query")
            state["query_context"] = {"answer_source": "rag_pipeline"}
            result = semantic_router_node(state)
            qc = result.get("query_context") or {}
            assert qc.get("track") == "poetic_rag"

    def test_timeout_falls_back_to_poetic_rag(self):
        """Simulated timeout: thread stays alive → default to poetic_rag."""
        import threading
        import time
        from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
            semantic_router_node, _classify_cached,
        )
        _classify_cached.cache_clear()

        def _slow_chat(*args, **kwargs):
            time.sleep(10)
            return '{"track": "capabilities", "confidence": 0.9, "subintent": null, "alt_family": null, "reasoning": "slow"}'

        with patch("fatat_al_arab.agent1_query_understanding.nodes.semantic_router.chat", _slow_chat):
            with patch("fatat_al_arab.agent1_query_understanding.nodes.semantic_router.ROUTER_TIMEOUT_S", 0.05):
                state = _make_state("timeout test query abc456")
                state["query_context"] = {"answer_source": "rag_pipeline"}
                result = semantic_router_node(state)
                qc = result.get("query_context") or {}
                assert qc.get("track") == "poetic_rag"


# ── Orchestrator integration ──────────────────────────────────────────────────

class TestOrchestratorRouting:
    def test_run_agent1_returns_capabilities_track(self):
        from fatat_al_arab.orchestrator import run_agent1
        state = run_agent1("how can you help me?")
        qc = state.get("query_context") or {}
        assert qc.get("track") == "capabilities"
        assert qc.get("answer_source") == "registry_lookup"

    def test_run_agent1_returns_registry_track_for_counting(self):
        from fatat_al_arab.orchestrator import run_agent1
        state = run_agent1("how many poems are in the corpus?")
        qc = state.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup"
        assert qc.get("track") == "registry_lookup"

    def test_run_agent1_returns_poetic_rag_for_poetry_search(self):
        from fatat_al_arab.orchestrator import run_agent1
        state = run_agent1("poems about longing")
        qc = state.get("query_context") or {}
        assert qc.get("track") == "poetic_rag"

    def test_run_agent2_capabilities_renders_blurb(self):
        from fatat_al_arab.orchestrator import run_agent1, run_agent2
        state = run_agent1("how can you help me?")
        result = run_agent2(state)
        final = result.get("final_response", "")
        flags = result.get("guardrail_flags") or []
        assert "capabilities" in flags
        # Should mention either "Cultural Institutions" or the Arabic equivalent
        assert ("Cultural Institutions" in final or "المؤسسات الثقافية" in final)

    def test_run_agent2_pipeline_debug_no_snapshot_returns_gentle_refusal(self):
        from fatat_al_arab.orchestrator import run_agent1, run_agent2
        state = run_agent1("what was your crag verdict?")
        # Ensure no debug_snapshot
        state.pop("debug_snapshot", None)
        result = run_agent2(state)
        final = result.get("final_response", "")
        flags = result.get("guardrail_flags") or []
        assert "pipeline_debug" in flags
        assert "snapshot" in final.lower() or "prior" in final.lower() or "لقطة" in final

    def test_run_agent2_unsupported_dimension_does_not_call_retriever(self):
        from fatat_al_arab.orchestrator import run_agent1, run_agent2
        state = run_agent1("find my family's wasm")
        result = run_agent2(state)
        flags = result.get("guardrail_flags") or []
        assert "unsupported_dimension" in flags
        # retrieval fields should not be populated
        assert not result.get("rrf_top5")
        assert not result.get("bm25_results")

    def test_routing_invariant_raises_on_mismatch(self):
        from fatat_al_arab.state import validate_query_context
        with pytest.raises(ValueError, match="routing invariant"):
            validate_query_context({"track": "capabilities", "answer_source": "rag_pipeline"})


# ── Optional: eval harness (gated by RUN_EVAL=1) ─────────────────────────────

@pytest.mark.skipif(os.getenv("RUN_EVAL") != "1", reason="Set RUN_EVAL=1 to run evaluation")
def test_eval_harness():
    """
    Run the 50-question gold set and assert ≥90% per-track precision.
    Uses the live LLM provider (not stub); requires LLM_API_KEY.
    """
    import json as json_mod
    from fatat_al_arab.agent1_query_understanding.nodes.semantic_router import (
        semantic_router_node, _classify_cached,
    )

    fixtures_path = Path(__file__).parent / "fixtures" / "intent_router_eval.jsonl"
    if not fixtures_path.exists():
        pytest.skip("fixtures/intent_router_eval.jsonl not found")

    records = [json_mod.loads(line) for line in fixtures_path.read_text().splitlines() if line.strip()]
    _classify_cached.cache_clear()

    from collections import defaultdict
    correct: defaultdict = defaultdict(int)
    total:   defaultdict = defaultdict(int)
    misses = []

    for rec in records:
        query    = rec["query"]
        expected = rec["track"]
        total[expected] += 1

        from fatat_al_arab.state import make_agent_state
        state = make_agent_state(query)
        state["query_context"] = {"answer_source": "rag_pipeline"}
        result = semantic_router_node(state)
        got = (result.get("query_context") or {}).get("track", "poetic_rag")

        if got == expected:
            correct[expected] += 1
        else:
            reasoning = (result.get("query_context") or {}).get("intent_router_reasoning", "")
            misses.append({"query": query, "expected": expected, "got": got, "reasoning": reasoning})

    print("\n=== Intent Router Eval ===")
    all_tracks = sorted(set(total.keys()))
    for t in all_tracks:
        prec = correct[t] / total[t] if total[t] else 0
        print(f"  {t:20s}  precision={prec:.0%}  ({correct[t]}/{total[t]})")

    if misses:
        print("\nMisses:")
        for m in misses:
            print(f"  Q: {m['query'][:60]}")
            print(f"     expected={m['expected']}  got={m['got']}")
            print(f"     reasoning: {m['reasoning']}")

    # Assert ≥90% precision per track
    for t in all_tracks:
        prec = correct[t] / total[t] if total[t] else 0
        assert prec >= 0.90, f"Track '{t}' precision {prec:.0%} < 90%"
