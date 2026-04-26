"""
tests/test_prototype_router.py
================================
Why this file exists: verify Tier 2 of the four-tier intent funnel —
the embedding-similarity prototype router. Tier 2 sits between Tier 1
(regex, intent_router) and Tier 3 (LLM, semantic_router) and short-
circuits paraphrased meta-questions ("the names of the writers", "tally
of authors") to the deterministic answer node without spending an LLM
call.

These tests run in stub mode so they are deterministic and CI-safe. They
exercise the TF-IDF fallback path (since sentence-transformers is not a
hard dependency); the AraBERT path is loaded lazily and tested only when
the model is locally available.

Run with: PYTHONPATH=src LLM_PROVIDER=stub pytest tests/test_prototype_router.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ── Ensure src on path ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("LLM_PROVIDER", "stub")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_state(query: str) -> dict:
    from fatat_al_arab.state import make_agent_state
    return make_agent_state(query)


@pytest.fixture(autouse=True)
def _reset_router():
    """Force a clean encoder load between tests so threshold tweaks land."""
    from fatat_al_arab.agent1_query_understanding.prototype_router import reset_for_tests
    reset_for_tests()
    yield
    reset_for_tests()


# ── Library invariants ───────────────────────────────────────────────────────

class TestPrototypeLibraryShape:
    def test_every_intent_has_at_least_ten_prototypes(self):
        from fatat_al_arab.agent1_query_understanding.prototype_router import PROTOTYPES
        thin = {k: len(v) for k, v in PROTOTYPES.items() if len(v) < 10}
        assert not thin, f"intents with <10 prototypes: {thin}"

    def test_every_intent_has_arabic_and_english_prototypes(self):
        from fatat_al_arab.agent1_query_understanding.prototype_router import PROTOTYPES
        for intent, protos in PROTOTYPES.items():
            has_ar = any(any("؀" <= c <= "ۿ" for c in p) for p in protos)
            has_en = any(any("a" <= c.lower() <= "z" for c in p) for p in protos)
            assert has_ar, f"intent {intent!r} has no Arabic prototypes"
            assert has_en, f"intent {intent!r} has no English prototypes"

    def test_total_prototype_count(self):
        """Documenting the size of the library — guard against accidental cuts."""
        from fatat_al_arab.agent1_query_understanding.prototype_router import prototype_count
        assert prototype_count() >= 100


# ── Encoder loading ──────────────────────────────────────────────────────────

class TestEncoderLoad:
    def test_encoder_loads_to_a_known_kind(self):
        """In any environment we should land on arabert OR tfidf, never silent fail."""
        from fatat_al_arab.agent1_query_understanding.prototype_router import classify
        result = classify("how many manuscripts are in your archive?")
        assert result["encoder"] in ("arabert", "tfidf"), (
            "prototype_router fell back to 'fallback' — install sklearn or sentence-transformers"
        )


# ── Direct classify() tests ──────────────────────────────────────────────────

class TestPrototypeClassifyParaphrases:
    @pytest.mark.parametrize("query, expected_intent", [
        # Paraphrased counts that the regex layer misses
        ("tally of unique poets", "count_poets"),
        ("size of the poet roster", "count_poets"),
        # Paraphrased "list poets" without the regex trigger words
        ("name the writers in this corpus", "list_poets"),
        ("the names of the poets please", "list_poets"),
        # Paraphrased manuscript count
        ("size of the manuscript collection", "count_manuscripts"),
        # Grief-elegy paraphrases (regex catches some, prototype catches more)
        ("elegies mourning the death of a child", "count_child_grief_poems"),
        ("قصائد رثاء الأبناء", "count_child_grief_poems"),
        # Capabilities-style paraphrase
        ("tell me your skills", "capabilities"),
    ])
    def test_paraphrase_routes_to_expected_intent(self, query, expected_intent):
        from fatat_al_arab.agent1_query_understanding.prototype_router import classify
        result = classify(query)
        assert result["fired"] is True, f"Tier 2 abstained on {query!r}: {result}"
        assert result["intent"] == expected_intent, (
            f"{query!r}: expected {expected_intent}, got {result['intent']} "
            f"(conf={result['confidence']:.3f}, runner_up={result['runner_up_intent']})"
        )


class TestPrototypeAbstainsOnPoetryQueries:
    """Genuine poetry queries must NOT fire Tier 2 — they must fall through to RAG."""
    @pytest.mark.parametrize("query", [
        "show me love poems by Al-Hazani",
        "verses about the desert at dusk",
        "what does this verse mean?",
        "قصائد عن الإبل في الصحراء",
        "explain the imagery in this qasida",
    ])
    def test_poetry_query_does_not_fire(self, query):
        from fatat_al_arab.agent1_query_understanding.prototype_router import classify
        result = classify(query)
        assert result["fired"] is False, (
            f"{query!r} incorrectly fired Tier 2 as {result['intent']!r} "
            f"(conf={result['confidence']:.3f})"
        )


class TestPrototypeMarginGate:
    """A query that's genuinely ambiguous between two intents must abstain."""
    def test_ambiguous_query_abstains_via_margin_gate(self):
        from fatat_al_arab.agent1_query_understanding import prototype_router
        # Synthesise an artificial tied-score scenario by monkey-patching the
        # cosine scorer to return a near-tie. This proves the MARGIN check
        # actually runs even when the absolute threshold is satisfied.
        prototype_router._ensure_loaded()
        intents = sorted(prototype_router.PROTOTYPES.keys())
        # Build a fake score vector: top intent at 0.80, runner at 0.78
        # (margin = 0.02 < MARGIN = 0.05).
        n_protos = len(prototype_router._proto_meta)
        target_a, target_b = intents[0], intents[1]
        fake_scores = []
        for intent, _ in prototype_router._proto_meta:
            if intent == target_a:
                fake_scores.append(0.80)
            elif intent == target_b:
                fake_scores.append(0.78)
            else:
                fake_scores.append(0.10)
        original = prototype_router._cosine_scores
        prototype_router._cosine_scores = lambda _q: fake_scores
        try:
            result = prototype_router.classify("synthetic ambiguous query")
        finally:
            prototype_router._cosine_scores = original

        assert result["fired"] is False
        assert result["intent"] is None
        assert result["confidence"] >= 0.78  # the score did clear threshold
        assert result["runner_up_intent"] is not None


# ── Integration with the real intent_router_node ─────────────────────────────

class TestIntentRouterTier2Wiring:
    """End-to-end: paraphrases that ONLY Tier 2 can catch must reach
    answer_source='registry_lookup' with router_source starting with 'prototype_'."""

    @pytest.mark.parametrize("query, expected_intent", [
        ("the names of the poets please",        "list_poets"),
        ("size of the manuscript collection",    "count_manuscripts"),
        ("tally of unique poets",                "count_poets"),
        ("elegies mourning the death of a child", "count_child_grief_poems"),
    ])
    def test_paraphrase_short_circuits_through_tier_2(self, query, expected_intent):
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state(query)
        out = intent_router_node(state)
        qc = out.get("query_context") or {}
        assert qc.get("answer_source") == "registry_lookup", (
            f"{query!r} did not short-circuit: qc={qc}"
        )
        assert qc.get("deterministic_intent") == expected_intent
        assert (qc.get("router_source") or "").startswith("prototype_"), (
            f"expected prototype_*, got router_source={qc.get('router_source')!r}"
        )

    def test_genuine_poetry_query_falls_through_to_rag(self):
        """The router must never gobble a real poetry question."""
        from fatat_al_arab.agent1_query_understanding.nodes.intent_router import intent_router_node
        state = _make_state("verses about the desert at dusk")
        out = intent_router_node(state)
        qc = out.get("query_context") or {}
        assert qc.get("answer_source") == "rag_pipeline"
        assert qc.get("deterministic_intent") is None
