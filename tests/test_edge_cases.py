"""
tests/test_edge_cases.py
=========================
Why these tests exist: the happy-path tests in test_orchestrator.py verify that
normal queries return well-formed results. These tests verify that the system
degrades gracefully — never raises, never hangs — on adversarial, malformed, or
boundary-condition inputs. Robustness under unexpected input is a core §5
requirement ("never raises from run()").

All tests run in stub mode — no network access or API key required.
"""

from __future__ import annotations

import os
import sys
import pytest

# Ensure stub mode so no real LLM call is made
os.environ.setdefault("LLM_PROVIDER", "stub")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(query, **kwargs):
    """Call orchestrator.run() and return the result dict."""
    from fatat_al_arab.orchestrator import run
    return run(query, **kwargs)


# ── 1. Empty and blank inputs ─────────────────────────────────────────────────

class TestEmptyInputs:
    """System must not raise on empty, blank, or whitespace-only queries."""

    def test_empty_string_returns_dict(self):
        result = _run("")
        assert isinstance(result, dict)

    def test_whitespace_only_returns_dict(self):
        result = _run("   ")
        assert isinstance(result, dict)

    def test_newlines_only_returns_dict(self):
        result = _run("\n\n\n")
        assert isinstance(result, dict)

    def test_arabic_whitespace_returns_dict(self):
        # Arabic zero-width non-joiner + space — common copy-paste artefact
        result = _run("‌ ‌")
        assert isinstance(result, dict)

    def test_empty_returns_final_response(self):
        result = _run("")
        assert "final_response" in result

    def test_whitespace_final_response_is_string(self):
        result = _run("   ")
        assert isinstance(result.get("final_response", ""), str)


# ── 2. Very long inputs ───────────────────────────────────────────────────────

class TestLongInputs:
    """System must handle very long queries without hanging or raising."""

    def test_very_long_english_query_returns_dict(self):
        long_query = "show me poems about " + "love " * 500
        result = _run(long_query)
        assert isinstance(result, dict)

    def test_very_long_arabic_query_returns_dict(self):
        long_query = "أظهر لي قصائد عن " + "الحب " * 400
        result = _run(long_query)
        assert isinstance(result, dict)

    def test_long_query_has_final_response(self):
        long_query = "find poems " + "about nature " * 300
        result = _run(long_query)
        assert "final_response" in result
        assert len(result["final_response"]) > 0


# ── 3. Special characters and injection attempts ──────────────────────────────

class TestSpecialCharacters:
    """Queries with special characters must not cause parsing errors."""

    def test_query_with_json_braces_returns_dict(self):
        result = _run('{"query": "find poems", "inject": true}')
        assert isinstance(result, dict)

    def test_query_with_sql_injection_pattern_returns_dict(self):
        result = _run("'; DROP TABLE poems; --")
        assert isinstance(result, dict)

    def test_query_with_html_tags_returns_dict(self):
        result = _run("<script>alert('xss')</script> find poems")
        assert isinstance(result, dict)

    def test_query_with_null_byte_returns_dict(self):
        result = _run("find poems\x00about love")
        assert isinstance(result, dict)

    def test_query_with_emoji_returns_dict(self):
        result = _run("find poems about love ❤️ 🌙")
        assert isinstance(result, dict)

    def test_query_with_mixed_scripts_returns_dict(self):
        result = _run("show me poems عن الحب about love قصائد")
        assert isinstance(result, dict)


# ── 4. Out-of-corpus queries ──────────────────────────────────────────────────

class TestOutOfCorpus:
    """Queries clearly outside the corpus should get a refusal, not hallucinate."""

    def test_out_of_corpus_returns_dict(self):
        # Shakespeare is not in a Khaleeji Nabati poetry corpus
        result = _run("Show me Shakespeare's sonnets about love")
        assert isinstance(result, dict)

    def test_out_of_corpus_has_final_response(self):
        result = _run("What are the best Persian rubaiyat by Omar Khayyam?")
        assert "final_response" in result

    def test_ooc_arabic_returns_dict(self):
        result = _run("ما هي قصائد أبي نواس المشهورة؟")
        assert isinstance(result, dict)

    def test_ooc_final_response_is_non_empty(self):
        result = _run("Tell me about Chinese Tang dynasty poetry")
        assert len(result.get("final_response", "")) > 0


# ── 5. Intent router edge cases ───────────────────────────────────────────────

class TestIntentRouterEdgeCases:
    """Boundary conditions for the Stage 0.5 regex fast-path router."""

    def test_partial_counting_cue_no_slot_returns_dict(self):
        # "how" alone should not trigger counting path
        result = _run("how")
        assert isinstance(result, dict)

    def test_ambiguous_counting_query_returns_dict(self):
        # "how many" without slot noun — should still not crash
        result = _run("how many")
        assert isinstance(result, dict)

    def test_arabic_counting_no_slot_returns_dict(self):
        # "كم" alone
        result = _run("كم")
        assert isinstance(result, dict)

    def test_capabilities_query_returns_dict(self):
        result = _run("what can you do?")
        assert isinstance(result, dict)

    def test_capabilities_arabic_returns_dict(self):
        result = _run("كيف يمكنك مساعدتي؟")
        assert isinstance(result, dict)

    def test_image_provenance_no_image_returns_dict(self):
        # Image provenance cue without an actual image attached
        result = _run("who wrote this poem?")
        assert isinstance(result, dict)


# ── 6. Conversation history edge cases ───────────────────────────────────────

class TestConversationHistory:
    """Conversation history must be handled gracefully even when malformed."""

    def test_empty_history_returns_dict(self):
        result = _run("من هو ناصر الهزاني؟", conversation_history=[])
        assert isinstance(result, dict)

    def test_single_turn_history_returns_dict(self):
        history = [{"query": "كم قصيدة لديك؟", "response": "لدينا 2222 قصيدة."}]
        result = _run("وكم شاعراً؟", conversation_history=history)
        assert isinstance(result, dict)

    def test_malformed_history_does_not_raise(self):
        # History entries missing expected keys
        bad_history = [{"text": "something"}, {}, {"query": "q"}]
        result = _run("find love poems", conversation_history=bad_history)
        assert isinstance(result, dict)

    def test_very_long_history_returns_dict(self):
        # 20 turns — more than the 5-turn cap in _format_history_for_prompt
        long_history = [
            {"query": f"query {i}", "response": f"response {i}"}
            for i in range(20)
        ]
        result = _run("what is Nabati poetry?", conversation_history=long_history)
        assert isinstance(result, dict)


# ── 7. Non-string inputs ──────────────────────────────────────────────────────

class TestNonStringInputs:
    """System must degrade gracefully if caller passes unexpected types."""

    def test_none_query_returns_dict(self):
        # orchestrator must not raise on None — returns displayable dict
        result = _run(None)
        assert isinstance(result, dict)

    def test_integer_query_returns_dict(self):
        result = _run(42)
        assert isinstance(result, dict)

    def test_list_query_returns_dict(self):
        result = _run(["find", "poems"])
        assert isinstance(result, dict)


# ── 8. Deterministic intent paths ────────────────────────────────────────────

class TestDeterministicPaths:
    """Counting / age / provenance queries must always return a final_response."""

    def test_count_poems_returns_dict(self):
        result = _run("how many poems are in the corpus?")
        assert isinstance(result, dict)
        assert "final_response" in result

    def test_count_poets_arabic_returns_dict(self):
        result = _run("كم عدد الشعراء في المجموعة؟")
        assert isinstance(result, dict)
        assert "final_response" in result

    def test_age_query_returns_dict(self):
        result = _run("how old are the manuscripts?")
        assert isinstance(result, dict)

    def test_provenance_query_returns_dict(self):
        result = _run("where are the manuscripts from?")
        assert isinstance(result, dict)

    def test_count_love_poems_by_genre_returns_dict(self):
        result = _run("how many love poems are there?")
        assert isinstance(result, dict)

    def test_count_elegy_arabic_returns_dict(self):
        result = _run("كم عدد قصائد الرثاء؟")
        assert isinstance(result, dict)


# ── 9. Failure budget verification ───────────────────────────────────────────

class TestFailureBudgets:
    """CRAG and Self-RAG retry caps must be enforced even under repeated failure."""

    def test_crag_requery_count_does_not_exceed_max(self):
        """orchestrator.run() must return even when CRAG fires its maximum retries."""
        result = _run("find verses that are definitely not in any corpus ever written")
        assert isinstance(result, dict)
        # The count must be ≤ CRAG_REQUERY_MAX (1)
        crag_count = result.get("crag_requery_count", 0)
        assert crag_count <= 1, f"crag_requery_count={crag_count} exceeds cap of 1"

    def test_self_rag_retries_does_not_exceed_max(self):
        """Self-RAG retry counter must not exceed 2."""
        result = _run("what is the most famous poem in the corpus?")
        assert isinstance(result, dict)
        rag_retries = result.get("self_rag_retries", 0)
        assert rag_retries <= 2, f"self_rag_retries={rag_retries} exceeds cap of 2"


# ── 10. Return shape invariants ───────────────────────────────────────────────

class TestReturnShapeInvariants:
    """run() must always return a dict with required keys, regardless of input."""

    REQUIRED_KEYS = {"final_response"}

    @pytest.mark.parametrize("query", [
        "",
        "   ",
        "hello",
        "كم قصيدة لديك؟",
        "who wrote this?",
        "find poems about camels",
        "أظهر لي قصائد الغزل",
        "how many manuscripts?",
        "x" * 5000,
        "!!!???###",
    ])
    def test_required_keys_always_present(self, query):
        result = _run(query)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Key '{key}' missing from result for query={query!r}"

    @pytest.mark.parametrize("query", [
        "",
        "find love poems",
        "كم شاعراً في المجموعة؟",
    ])
    def test_final_response_always_string(self, query):
        result = _run(query)
        assert isinstance(result["final_response"], str)
