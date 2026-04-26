"""
tests/test_orchestrator.py
===========================
Why these tests exist: M7 acceptance gate — the orchestrator is the single
public entry point for the full RAG pipeline. These tests verify that it
returns a well-formed AgentState regardless of whether LangGraph is installed,
and that it never raises even under adversarial inputs.

All LLM calls are routed through the stub provider (LLM_PROVIDER=stub or
absent LLM_API_KEY), so no network access is required.
"""

from __future__ import annotations

import pytest

# ── Shared helper ─────────────────────────────────────────────────────────────

def _run(query="من هو ناصر الهزاني؟", **kwargs):
    """Call orchestrator.run() and return the result dict."""
    from fatat_al_arab.orchestrator import run
    return run(query, **kwargs)


# ── 1. Return type ────────────────────────────────────────────────────────────

class TestReturnType:
    def test_run_returns_dict(self):
        """run() must always return a dict, never raise."""
        result = _run()
        assert isinstance(result, dict)

    def test_run_returns_dict_on_english_query(self):
        result = _run("Who is Nasser Al-Hazani?")
        assert isinstance(result, dict)


# ── 2. final_response ─────────────────────────────────────────────────────────

class TestFinalResponse:
    def test_final_response_key_present(self):
        result = _run()
        assert "final_response" in result

    def test_final_response_is_string(self):
        result = _run()
        assert isinstance(result["final_response"], str)

    def test_final_response_non_empty(self):
        result = _run()
        assert len(result["final_response"]) > 0

    def test_out_of_corpus_query_returns_refusal_in_stub_mode(self, monkeypatch):
        """OOC fixture queries must hit the scoped refusal path offline."""
        from fatat_al_arab import llm
        from fatat_al_arab.guardrails import REFUSAL_TEMPLATE_AR

        monkeypatch.setattr(llm, "PROVIDER", "stub")

        result = _run("ما هي قصائد أبي نواس المشهورة؟")

        assert result.get("is_refusal") is True
        assert REFUSAL_TEMPLATE_AR in result.get("final_response", "")


# ── 3. formatted_response structure ──────────────────────────────────────────

class TestFormattedResponse:
    def test_formatted_response_key_present(self):
        result = _run()
        assert "formatted_response" in result

    def test_formatted_response_has_al_maktub(self):
        result = _run()
        assert "al_maktub" in result["formatted_response"]

    def test_formatted_response_has_orthographic(self):
        result = _run()
        assert "orthographic" in result["formatted_response"]

    def test_formatted_response_has_al_mantuq(self):
        result = _run()
        assert "al_mantuq" in result["formatted_response"]

    def test_formatted_response_has_citations(self):
        result = _run()
        assert "citations" in result["formatted_response"]

    def test_citations_is_list(self):
        result = _run()
        assert isinstance(result["formatted_response"]["citations"], list)


# ── 4. guardrail_passed ───────────────────────────────────────────────────────

class TestGuardrailPassed:
    def test_guardrail_passed_key_present(self):
        result = _run()
        assert "guardrail_passed" in result

    def test_guardrail_passed_is_bool(self):
        result = _run()
        assert isinstance(result["guardrail_passed"], bool)


# ── 5. crag_verdict ───────────────────────────────────────────────────────────

class TestCragVerdict:
    def test_crag_verdict_key_present(self):
        result = _run()
        assert "crag_verdict" in result

    def test_crag_verdict_valid_value(self):
        result = _run()
        assert result["crag_verdict"] in ("Correct", "Ambiguous", "Incorrect")


# ── 6. self_rag_verdict ───────────────────────────────────────────────────────

class TestSelfRagVerdict:
    def test_self_rag_verdict_key_present(self):
        result = _run()
        assert "self_rag_verdict" in result

    def test_self_rag_verdict_valid_value(self):
        result = _run()
        assert result["self_rag_verdict"] in ("pass", "retry", "flag")


# ── 7. stage_timings ─────────────────────────────────────────────────────────

class TestStageTimings:
    def test_stage_timings_key_present(self):
        result = _run()
        assert "stage_timings" in result

    def test_stage_timings_is_dict(self):
        result = _run()
        assert isinstance(result["stage_timings"], dict)

    def test_stage_timings_has_agent1_ms(self):
        result = _run()
        assert "agent1_ms" in result["stage_timings"]

    def test_stage_timings_has_agent2_ms(self):
        result = _run()
        assert "agent2_ms" in result["stage_timings"]

    def test_stage_timings_are_numeric(self):
        result = _run()
        for key, val in result["stage_timings"].items():
            assert isinstance(val, (int, float)), f"{key} timing is not numeric: {val!r}"


# ── 8. conversation_id preserved ─────────────────────────────────────────────

class TestConversationId:
    def test_conversation_id_preserved(self):
        result = _run(conversation_id="session-abc-123")
        assert result.get("conversation_id") == "session-abc-123"

    def test_conversation_id_none_when_not_provided(self):
        result = _run()
        # conversation_id may be absent or None — either is fine
        assert result.get("conversation_id") is None or "conversation_id" not in result


# ── 9. turn_index preserved ───────────────────────────────────────────────────

class TestTurnIndex:
    def test_turn_index_preserved_default(self):
        result = _run()
        assert result.get("turn_index") == 0

    def test_turn_index_preserved_nonzero(self):
        result = _run(turn_index=3)
        assert result.get("turn_index") == 3


# ── 10. Catastrophic failure safety ──────────────────────────────────────────

class TestCatastrophicFailure:
    def test_none_query_does_not_raise(self):
        """Passing None as query must never raise — returns a refusal dict."""
        from fatat_al_arab.orchestrator import run
        result = run(None)  # type: ignore[arg-type]
        assert isinstance(result, dict)

    def test_none_query_returns_refusal_response(self):
        from fatat_al_arab.orchestrator import run
        from fatat_al_arab.guardrails import REFUSAL_TEMPLATE_AR, REFUSAL_TEMPLATE_EN
        result = run(None)  # type: ignore[arg-type]
        assert "final_response" in result
        # Must contain one of the refusal template strings
        fr = result["final_response"]
        assert REFUSAL_TEMPLATE_AR in fr or REFUSAL_TEMPLATE_EN in fr

    def test_empty_string_query_does_not_raise(self):
        from fatat_al_arab.orchestrator import run
        result = run("")
        assert isinstance(result, dict)

    def test_empty_string_returns_refusal(self):
        from fatat_al_arab.orchestrator import run
        from fatat_al_arab.guardrails import REFUSAL_TEMPLATE_AR, REFUSAL_TEMPLATE_EN
        result = run("")
        fr = result.get("final_response", "")
        assert REFUSAL_TEMPLATE_AR in fr or REFUSAL_TEMPLATE_EN in fr

    def test_catastrophic_failure_sets_agent2_error(self):
        """When Agent 2 raises, agent2_error is set in the returned state."""
        from fatat_al_arab.orchestrator import run
        from unittest.mock import patch

        def _boom(state):
            raise RuntimeError("simulated total collapse")

        with patch(
            "fatat_al_arab.orchestrator._run_agent2",
            side_effect=RuntimeError("simulated total collapse"),
        ):
            result = run("اختبار الفشل الكارثي")

        assert isinstance(result, dict)
        assert "agent2_error" in result
        assert "simulated total collapse" in result["agent2_error"]

    def test_catastrophic_failure_final_response_is_refusal(self):
        """On catastrophic Agent 2 failure, final_response must be the refusal template."""
        from fatat_al_arab.orchestrator import run
        from fatat_al_arab.guardrails import REFUSAL_TEMPLATE_AR, REFUSAL_TEMPLATE_EN
        from unittest.mock import patch

        with patch(
            "fatat_al_arab.orchestrator._run_agent2",
            side_effect=RuntimeError("kaboom"),
        ):
            result = run("سؤال")

        fr = result.get("final_response", "")
        assert REFUSAL_TEMPLATE_AR in fr or REFUSAL_TEMPLATE_EN in fr

    def test_catastrophic_failure_conversation_id_preserved(self):
        """conversation_id must survive even a catastrophic failure."""
        from fatat_al_arab.orchestrator import run
        from unittest.mock import patch

        with patch(
            "fatat_al_arab.orchestrator._run_agent2",
            side_effect=RuntimeError("kaboom"),
        ):
            result = run("سؤال", conversation_id="keep-me-42")

        assert result.get("conversation_id") == "keep-me-42"


# ── 11. input_image_path forwarded ───────────────────────────────────────────

class TestImagePath:
    def test_input_image_path_set_in_state(self):
        """If an image path is provided it should appear in the returned state."""
        result = _run(input_image_path="/tmp/fake_image.png")
        # The state should carry the path (bilingual_analyzer reads it)
        assert result.get("input_image_path") == "/tmp/fake_image.png"


# ── 12. raw_query preserved ───────────────────────────────────────────────────

class TestRawQuery:
    def test_raw_query_preserved_in_state(self):
        result = _run("قصيدة في المطر")
        assert result.get("raw_query") == "قصيدة في المطر"
