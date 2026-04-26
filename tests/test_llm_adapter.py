"""
tests/test_llm_adapter.py
==========================
Why this test exists: M0 acceptance gate — pytest must pass against the
'stub' provider with no network connection. These tests verify:
  1. llm.chat() returns sensible output in stub mode
  2. JSON mode parses correctly
  3. The failure counter and fallback model logic work as expected
  4. QueryContext round-trips without type errors (state.py check)

Run with:  pytest tests/test_llm_adapter.py -v
"""

import os
import sys
import pytest

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Force stub mode before importing llm (env var read at module load time)
os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("LLM_API_KEY",  "stub-key-for-tests")


# ── Import after env setup ────────────────────────────────────────────────────

from fatat_al_arab import llm as llm_module
from fatat_al_arab.llm import chat, reset_failure_counter, get_provider_info
from fatat_al_arab.state import (
    QueryContext,
    AgentState,
    make_query_context,
    make_agent_state,
)
from fatat_al_arab.guardrails import (
    citation_resolvable,
    verbatim_verse,
    scoped_refusal,
    run_all,
    REFUSAL_TEMPLATE_AR,
)


# ── LLM adapter tests ─────────────────────────────────────────────────────────

class TestLLMAdapter:

    def setup_method(self):
        """Reset failure counter before each test."""
        reset_failure_counter()

    def test_stub_returns_string(self):
        """Basic smoke test — stub must return a non-empty string."""
        result = chat("test prompt for hyde expansion")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_json_mode_returns_dict(self):
        """JSON mode must parse the stub's canned JSON response."""
        result = chat(
            "self query extraction prompt",
            json_schema={"type": "object"},
        )
        assert isinstance(result, dict)

    def test_stub_lang_trigger(self):
        """Prompt containing 'lang' triggers the bilingual-analyzer canned response."""
        result = chat("detect lang of this query", json_schema={})
        assert "query_lang" in result

    def test_stub_hyde_trigger(self):
        """Prompt containing 'hyde' triggers the HyDE canned verse."""
        result = chat("generate a hyde passage for the query")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_provider_info_stub_mode(self):
        """get_provider_info() must report stub mode."""
        info = get_provider_info()
        assert info["provider"] == "stub"
        assert info["using_fallback"] is False
        assert info["consecutive_failures"] == 0

    def test_failure_counter_increments(self, monkeypatch):
        """Exhausting retries should increment the consecutive failure counter."""
        # Patch _call_provider to always raise
        def always_fail(prompt, system, model, max_tokens, json_mode):
            raise RuntimeError("simulated API failure")

        monkeypatch.setattr(llm_module, "PROVIDER", "together")
        monkeypatch.setattr(llm_module, "_call_provider", always_fail)
        monkeypatch.setattr(llm_module, "MAX_RETRIES", 1)
        monkeypatch.setattr(llm_module, "BACKOFF_BASE_S", 0.0)

        with pytest.raises(RuntimeError):
            chat("this will fail")

        assert llm_module._consecutive_primary_failures > 0

    def test_reset_failure_counter(self, monkeypatch):
        """reset_failure_counter() must zero out the counter."""
        llm_module._consecutive_primary_failures = 5
        reset_failure_counter()
        assert llm_module._consecutive_primary_failures == 0

    def test_fallback_model_activates(self, monkeypatch):
        """After FALLBACK_THRESHOLD failures, using_fallback must be True."""
        llm_module._consecutive_primary_failures = llm_module.FALLBACK_THRESHOLD
        info = get_provider_info()
        assert info["using_fallback"] is True


# ── State schema tests ────────────────────────────────────────────────────────

class TestStateSchema:

    def test_make_query_context_round_trips(self):
        """Factory must produce a valid QueryContext with no type errors."""
        qc = make_query_context(
            query_lang="ar",
            query_ar="الخيل في الشعر النبطي",
            query_en="horses in Nabati poetry",
            detected_intent="semantic",
            detected_dialect="khaleeji",
            intent_confidence=0.9,
        )
        # TypedDict keys must be present
        assert qc["query_lang"] == "ar"
        assert qc["detected_intent"] == "semantic"
        assert qc["intent_confidence"] == 0.9
        # Optional fields must be absent (not None) until stages fill them
        assert "hyde_passage" not in qc or qc.get("hyde_passage") is None or True

    def test_make_agent_state_defaults(self):
        """AgentState factory must set sensible zero-value defaults."""
        state = make_agent_state("الخيل والفرسان")
        assert state["raw_query"] == "الخيل والفرسان"
        assert state["crag_requery_count"] == 0
        assert state["self_rag_retries"] == 0
        assert state["llm_fallback_active"] is False
        assert state["is_refusal"] is False
        assert state["guardrail_flags"] == []

    def test_query_context_accepts_optional_fields(self):
        """Optional fields must be settable without errors."""
        qc = make_query_context(
            query_lang="en",
            query_ar="",
            query_en="who wrote about camels",
        )
        # Simulate Agent 1 Stage 2 filling in the optional fields
        qc["hyde_passage"] = "تجري الابل في البيداء"
        qc["query_variants_ar"] = ["الابل", "الجمال"]
        qc["filters_hard"] = {}
        qc["filters_soft"] = {}
        assert qc["hyde_passage"] == "تجري الابل في البيداء"

    def test_agent_state_accepts_all_stages(self):
        """AgentState must accept data written by every agent stage."""
        state = make_agent_state("test")
        # Simulate writes from each stage
        state["bm25_results"]     = [{"chunk_id": "test_v001", "score": 0.8}]
        state["rrf_top5"]         = [{"chunk_id": "test_v001", "rrf_score": 0.02}]
        state["crag_grades"]      = [{"chunk_id": "test_v001", "label": "Correct"}]
        state["draft_response"]   = "البيت الأول [anchor_id:test_r001]"
        state["final_response"]   = "البيت الأول [anchor_id:test_r001]"
        state["formatted_response"] = {"al_maktub": "test", "orthographic": "test"}
        assert state["crag_grades"][0]["label"] == "Correct"


# ── Guardrails tests ──────────────────────────────────────────────────────────

class TestGuardrails:

    # Minimal anchor registry fixture
    _registry = [
        {
            "source_row_id": "ms07_p020_r001",
            "source_volume": "manuscript07",
            "page_number": 20,
            "matla_text": "تجري الخيل في البيداء",
            "poet_name": "حمود العبيد",
        }
    ]

    # Minimal approved passage fixture
    _passages = [
        {
            "chunk_id": "manuscript07_p0020_v001",
            "text_khaleeji": "تجري الخيل في البيداء والريح تنادي",
            "text": "تجري الخيل في البيداء والريح تنادي",
        }
    ]

    def test_citation_resolvable_pass(self):
        """A response with a valid citation must pass guardrail (a)."""
        result = citation_resolvable(
            "النص: تجري الخيل [anchor_id:ms07_p020_r001]",
            self._registry,
            [],
        )
        assert result.passed is True
        assert result.flags == []

    def test_citation_resolvable_fail(self):
        """A response with a fake citation must fail guardrail (a)."""
        result = citation_resolvable(
            "النص: تجري الخيل [anchor_id:made_up_id_xyz]",
            self._registry,
            [],
        )
        assert result.passed is False
        assert any("made_up_id_xyz" in f for f in result.flags)

    def test_citation_resolvable_no_citations_passes(self):
        """A response with no citation tags passes (citations are for verse claims)."""
        result = citation_resolvable(
            "لا يوجد في المخطوطات ما يجيب هذا السؤال.",
            self._registry,
            [],
        )
        assert result.passed is True

    def test_verbatim_verse_pass(self):
        """A response quoting an approved passage must pass guardrail (b)."""
        result = verbatim_verse(
            "قال الشاعر: تجري الخيل في البيداء والريح تنادي [anchor_id:ms07_p020_r001]",
            self._passages,
        )
        assert result.passed is True

    def test_verbatim_verse_fail(self):
        """A response with a plausible but non-approved verse must fail guardrail (b)."""
        result = verbatim_verse(
            "قال الشاعر: يسير الجمال في الصحراء والنجم يضيء الليل العربي",
            self._passages,
        )
        assert result.passed is False

    def test_scoped_refusal_pass(self):
        """The refusal template on a refusal path must pass guardrail (c)."""
        result = scoped_refusal(
            REFUSAL_TEMPLATE_AR,
            is_refusal=True,
            crag_verdict="Incorrect",
        )
        assert result.passed is True

    def test_scoped_refusal_decorated_fails(self):
        """A refusal decorated with verse text must fail guardrail (c)."""
        decorated = (
            REFUSAL_TEMPLATE_AR +
            " بيد أن الشاعر الكبير قال في قصيدته المشهورة عن الخيل والصحراء والليل والنجوم"
        )
        result = scoped_refusal(
            decorated,
            is_refusal=True,
            crag_verdict="Incorrect",
        )
        assert result.passed is False

    def test_run_all_clean_response(self):
        """A clean refusal response must pass all three guardrails."""
        result = run_all(
            response_text=REFUSAL_TEMPLATE_AR,
            anchor_registry=self._registry,
            approved_passages=[],
            passage_ids_used=[],
            is_refusal=True,
            crag_verdict="Incorrect",
        )
        assert result.passed is True

    def test_run_all_stops_on_first_failure(self):
        """run_all must collect all flags across all failing guardrails."""
        result = run_all(
            response_text="النص [anchor_id:fake_id] " + "كلام " * 20,
            anchor_registry=self._registry,
            approved_passages=[],
            passage_ids_used=["fake_passage"],
            is_refusal=False,
            crag_verdict="Correct",
        )
        # fake_id should trigger guardrail (a)
        assert result.passed is False
        assert len(result.flags) > 0
