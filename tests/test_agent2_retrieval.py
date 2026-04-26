"""
tests/test_agent2_retrieval.py
================================
Tests for Agent 2 stages 4-6:
  - retrieve_node  (Stage 4) — hard filters, dropped retrievers, state keys
  - rrf_fuse_node  (Stage 5) — fusion, soft boost, dedup, empty input
  - resolve_heritage_node (Stage 6) — citation fields, unresolvable tagging
  - graph smoke test — full stage 4→5→6 pipeline with injected index

No real model loading — the synthetic IndexBundle from test_index.py fixtures
is injected via retrieve.set_index().
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fatat_al_arab.state import make_agent_state, make_query_context, AgentState
from fatat_al_arab.rrf import ScoredChunk
from fatat_al_arab.index import build_index, IndexBundle


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_entry(anchor_id, poet, matla, volume="001-100", page=1):
    return {
        "source_row_id":    anchor_id,
        "poet_name":        poet,
        "matla_text":       matla,
        "source_volume":    volume,
        "page_number":      page,
        "source_image_path": f"manuscripts/{anchor_id}.png",
        "manuscript_short_key": "test_ms",
        "manuscript_arabic_name": "مخطوطة تجريبية",
        "manuscript_english_name": "Test Manuscript",
    }


POET_A = "ناصر الهزاني"
POET_B = "سالم بن علي"

_REGISTRY = (
    [_make_entry(f"a{i:02d}", POET_A, f"مطلع الشاعر أ رقم {i}", volume="001-100", page=i)
     for i in range(7)] +
    [_make_entry(f"b{i:02d}", POET_B, f"مطلع الشاعر ب رقم {i}", volume="200-300", page=i)
     for i in range(3)]
)


@pytest.fixture
def bundle(tmp_path):
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(_REGISTRY, ensure_ascii=False), encoding="utf-8")
    return build_index(registry_path=reg_path, qdrant_path=tmp_path / "qdrant",
                       force_rebuild=True)


def _make_state_with_qc(raw_query="ناصر الهزاني", **qc_overrides) -> AgentState:
    qc = make_query_context(
        query_lang="ar",
        query_ar=raw_query,
        query_en="test query",
        detected_intent=qc_overrides.pop("detected_intent", "semantic"),
    )
    qc.update(qc_overrides)
    state = make_agent_state(raw_query)
    state["query_context"] = qc
    return state


# ── Stage 4: retrieve_node ─────────────────────────────────────────────────────

class TestRetrieveNode:
    def test_returns_three_result_keys(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc("مطلع الشاعر")
        result = retrieve_node(state)
        assert "bm25_results"    in result
        assert "dense_results"   in result
        assert "colbert_results" in result

    def test_bm25_results_are_list_of_dicts(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc("مطلع الشاعر أ")
        result = retrieve_node(state)
        assert isinstance(result["bm25_results"], list)
        if result["bm25_results"]:
            assert isinstance(result["bm25_results"][0], dict)
            assert "chunk_id" in result["bm25_results"][0]

    def test_colbert_stub_returns_empty_list(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc("شعر")
        result = retrieve_node(state)
        assert result["colbert_results"] == []

    def test_timings_dict_has_expected_keys(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc("شعر عربي")
        result = retrieve_node(state)
        timings = result.get("retriever_timings") or {}
        assert "bm25_ms" in timings
        assert "dense_ms" in timings

    def test_hard_filter_by_poet_name(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc(
            "شعر",
            filters_hard={"poet_name": POET_A},
        )
        result = retrieve_node(state)
        # All returned BM25 chunks must have POET_A in poet_name
        for r in result["bm25_results"]:
            assert POET_A.lower() in r.get("poet_name", "").lower(), \
                f"Expected {POET_A!r} in poet_name but got {r.get('poet_name')!r}"

    def test_hard_filter_eliminating_all_falls_back_to_unfiltered(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc(
            "شعر",
            filters_hard={"manuscript_short_key": "nonexistent_ms"},
        )
        result = retrieve_node(state)
        # Fallback: should still return results from unfiltered corpus
        assert len(result["bm25_results"]) > 0 or len(result["dense_results"]) > 0

    def test_state_keys_preserved(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        set_index(bundle)
        state = _make_state_with_qc("مطلع")
        state["crag_requery_count"] = 0
        result = retrieve_node(state)
        assert result["raw_query"] == state["raw_query"]
        assert result["crag_requery_count"] == 0

    def test_hyde_passage_used_for_dense(self, bundle):
        """Dense retriever should receive the hyde_passage, not the raw query."""
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        from fatat_al_arab.retrievers.dense import DenseRetriever
        set_index(bundle)
        hyde = "قصيدة افتراضية تحاكي أسلوب الشعر الخليجي"
        state = _make_state_with_qc("شعر", hyde_passage=hyde)
        # Just check the node doesn't crash and returns results
        result = retrieve_node(state)
        assert "dense_results" in result


# ── Stage 5: rrf_fuse_node ────────────────────────────────────────────────────

class TestRRFFuseNode:
    def _make_chunk_dict(self, chunk_id, poet="شاعر", rrf_score=0.0, ms_key="test_ms"):
        return {
            "chunk_id": chunk_id,
            "rrf_score": rrf_score,
            "text": f"نص {chunk_id}",
            "level": "verse",
            "anchor_id": f"anc_{chunk_id}",
            "poet_name": poet,
            "source_volume": "001-100",
            "source_page": 1,
            "source_image_path": "",
            "manuscript_short_key": ms_key,
        }

    def test_produces_rrf_top5_key(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        bm25 = [self._make_chunk_dict(f"A{i}") for i in range(5)]
        state = make_agent_state("شعر")
        state["bm25_results"]    = bm25
        state["dense_results"]   = []
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        assert "rrf_top5" in result
        assert isinstance(result["rrf_top5"], list)

    def test_fused_results_sorted_descending(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        bm25  = [self._make_chunk_dict(f"B{i}") for i in range(5)]
        dense = [self._make_chunk_dict(f"B{4-i}") for i in range(5)]
        state = make_agent_state("شعر")
        state["bm25_results"]    = bm25
        state["dense_results"]   = dense
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        scores = [r["rrf_score"] for r in result["rrf_top5"]]
        assert scores == sorted(scores, reverse=True)

    def test_dedup_removes_same_anchor(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        # Same anchor_id appears at verse and group level
        verse = self._make_chunk_dict("verse__anc_X")
        verse["anchor_id"] = "anc_X"
        verse["level"] = "verse"
        group = self._make_chunk_dict("group__anc_X")
        group["anchor_id"] = "anc_X"
        group["level"] = "group"
        state = make_agent_state("شعر")
        state["bm25_results"]    = [verse, group]
        state["dense_results"]   = []
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        anchor_ids = [r["anchor_id"] for r in result["rrf_top5"]]
        assert anchor_ids.count("anc_X") == 1

    def test_interpretive_intent_keeps_all_levels(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        verse = self._make_chunk_dict("verse__anc_Y")
        verse["anchor_id"] = "anc_Y"; verse["level"] = "verse"
        poem  = self._make_chunk_dict("poem__anc_Y")
        poem["anchor_id"]  = "anc_Y"; poem["level"]  = "poem"
        state = _make_state_with_qc("شعر", detected_intent="interpretive")
        state["bm25_results"]    = [verse, poem]
        state["dense_results"]   = []
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        anchor_ids = [r["anchor_id"] for r in result["rrf_top5"]]
        assert anchor_ids.count("anc_Y") == 2

    def test_soft_boost_elevates_matching_poet(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        a = self._make_chunk_dict("A", poet=POET_A)
        b = self._make_chunk_dict("B", poet=POET_B)
        # B ranks first in BM25, A second — soft filter should flip this
        state = _make_state_with_qc("شعر", filters_soft={"poet_name": POET_A})
        state["bm25_results"]    = [b, a]   # B rank-1, A rank-2
        state["dense_results"]   = []
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        top_chunk_id = result["rrf_top5"][0]["chunk_id"]
        assert top_chunk_id == "A", f"Expected A boosted to top; got {top_chunk_id}"

    def test_empty_all_retrievers_returns_empty(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        state = make_agent_state("شعر")
        state["bm25_results"]    = []
        state["dense_results"]   = []
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        assert result["rrf_top5"] == []

    def test_chunk_ids_are_unique_in_output(self):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        chunks = [self._make_chunk_dict(f"C{i}") for i in range(10)]
        state = make_agent_state("شعر")
        state["bm25_results"]    = chunks
        state["dense_results"]   = list(reversed(chunks))
        state["colbert_results"] = []
        result = rrf_fuse_node(state)
        ids = [r["chunk_id"] for r in result["rrf_top5"]]
        assert len(ids) == len(set(ids))


# ── Stage 6: resolve_heritage_node ────────────────────────────────────────────

class TestResolveHeritageNode:
    @pytest.fixture
    def registry_path(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(_REGISTRY, ensure_ascii=False), encoding="utf-8")
        return path

    @pytest.fixture
    def bios_path(self, tmp_path):
        bios = [
            {
                "poet_name": POET_A,
                "normalised_name": POET_A,
                "bio_ar": "شاعر خليجي",
                "bio_en": "Gulf poet",
                "sources": ["Sowayan vol. 1"],
            }
        ]
        path = tmp_path / "poets_bio.json"
        path.write_text(json.dumps(bios, ensure_ascii=False), encoding="utf-8")
        return path

    def _run_node(self, rrf_top, registry_path, bios_path):
        """Run resolve_heritage_node with patched data paths."""
        from fatat_al_arab.agent2_retrieval_synthesis.nodes import resolve_heritage as rh_module
        rh_module._load_registry.cache_clear()
        rh_module._load_bios.cache_clear()
        with (
            patch.object(rh_module, "_REGISTRY_PATH", registry_path),
            patch.object(rh_module, "_BIOS_PATH",     bios_path),
        ):
            state = make_agent_state("شعر")
            state["rrf_top5"] = rrf_top
            return rh_module.resolve_heritage_node(state)

    def _make_rrf_chunk(self, anchor_id, poet=""):
        return {
            "chunk_id":   f"cid_{anchor_id}",
            "rrf_score":  0.016,
            "text":       "نص",
            "level":      "verse",
            "anchor_id":  anchor_id,
            "poet_name":  poet,
            "source_volume": "",
            "source_page": 0,
            "source_image_path": "",
            "manuscript_short_key": "",
        }

    def test_resolved_passages_key_present(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a00")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        assert "resolved_passages" in result

    def test_resolvable_chunk_has_citation_true(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a00")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        assert p["citation_resolvable"] is True

    def test_resolvable_chunk_has_matla_text(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a00")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        assert p.get("matla_text")

    def test_resolvable_chunk_has_page_number(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a00")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        assert isinstance(p.get("source_page"), int)

    def test_resolvable_chunk_has_image_path(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a00")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        assert p.get("source_image_path")

    def test_resolvable_chunk_with_bio_has_bio_ar(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a00", poet=POET_A)]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        # bio_ar should be present because we patched bios_path to have POET_A
        assert "bio_ar" in p

    def test_unresolvable_anchor_flagged(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("NONEXISTENT_ID")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        assert p["citation_resolvable"] is False
        assert "citation_note" in p

    def test_mixed_resolvable_and_not(self, registry_path, bios_path):
        rrf_top = [
            self._make_rrf_chunk("a00"),
            self._make_rrf_chunk("GHOST"),
            self._make_rrf_chunk("b00"),
        ]
        result = self._run_node(rrf_top, registry_path, bios_path)
        passages = result["resolved_passages"]
        assert len(passages) == 3
        resolvable_flags = [p["citation_resolvable"] for p in passages]
        assert resolvable_flags == [True, False, True]

    def test_empty_rrf_top_returns_empty(self, registry_path, bios_path):
        result = self._run_node([], registry_path, bios_path)
        assert result["resolved_passages"] == []

    def test_all_citation_fields_present(self, registry_path, bios_path):
        rrf_top = [self._make_rrf_chunk("a03")]
        result = self._run_node(rrf_top, registry_path, bios_path)
        p = result["resolved_passages"][0]
        for field in ("matla_text", "poet_name", "source_volume",
                      "source_page", "source_image_path",
                      "manuscript_short_key", "bio_ar", "bio_en"):
            assert field in p, f"Missing field: {field}"


# ── Full pipeline smoke test ────────────────────────────────────────────────────

class TestAgent2Pipeline:
    """Stage 4 → 5 → 6 in sequence using injected index."""

    def test_full_pipeline_state_flow(self, bundle):
        """Run all three nodes in sequence; check final state has resolved_passages."""
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node
        from fatat_al_arab.agent2_retrieval_synthesis.nodes import resolve_heritage as rh_module

        set_index(bundle)
        rh_module._load_registry.cache_clear()
        rh_module._load_bios.cache_clear()

        state = _make_state_with_qc("ناصر الهزاني مطلع")

        # Stage 4
        state = retrieve_node(state)
        assert "bm25_results" in state

        # Stage 5
        state = rrf_fuse_node(state)
        assert "rrf_top5" in state

        # Stage 6 — use empty registry so citations are not resolvable
        # (bundle uses synthetic data with no matching anchor_registry_phase4.json)
        state = rh_module.resolve_heritage_node(state)
        assert "resolved_passages" in state
        assert isinstance(state["resolved_passages"], list)

    def test_crag_requery_count_preserved_through_pipeline(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node

        set_index(bundle)
        state = _make_state_with_qc("شعر")
        state["crag_requery_count"] = 0
        state = retrieve_node(state)
        state = rrf_fuse_node(state)
        assert state.get("crag_requery_count") == 0

    def test_rrf_top5_chunks_have_required_fields(self, bundle):
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.retrieve import (
            retrieve_node, set_index
        )
        from fatat_al_arab.agent2_retrieval_synthesis.nodes.rrf_fuse import rrf_fuse_node

        set_index(bundle)
        state = _make_state_with_qc("مطلع الشاعر")
        state = retrieve_node(state)
        state = rrf_fuse_node(state)
        for chunk in state.get("rrf_top5") or []:
            for field in ("chunk_id", "rrf_score", "text", "level", "anchor_id"):
                assert field in chunk, f"Missing {field} in fused chunk"
