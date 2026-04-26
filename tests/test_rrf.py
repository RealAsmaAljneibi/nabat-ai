"""
tests/test_rrf.py
=================
Tests for fatat_al_arab.rrf — Reciprocal Rank Fusion.
All tests use synthetic ScoredChunk fixtures; no model loading required.
"""

from __future__ import annotations

import pytest
from fatat_al_arab.rrf import ScoredChunk, fuse, scores_only


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str = "", score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        rrf_score=score,
        text=text or f"text for {chunk_id}",
        level="verse",
        anchor_id=f"anchor_{chunk_id}",
        poet_name="شاعر",
        source_volume="001",
        source_page=1,
        source_image_path="",
        manuscript_short_key="ibn_yahya_601_842",
    )


# ── RRF formula correctness ───────────────────────────────────────────────────

class TestRRFFormula:
    def test_single_list_rank1_score(self):
        """Rank-1 in a single list should give 1/(60+1)."""
        chunk = _make_chunk("A")
        result = fuse([[chunk]], k=60)
        assert len(result) == 1
        assert abs(result[0].rrf_score - 1 / 61) < 1e-9

    def test_single_list_rank2_score(self):
        a = _make_chunk("A")
        b = _make_chunk("B")
        result = fuse([[a, b]], k=60)
        assert abs(result[0].rrf_score - 1 / 61) < 1e-9
        assert abs(result[1].rrf_score - 1 / 62) < 1e-9

    def test_two_lists_same_doc_adds_scores(self):
        """A document appearing rank-1 in both lists gets 1/61 + 1/61."""
        a = _make_chunk("A")
        b = _make_chunk("B")
        # A is rank-1 in both lists; B only appears in list 2 at rank-2
        result = fuse([[a], [a, b]], k=60)
        a_result = next(r for r in result if r.chunk_id == "A")
        b_result = next(r for r in result if r.chunk_id == "B")
        expected_a = 1 / 61 + 1 / 61
        expected_b = 1 / 62
        assert abs(a_result.rrf_score - expected_a) < 1e-9
        assert abs(b_result.rrf_score - expected_b) < 1e-9

    def test_custom_k(self):
        chunk = _make_chunk("A")
        result = fuse([[chunk]], k=5)
        assert abs(result[0].rrf_score - 1 / 6) < 1e-9

    def test_descending_order(self):
        """Fused result must be sorted descending by rrf_score."""
        chunks_list1 = [_make_chunk(c) for c in ["A", "B", "C", "D"]]
        chunks_list2 = [_make_chunk(c) for c in ["C", "A", "D", "B"]]
        result = fuse([chunks_list1, chunks_list2])
        scores = scores_only(result)
        assert scores == sorted(scores, reverse=True)


class TestRRFEdgeCases:
    def test_empty_input(self):
        assert fuse([]) == []

    def test_single_empty_list(self):
        assert fuse([[]]) == []

    def test_mixed_empty_and_populated(self):
        a = _make_chunk("A")
        result = fuse([[], [a], []])
        assert len(result) == 1

    def test_top_n_truncation(self):
        chunks = [_make_chunk(str(i)) for i in range(20)]
        result = fuse([chunks], top_n=5)
        assert len(result) == 5

    def test_top_n_larger_than_result(self):
        chunks = [_make_chunk(str(i)) for i in range(3)]
        result = fuse([chunks], top_n=10)
        assert len(result) == 3

    def test_deduplication(self):
        """The same chunk appearing in multiple lists is fused, not duplicated."""
        a = _make_chunk("A")
        result = fuse([[a, a, a]])
        # Although 'A' appears 3 times in the list, positions 0,1,2 are distinct
        # The chunk_id deduplication means each unique id appears once.
        ids = [r.chunk_id for r in result]
        assert len(ids) == len(set(ids))

    def test_scores_are_positive(self):
        chunks = [_make_chunk(c) for c in "ABCDE"]
        result = fuse([chunks])
        assert all(r.rrf_score > 0 for r in result)


class TestRRFMetadata:
    def test_metadata_preserved(self):
        """Non-score fields from the input chunk must be preserved in output."""
        chunk = ScoredChunk(
            chunk_id="X",
            rrf_score=0.0,
            text="أمس الضحى انعليت",
            level="poem",
            anchor_id="anchor_X",
            poet_name="ناصر بن حمد الهزاني",
            source_volume="601-782",
            source_page=178,
            source_image_path="manuscripts/p178.png",
            manuscript_short_key="ibn_yahya_601_842",
        )
        result = fuse([[chunk]])
        r = result[0]
        assert r.text == chunk.text
        assert r.level == chunk.level
        assert r.anchor_id == chunk.anchor_id
        assert r.poet_name == chunk.poet_name
        assert r.source_volume == chunk.source_volume
        assert r.source_page == chunk.source_page
        assert r.source_image_path == chunk.source_image_path

    def test_rrf_score_overrides_original_score(self):
        """rrf_score in the output reflects the fused value, not the input score."""
        chunk = _make_chunk("A", score=999.0)
        result = fuse([[chunk]])
        assert result[0].rrf_score != 999.0
        assert result[0].rrf_score == pytest.approx(1 / 61)

    def test_tie_breaking_is_deterministic(self):
        """Ties are broken by chunk_id — repeated calls must produce same order."""
        a = _make_chunk("A")
        b = _make_chunk("B")
        # Both appear at the same rank in a single list → deterministic tie-break
        result1 = fuse([[a, b]])
        result2 = fuse([[a, b]])
        assert [r.chunk_id for r in result1] == [r.chunk_id for r in result2]

    def test_to_dict_roundtrip(self):
        chunk = _make_chunk("D", text="بيت شعري")
        fused = fuse([[chunk]])[0]
        d = fused.to_dict()
        assert d["chunk_id"] == "D"
        assert d["text"] == "بيت شعري"
        assert "rrf_score" in d

    def test_three_retriever_scenario(self):
        """Simulate BM25 + Dense + ColBERT (empty) fusion."""
        # BM25 list
        bm25_list = [_make_chunk(c) for c in ["A", "B", "C", "D", "E"]]
        # Dense list — different ranking
        dense_list = [_make_chunk(c) for c in ["C", "A", "E", "B", "D"]]
        # ColBERT stub — empty
        colbert_list: list[ScoredChunk] = []

        result = fuse([bm25_list, dense_list, colbert_list])
        # A appears rank-1 in BM25, rank-2 in dense → should rank highly
        # C appears rank-3 in BM25, rank-1 in dense → also competitive
        ids = [r.chunk_id for r in result]
        assert set(ids) == {"A", "B", "C", "D", "E"}
        # Both A and C should be in top-2
        assert set(ids[:2]) == {"A", "C"}
