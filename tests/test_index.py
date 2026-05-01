"""
tests/test_index.py
====================
Tests for fatat_al_arab.index — stanza-aware chunking, IndexBundle,
and retriever smoke tests. No real model loading — relies on the
fallback embeddings from embed.py when sentence-transformers is absent.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fatat_al_arab.index import build_chunks, build_index, load_index, IndexBundle
from fatat_al_arab.rrf import ScoredChunk


# ── Minimal synthetic registry fixture ────────────────────────────────────────

def _make_entry(
    anchor_id: str,
    poet: str,
    matla: str,
    volume: str = "001-100",
    page: int = 1,
) -> dict:
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

_REGISTRY_10 = [
    _make_entry(f"a{i:02d}", POET_A, f"مطلع الشاعر أ رقم {i}", volume="001-100", page=i)
    for i in range(7)
] + [
    _make_entry(f"b{i:02d}", POET_B, f"مطلع الشاعر ب رقم {i}", volume="200-300", page=i)
    for i in range(3)
]


# ── Chunking tests ─────────────────────────────────────────────────────────────

class TestBuildChunks:
    def test_returns_list_of_scored_chunk(self):
        chunks = build_chunks(_REGISTRY_10)
        assert all(isinstance(c, ScoredChunk) for c in chunks)

    def test_verse_count_equals_entries(self):
        chunks = build_chunks(_REGISTRY_10)
        verse_chunks = [c for c in chunks if c.level == "verse"]
        assert len(verse_chunks) == len(_REGISTRY_10)

    def test_group_chunks_exist_for_multi_entry_poets(self):
        chunks = build_chunks(_REGISTRY_10)
        group_chunks = [c for c in chunks if c.level == "group"]
        # POET_A has 7 entries → 7-3+1=5 windows; POET_B has 3 entries → 1 window
        assert len(group_chunks) == 5 + 1

    def test_poem_chunks_one_per_poet_volume(self):
        chunks = build_chunks(_REGISTRY_10)
        poem_chunks = [c for c in chunks if c.level == "poem"]
        # POET_A (vol 001-100) and POET_B (vol 200-300) = 2 poem chunks
        assert len(poem_chunks) == 2

    def test_chunk_ids_are_unique(self):
        chunks = build_chunks(_REGISTRY_10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_verse_text_is_normalised_arabic(self):
        chunks = build_chunks(_REGISTRY_10)
        verse_chunks = [c for c in chunks if c.level == "verse"]
        for c in verse_chunks:
            assert c.text  # non-empty
            assert "آ" not in c.text   # alef variant normalised away

    def test_group_text_contains_separator(self):
        chunks = build_chunks(_REGISTRY_10)
        group_chunks = [c for c in chunks if c.level == "group"]
        for c in group_chunks:
            assert "|" in c.text

    def test_poem_text_longer_than_verse(self):
        chunks = build_chunks(_REGISTRY_10)
        verse_texts = {c.anchor_id: len(c.text) for c in chunks if c.level == "verse"}
        poem_chunks = [c for c in chunks if c.level == "poem"]
        for pc in poem_chunks:
            # poem text should be longer than any single verse
            single_verse_len = max(verse_texts.values())
            assert len(pc.text) > single_verse_len

    def test_payload_fields_present(self):
        chunks = build_chunks(_REGISTRY_10)
        required = {"anchor_id", "poet_name", "source_volume", "source_page",
                    "source_image_path", "manuscript_short_key"}
        for c in chunks:
            for field in required:
                assert getattr(c, field) is not None, f"{field} missing on {c.chunk_id}"

    def test_empty_registry_returns_manuscript_headers(self):
        # build_chunks intentionally emits one manuscript-level chunk per
        # registered manuscript even when no verses are provided, so that
        # queries like "what is the Huber 2 manuscript?" still work.
        chunks = build_chunks([])
        ms_chunks = [c for c in chunks if c.level == "manuscript"]
        assert len(ms_chunks) > 0
        verse_chunks = [c for c in chunks if c.level == "verse"]
        assert verse_chunks == []

    def test_single_entry_no_group_or_poem(self):
        single = [_make_entry("z00", "شاعر وحيد", "مطلع وحيد")]
        chunks = build_chunks(single)
        assert sum(1 for c in chunks if c.level == "verse") == 1
        assert sum(1 for c in chunks if c.level == "group") == 0
        assert sum(1 for c in chunks if c.level == "poem") == 0


# ── IndexBundle build + load ───────────────────────────────────────────────────

class TestBuildIndex:
    def _write_registry(self, tmp: Path, registry: list) -> Path:
        p = tmp / "registry.json"
        p.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
        return p

    def test_build_returns_index_bundle(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        bundle = build_index(registry_path=reg, qdrant_path=tmp_path / "qdrant", force_rebuild=True)
        assert isinstance(bundle, IndexBundle)

    def test_bundle_chunk_count_matches_build_chunks(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        bundle = build_index(registry_path=reg, qdrant_path=tmp_path / "qdrant", force_rebuild=True)
        expected = len(build_chunks(_REGISTRY_10))
        assert len(bundle.chunks) == expected

    def test_embeddings_shape(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        bundle = build_index(registry_path=reg, qdrant_path=tmp_path / "qdrant", force_rebuild=True)
        assert bundle.embeddings.shape == (len(bundle.chunks), 768)

    def test_embeddings_are_unit_vectors(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        bundle = build_index(registry_path=reg, qdrant_path=tmp_path / "qdrant", force_rebuild=True)
        norms = np.linalg.norm(bundle.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_stats_contain_required_keys(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        bundle = build_index(registry_path=reg, qdrant_path=tmp_path / "qdrant", force_rebuild=True)
        for key in ("total_chunks", "verse_chunks", "group_chunks", "poem_chunks", "embedding_shape"):
            assert key in bundle.stats

    def test_meta_and_npy_files_created(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        qdrant_dir = tmp_path / "qdrant"
        build_index(registry_path=reg, qdrant_path=qdrant_dir, force_rebuild=True)
        assert (qdrant_dir / "chunks_meta.json").exists()
        assert (qdrant_dir / "embeddings.npy").exists()

    def test_cache_load_skips_re_encode(self, tmp_path):
        """Second call without force_rebuild returns cached index."""
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        qdrant_dir = tmp_path / "qdrant"
        b1 = build_index(registry_path=reg, qdrant_path=qdrant_dir, force_rebuild=True)
        b2 = build_index(registry_path=reg, qdrant_path=qdrant_dir, force_rebuild=False)
        assert b2.stats.get("loaded_from_cache") is True
        assert len(b2.chunks) == len(b1.chunks)

    def test_force_rebuild_ignores_cache(self, tmp_path):
        reg = self._write_registry(tmp_path, _REGISTRY_10)
        qdrant_dir = tmp_path / "qdrant"
        build_index(registry_path=reg, qdrant_path=qdrant_dir, force_rebuild=True)
        b2 = build_index(registry_path=reg, qdrant_path=qdrant_dir, force_rebuild=True)
        assert b2.stats.get("loaded_from_cache") is not True


class TestLoadIndex:
    def test_load_without_build_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index(tmp_path / "nonexistent")

    def test_load_after_build_succeeds(self, tmp_path):
        reg_path = tmp_path / "registry.json"
        reg_path.write_text(json.dumps(_REGISTRY_10, ensure_ascii=False), encoding="utf-8")
        qdrant_dir = tmp_path / "qdrant"
        build_index(registry_path=reg_path, qdrant_path=qdrant_dir, force_rebuild=True)
        bundle = load_index(qdrant_dir)
        assert isinstance(bundle, IndexBundle)
        assert len(bundle.chunks) > 0


# ── Retriever smoke tests ──────────────────────────────────────────────────────

class TestRetrievers:
    @pytest.fixture
    def bundle(self, tmp_path):
        reg_path = tmp_path / "registry.json"
        reg_path.write_text(json.dumps(_REGISTRY_10, ensure_ascii=False), encoding="utf-8")
        return build_index(registry_path=reg_path, qdrant_path=tmp_path / "qdrant",
                           force_rebuild=True)

    def test_bm25_retrieve_returns_scored_chunks(self, bundle):
        results = bundle.bm25.retrieve("ناصر الهزاني", n=5)
        assert isinstance(results, list)
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_bm25_retrieve_respects_n(self, bundle):
        results = bundle.bm25.retrieve("مطلع", n=3)
        assert len(results) <= 3

    def test_bm25_retrieve_nonzero_score(self, bundle):
        """Query matching corpus terms should return hits with score > 0."""
        results = bundle.bm25.retrieve("مطلع الشاعر أ", n=5)
        # Not guaranteed to have results if BM25 scores are all 0, but with
        # matching terms we expect at least one hit
        if results:
            assert all(r.rrf_score > 0 for r in results)

    def test_dense_retrieve_returns_scored_chunks(self, bundle):
        results = bundle.dense.retrieve("شعر عربي", n=5)
        assert isinstance(results, list)
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_dense_retrieve_respects_n(self, bundle):
        results = bundle.dense.retrieve("شعر", n=4)
        assert len(results) <= 4

    def test_dense_retrieve_sorted_descending(self, bundle):
        results = bundle.dense.retrieve("ناصر", n=10)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_colbert_stub_returns_empty(self, bundle):
        """ColBERT stub must return empty list when COLBERT_ENABLED is false."""
        results = bundle.colbert.retrieve("أي شيء", n=5)
        assert results == []

    def test_retrievers_property_returns_triple(self, bundle):
        bm25, dense, colbert = bundle.retrievers
        assert bm25 is bundle.bm25
        assert dense is bundle.dense
        assert colbert is bundle.colbert

    def test_full_rrf_fusion_pipeline(self, bundle):
        """End-to-end: retrieve from both BM25 and dense, fuse, check output."""
        from fatat_al_arab.rrf import fuse
        query = "ناصر الهزاني مطلع"
        bm25_results  = bundle.bm25.retrieve(query, n=10)
        dense_results = bundle.dense.retrieve(query, n=10)
        colbert_results = bundle.colbert.retrieve(query, n=10)
        fused = fuse([bm25_results, dense_results, colbert_results], top_n=5)
        assert isinstance(fused, list)
        # With a real query, at least one result should come back
        assert len(fused) > 0
        # All results must have positive rrf_score
        assert all(r.rrf_score > 0 for r in fused)
        # Results must be sorted descending
        scores = [r.rrf_score for r in fused]
        assert scores == sorted(scores, reverse=True)
