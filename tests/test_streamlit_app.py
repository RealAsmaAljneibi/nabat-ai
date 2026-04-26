"""
tests/test_streamlit_app.py
============================
Why this file exists: M8 acceptance gate — the three module-level helpers in
app/streamlit_app.py must be unit-testable without a running Streamlit server
or any heavy dependencies (LLM SDK, Qdrant, pytesseract).

Run with:
    PYTHONPATH=src:app python -m pytest tests/test_streamlit_app.py -v

All tests are pure-Python: no network, no filesystem writes (except where a
tmp path is injected), and no Streamlit server required.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# ── Add app/ to path so `import streamlit_app` resolves ───────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR   = str(_REPO_ROOT / "app")
_SRC_DIR   = str(_REPO_ROOT / "src")

for _p in [_APP_DIR, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Module imports without error (even without streamlit installed)
# ══════════════════════════════════════════════════════════════════════════════

class TestModuleImport:
    """Import guard: the app must load cleanly even without heavy dependencies."""

    def test_import_streamlit_app(self):
        """
        Why this test: if the module crashes on import the entire demo fails
        before the doctor even sees a UI. This test proves the lazy-import
        pattern works correctly.
        """
        mod = importlib.import_module("streamlit_app")
        assert mod is not None

    def test_module_exposes_helpers(self):
        """The three testable helpers must be module-level attributes."""
        mod = importlib.import_module("streamlit_app")
        assert callable(getattr(mod, "_index_exists", None)), \
            "_index_exists must be a module-level callable"
        assert callable(getattr(mod, "_format_citation", None)), \
            "_format_citation must be a module-level callable"
        assert callable(getattr(mod, "_trim_history", None)), \
            "_trim_history must be a module-level callable"


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — _index_exists()
# ══════════════════════════════════════════════════════════════════════════════

class TestIndexExists:
    """_index_exists checks for data/qdrant/chunks_meta.json under repo root."""

    def _get(self):
        return importlib.import_module("streamlit_app")._index_exists

    def test_returns_false_when_file_absent(self, tmp_path, monkeypatch):
        """
        Why: the Qdrant index is built separately (fatat_al_arab.embed); it
        won't be present on a fresh clone. The debug sidebar must reflect this
        honestly rather than silently claiming the index is ready.
        """
        mod = importlib.import_module("streamlit_app")
        # Redirect _REPO_ROOT to a fresh tmp directory with no qdrant/ folder
        monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
        assert mod._index_exists() is False

    def test_returns_true_when_file_present(self, tmp_path, monkeypatch):
        """
        Why: after `python -m fatat_al_arab.embed` the sidebar must show green.
        """
        mod = importlib.import_module("streamlit_app")
        # Create the expected path
        index_file = tmp_path / "data" / "qdrant" / "chunks_meta.json"
        index_file.parent.mkdir(parents=True)
        index_file.write_text('{"chunks": []}')

        monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
        assert mod._index_exists() is True


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — _format_citation()
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatCitation:
    """_format_citation must produce consistent strings and handle missing keys."""

    def _fn(self):
        return importlib.import_module("streamlit_app")._format_citation

    def test_full_citation(self):
        """All three fields present → poet — Vol. X — p. Y."""
        fn = self._fn()
        result = fn({
            "poet_name":     "ابن ذريل",
            "source_volume": "ms07",
            "source_page":   "20",
        })
        assert "ابن ذريل" in result
        assert "ms07" in result
        assert "20" in result

    def test_missing_volume(self):
        """Volume absent → still renders poet name without crashing."""
        fn = self._fn()
        result = fn({"poet_name": "Test Poet", "source_page": "5"})
        assert "Test Poet" in result
        assert "5" in result
        # Should not contain a dangling separator for the missing volume
        assert "Vol." not in result

    def test_missing_page(self):
        """Page absent → renders poet and volume, no 'p.' fragment."""
        fn = self._fn()
        result = fn({"poet_name": "Poet", "source_volume": "ms14"})
        assert "Poet" in result
        assert "ms14" in result
        assert "p." not in result

    def test_empty_dict_does_not_crash(self):
        """
        Why: a citation dict may arrive with all fields absent if the anchor
        registry entry is incomplete. The UI must not crash.
        """
        fn = self._fn()
        result = fn({})
        # Returns something (the Unknown poet fallback)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_alternate_field_names(self):
        """
        Why: anchor_registry_phase4 uses 'poet' in some entries, 'poet_name'
        in others. The formatter must handle both.
        """
        fn = self._fn()
        result = fn({"poet": "Ibn Rashid", "volume": "ms22", "page": "7"})
        assert "Ibn Rashid" in result


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — _trim_history()
# ══════════════════════════════════════════════════════════════════════════════

class TestTrimHistory:
    """_trim_history must keep the last max_turns items and never crash."""

    def _fn(self):
        return importlib.import_module("streamlit_app")._trim_history

    def _make_turn(self, n: int) -> dict:
        return {"query": f"q{n}", "response": f"r{n}", "is_refusal": False}

    def test_keeps_last_5_of_7(self):
        """
        Why: the M8 spec says keep last 5 turns. A demo that runs 7 queries
        must drop the earliest two, not the latest.
        """
        fn = self._fn()
        history = [self._make_turn(i) for i in range(7)]
        trimmed = fn(history, max_turns=5)
        assert len(trimmed) == 5
        # Last item should be turn 6 (index 6), not turn 4
        assert trimmed[-1]["query"] == "q6"
        assert trimmed[0]["query"] == "q2"

    def test_does_not_trim_when_within_limit(self):
        """Fewer than max_turns items → returned unchanged."""
        fn = self._fn()
        history = [self._make_turn(i) for i in range(3)]
        trimmed = fn(history, max_turns=5)
        assert len(trimmed) == 3

    def test_empty_history_returns_empty(self):
        """Edge case: no history yet (start of session)."""
        fn = self._fn()
        assert fn([], max_turns=5) == []

    def test_exactly_max_turns_unchanged(self):
        """Exactly max_turns items → no items dropped."""
        fn = self._fn()
        history = [self._make_turn(i) for i in range(5)]
        trimmed = fn(history, max_turns=5)
        assert len(trimmed) == 5

    def test_custom_max_turns(self):
        """max_turns parameter is respected (not hard-coded to 5)."""
        fn = self._fn()
        history = [self._make_turn(i) for i in range(10)]
        trimmed = fn(history, max_turns=3)
        assert len(trimmed) == 3
        assert trimmed[-1]["query"] == "q9"
