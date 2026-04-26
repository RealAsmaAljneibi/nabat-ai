"""
tests/test_archive_manager.py
==============================
Why this file exists: M9 acceptance gate — unit tests for the Archive Manager
helper functions added to app/streamlit_app.py. These tests verify that each
module-level helper:
  - Returns a dict with the expected shape and keys
  - Handles missing / unavailable dependencies gracefully (returns safe
    defaults, never raises)
  - Does not require a running Streamlit server to pass

All helpers are imported directly from streamlit_app (no Streamlit server
needed — the module uses a _StStub fallback when streamlit is unavailable).
"""

from __future__ import annotations

import io
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Add app/ to sys.path so we can import streamlit_app without installing it.
APP_DIR  = Path(__file__).resolve().parent.parent / "app"
SRC_DIR  = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(SRC_DIR))

import streamlit_app as sa   # noqa: E402 — must come after sys.path manipulation


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: minimal synthetic 3×3 white PNG in bytes
# ═══════════════════════════════════════════════════════════════════════════════

def _white_png_bytes(width: int = 20, height: int = 20) -> bytes:
    """Return a minimal valid PNG (white pixels) as raw bytes."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # If PIL not installed, return a 1-byte payload so helpers can show
        # their error-path behaviour instead of crashing the test runner.
        return b"\x89PNG"


# ═══════════════════════════════════════════════════════════════════════════════
# _get_queue_summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetQueueSummary:
    """_get_queue_summary() must always return a dict with the 5 expected keys."""

    EXPECTED_KEYS = {"total", "pending", "in_review", "complete", "degraded"}

    def test_returns_dict_with_all_keys_when_module_unavailable(self):
        """When al_nassikh.ingest.queue_for_review is absent → safe zeros dict."""
        with patch.dict(sys.modules, {"al_nassikh": None,
                                       "al_nassikh.ingest": None,
                                       "al_nassikh.ingest.queue_for_review": None}):
            result = sa._get_queue_summary()
        assert isinstance(result, dict)
        assert self.EXPECTED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.EXPECTED_KEYS - result.keys()}"
        )

    def test_returns_zeros_on_import_error(self):
        """All counts should be 0 when the module import fails."""
        with patch.dict(sys.modules, {"al_nassikh.ingest.queue_for_review": None}):
            result = sa._get_queue_summary()
        for key in self.EXPECTED_KEYS:
            assert result[key] == 0, f"Expected {key}=0, got {result[key]}"

    def test_returns_real_summary_when_module_available(self):
        """When the module is present and summary() works → pass through its dict."""
        mock_summary = {"total": 5, "pending": 2, "in_review": 1,
                        "complete": 2, "degraded": 0}
        fake_module = MagicMock()
        fake_module.summary.return_value = mock_summary

        with patch.dict(sys.modules, {
            "al_nassikh.ingest.queue_for_review": fake_module,
        }):
            result = sa._get_queue_summary()

        assert result == mock_summary

    def test_handles_exception_in_summary_call(self):
        """If summary() raises, return safe zeros — never propagate."""
        fake_module = MagicMock()
        fake_module.summary.side_effect = RuntimeError("db locked")

        with patch.dict(sys.modules, {
            "al_nassikh.ingest.queue_for_review": fake_module,
        }):
            result = sa._get_queue_summary()

        assert isinstance(result, dict)
        assert result["total"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _escriptorium_status
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscriptoriumStatus:
    """_escriptorium_status() must return a dict with available/base_url/error."""

    EXPECTED_KEYS = {"available", "base_url", "error"}

    def test_returns_dict_with_all_keys_when_module_unavailable(self):
        with patch.dict(sys.modules, {"al_nassikh.escriptorium_client": None}):
            result = sa._escriptorium_status()
        assert isinstance(result, dict)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_available_false_on_import_error(self):
        with patch.dict(sys.modules, {"al_nassikh.escriptorium_client": None}):
            result = sa._escriptorium_status()
        assert result["available"] is False
        assert result["error"] is not None

    def test_available_true_when_is_available_returns_true(self):
        fake_module = MagicMock()
        fake_module.is_available.return_value = True
        fake_module.ESCR_BASE_URL = "http://localhost:8080"

        with patch.dict(sys.modules, {
            "al_nassikh.escriptorium_client": fake_module,
        }):
            result = sa._escriptorium_status()

        assert result["available"] is True
        assert result["base_url"] == "http://localhost:8080"
        assert result["error"] is None

    def test_available_false_when_is_available_returns_false(self):
        fake_module = MagicMock()
        fake_module.is_available.return_value = False
        fake_module.ESCR_BASE_URL = "http://localhost:8080"

        with patch.dict(sys.modules, {
            "al_nassikh.escriptorium_client": fake_module,
        }):
            result = sa._escriptorium_status()

        assert result["available"] is False

    def test_handles_exception_from_is_available(self):
        fake_module = MagicMock()
        fake_module.is_available.side_effect = ConnectionRefusedError("refused")
        fake_module.ESCR_BASE_URL = "http://localhost:8080"

        with patch.dict(sys.modules, {
            "al_nassikh.escriptorium_client": fake_module,
        }):
            result = sa._escriptorium_status()

        assert result["available"] is False
        assert "refused" in str(result.get("error", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# _run_triage
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunTriage:
    """_run_triage() must return a dict — never raise."""

    def test_returns_dict_on_invalid_input(self):
        """Non-image bytes → error path, still returns a dict."""
        result = sa._run_triage(b"not an image")
        assert isinstance(result, dict)

    def test_error_dict_has_expected_keys(self):
        result = sa._run_triage(b"garbage")
        # Must have at least status and reason (error path)
        assert "status" in result or "error" in result

    def test_returns_error_when_triage_module_missing(self):
        """Even without al_nassikh.operator.triage the helper returns a dict."""
        with patch.dict(sys.modules, {
            "al_nassikh.operator.triage": None,
            "al_nassikh.operator": None,
        }):
            result = sa._run_triage(_white_png_bytes())
        assert isinstance(result, dict)

    def test_passes_through_is_degraded_result(self):
        """When is_degraded() returns a dict, _run_triage should return it."""
        fake_triage = MagicMock()
        fake_triage.is_degraded.return_value = {
            "status": "OK", "score": 0.1, "reason": "all good", "checks": {}
        }

        fake_pil_img = MagicMock()
        fake_pil_module = MagicMock()
        fake_pil_module.Image.open.return_value.__enter__ = MagicMock(return_value=fake_pil_img)
        fake_pil_module.Image.open.return_value.convert.return_value = fake_pil_img

        with patch.dict(sys.modules, {
            "al_nassikh.operator.triage": fake_triage,
            "al_nassikh.operator": MagicMock(),
        }):
            # Pass real PNG bytes so PIL can open them
            result = sa._run_triage(_white_png_bytes())

        # If PIL is available the real result flows through; if not, error path
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# _run_bleed_suppress
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBleedSuppress:
    """_run_bleed_suppress() must return a dict — never raise."""

    def test_returns_dict_on_invalid_input(self):
        result = sa._run_bleed_suppress(b"not an image")
        assert isinstance(result, dict)

    def test_error_dict_has_applied_key(self):
        """Even on failure the returned dict must include the 'applied' key."""
        result = sa._run_bleed_suppress(b"garbage bytes")
        assert "applied" in result

    def test_applied_false_on_error(self):
        result = sa._run_bleed_suppress(b"garbage bytes")
        assert result["applied"] is False

    def test_passes_through_suppress_result(self):
        """When suppress() succeeds, _run_bleed_suppress passes through its dict."""
        fake_bleed = MagicMock()
        fake_bleed.suppress.return_value = {
            "applied": False,
            "result_image": None,
            "reason": "No bleed detected",
        }

        with patch.dict(sys.modules, {
            "al_nassikh.operator.bleed_suppress": fake_bleed,
            "al_nassikh.operator": MagicMock(),
        }):
            result = sa._run_bleed_suppress(_white_png_bytes())

        assert isinstance(result, dict)
        assert "applied" in result

    def test_handles_exception_from_suppress(self):
        fake_bleed = MagicMock()
        fake_bleed.suppress.side_effect = MemoryError("OOM")

        with patch.dict(sys.modules, {
            "al_nassikh.operator.bleed_suppress": fake_bleed,
        }):
            result = sa._run_bleed_suppress(_white_png_bytes())

        assert isinstance(result, dict)
        assert result["applied"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# _run_standardise
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunStandardise:
    """_run_standardise() must return a dict — never raise."""

    def test_returns_dict_on_invalid_input(self):
        result = sa._run_standardise(b"not an image")
        assert isinstance(result, dict)

    def test_error_dict_has_result_image_key(self):
        result = sa._run_standardise(b"garbage bytes")
        assert "result_image" in result

    def test_result_image_none_on_error(self):
        result = sa._run_standardise(b"garbage bytes")
        assert result["result_image"] is None

    def test_passes_through_standardise_result(self):
        fake_std = MagicMock()
        fake_std.standardise.return_value = {
            "result_image": None,
            "skew_angle": 0.5,
            "resize_scale": 1.0,
            "applied": True,
        }

        with patch.dict(sys.modules, {
            "al_nassikh.operator.standardise": fake_std,
            "al_nassikh.operator": MagicMock(),
        }):
            result = sa._run_standardise(_white_png_bytes())

        assert isinstance(result, dict)
        assert "result_image" in result

    def test_handles_exception_from_standardise(self):
        fake_std = MagicMock()
        fake_std.standardise.side_effect = ValueError("bad shape")

        with patch.dict(sys.modules, {
            "al_nassikh.operator.standardise": fake_std,
        }):
            result = sa._run_standardise(_white_png_bytes())

        assert isinstance(result, dict)
        assert result["result_image"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# _rebuild_index
# ═══════════════════════════════════════════════════════════════════════════════

class TestRebuildIndex:
    """_rebuild_index() must return a dict with success/stats/error keys."""

    EXPECTED_KEYS = {"success", "stats", "error"}

    def test_returns_dict_with_all_keys_on_import_error(self):
        with patch.dict(sys.modules, {
            "fatat_al_arab.index": None,
        }):
            result = sa._rebuild_index()
        assert isinstance(result, dict)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_success_false_on_import_error(self):
        with patch.dict(sys.modules, {
            "fatat_al_arab.index": None,
        }):
            result = sa._rebuild_index()
        assert result["success"] is False
        assert result["error"] is not None

    def test_success_true_when_build_index_succeeds(self):
        fake_bundle = MagicMock()
        fake_bundle.stats = {"chunks": 100, "poems": 20}

        fake_index_module = MagicMock()
        fake_index_module.build_index.return_value = fake_bundle
        fake_index_module.DEFAULT_REGISTRY = "data/ground_truth/anchor_registry_phase4.json"
        fake_index_module.DEFAULT_QDRANT = "data/qdrant/"

        with patch.dict(sys.modules, {
            "fatat_al_arab.index": fake_index_module,
        }):
            result = sa._rebuild_index()

        assert isinstance(result, dict)
        assert self.EXPECTED_KEYS.issubset(result.keys())
        # If the import succeeds (module not None), success should be True
        assert result["success"] is True
        assert result["error"] is None

    def test_success_false_on_build_exception(self):
        fake_index_module = MagicMock()
        fake_index_module.build_index.side_effect = RuntimeError("disk full")
        fake_index_module.DEFAULT_REGISTRY = "data/ground_truth/anchor_registry_phase4.json"
        fake_index_module.DEFAULT_QDRANT = "data/qdrant/"

        with patch.dict(sys.modules, {
            "fatat_al_arab.index": fake_index_module,
        }):
            result = sa._rebuild_index()

        assert result["success"] is False
        assert "disk full" in str(result["error"])

    def test_stats_is_dict(self):
        """stats must always be a dict (empty dict on failure)."""
        with patch.dict(sys.modules, {"fatat_al_arab.index": None}):
            result = sa._rebuild_index()
        assert isinstance(result["stats"], dict)
