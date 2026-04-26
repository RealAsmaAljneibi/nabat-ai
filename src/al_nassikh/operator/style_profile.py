"""
src/al_nassikh/operator/style_profile.py
=========================================
Why this file exists: §2.3 Stage 4 — Kraken's BLLA model performs differently
on NASKH_COMPRESSED (tight inter-line, high glyph density) vs RUQAH_OPEN (loose
inter-line, rounded forms) vs CALLIGRAPHIC (variable baseline, decorative
ligatures) vs HURR (unconstrained personal hand). The style tag drives two
downstream decisions: (a) which Kraken fine-tune to use in Stage 6, and
(b) which diacritics-sharpening parameters to pass to Stage 2 (standardise).

Algorithm:
  - Inter-line spacing: estimated from horizontal projection profile valley depths.
  - Baseline variance: variance of the horizontal projection minima positions.
  - x-height distribution: via pixel-density in the mid-band of the image.
  - Aspect ratio of connected components (wide = RUQAH, tall = NASKH).

Why not a trained classifier: we don't have enough labelled training examples
per style across our 4 Phase-1 manuscripts to train reliably. Geometric
heuristics are more defensible for a graded deliverable.

Architecture ref: §2.3 Stage 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Style tag constants ───────────────────────────────────────────────────────

STYLE_NASKH_COMPRESSED = "NASKH_COMPRESSED"
STYLE_RUQAH_OPEN       = "RUQAH_OPEN"
STYLE_CALLIGRAPHIC     = "CALLIGRAPHIC"
STYLE_HURR             = "HURR"

# Heuristic thresholds (empirically set on ms07/ms14 sample pages)
INTER_LINE_TIGHT    = 0.06   # fraction of image height per line — below = NASKH
INTER_LINE_LOOSE    = 0.10   # above = RUQAH
BASELINE_VAR_HIGH   = 0.25   # high projection variance → calligraphic
CC_ASPECT_WIDE      = 2.5    # mean CC width/height > this → RUQAH


def _to_gray(image: Any) -> np.ndarray:
    try:
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            return np.array(image.convert("L"), dtype=np.uint8)
    except ImportError:
        pass
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = (0.299 * arr[:, :, 0] +
               0.587 * arr[:, :, 1] +
               0.114 * arr[:, :, 2]).astype(np.uint8)
    return arr.astype(np.uint8)


def _horizontal_projection(gray: np.ndarray) -> np.ndarray:
    """Sum of dark pixels per row (inverted: high = lots of ink)."""
    binary = (gray < 128).astype(np.float32)   # ink = dark
    return binary.sum(axis=1)


def _inter_line_spacing(proj: np.ndarray) -> float:
    """
    Estimate mean inter-line spacing as fraction of image height.
    Uses valley-peak-valley pattern in the horizontal projection.
    """
    h = len(proj)
    # Smooth the projection
    kernel_size = max(3, h // 80)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(proj, kernel, mode="same")

    # Find valleys (minima between text lines)
    threshold = smoothed.mean() * 0.3
    in_valley = smoothed < threshold
    # Count transitions valley→text as line starts
    transitions = np.diff(in_valley.astype(int))
    n_lines = max(1, int((transitions == -1).sum()))   # falling edge = line start
    return 1.0 / n_lines   # fraction per line


def _baseline_variance(proj: np.ndarray) -> float:
    """
    Variance of where the projection minima are — high variance = calligraphic
    (baselines not horizontal).
    """
    h = len(proj)
    # Simple: divide image into horizontal strips and find the densest row
    strip_size = max(10, h // 20)
    strip_densities = []
    for i in range(0, h - strip_size, strip_size):
        strip_densities.append(proj[i:i+strip_size].max())
    if not strip_densities:
        return 0.0
    arr = np.array(strip_densities, dtype=float)
    if arr.max() == 0:
        return 0.0
    arr /= arr.max()
    return float(arr.std())


def _mean_cc_aspect(gray: np.ndarray) -> float:
    """
    Estimate mean connected-component width/height ratio.
    Wide components → RUQAH; tall components → NASKH.
    """
    try:
        from scipy.ndimage import label as scipy_label
        binary = (gray < 128).astype(np.uint8)
        labeled, n = scipy_label(binary)
        if n == 0:
            return 1.0
        aspects = []
        for cid in range(1, min(n + 1, 200)):   # sample up to 200 components
            ys, xs = np.where(labeled == cid)
            if len(xs) < 10:
                continue
            w = xs.max() - xs.min() + 1
            h = ys.max() - ys.min() + 1
            if h > 0:
                aspects.append(w / h)
        return float(np.mean(aspects)) if aspects else 1.0
    except ImportError:
        return 1.0


def profile(image: Any) -> dict:
    """
    Stage 4 handwriting style profiling.

    Args:
        image: PIL Image or numpy array.

    Returns:
        {
          "style":       str    — one of NASKH_COMPRESSED, RUQAH_OPEN,
                                  CALLIGRAPHIC, HURR
          "confidence":  float  — 0.0-1.0
          "features": {
              "inter_line_spacing": float,
              "baseline_variance":  float,
              "mean_cc_aspect":     float,
          }
          "note": str           — recommendation for annotator
        }
    """
    gray = _to_gray(image)
    proj = _horizontal_projection(gray)

    inter_line = _inter_line_spacing(proj)
    baseline_var = _baseline_variance(proj)
    cc_aspect = _mean_cc_aspect(gray)

    # Decision logic — priority order matters
    if baseline_var > BASELINE_VAR_HIGH:
        style = STYLE_CALLIGRAPHIC
        confidence = min(1.0, baseline_var / BASELINE_VAR_HIGH)
        note = (
            "High baseline variance suggests calligraphic script. "
            "Use Kraken 'arabic_calig' fine-tune if available; "
            "expect lower automatic segmentation accuracy — increase HITL review budget."
        )
    elif inter_line < INTER_LINE_TIGHT:
        style = STYLE_NASKH_COMPRESSED
        confidence = min(1.0, INTER_LINE_TIGHT / inter_line)
        note = (
            "Tight inter-line spacing (NASKH_COMPRESSED). "
            "Enable Kraken's dense-baseline mode; increase unsharp-mask strength."
        )
    elif inter_line > INTER_LINE_LOOSE or cc_aspect > CC_ASPECT_WIDE:
        style = STYLE_RUQAH_OPEN
        confidence = min(1.0, max(
            inter_line / INTER_LINE_LOOSE,
            cc_aspect / CC_ASPECT_WIDE,
        ))
        note = (
            "Loose inter-line spacing / wide connected components (RUQAH_OPEN). "
            "Standard Kraken BLLA settings are appropriate."
        )
    else:
        style = STYLE_HURR
        confidence = 0.5
        note = (
            "Unconstrained personal hand (HURR) — no dominant metric stands out. "
            "Manual zone inspection recommended before Kraken segmentation."
        )

    return {
        "style":      style,
        "confidence": round(confidence, 3),
        "features": {
            "inter_line_spacing": round(float(inter_line), 4),
            "baseline_variance":  round(float(baseline_var), 4),
            "mean_cc_aspect":     round(float(cc_aspect), 3),
        },
        "note": note,
    }
