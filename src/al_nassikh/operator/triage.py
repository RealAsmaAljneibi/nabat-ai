"""
src/al_nassikh/operator/triage.py
==================================
Why this file exists: §2.3 Stage 0 of the architecture — every new manuscript
page enters the pipeline through a quality gate before any transcription work
begins. A DEGRADED page (extreme bleed-through, physical damage, heavy noise)
sent directly to Kraken will produce garbage baselines that mislead the HITL
annotator. The triage step surfaces these early so they can go to a specialist
track instead.

The function is intentionally a pure predicate — it takes an image and returns
a structured decision dict. The Streamlit tab renders this; the annotator
confirms or overrides. Nothing is written to disk here.

Architecture ref: §2.3 Stage 0 (ms09 degradation pattern is the primary motivator).
Challenge mapping: Challenge 2 (physical damage), Challenge 3 (bleed-through),
Challenge 4 (paper yellowing / extreme contrast loss).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
# Why named constants: §5 says every threshold is a named constant, not a magic
# literal. The values below were chosen empirically on the ms07/ms09 test set.

DEGRADED_DARK_FRACTION  = 0.60   # > 60% of pixels below INTENSITY_DARK → DEGRADED
DEGRADED_LIGHT_FRACTION = 0.85   # > 85% of pixels above INTENSITY_LIGHT → DEGRADED (washed out)
INTENSITY_DARK          = 50     # pixel intensity ≤ this is "dark" (0-255 scale)
INTENSITY_LIGHT         = 220    # pixel intensity ≥ this is "light" (washed out)
EDGE_COHERENCE_THRESHOLD = 0.03  # Sobel edge fraction below this → blurry/featureless → DEGRADED
BLEED_STD_THRESHOLD     = 45.0   # grayscale std-dev above this suggests bleed-through noise


def _to_gray(image: Any) -> np.ndarray:
    """
    Convert an image to a 2D uint8 grayscale numpy array.
    Accepts: PIL Image, numpy array (RGB/RGBA/gray).
    Why here: callers shouldn't need to know whether we got a PIL or numpy input.
    """
    # Lazy import — PIL is not always installed in stub/test environments
    try:
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            if image.mode != "L":
                image = image.convert("L")
            return np.array(image, dtype=np.uint8)
    except ImportError:
        pass

    arr = np.asarray(image)
    if arr.ndim == 3:
        # RGB → weighted luminance
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]   # drop alpha
        gray = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2]).astype(np.uint8)
        return gray
    return arr.astype(np.uint8)


def _edge_fraction(gray: np.ndarray) -> float:
    """
    Simple edge-energy proxy: mean absolute difference between adjacent pixels
    normalised by pixel range. Higher = more edges = more coherent text.
    Why not Canny: avoids OpenCV dependency; this proxy is sufficient for a
    binary NORMAL/DEGRADED gate.
    """
    dy = np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float))
    dx = np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float))
    edge_energy = (dy.mean() + dx.mean()) / 2
    return edge_energy / 255.0   # normalise to [0, 1]


def is_degraded(image: Any) -> dict:
    """
    Stage 0 quality gate — assess whether a manuscript page is too degraded
    for standard Kraken segmentation.

    Args:
        image: PIL Image or numpy array (RGB, RGBA, or grayscale).

    Returns:
        {
          "status":      "NORMAL" | "DEGRADED",
          "score":       float 0.0-1.0  (higher = more degraded),
          "reason":      str            (human-readable flag, empty if NORMAL),
          "checks": {
              "dark_fraction":   float,
              "light_fraction":  float,
              "edge_fraction":   float,
              "bleed_std":       float,
          }
        }

    Why the 'checks' sub-dict: the Streamlit tab surfaces per-check readings
    in the debug panel so the archivist understands *why* a page was flagged.
    """
    gray = _to_gray(image)
    total_pixels = gray.size

    dark_fraction  = float(np.sum(gray <= INTENSITY_DARK)  / total_pixels)
    light_fraction = float(np.sum(gray >= INTENSITY_LIGHT) / total_pixels)
    edge_frac      = _edge_fraction(gray)
    bleed_std      = float(gray.std())

    reasons: list[str] = []

    if dark_fraction > DEGRADED_DARK_FRACTION:
        reasons.append(
            f"extreme darkness ({dark_fraction:.0%} pixels ≤ {INTENSITY_DARK}) — "
            "possible heavy water damage or charring"
        )
    if light_fraction > DEGRADED_LIGHT_FRACTION:
        reasons.append(
            f"washed-out ({light_fraction:.0%} pixels ≥ {INTENSITY_LIGHT}) — "
            "possible severe foxing or bleaching"
        )
    if edge_frac < EDGE_COHERENCE_THRESHOLD:
        reasons.append(
            f"low edge coherence ({edge_frac:.3f}) — image is blurry or featureless; "
            "Kraken baselines will be unreliable"
        )

    # Bleed-through: high std-dev on a mostly-light page signals ink from reverse side
    if light_fraction > 0.4 and bleed_std > BLEED_STD_THRESHOLD:
        reasons.append(
            f"probable bleed-through (std={bleed_std:.1f} on {light_fraction:.0%} light page) — "
            "run bleed_suppress before segmentation"
        )

    # Degradation score: weighted sum of sub-scores, normalised to [0, 1]
    score = min(1.0, (
        0.35 * min(dark_fraction / DEGRADED_DARK_FRACTION, 1.0) +
        0.25 * min(light_fraction / DEGRADED_LIGHT_FRACTION, 1.0) +
        0.25 * max(0.0, 1.0 - edge_frac / EDGE_COHERENCE_THRESHOLD) +
        0.15 * min(bleed_std / (BLEED_STD_THRESHOLD * 2), 1.0)
    ))

    status = "DEGRADED" if reasons else "NORMAL"

    return {
        "status":  status,
        "score":   round(score, 4),
        "reason":  "; ".join(reasons),
        "checks": {
            "dark_fraction":  round(dark_fraction, 4),
            "light_fraction": round(light_fraction, 4),
            "edge_fraction":  round(edge_frac, 4),
            "bleed_std":      round(bleed_std, 2),
        },
    }
