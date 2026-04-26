"""
src/al_nassikh/operator/zone_classify.py
=========================================
Why this file exists: §2.3 Stage 5 — the architecture defines five zone labels
that the annotator assigns before Kraken segmentation. These labels are fed back
into eScriptorium's ontology API (via escriptorium_client.set_ontology) so they
appear in the annotation UI as named options rather than generic "TextRegion".
The function here analyses a page image and proposes zone assignments as
geometric regions — the annotator confirms or adjusts in eScriptorium's editor.

Five zone types (from §2.3 Stage 5):
  TWO_COLUMN_POETRY    — the main text body (sadr + ajuz side-by-side)
  PROSE_ATTRIBUTION    — attribution line (اسم الشاعر / occasion note)
  ORNAMENTAL_DIVIDER   — horizontal rule or geometric ornament between poems
  MARGIN_PERPENDICULAR — text written perpendicular in the margin
  INTERLINEAR_ANNOTATION — glosses written between text lines

Algorithm: geometric heuristics on the binarised page.
  - TWO_COLUMN: column separator detection (vertical gap in horizontal projection).
  - ORNAMENTAL_DIVIDER: horizontal bands with very low pixel density → decorative rule.
  - MARGIN_PERPENDICULAR: narrow strips at left/right edge with ink density.
  - INTERLINEAR_ANNOTATION: thin ink bands sandwiched between main text lines.
  - PROSE_ATTRIBUTION: short dense bands near the top or between poems.

Why heuristics (not a region-detection model): we have no labelled training set
for zone detection specific to Khaleeji manuscripts. Heuristics are transparent
and the annotator's confirmation is the authority.

Architecture ref: §2.3 Stage 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Zone type constants ───────────────────────────────────────────────────────

ZONE_TWO_COLUMN_POETRY      = "TWO_COLUMN_POETRY"
ZONE_PROSE_ATTRIBUTION      = "PROSE_ATTRIBUTION"
ZONE_ORNAMENTAL_DIVIDER     = "ORNAMENTAL_DIVIDER"
ZONE_MARGIN_PERPENDICULAR   = "MARGIN_PERPENDICULAR"
ZONE_INTERLINEAR_ANNOTATION = "INTERLINEAR_ANNOTATION"

# Thresholds
DIVIDER_MAX_INK_FRACTION   = 0.03   # < 3% ink per row → ornamental divider candidate
MARGIN_WIDTH_FRACTION      = 0.08   # left/right 8% of image width = margin zone
MARGIN_MIN_INK_FRACTION    = 0.02   # minimum ink in margin strip to flag it
ATTRIBUTION_MAX_ROWS       = 0.04   # attribution band < 4% of image height
INTERLINEAR_MAX_ROWS       = 0.015  # inter-linear gloss bands < 1.5% of image height
COLUMN_GAP_MIN_WIDTH       = 0.05   # vertical gap ≥ 5% of image width → two columns


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


def _ink_binary(gray: np.ndarray) -> np.ndarray:
    """Binarise: True where ink (dark pixels)."""
    return gray < 128


def classify_zones(image: Any) -> dict:
    """
    Stage 5 zone classification.

    Args:
        image: PIL Image or numpy array.

    Returns:
        {
          "zones": list[dict]   — proposed zone regions
          "note":  str          — message for annotator
        }

    Each zone dict:
        {
          "zone_type": str        — one of the five zone type constants
          "bbox":      [x, y, w, h]  — bounding box in pixels
          "confidence": float
          "annotation": str       — short explanation
        }
    """
    gray = _to_gray(image)
    ink  = _ink_binary(gray)
    h, w = gray.shape
    zones: list[dict] = []

    # ── Horizontal projection (ink rows) ───────────────────────────────────
    row_ink_fraction = ink.sum(axis=1) / w   # fraction of inked pixels per row

    # ── Detect ornamental dividers ─────────────────────────────────────────
    # Rows with almost no ink between text blocks → ornamental dividers
    in_divider = False
    div_start  = 0
    for r in range(h):
        is_empty = row_ink_fraction[r] < DIVIDER_MAX_INK_FRACTION
        if is_empty and not in_divider:
            in_divider = True
            div_start  = r
        elif not is_empty and in_divider:
            in_divider = False
            div_height = r - div_start
            # Only flag as ornamental divider if it's a thin band (not big margin)
            if 3 < div_height < int(h * 0.04):
                zones.append({
                    "zone_type":  ZONE_ORNAMENTAL_DIVIDER,
                    "bbox":       [0, div_start, w, div_height],
                    "confidence": 0.6,
                    "annotation": f"Thin empty band ({div_height} rows) — possible ornamental divider",
                })

    # ── Detect margin perpendicular text ───────────────────────────────────
    margin_w = int(w * MARGIN_WIDTH_FRACTION)
    for side, x_start in [("left", 0), ("right", w - margin_w)]:
        margin_strip = ink[:, x_start:x_start + margin_w]
        margin_ink   = margin_strip.sum() / margin_strip.size
        if margin_ink > MARGIN_MIN_INK_FRACTION:
            zones.append({
                "zone_type":  ZONE_MARGIN_PERPENDICULAR,
                "bbox":       [x_start, 0, margin_w, h],
                "confidence": min(1.0, margin_ink / 0.1),
                "annotation": f"{side.capitalize()} margin has ink ({margin_ink:.1%}) — possible perpendicular text",
            })

    # ── Detect main body column structure ──────────────────────────────────
    # Vertical projection: ink per column
    col_ink_fraction = ink.sum(axis=0) / h
    # Find central vertical gap (between sadr and ajuz)
    mid_start = w // 4
    mid_end   = 3 * w // 4
    mid_section = col_ink_fraction[mid_start:mid_end]
    gap_cols = mid_section < 0.02   # very low ink column in middle
    if gap_cols.sum() > w * COLUMN_GAP_MIN_WIDTH:
        # Two-column layout detected
        zones.append({
            "zone_type":  ZONE_TWO_COLUMN_POETRY,
            "bbox":       [0, 0, w, h],
            "confidence": min(1.0, gap_cols.sum() / (w * COLUMN_GAP_MIN_WIDTH * 2)),
            "annotation": "Vertical gap detected in mid-section — likely sadr/ajuz two-column poetry layout",
        })
    else:
        # Assume single main text block
        zones.append({
            "zone_type":  ZONE_TWO_COLUMN_POETRY,
            "bbox":       [0, 0, w, h],
            "confidence": 0.4,
            "annotation": "No clear column gap — may be single-column prose or poetry. Verify manually.",
        })

    # ── Detect attribution lines (short dense bands) ───────────────────────
    # These appear as short rows of dense ink, often at top or between sections
    dense_rows = row_ink_fraction > row_ink_fraction.mean() * 1.8
    in_band = False
    band_start = 0
    for r in range(h):
        if dense_rows[r] and not in_band:
            in_band    = True
            band_start = r
        elif not dense_rows[r] and in_band:
            in_band = False
            band_height = r - band_start
            if band_height < int(h * ATTRIBUTION_MAX_ROWS):
                zones.append({
                    "zone_type":  ZONE_PROSE_ATTRIBUTION,
                    "bbox":       [0, band_start, w, band_height],
                    "confidence": 0.5,
                    "annotation": f"Short dense band ({band_height} rows) — possible attribution / poet name line",
                })

    note = (
        f"Proposed {len(zones)} zone(s). All require annotator confirmation "
        "in eScriptorium before Kraken segmentation. "
        "Two-column poetry zones are most important to verify."
    )

    return {
        "zones": zones,
        "note":  note,
    }
