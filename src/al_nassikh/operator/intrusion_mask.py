"""
src/al_nassikh/operator/intrusion_mask.py
==========================================
Why this file exists: §2.3 Stage 3 — manuscript pages from personal and
institutional collections often carry rubber stamps (library ownership marks,
archival accession numbers) and adhesive labels (barcode stickers, sale labels).
These are Challenge 7 in the architecture doc. Kraken treats stamp ink as text
and generates spurious text lines from them. The function here detects candidate
intrusion regions and returns mask polygons for the annotator to confirm.
Where it's called: tests. Purpose: Masks stamps and labels so HTR doesn't transcribe them as verse

Algorithm:
  Stamps: Circular Hough transform on a Canny-edge binary. Circles with radius
    in [stamp_r_min, stamp_r_max] are stamp candidates.
  Labels: Rectangular blob detection via connected-component analysis on a
    thresholded diff from the background. Rectangles with aspect ratio in a
    "label-like" range are label candidates.

Why return polygons (not applied mask): the annotator must confirm before
anything is erased. Accidental masking of genuine verse text is catastrophic.

Architecture ref: §2.3 Stage 3. Challenge 7.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

STAMP_RADIUS_MIN     = 30    # pixels; smaller → likely noise
STAMP_RADIUS_MAX     = 200   # pixels; larger → unlikely to be a stamp
LABEL_MIN_AREA       = 2000  # pixels²; minimum area to report a label region
LABEL_ASPECT_MIN     = 1.5   # width/height ratio minimum for "label-like"
LABEL_ASPECT_MAX     = 8.0   # width/height ratio maximum
CANNY_LOW            = 30
CANNY_HIGH           = 100
BINARISE_THRESHOLD   = 128


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


def _canny_edges(gray: np.ndarray) -> np.ndarray:
    """Simplified Canny: Sobel gradient magnitude thresholded."""
    gx = np.abs(np.diff(gray.astype(float), axis=1))
    gy = np.abs(np.diff(gray.astype(float), axis=0))
    # Pad to same size
    gx = np.hstack([gx, gx[:, -1:]])
    gy = np.vstack([gy, gy[-1:, :]])
    mag = np.hypot(gx, gy)
    edges = ((mag > CANNY_LOW) & (mag > mag.mean())).astype(np.uint8) * 255
    return edges


def _detect_circular_stamps(gray: np.ndarray) -> list[dict]:
    """
    Detect circular stamps using a simplified Hough circle transform.
    Returns list of {cx, cy, radius, confidence} dicts.
    """
    try:
        from skimage.transform import hough_circle, hough_circle_peaks
        from skimage.feature import canny
        edges = canny(gray, sigma=2.0)
        radii = np.arange(STAMP_RADIUS_MIN, STAMP_RADIUS_MAX, 5)
        hough_res = hough_circle(edges, radii)
        accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
            hough_res, radii,
            min_xdistance=STAMP_RADIUS_MIN * 2,
            min_ydistance=STAMP_RADIUS_MIN * 2,
            threshold=0.4,
            num_peaks=5,
        )
        return [
            {
                "type":       "stamp_circle",
                "cx":         int(cx),
                "cy":         int(cy),
                "radius":     int(r),
                "confidence": float(round(acc, 3)),
                "polygon":    _circle_polygon(int(cx), int(cy), int(r)),
            }
            for acc, cx, cy, r in zip(accums, cx_arr, cy_arr, rad_arr)
        ]
    except ImportError:
        # Fallback: no skimage — return empty (HITL annotator can add manually)
        return []


def _circle_polygon(cx: int, cy: int, r: int, n_points: int = 16) -> list[list[int]]:
    """Approximate a circle as a polygon with n_points vertices."""
    import math
    return [
        [int(cx + r * math.cos(2 * math.pi * i / n_points)),
         int(cy + r * math.sin(2 * math.pi * i / n_points))]
        for i in range(n_points)
    ]


def _detect_rectangular_labels(gray: np.ndarray) -> list[dict]:
    """
    Detect rectangular label stickers using connected-component analysis on a
    binary image derived from background subtraction.
    Returns list of {x, y, w, h, confidence, polygon} dicts.
    """
    try:
        from scipy.ndimage import label as scipy_label
    except ImportError:
        return []   # skip if scipy unavailable

    # Binarise: light regions that are uniform (labels are bright + low texture)
    binary = (gray > BINARISE_THRESHOLD).astype(np.uint8)

    # Local texture: std in a small window; low std = flat (label-like)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray.astype(float), size=15)
    local_sq   = uniform_filter((gray.astype(float))**2, size=15)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
    flat_mask  = (local_std < 20) & binary   # bright + flat texture

    labeled, n_components = scipy_label(flat_mask)
    candidates = []
    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        area = mask.sum()
        if area < LABEL_MIN_AREA:
            continue
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            continue
        aspect = w / h
        if not (LABEL_ASPECT_MIN <= aspect <= LABEL_ASPECT_MAX):
            continue
        candidates.append({
            "type":       "label_rectangle",
            "x": x1, "y": y1, "w": w, "h": h,
            "area":       int(area),
            "confidence": round(min(1.0, area / 10000), 3),
            "polygon": [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ],
        })

    return candidates


def detect_intrusions(image: Any) -> dict:
    """
    Stage 3 intrusion detection — stamps and labels.

    Args:
        image: PIL Image or numpy array.

    Returns:
        {
          "intrusions":  list[dict]  — each item is a detected region
          "count":       int
          "note":        str         — message for the annotator
        }

    Each intrusion dict has:
        type:       "stamp_circle" | "label_rectangle"
        polygon:    list of [x, y] points — confirm or adjust in the UI
        confidence: float 0-1
        (+ type-specific fields: cx/cy/radius or x/y/w/h/area)

    Why return polygons for confirmation: an accidental mask on real verse text
    is irreversible. The annotator always has the final word.
    """
    gray = _to_gray(image)
    circles = _detect_circular_stamps(gray)
    rects   = _detect_rectangular_labels(gray)
    all_intrusions = circles + rects

    if not all_intrusions:
        note = "No stamp or label intrusions detected."
    else:
        n_stamps = len(circles)
        n_labels = len(rects)
        parts = []
        if n_stamps:
            parts.append(f"{n_stamps} circular stamp(s)")
        if n_labels:
            parts.append(f"{n_labels} label(s)")
        note = (
            f"Detected {' and '.join(parts)}. "
            "Please confirm each polygon before masking — false positives are possible."
        )

    return {
        "intrusions": all_intrusions,
        "count":      len(all_intrusions),
        "note":       note,
    }
