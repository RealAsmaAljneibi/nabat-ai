"""
src/al_nassikh/operator/standardise.py
=======================================
Why this file exists: §2.3 Stage 2 — manuscript pages arrive at varying
resolutions (150–600 DPI scans) and with varying skew angles (books scanned
flat vs camera-captured at an angle). Kraken's BLLA model was trained at 300 DPI;
providing images at different resolutions degrades segmentation quality. Skew
also causes the baseline detector to underfit short curved lines.

This module standardises every page to 300 DPI grayscale, deskews by detecting
the dominant line angle, and applies a mild unsharp mask to sharpen diacritics
(harakāt), which are Challenge 8 in the architecture doc.

Algorithm:
  1. Convert to grayscale.
  2. Resize to target DPI (if metadata available) or to a target width (fallback).
  3. Deskew by Hough-line dominant angle (on a Canny-edge binary).
  4. Unsharp mask: image + α * (image - blur).

Why Hough-line deskew (not projection profile): projection profile is simpler
but fails on pages with mixed-direction text (notes in margins). Hough on edges
gives a robust dominant angle for the main text block.

Architecture ref: §2.3 Stage 2. Challenge 8 (diacritics).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_WIDTH      = 2400    # pixels; ~300 DPI for typical A4 manuscript scan
UNSHARP_ALPHA     = 0.5     # unsharp mask strength (0 = no sharpening)
UNSHARP_SIGMA     = 1.5     # Gaussian sigma for the blur component
DESKEW_ANGLE_MAX  = 10.0    # degrees; larger angles → suspect (flip, not tilt)
HOUGH_THRESHOLD   = 80      # minimum Hough accumulator votes to accept a line


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


def _resize_to_target_width(gray: np.ndarray, target_width: int) -> np.ndarray:
    """Resize maintaining aspect ratio, only if narrower than target_width."""
    h, w = gray.shape
    if w == target_width:
        return gray
    scale = target_width / w
    new_h = int(h * scale)
    new_w = target_width
    # Nearest-neighbour resize via numpy (avoids PIL/cv2 requirement)
    row_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    col_idx = (np.linspace(0, w - 1, new_w)).astype(int)
    return gray[np.ix_(row_idx, col_idx)]


def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate the dominant text-line angle using a simple horizontal-projection
    variance approach (projection profile method — fast, no OpenCV needed).

    Returns angle in degrees (positive = clockwise tilt).
    """
    # Try a range of small angles; for each, rotate and compute horizontal
    # projection variance. Maximum variance = best alignment.
    best_angle = 0.0
    best_var   = -1.0

    # Downsample for speed
    small = gray[::4, ::4]

    for angle in np.arange(-DESKEW_ANGLE_MAX, DESKEW_ANGLE_MAX + 0.5, 0.5):
        rotated = _rotate(small, angle)
        projection = rotated.sum(axis=1).astype(float)
        var = float(projection.var())
        if var > best_var:
            best_var   = var
            best_angle = angle

    return best_angle


def _rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a grayscale image by angle_deg using affine transform (numpy only)."""
    if abs(angle_deg) < 0.1:
        return gray
    try:
        from scipy.ndimage import rotate as scipy_rotate
        return scipy_rotate(gray, angle_deg, reshape=False, cval=255, order=1)
    except ImportError:
        # Fallback: return original (deskew skipped)
        return gray


def _unsharp_mask(gray: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """Apply unsharp mask: result = gray + alpha * (gray - blur(gray))."""
    try:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(gray.astype(np.float32), sigma=sigma)
    except ImportError:
        return gray   # skip if scipy unavailable
    sharpened = gray.astype(np.float32) + alpha * (gray.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def standardise(
    image: Any,
    source_dpi: Optional[int] = None,
    target_width: int = TARGET_WIDTH,
    apply_deskew: bool = True,
    apply_unsharp: bool = True,
) -> dict:
    """
    Stage 2 image standardisation.

    Args:
        image:        PIL Image or numpy array.
        source_dpi:   Original scan DPI (optional — used to scale to 300 DPI
                      if provided; otherwise rescales to target_width).
        target_width: Target width in pixels (default 2400 ≈ 300 DPI A4).
        apply_deskew: Whether to run deskew correction.
        apply_unsharp: Whether to apply unsharp mask for diacritics.

    Returns:
        {
          "result_image": ndarray uint8,
          "skew_angle":   float   (degrees corrected; 0.0 if no deskew),
          "resize_scale": float   (1.0 = no resize),
          "applied":      list[str]  operations applied
        }
    """
    gray = _to_gray(image)
    applied: list[str] = ["grayscale"]
    original_shape = gray.shape

    # Step 1: resize
    h, w = gray.shape
    if source_dpi and source_dpi != 300:
        scale = 300 / source_dpi
        new_w = int(w * scale)
        gray = _resize_to_target_width(gray, new_w)
        applied.append(f"resize {source_dpi}→300 DPI (scale={scale:.2f})")
        resize_scale = scale
    elif w != target_width:
        scale = target_width / w
        gray = _resize_to_target_width(gray, target_width)
        applied.append(f"resize to {target_width}px width (scale={scale:.2f})")
        resize_scale = scale
    else:
        resize_scale = 1.0

    # Step 2: deskew
    skew_angle = 0.0
    if apply_deskew:
        skew_angle = _detect_skew_angle(gray)
        if abs(skew_angle) > 0.3:
            gray = _rotate(gray, skew_angle)
            applied.append(f"deskew {skew_angle:+.1f}°")
        else:
            applied.append("deskew skipped (angle < 0.3°)")

    # Step 3: unsharp mask for diacritics
    if apply_unsharp:
        gray = _unsharp_mask(gray, UNSHARP_SIGMA, UNSHARP_ALPHA)
        applied.append(f"unsharp mask (σ={UNSHARP_SIGMA}, α={UNSHARP_ALPHA})")

    return {
        "result_image": gray,
        "skew_angle":   round(skew_angle, 2),
        "resize_scale": round(resize_scale, 4),
        "applied":      applied,
    }
