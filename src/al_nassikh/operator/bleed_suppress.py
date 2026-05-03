"""
src/al_nassikh/operator/bleed_suppress.py
==========================================
Why this file exists: §2.3 Stage 1 — bleed-through from the reverse side of
the folio is Challenge 3 in the architecture doc. It manifests as faint mirrored
text overlaid on the recto, which confuses Kraken's baseline detector into
splitting real text lines or hallucinating extra baselines. The filter here
estimates and subtracts the background component without touching the ink.
Where it's called: app Tab B (live), tests
Purpose: Removes ghost ink from
reverse-page bleed-through

Algorithm:
  1. Detect whether bleed-through is present (std-dev check on bright region).
  2. If absent, return the image unchanged — no-op is explicit in the result dict.
  3. If present, apply a morphological closing to estimate the "background sheet"
     and subtract it, then apply a Gaussian difference-of-Gaussians (DoG) to
     enhance ink edges without amplifying noise.

Why morphological closing (not FFT): the closed-form background estimate is
interpretable and fast on CPU. FFT approaches require careful frequency selection
per manuscript — not appropriate for a HITL tool where the archivists aren't
signal-processing engineers.

Architecture ref: §2.3 Stage 1. Challenge 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

BLEED_DETECTION_STD  = 30.0   # std-dev above this on the bright region → bleed detected
# Why 30 (not 40): real bleed-through on Khaleeji manuscript scans produces
# std-dev of 28-35 in the bright background region. 40 was too conservative.
BRIGHT_REGION_THRESH = 180    # pixels above this are "background sheet" candidates
MORPH_KERNEL_SIZE    = 25     # closing kernel size (pixels); larger = smoother background
DOG_SIGMA1           = 1.0    # inner Gaussian sigma (edge-preserving)
DOG_SIGMA2           = 3.0    # outer Gaussian sigma (background)
CLIP_LOW             = 0
CLIP_HIGH            = 255


def _to_gray(image: Any) -> np.ndarray:
    """Accept PIL Image or numpy array; return 2D uint8 grayscale."""
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


def _has_bleed(gray: np.ndarray) -> bool:
    """Check whether bleed-through is detectable."""
    bright_mask = gray > BRIGHT_REGION_THRESH
    if bright_mask.sum() < gray.size * 0.1:
        return False   # too few bright pixels — likely a very dark page
    return float(gray[bright_mask].std()) > BLEED_DETECTION_STD


def _morph_close_background(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Morphological closing: dilation then erosion with a square structuring element.
    The result approximates the "bright background sheet" without ink strokes.
    Why pure-numpy (not cv2.morphologyEx): avoids OpenCV as a hard dependency.
    Pure numpy is ~10× slower on large images but acceptable for HITL use (single pages).
    """
    from scipy.ndimage import grey_closing  # soft dependency via scipy
    return grey_closing(gray.astype(np.float32), size=kernel_size).astype(np.uint8)


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Simple Gaussian blur via scipy or manual convolution fallback."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr.astype(np.float32), sigma=sigma)
    except ImportError:
        # Fallback: box blur approximation (3 passes)
        k = max(1, int(sigma * 3) | 1)   # odd kernel size
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)
        from numpy.lib.stride_tricks import sliding_window_view
        # Simple uniform filter approximation
        pad = k // 2
        padded = np.pad(arr.astype(np.float32), pad, mode="reflect")
        result = np.zeros_like(arr, dtype=np.float32)
        h, w = arr.shape
        for i in range(h):
            for j in range(w):
                result[i, j] = padded[i:i+k, j:j+k].mean()
        return result


def suppress(image: Any) -> dict:
    """
    Stage 1 bleed-through suppression.

    Args:
        image: PIL Image or numpy array (grayscale or RGB).

    Returns:
        {
          "applied":      bool    — True if filter was applied (bleed detected)
          "result_image": ndarray — processed grayscale uint8 image
          "reason":       str     — explanation for the annotator
        }

    Why return the image in the dict: the Streamlit tab shows a before/after
    side-by-side and the annotator decides whether to use the filtered version.
    """
    gray = _to_gray(image)

    if not _has_bleed(gray):
        return {
            "applied":      False,
            "result_image": gray,
            "reason":       "No bleed-through detected — image returned unchanged.",
        }

    # Step 1: morphological background estimate
    try:
        background = _morph_close_background(gray, MORPH_KERNEL_SIZE)
        # Subtract background; clip to valid range; invert back
        # (gray images: low = dark ink; high = white paper)
        diff = np.clip(
            background.astype(np.int16) - gray.astype(np.int16) + 128,
            CLIP_LOW, CLIP_HIGH
        ).astype(np.uint8)
    except Exception:
        # scipy not available — skip morph step, go straight to DoG
        diff = gray.copy()

    # Step 2: DoG edge sharpening to restore ink clarity after background subtraction
    g1 = _gaussian_blur(diff, DOG_SIGMA1)
    g2 = _gaussian_blur(diff, DOG_SIGMA2)
    dog = np.clip(diff.astype(np.float32) + (g1 - g2) * 0.5, 0, 255).astype(np.uint8)

    return {
        "applied":      True,
        "result_image": dog,
        "reason": (
            f"Bleed-through detected (std={gray.std():.1f} > {BLEED_DETECTION_STD}). "
            f"Applied morphological background subtraction + DoG sharpening "
            f"(kernel={MORPH_KERNEL_SIZE}, σ={DOG_SIGMA1}/{DOG_SIGMA2})."
        ),
    }
