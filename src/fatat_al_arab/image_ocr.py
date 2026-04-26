"""
src/fatat_al_arab/image_ocr.py
================================
Why this file exists: §2.4 Stage 1 allows the user to submit a question as an
image (handwritten or photographed Arabic). This module turns that image into
question text using pytesseract (Arabic language pack) with an LLM cleanup pass
for low-confidence output. The cleaned text is then handed to translate.py and
the rest of Agent 1.

Architecture refs: §2.4 Stage 1 (image input path), §2.8 "Agent 1 image OCR"
(honest refusal when OCR confidence is too low — we don't silently embed garbage).

Failure behaviour (§5): if pytesseract is unavailable, or if the OCR confidence
is below MIN_CONFIDENCE, the function raises OcrLowConfidenceError and the
Streamlit UI renders an honest "couldn't read this image — please type your
question" message rather than propagating bad text into the retrieval pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .llm import chat

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Why a named constant: §5 says every budget/threshold is a constant, not a
# magic literal. Tweaking OCR sensitivity is a one-line change here.

MIN_CONFIDENCE = 60.0   # pytesseract confidence threshold (0-100); below → LLM cleanup pass
LLM_CLEANUP_CONFIDENCE = 30.0   # below this even LLM cleanup may not help → raise

_SYSTEM_OCR_CLEANUP = (
    "You are an expert in classical and Gulf-dialect Arabic (Nabati poetry). "
    "The following text was produced by an OCR engine reading handwritten Arabic "
    "and may contain recognition errors. Clean up the text: fix obvious OCR mistakes, "
    "correct word boundaries, restore diacritics where unambiguous, and preserve "
    "all content. Return ONLY the corrected Arabic text — no explanation."
)


class OcrLowConfidenceError(RuntimeError):
    """
    Why a specific exception type: the Streamlit UI catches this and renders a
    user-friendly message; other RuntimeErrors propagate as unexpected failures.
    """
    pass


def _pytesseract_available() -> bool:
    """Lazy check so the module imports cleanly without the system package."""
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def ocr_image(image_path: str | Path) -> str:
    """
    Run pytesseract (Arabic language pack) on *image_path* and return the
    extracted question text.

    Why this order of operations:
    1. pytesseract gives us a confidence score per word (--psm 3 gives page-level segmentation).
    2. If mean confidence ≥ MIN_CONFIDENCE the raw text is already usable.
    3. If mean confidence is in [LLM_CLEANUP_CONFIDENCE, MIN_CONFIDENCE) an LLM pass
       fixes the most common OCR artefacts (ي/ى confusion, missing hamzas, joined words).
    4. Below LLM_CLEANUP_CONFIDENCE we raise OcrLowConfidenceError — the image is
       too degraded for the system to read reliably.

    Args:
        image_path: Path to a PNG/JPEG image file containing the handwritten question.

    Returns:
        Cleaned Arabic question text as a string.

    Raises:
        OcrLowConfidenceError: if OCR confidence is too low even after LLM cleanup.
        FileNotFoundError: if image_path does not exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image_ocr: image not found: {image_path}")

    if not _pytesseract_available():
        # Why not silently return garbage: §5 says honest refusal is better than
        # embedding unreadable input into the retrieval pipeline.
        raise OcrLowConfidenceError(
            "pytesseract is not installed. Install it with: "
            "pip install pytesseract  (and install tesseract-ocr + ara language pack)."
        )

    import pytesseract
    from PIL import Image

    img = Image.open(image_path)

    # Why --oem 3 --psm 3: LSTM engine (oem 3) gives the best Arabic results;
    # psm 3 (fully automatic page segmentation) handles single questions well.
    ocr_config = "--oem 3 --psm 3 -l ara"

    # Get both the text and per-word confidence data
    raw_text: str = pytesseract.image_to_string(img, config=ocr_config).strip()
    data: dict = pytesseract.image_to_data(
        img, config=ocr_config, output_type=pytesseract.Output.DICT
    )

    # Compute mean confidence from words with valid confidence values
    confidences = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c >= 0]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    logger.debug(
        "image_ocr: file=%s raw_text=%r mean_conf=%.1f",
        image_path.name, raw_text[:50], mean_conf
    )

    if mean_conf >= MIN_CONFIDENCE:
        return raw_text

    if mean_conf >= LLM_CLEANUP_CONFIDENCE:
        logger.info(
            "image_ocr: low confidence (%.1f < %.1f), running LLM cleanup pass.",
            mean_conf, MIN_CONFIDENCE
        )
        cleaned = chat(
            prompt=raw_text,
            system=_SYSTEM_OCR_CLEANUP,
            max_tokens=512,
        )
        return cleaned.strip()

    # Below LLM_CLEANUP_CONFIDENCE — image is too degraded
    raise OcrLowConfidenceError(
        f"Image OCR confidence too low ({mean_conf:.1f} < {LLM_CLEANUP_CONFIDENCE}). "
        "The handwriting could not be read reliably. Please type your question instead."
    )


def ocr_image_to_query(image_path: str | Path) -> Optional[str]:
    """
    Why this wrapper: the Streamlit UI calls this function and handles
    OcrLowConfidenceError itself (shows the error message). Returns None on
    any failure so the UI can display a fallback gracefully rather than crashing.
    """
    try:
        return ocr_image(image_path)
    except OcrLowConfidenceError as exc:
        logger.warning("image_ocr: low confidence — %s", exc)
        return None
    except FileNotFoundError as exc:
        logger.error("image_ocr: file error — %s", exc)
        return None
    except Exception as exc:
        logger.error("image_ocr: unexpected error — %s", exc)
        return None
