"""
src/al_nassikh/ingest/pdf_to_pages.py
=======================================
Why this file exists: §M2 item 8 — new manuscripts arrive as scanned PDFs.
Before any operator helper or eScriptorium upload can run, the PDF must be
split into individual page images at 300 DPI (the resolution Kraken's BLLA
model was trained on). This module handles that extraction and writes PNGs
into a per-phase directory under `manuscripts/MVP_Ground_Truth_Images/`.

Why pdf2image + poppler (not pypdf.PdfReader): pypdf can extract embedded images
but most manuscript PDFs are scanned (raster) — the embedded "image" is the
full-page JPEG/TIFF. pdf2image with poppler re-renders each page at a controlled
DPI, guaranteeing consistent pixel density regardless of the original scan DPI.

Architecture ref: §M2 item 8, §2.3 Stage 0 (triage runs on these PNGs).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DPI      = 300
DEFAULT_FORMAT   = "PNG"
PHASE_DIR_PREFIX = "Phase"   # output goes to manuscripts/MVP_Ground_Truth_Images/Phase_N/


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path(".")


def pdf_to_pages(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    phase: int = 1,
    dpi: int = DEFAULT_DPI,
    fmt: str = DEFAULT_FORMAT,
    first_page: Optional[int] = None,
    last_page:  Optional[int] = None,
) -> dict:
    """
    Convert a PDF manuscript to per-page PNG images at target DPI.

    Args:
        pdf_path:   Path to the source PDF.
        output_dir: Where to write PNGs. Defaults to
                    <repo_root>/manuscripts/MVP_Ground_Truth_Images/Phase_{phase}/.
        phase:      Which ingestion phase (1, 2, 3) — determines output subdirectory.
        dpi:        Render DPI (default 300).
        fmt:        Output image format ("PNG" or "TIFF").
        first_page: First page to extract (1-indexed, inclusive; None = first page).
        last_page:  Last page to extract (1-indexed, inclusive; None = last page).

    Returns:
        {
          "pdf_path":    str,
          "output_dir":  str,
          "page_count":  int,
          "pages":       list[str]   — absolute paths to written PNGs
          "dpi":         int,
          "errors":      list[str]   — any non-fatal per-page errors
        }
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image not installed. Run: pip install pdf2image\n"
            "Also requires poppler: macOS: brew install poppler | "
            "Ubuntu: apt-get install poppler-utils"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_dir is None:
        output_dir = _repo_root() / "manuscripts" / "MVP_Ground_Truth_Images" / f"{PHASE_DIR_PREFIX}_{phase}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem

    logger.info("Converting %s → %s at %d DPI…", pdf_path.name, output_dir, dpi)

    convert_kwargs: dict = {
        "pdf_path": str(pdf_path),
        "dpi":      dpi,
        "fmt":      fmt.lower(),
    }
    if first_page:
        convert_kwargs["first_page"] = first_page
    if last_page:
        convert_kwargs["last_page"] = last_page

    images = convert_from_path(**convert_kwargs)

    written_paths: list[str] = []
    errors:        list[str] = []
    start_page = (first_page or 1)

    for i, img in enumerate(images):
        page_num = start_page + i
        out_path = output_dir / f"{stem}_p{page_num:04d}.{fmt.lower()}"
        try:
            img.save(str(out_path))
            written_paths.append(str(out_path))
            logger.debug("Wrote %s", out_path.name)
        except Exception as exc:
            errors.append(f"Page {page_num}: {exc}")
            logger.warning("Failed to write page %d: %s", page_num, exc)

    logger.info(
        "Extracted %d pages from %s → %s",
        len(written_paths), pdf_path.name, output_dir
    )

    return {
        "pdf_path":   str(pdf_path),
        "output_dir": str(output_dir),
        "page_count": len(written_paths),
        "pages":      written_paths,
        "dpi":        dpi,
        "errors":     errors,
    }
