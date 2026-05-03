"""
al_nassikh/folio_gallery.py
============================
Why this exists: the Streamlit Scholar Workbench should let users browse actual
manuscript folio images for a chosen poet or manuscript — not just read text
excerpts.  Where it's called: app (live query)
  Purpose: Resolves manuscript folio images 
    by poet/page for the image viewer in Tab A
    test. Purpose: Resolves manuscript folio images by poet/page for the image viewer in Tab A
This module resolves the image file names stored in the anchor registry
to actual paths on disk, then provides two lookup helpers:

  get_images_for_manuscript(short_key)  → sorted list of Path objects
  get_images_for_poet(poet_name)        → {short_key: [Path, ...], ...}
  resolve_citation_image(source_image_path) → Path | None

The image files live in handwritten-poems/manuscripts/ and its sub-directories.
Registry entries store paths like "manuscripts/MVP_Ground_Truth_Images/manuscript01_p70.png"
but the actual files may be in any sub-directory, so we build a filename→path index
once and cache it.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HERE      = Path(__file__).resolve().parent          # src/al_nassikh/
_REPO_ROOT = _HERE.parent.parent                      # handwritten-poems/
_MS_ROOT   = _REPO_ROOT / "manuscripts"
_REGISTRY_PATH = _REPO_ROOT / "data" / "ground_truth" / "anchor_registry_full_enriched.json"
_MS_REG_PATH   = _REPO_ROOT / "data" / "ground_truth" / "manuscript_registry.json"


# ── Image index ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_image_index() -> dict[str, Path]:
    """
    Why cached: scanning ~300 files across sub-directories is slow for a web app.
    We do it once per process and return a filename→absolute_path mapping.
    Only indexes files whose stem matches `manuscriptXX_pYYY` to skip UI assets.
    """
    index: dict[str, Path] = {}
    if not _MS_ROOT.exists():
        logger.warning("folio_gallery: manuscripts/ directory not found at %s", _MS_ROOT)
        return index

    for img in _MS_ROOT.rglob("*.png"):
        if ".venv" in str(img) or "__pycache__" in str(img):
            continue
        stem = img.stem.lower()
        if "manuscript" in stem and "_p" in stem:
            index[img.name] = img.resolve()

    logger.info("folio_gallery: indexed %d manuscript page images", len(index))
    return index


def resolve_citation_image(source_image_path: str) -> Optional[Path]:
    """
    Given a source_image_path from the anchor registry (e.g.
    "manuscripts/MVP_Ground_Truth_Images/manuscript01_p70.png"),
    return the resolved absolute Path on disk, or None if not found.
    """
    if not source_image_path:
        return None
    name = Path(source_image_path).name
    return _build_image_index().get(name)


# ── Per-manuscript lookup ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_registry() -> list[dict]:
    try:
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("folio_gallery: registry not found at %s", _REGISTRY_PATH)
        return []


@lru_cache(maxsize=1)
def _load_ms_registry() -> list[dict]:
    try:
        with open(_MS_REG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


@lru_cache(maxsize=64)
def get_images_for_manuscript(short_key: str) -> list[Path]:
    """
    Return a sorted (by page number) list of resolved image Paths for every
    anchor in the registry whose manuscript_short_key matches short_key.
    Duplicates are removed — multiple anchors on the same page share one image.
    """
    index  = _build_image_index()
    seen: dict[str, Path] = {}
    for entry in _load_registry():
        if entry.get("manuscript_short_key") != short_key:
            continue
        ip = entry.get("source_image_path", "")
        if not ip:
            continue
        name = Path(ip).name
        if name in index and name not in seen:
            seen[name] = index[name]

    def _page_num(p: Path) -> int:
        stem = p.stem  # e.g. manuscript01_p70
        try:
            return int(stem.rsplit("_p", 1)[-1])
        except ValueError:
            return 0

    return sorted(seen.values(), key=_page_num)


@lru_cache(maxsize=256)
def get_images_for_poet(poet_name: str) -> dict[str, list[Path]]:
    """
    Return {short_key: [Path, ...]} for all manuscripts that contain at least
    one anchor attributed to poet_name.  The poet name match is exact (stripped).
    Use list_poets_with_images() to get valid poet names.
    """
    poet_name = poet_name.strip()
    # Find which manuscript short_keys have this poet
    ms_keys: set[str] = set()
    for entry in _load_registry():
        if (entry.get("poet_name") or "").strip() == poet_name:
            sk = entry.get("manuscript_short_key")
            if sk:
                ms_keys.add(sk)

    return {sk: get_images_for_manuscript(sk) for sk in ms_keys if get_images_for_manuscript(sk)}


def list_manuscripts_with_images() -> list[dict]:
    """
    Return a list of {short_key, arabic_name, english_name, image_count} dicts
    for every manuscript that has at least one resolved image, sorted by image_count desc.
    """
    ms_reg = {m["short_key"]: m for m in _load_ms_registry() if m.get("short_key")}
    results = []
    covered: set[str] = set()
    index = _build_image_index()
    for entry in _load_registry():
        sk  = entry.get("manuscript_short_key", "")
        ip  = entry.get("source_image_path", "")
        if sk and ip and Path(ip).name in index:
            covered.add(sk)

    for sk in covered:
        m = ms_reg.get(sk, {})
        imgs = get_images_for_manuscript(sk)
        results.append({
            "short_key":    sk,
            "arabic_name":  m.get("arabic_name", sk),
            "english_name": m.get("english_name", sk),
            "image_count":  len(imgs),
        })

    return sorted(results, key=lambda x: -x["image_count"])


def list_poets_with_images(min_images: int = 1) -> list[str]:
    """
    Return a sorted list of poet names that appear in at least one manuscript
    with resolvable images.  Names like 'unknown', 'مجهول', 'غير محدد' are
    filtered out as they are not useful for browsing.
    """
    _ANON = {"unknown", "مجهول", "غير محدد", "مخطوطة"}
    _ANON_SUBSTRINGS = {"unknown", "مجهول", "غير محدد"}
    index = _build_image_index()
    poets_with_imgs: set[str] = set()
    for entry in _load_registry():
        pn = (entry.get("poet_name") or "").strip()
        if not pn or pn in _ANON or any(a in pn for a in _ANON_SUBSTRINGS):
            continue
        # Skip junk: must start with Arabic letter (not Indic digit ٠-٩), ≥4 Arabic letters
        arabic_chars = sum(1 for c in pn if "؀" <= c <= "ۿ")
        arabic_indic_digit = "٠" <= pn[0] <= "٩"
        is_arabic_letter_start = "؀" <= pn[0] <= "ۿ" and not arabic_indic_digit
        if len(pn) < 4 or arabic_chars < 4 or not is_arabic_letter_start:
            continue
        ip = entry.get("source_image_path", "")
        if ip and Path(ip).name in index:
            poets_with_imgs.add(pn)

    return sorted(poets_with_imgs)


def clear_caches() -> None:
    """Clear LRU caches — used in tests."""
    _build_image_index.cache_clear()
    _load_registry.cache_clear()
    _load_ms_registry.cache_clear()
    get_images_for_manuscript.cache_clear()
    get_images_for_poet.cache_clear()
