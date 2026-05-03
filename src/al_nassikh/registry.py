"""
src/al_nassikh/registry.py
==========================
Why this file exists: §M1 — the Streamlit Archive Manager (Tab A) needs to show
the manuscript inventory in human-readable form ("Ibn Yahya Manuscript (401-600)")
rather than the internal short keys used throughout the data pipeline. This module
loads manuscript_registry.json once at import time and exposes four stable helper
functions the UI, the retriever, and the evaluation harness can all use without
re-reading the file.  Where it's called: self_query.py (live query), testsPurpose: Manuscript registry phone book, by_short_key, by_filename, list_all

Architecture ref: §2.3 (Al-Nassikh) states that every manuscript identifier in
the pipeline must resolve to a canonical Arabic + English name pair via this
registry so that citations in Stage 8 (Synthesis) are human-readable.

Public API:
    by_short_key(key: str) -> dict | None
    by_number(n: int) -> dict | None
    by_filename(path_or_stem: str) -> dict | None
    list_all() -> list[dict]

Each returned dict has:
    number        int    — canonical manuscript number (1-25)
    short_key     str    — internal pipeline identifier
    arabic_name   str    — Arabic display name
    english_name  str    — English display name
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ── Path resolution ───────────────────────────────────────────────────────────

def _repo_root() -> Path:
    """Walk up from this file until we find requirements.txt (repo root)."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path(os.getcwd())


_REGISTRY_PATH = _repo_root() / "data" / "ground_truth" / "manuscript_registry.json"
_FILENAME_MAP_PATH = _repo_root() / "data" / "ground_truth" / "manuscript_filename_map.json"


# ── Load at import time ───────────────────────────────────────────────────────
# Why module-level: every node in the LangGraph pipeline may call by_short_key()
# on every retrieved chunk. Loading once at import is far cheaper than opening
# the JSON file on every call.

def _load() -> tuple[list[dict], dict[str, dict], dict[int, dict], dict[str, dict]]:
    """
    Returns (all_entries, by_key_index, by_number_index, by_filename_index).
    Called once at module load; results are stored in module-level variables.
    """
    with _REGISTRY_PATH.open(encoding="utf-8") as f:
        entries: list[dict] = json.load(f)

    by_key: dict[str, dict]  = {e["short_key"]: e for e in entries}
    by_num: dict[int, dict]  = {e["number"]: e for e in entries}

    # Filename map: stem → short_key → registry entry
    by_stem: dict[str, dict] = {}
    if _FILENAME_MAP_PATH.exists():
        with _FILENAME_MAP_PATH.open(encoding="utf-8") as f:
            filename_map: list[dict] = json.load(f)
        for row in filename_map:
            stem = row.get("filename_stem", "")
            key  = row.get("short_key", "")
            if stem and key and key in by_key:
                by_stem[stem] = by_key[key]

    return entries, by_key, by_num, by_stem


_ALL_ENTRIES, _BY_KEY, _BY_NUMBER, _BY_STEM = _load()


# ── Public helpers ────────────────────────────────────────────────────────────

def by_short_key(key: str) -> Optional[dict]:
    """
    Look up a manuscript by its pipeline short_key (e.g., 'ibn_yahya_401_600').
    Returns None if not found — callers must handle gracefully.

    Why None-safe: Stage 8 (Synthesis) calls this on every citation it builds.
    An unknown key must produce a None (logged as an audit warning) rather than
    crashing the pipeline mid-response.
    """
    return _BY_KEY.get(key)


def by_number(n: int) -> Optional[dict]:
    """
    Look up a manuscript by its canonical number (1-25).
    Used by the Streamlit Archive Manager to render the numbered inventory list.
    """
    return _BY_NUMBER.get(n)


def by_filename(path_or_stem: str) -> Optional[dict]:
    """
    Resolve a filesystem path or bare stem (e.g., 'manuscript07',
    'manuscripts/MVP_Ground_Truth_Images/601-782_p178.png') to a registry
    entry via manuscript_filename_map.json.

    Why volume-stem extraction: the anchor registry stores full image paths like
    'manuscripts/MVP_Ground_Truth_Images/601-782_p178.png'. We extract the
    volume identifier ('601-782') and look it up in the filename map.
    """
    # Strip to stem — handle full paths
    stem = Path(path_or_stem).stem if path_or_stem else ""
    # If stem contains '_p' page suffix (e.g., '601-782_p178'), strip page part
    if "_p" in stem:
        stem = stem.split("_p")[0]
    return _BY_STEM.get(stem)


def list_all() -> list[dict]:
    """
    Return all 25 manuscript entries in canonical order (by number).
    Used by the Streamlit Archive Manager sidebar to render the corpus inventory.
    """
    return sorted(_ALL_ENTRIES, key=lambda e: e["number"])
