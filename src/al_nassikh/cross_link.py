"""
src/al_nassikh/cross_link.py
============================
Why this file exists: §M1 of the implementation plan — the Al-Nassikh ETL
produces two independent artefacts (phase{1,2,3}_poems.json from manuscript
transcription, anchor_registry_phase4.json from TOC images) that share poets
and matla lines but were built from different sources. This script joins them
so Agent 2 can navigate from a verse chunk all the way to the TOC entry that
gives the poet's full name, page citation, and source volume.

Algorithm (§M1 spec):
  1. Normalise (poet, matla) keys from both sides — strip harakat, unify alef
     variants, lowercase, strip punctuation.
  2. Use rapidfuzz WRatio (token-set ratio) on the composite key
     "{poet_name} :: {matla_first_10_tokens}" to tolerate OCR drift.
  3. Assign decision:
       linked  — best_score ≥ 0.85 (direct join; trustworthy for RAG)
       review  — 0.70 ≤ best_score < 0.85 (human review needed before M6)
       unlinked — best_score < 0.70 (no match found)
  4. Write data/ground_truth/verse_anchor_crosslink.json.

Run with:
    python -m src.al_nassikh.cross_link          # from repo root
    python cross_link.py                          # from src/al_nassikh/

Architecture refs: §M1 acceptance gate (≥ 70 % of Phase-1 verses linked ≥ 0.85),
§2.5 Stage 6 (resolve_heritage uses this file), §3.3 (citation chain).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Dependency guard ──────────────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz, process as rfprocess
except ImportError:
    sys.exit(
        "rapidfuzz not installed. Run: pip install rapidfuzz  (or pip install -r requirements.txt)"
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ── Path resolution ───────────────────────────────────────────────────────────
# Why dual resolution: the module is run both as __main__ (from repo root via
# `python -m src.al_nassikh.cross_link`) and imported in tests. Both need to
# find data/ground_truth/ regardless of cwd.

def _repo_root() -> Path:
    """Walk up from this file until we find requirements.txt (repo root)."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "requirements.txt").exists():
            return candidate
    # Fallback — trust cwd
    return Path(os.getcwd())


REPO_ROOT    = _repo_root()
GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth"

PHASE_FILES = [
    GROUND_TRUTH / "phase1_poems.json",
    GROUND_TRUTH / "phase2_poems.json",
    GROUND_TRUTH / "phase3_poems.json",
]
REGISTRY_FILE   = GROUND_TRUTH / "anchor_registry_phase4.json"
OUTPUT_FILE     = GROUND_TRUTH / "verse_anchor_crosslink.json"

# ── Thresholds (§M1) ─────────────────────────────────────────────────────────
LINKED_THRESHOLD  = 85   # rapidfuzz uses 0-100 scale
REVIEW_THRESHOLD  = 70


# ── Arabic normalisation ──────────────────────────────────────────────────────
# Why shared normalisation: OCR of handwritten Khaleeji manuscripts introduces
# hamza/alef variants, missing harakat, and tatweel. Matching on raw Unicode
# fails; we need to collapse these before comparison.

_ALEF_VARIANTS = "أإآٱ"
_HARAKAT_RE    = re.compile(r"[\u064B-\u065F\u0670]")  # fathatan … superscript alef
_PUNCT_RE      = re.compile(r"[^\u0600-\u06FF\s]")     # keep only Arabic + space


def _normalise(text: str) -> str:
    """
    Why each step:
      alef unification  — OCR renders أ / إ / آ / ٱ inconsistently; collapse to ا.
      ة → ه            — taa marbuta is often rendered as haa in Nabati manuscripts.
      strip harakat     — vocalization marks absent from most manuscript scans.
      strip tatweel     — kashida used inconsistently in older typesetting.
      strip punctuation — non-Arabic punctuation adds noise to token comparison.
      whitespace norm   — multiple spaces / newlines collapsed to single space.
    """
    if not text:
        return ""
    for v in _ALEF_VARIANTS:
        text = text.replace(v, "ا")
    text = text.replace("ة", "ه")
    text = _HARAKAT_RE.sub("", text)
    text = text.replace("\u0640", "")      # tatweel
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_n_tokens(text: str, n: int = 10) -> str:
    """Return the first n whitespace-separated tokens of a string."""
    return " ".join(text.split()[:n])


def _composite_key(poet: str, matla: str) -> str:
    """
    Why composite: matching on poet name alone fails when names are transliterated
    differently across the two pipelines (TOC images vs PAGE-XML). Appending the
    first 10 matla tokens forces the matcher to require agreement on both axes,
    reducing false positives from common poets (e.g., محمد) without a matla.

    Why matla-only when poet is unknown: when the transcription pipeline labels
    the poet as "unknown" or "مجهول", the composite key degenerates to just the
    matla tokens. We still attempt matching but only on matla text similarity.
    """
    norm_poet  = _normalise(poet)
    # Treat "unknown" / "مجهول" / blank as genuinely missing — don't let the
    # literal string "unknown" pollute the match key.
    if norm_poet.lower() in ("unknown", "مجهول", ""):
        norm_poet = ""
    norm_matla = _normalise(_first_n_tokens(matla, 10))
    if norm_poet:
        return f"{norm_poet} :: {norm_matla}"
    return norm_matla   # matla-only key when poet unknown


def _is_unknown_poet(poet: str) -> bool:
    """True when the poet field carries no useful identity information."""
    norm = _normalise(poet).lower()
    return norm in ("unknown", "مجهول", "")


# ── Load phase poems ──────────────────────────────────────────────────────────

def _load_phase_poems() -> list[dict]:
    """
    Why flatten: phase{1,2,3}_poems.json are dicts keyed by page_id (e.g.,
    "manuscript07_p0020"). We flatten to a list so the matcher can iterate
    without knowing the page structure.

    Each output dict has:
      verse_id      — the page key (used as the cross-link primary key)
      poet_name     — from poet.name
      matla_text    — best available matla: standard > dialectal > manuscript
      source_phase  — "phase1" | "phase2" | "phase3"
      source_volume — manuscript identifier
      source_page   — page number (int)
    """
    rows: list[dict] = []
    for phase_file in PHASE_FILES:
        if not phase_file.exists():
            logger.warning("Phase file not found, skipping: %s", phase_file)
            continue
        phase_label = phase_file.stem.split("_")[0]   # "phase1" / "phase2" / "phase3"
        with phase_file.open(encoding="utf-8") as f:
            data: dict = json.load(f)

        for page_key, entry in data.items():
            matla = entry.get("matla") or {}
            matla_text = (
                matla.get("standard") or
                matla.get("dialectal") or
                matla.get("manuscript") or
                ""
            )
            poet_info  = entry.get("poet") or {}
            poet_name  = poet_info.get("name", "") or ""

            rows.append({
                "verse_id":     page_key,
                "poet_name":    poet_name,
                "matla_text":   matla_text,
                "source_phase": phase_label,
                "source_volume": entry.get("source_volume", ""),
                "source_page":  entry.get("source_page"),
            })

    logger.info("Loaded %d verse entries from phase files.", len(rows))
    return rows


# ── Load anchor registry ──────────────────────────────────────────────────────

def _load_registry() -> list[dict]:
    with REGISTRY_FILE.open(encoding="utf-8") as f:
        entries: list[dict] = json.load(f)
    logger.info("Loaded %d entries from anchor registry.", len(entries))
    return entries


# ── Cross-link algorithm ──────────────────────────────────────────────────────

def _build_registry_index(registry: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Build a list of composite keys parallel to the registry list so rapidfuzz
    can do a single extractOne call per verse — O(V × R) in practice but
    vectorised inside rapidfuzz's C extension for speed.
    """
    keys: list[str] = []
    for entry in registry:
        poet   = entry.get("poet_name", "") or ""
        matla  = entry.get("matla_text", "") or ""
        keys.append(_composite_key(poet, matla))
    return keys, registry


def crosslink(
    verse_rows: Optional[list[dict]] = None,
    registry: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Why a function (not just __main__): tests call this directly with fixture
    data so the acceptance gate runs without touching the file system.

    Returns a list of crosslink rows:
      {
        verse_id, poet_name, matla_text, source_phase, source_volume, source_page,
        anchor_id, anchor_poet, anchor_matla, confidence, decision
      }
    """
    if verse_rows is None:
        verse_rows = _load_phase_poems()
    if registry is None:
        registry = _load_registry()

    reg_keys, reg_entries = _build_registry_index(registry)

    results: list[dict] = []

    for verse in verse_rows:
        verse_poet  = verse["poet_name"]
        verse_matla = verse["matla_text"]
        verse_key   = _composite_key(verse_poet, verse_matla)

        if not verse_key.strip() and not verse_matla.strip():
            # Both poet and matla are truly empty — cannot match at all
            results.append({
                **verse,
                "anchor_id":    None,
                "anchor_poet":  None,
                "anchor_matla": None,
                "confidence":   0.0,
                "decision":     "unlinked",
            })
            continue

        unknown_poet = _is_unknown_poet(verse_poet)

        # rapidfuzz.process.extractOne returns (best_match_key, score, index)
        match = rfprocess.extractOne(
            verse_key,
            reg_keys,
            scorer=fuzz.WRatio,
            score_cutoff=0,    # always return best, even if low
        )

        if match is None:
            best_score = 0.0
            best_idx   = None
        else:
            _, best_score, best_idx = match

        # Extra matla-level verification: when the poet is unknown the composite
        # key is matla-only, so the WRatio score can be artificially high if both
        # keys are short. We add a secondary check: the matla tokens must overlap
        # ≥ 50% with the candidate's matla tokens for the match to be trustworthy.
        if best_idx is not None and unknown_poet:
            candidate_matla = _normalise(reg_entries[best_idx].get("matla_text", ""))
            verse_matla_norm = _normalise(verse_matla)
            v_tokens = set(verse_matla_norm.split())
            c_tokens = set(candidate_matla.split())
            if v_tokens and c_tokens:
                token_overlap = len(v_tokens & c_tokens) / max(len(v_tokens), len(c_tokens))
            else:
                token_overlap = 0.0
            # If token overlap is too low, penalise the score
            if token_overlap < 0.30:
                best_score = min(best_score, REVIEW_THRESHOLD - 1)

        if best_idx is not None and best_score >= REVIEW_THRESHOLD:
            anchor     = reg_entries[best_idx]
            anchor_id  = anchor.get("source_row_id", "")
            anchor_poet = anchor.get("poet_name", "")
            anchor_matla = anchor.get("matla_text", "")
        else:
            anchor_id   = None
            anchor_poet = None
            anchor_matla = None

        if best_score >= LINKED_THRESHOLD:
            decision = "linked"
        elif best_score >= REVIEW_THRESHOLD:
            decision = "review"
        else:
            decision = "unlinked"

        results.append({
            "verse_id":     verse["verse_id"],
            "poet_name":    verse["poet_name"],
            "matla_text":   verse["matla_text"],
            "source_phase": verse["source_phase"],
            "source_volume": verse["source_volume"],
            "source_page":  verse["source_page"],
            "anchor_id":    anchor_id,
            "anchor_poet":  anchor_poet,
            "anchor_matla": anchor_matla,
            "confidence":   round(best_score / 100, 4),  # normalise to 0-1
            "decision":     decision,
        })

    # Log summary
    from collections import Counter
    decisions = Counter(r["decision"] for r in results)
    logger.info(
        "Crosslink complete — linked: %d, review: %d, unlinked: %d (total: %d)",
        decisions["linked"], decisions["review"], decisions["unlinked"], len(results),
    )
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Starting Al-Nassikh cross-link (M1)…")
    rows = crosslink()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d rows → %s", len(rows), OUTPUT_FILE)

    # Print a quick summary table
    from collections import Counter
    decisions = Counter(r["decision"] for r in rows)
    total = len(rows)
    print("\n── M1 Cross-link Summary ───────────────────────────────────────")
    print(f"  linked  (≥ 0.85) : {decisions['linked']:4d}  ({decisions['linked']/total:.0%})")
    print(f"  review  (0.70-0.85): {decisions['review']:4d}  ({decisions['review']/total:.0%})")
    print(f"  unlinked (< 0.70) : {decisions['unlinked']:4d}  ({decisions['unlinked']/total:.0%})")
    print(f"  total             : {total:4d}")

    # Show Phase-1-only acceptance gate stat
    phase1 = [r for r in rows if r["source_phase"] == "phase1"]
    if phase1:
        p1_linked = sum(1 for r in phase1 if r["decision"] == "linked")
        p1_rate = p1_linked / len(phase1)
        gate_ok = "✅ PASS" if p1_rate >= 0.70 else "❌ FAIL"
        print(f"\n  Phase-1 acceptance gate (≥ 70 % linked): "
              f"{p1_linked}/{len(phase1)} = {p1_rate:.0%}  {gate_ok}")
    print("────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
