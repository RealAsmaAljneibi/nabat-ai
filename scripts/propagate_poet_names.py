#!/usr/bin/env python3
"""
scripts/propagate_poet_names.py
================================
Why this script exists: Phase 1/2/3 PAGE-XML entries were ingested with
poet_name="غير محدد" because the XML files carry text only — no poet attribution.
The Phase 4 TOC registry (anchor_registry_phase4_enriched.json) has poet names
for first verses. This script bridges the two:

  1. For each page in phase1/2/3_poems.json, take the manually-annotated matla
     text (first verse on that page).
  2. Fuzzy-match it against Phase 4 TOC entries from the SAME manuscript using
     rapidfuzz WRatio (same normalisation as cross_link.py).
  3. When score ≥ LINKED_THRESHOLD (0.70), copy the poet name from the TOC entry
     to ALL verse entries for that page in anchor_registry_full_enriched.json.
  4. Write the updated registry and print a coverage report.

Why page-level matching (not verse-level): the phase JSON files record the
FIRST verse (matla) of the poem per page — exactly what Phase 4 TOC stores.
Matching on the matla gives the highest signal. Once the page's poet is
established, all verses on that page inherit the attribution.

Usage:
    python scripts/propagate_poet_names.py [--dry-run]
    python scripts/propagate_poet_names.py --threshold 0.70
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rapidfuzz import fuzz, process as rfprocess
except ImportError:
    sys.exit("rapidfuzz not installed. Run: pip install rapidfuzz")

_REPO    = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO / "data" / "ground_truth"

PHASE_FILES = [
    DATA_DIR / "phase1_poems.json",
    DATA_DIR / "phase2_poems.json",
    DATA_DIR / "phase3_poems.json",
]
PHASE4_REGISTRY = DATA_DIR / "anchor_registry_phase4_enriched.json"
FULL_REGISTRY   = DATA_DIR / "anchor_registry_full_enriched.json"

# Minimum WRatio score (0-100 scale) to accept a match
DEFAULT_THRESHOLD = 70

# ── Arabic normalisation (mirrors cross_link.py) ──────────────────────────────

_ALEF_VARIANTS = "أإآٱ"
_HARAKAT_RE    = re.compile(r"[\u064B-\u065F\u0670]")
_PUNCT_RE      = re.compile(r"[^\u0600-\u06FF\s]")


def _normalise(text: str) -> str:
    if not text:
        return ""
    for v in _ALEF_VARIANTS:
        text = text.replace(v, "ا")
    text = text.replace("ة", "ه")
    text = _HARAKAT_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_useless(text: str) -> bool:
    """True for empty, page-number-only, or very short (< 4 Arabic chars) texts."""
    norm = _normalise(text)
    if not norm:
        return True
    arabic_chars = [c for c in norm if "\u0600" <= c <= "\u06FF"]
    return len(arabic_chars) < 4


def _is_useless_poet(name: str) -> bool:
    norm = _normalise(name).lower()
    return norm in ("", "unknown", "مجهول", "غير محدد", "٠٠")


# ── Page-key normalisation ────────────────────────────────────────────────────
# phase_poems.json keys:  "manuscript05_p0015"  (zero-padded 4-digit page)
# PAGE-XML source_page_key: "manuscript05_p15"  (bare page number, no padding)

def _page_key_to_stem_page(key: str) -> tuple[str, int]:
    """Extract (manuscript_stem, page_int) from either key format."""
    m = re.match(r"^(manuscript\d+)_p0*(\d+)$", key)
    if m:
        return m.group(1), int(m.group(2))
    m2 = re.match(r"^(manuscript\d+)_p(\d+)$", key)
    if m2:
        return m2.group(1), int(m2.group(2))
    return key, 0


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_phase_matlas() -> dict[tuple[str, int], str]:
    """
    Return {(manuscript_stem, page_int) → matla_text} from phase JSON files.
    Only pages with a usable matla text (≥ 4 Arabic chars) are included.
    """
    result: dict[tuple[str, int], str] = {}
    for phase_file in PHASE_FILES:
        if not phase_file.exists():
            print(f"  WARNING: {phase_file.name} not found — skipping")
            continue
        with phase_file.open(encoding="utf-8") as f:
            data: dict = json.load(f)
        for page_key, entry in data.items():
            matla_obj  = entry.get("matla") or {}
            matla_text = (
                matla_obj.get("standard") or
                matla_obj.get("dialectal") or
                matla_obj.get("manuscript") or ""
            )
            if _is_useless(matla_text):
                continue
            stem, page = _page_key_to_stem_page(page_key)
            result[(stem, page)] = matla_text
    print(f"  Loaded {len(result)} usable page matlas from phase JSON files.")
    return result


def _load_phase4_by_manuscript(
) -> dict[str, list[dict]]:
    """
    Return {manuscript_short_key → [Phase-4 entries with usable poet + matla]}.
    Entries whose poet is useless or whose matla is empty are skipped.
    """
    with PHASE4_REGISTRY.open(encoding="utf-8") as f:
        entries: list[dict] = json.load(f)

    by_ms: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        poet  = e.get("poet_name") or ""
        matla = e.get("matla_text") or ""
        if _is_useless_poet(poet) or _is_useless(matla):
            continue
        sk = e.get("manuscript_short_key", "")
        if sk:
            by_ms[sk].append(e)

    for sk, lst in by_ms.items():
        print(f"    {sk:<35} {len(lst):>4} matchable Phase-4 entries")
    return dict(by_ms)


# ── Filename-map helper ───────────────────────────────────────────────────────

def _load_stem_to_short_key() -> dict[str, str]:
    fname = DATA_DIR / "manuscript_filename_map.json"
    with fname.open(encoding="utf-8") as f:
        entries = json.load(f)
    return {e["filename_stem"]: e["short_key"] for e in entries}


# ── Matching ──────────────────────────────────────────────────────────────────

def _match_matla(
    query_matla: str,
    candidates: list[dict],
    threshold: int,
) -> dict | None:
    """
    Return the best-matching Phase-4 entry for query_matla, or None.
    Uses rapidfuzz WRatio on normalised matla text.
    """
    if not candidates:
        return None
    norm_query = _normalise(query_matla)
    if not norm_query:
        return None

    norm_keys = [_normalise(c.get("matla_text") or "") for c in candidates]

    match = rfprocess.extractOne(
        norm_query,
        norm_keys,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    if match is None:
        return None
    _, score, idx = match
    return candidates[idx]


# ── Main propagation ──────────────────────────────────────────────────────────

def propagate(threshold: int = DEFAULT_THRESHOLD, dry_run: bool = False) -> None:
    print("\nLoading phase matlas …")
    page_matlas = _load_phase_matlas()        # {(stem, page) → matla_text}

    print("\nLoading Phase-4 TOC entries by manuscript …")
    phase4_by_ms = _load_phase4_by_manuscript()

    stem_to_sk = _load_stem_to_short_key()   # manuscript07 → al_shuraiti

    # Build the page → poet_name mapping by matching
    print(f"\nMatching page matlas against Phase-4 TOC (threshold={threshold}/100) …")
    page_to_poet: dict[tuple[str, int], str] = {}       # {(stem, page) → poet_name}
    page_to_confidence: dict[tuple[str, int], float] = {}

    matched = unmatched = no_coverage = 0

    for (stem, page), matla in sorted(page_matlas.items()):
        sk = stem_to_sk.get(stem)
        if sk is None or sk not in phase4_by_ms:
            no_coverage += 1
            continue   # manuscript has no Phase-4 coverage

        best = _match_matla(matla, phase4_by_ms[sk], threshold)
        if best is None:
            unmatched += 1
            continue

        poet = best.get("poet_name", "")
        conf = fuzz.WRatio(_normalise(matla), _normalise(best.get("matla_text") or ""))
        page_to_poet[(stem, page)]      = poet
        page_to_confidence[(stem, page)] = round(conf / 100, 3)
        matched += 1
        print(f"  ✓ {stem}_p{page:04d}  →  {poet[:35]}  (score={conf:.0f})")

    print(f"\n  Matched:     {matched}")
    print(f"  Unmatched:   {unmatched}")
    print(f"  No coverage: {no_coverage} (manuscripts absent from Phase-4 TOC)")

    if not page_to_poet:
        print("\nNo matches found — nothing to propagate.")
        return

    # Load the full registry and apply poet names
    print(f"\nLoading full registry: {FULL_REGISTRY.name} …")
    with FULL_REGISTRY.open(encoding="utf-8") as f:
        registry: list[dict] = json.load(f)

    updated = 0
    skipped_already_known = 0

    for entry in registry:
        # Only touch Phase 1-2-3 entries whose poet is still unknown
        if entry.get("content_phase") not in ("phase1", "phase2", "phase3"):
            continue
        if not _is_useless_poet(entry.get("poet_name", "")):
            skipped_already_known += 1
            continue

        source_page_key = entry.get("source_page_key", "")  # e.g. "manuscript22_p12"
        stem, page = _page_key_to_stem_page(source_page_key)

        poet = page_to_poet.get((stem, page))
        if poet:
            entry["poet_name"]            = poet
            entry["poet_attribution"]     = "phase4_crosslink"
            entry["poet_conf"]            = page_to_confidence.get((stem, page), 0.0)
            updated += 1

    print(f"\n  Verse entries updated:       {updated}")
    print(f"  Entries already attributed:  {skipped_already_known}")
    print(f"  Still unknown:               {sum(1 for e in registry if e.get('content_phase') in ('phase1','phase2','phase3') and _is_useless_poet(e.get('poet_name','')))} ")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    with FULL_REGISTRY.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Updated registry written → {FULL_REGISTRY.name}")
    print(f"\nNext step: python scripts/rebuild_index.py --registry {FULL_REGISTRY} --force")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate poet names from Phase-4 TOC to Phase 1/2/3 verse entries."
    )
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would be updated without writing.")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"RapidFuzz WRatio score threshold 0-100 (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()
    propagate(threshold=args.threshold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
