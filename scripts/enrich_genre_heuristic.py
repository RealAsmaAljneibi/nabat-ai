#!/usr/bin/env python3
"""
scripts/enrich_genre_heuristic.py
==================================
Why this file exists: The retrieval layer wants `genre` and `emotions` on
every anchor so the Scholar Workbench can render facet filters and the
Self-Query extractor can emit structured filters. This script runs the
keyword heuristic from `al_nassikh.genre_heuristic` over every entry in
`anchor_registry_phase4.json` and writes an enriched copy alongside it.

Why alongside and not in-place: the original file is Phase-4 ground truth.
Overwriting it would conflate human TOC transcription with heuristic
classifier output. Writing `anchor_registry_phase4_enriched.json` keeps
provenance clean — `registry_join.py` and `embed.py` prefer the enriched
file when it exists, fall back to the original otherwise.

Usage:
    PYTHONPATH=src python3 scripts/enrich_genre_heuristic.py
    PYTHONPATH=src python3 scripts/enrich_genre_heuristic.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make the script runnable from the repo root without fiddling with PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from al_nassikh.genre_heuristic import classify   # noqa: E402
from al_nassikh.nabati_taxonomy import GENRES, EMOTIONS  # noqa: E402


GROUND_TRUTH = _REPO_ROOT / "data" / "ground_truth"
INPUT_FILE  = GROUND_TRUTH / "anchor_registry_phase4.json"
OUTPUT_FILE = GROUND_TRUTH / "anchor_registry_phase4_enriched.json"
AUDIT_FILE  = GROUND_TRUTH / "genre_enrichment_audit.json"


def enrich_one(anchor: dict) -> dict:
    """
    Add genre + emotion fields to one anchor. Preserves every original field.

    Why not mutate in place: some callers already hold references to the
    anchor dict; returning a new dict is predictable.
    """
    result = classify(
        matla_text = anchor.get("matla_text", ""),
        occasion   = anchor.get("occasion"),
    )
    return {
        **anchor,
        "genre":              result["genre"],
        "genre_confidence":   result["genre_confidence"],
        "genre_source":       "heuristic_v1",
        "genre_evidence":     result["genre_evidence"],
        "occasion_hint_used": result["occasion_hint"],
        "emotions":           result["emotions"],
    }


def build_audit(enriched: list[dict]) -> dict:
    """
    Compute the per-genre distribution, coverage rates, and agreement-with-TOC
    statistics. This dict is what the user reviews to decide if the heuristic
    is shippable before looking at individual verses.
    """
    n = len(enriched)
    genre_counter = Counter(a["genre"] for a in enriched)
    emotion_counter: Counter[str] = Counter()
    for a in enriched:
        emotion_counter.update(a["emotions"])

    with_occasion = [a for a in enriched if (a.get("occasion") or "").strip()]
    with_occasion_hint_used = [a for a in enriched if a.get("occasion_hint_used")]

    # Confidence histogram bucketed for readability
    conf_buckets = {"0.0": 0, "0.0-0.3": 0, "0.3-0.6": 0, "0.6-0.9": 0, "0.9-1.0": 0}
    for a in enriched:
        c = a["genre_confidence"]
        if c == 0.0:            conf_buckets["0.0"] += 1
        elif c < 0.3:           conf_buckets["0.0-0.3"] += 1
        elif c < 0.6:           conf_buckets["0.3-0.6"] += 1
        elif c < 0.9:           conf_buckets["0.6-0.9"] += 1
        else:                   conf_buckets["0.9-1.0"] += 1

    return {
        "total_anchors":        n,
        "coverage": {
            "classified":       n - genre_counter["غير_محدد"],
            "abstained":        genre_counter["غير_محدد"],
            "abstain_rate":     round(genre_counter["غير_محدد"] / n, 3),
        },
        "genre_distribution": {
            g: genre_counter[g] for g in GENRES
        },
        "emotion_distribution": {
            e: emotion_counter[e] for e in EMOTIONS
        },
        "verses_with_any_emotion":     sum(1 for a in enriched if a["emotions"]),
        "verses_with_occasion_field":  len(with_occasion),
        "occasion_hint_used_count":    len(with_occasion_hint_used),
        "confidence_histogram":        conf_buckets,
        "classifier_version":          "heuristic_v1",
        "notes": [
            "No human-labelled gold set — we cannot report accuracy numbers.",
            "Confidence is margin-based, not probabilistic. Treat it as a relative ranking.",
            "The UI should surface 🔸 on any tag and explicitly state this is a keyword classifier.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print audit without writing files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N anchors (for testing).")
    parser.add_argument("--registry", default=None,
                        help="Custom input registry path (default: anchor_registry_phase4.json).")
    parser.add_argument("--output", default=None,
                        help="Custom output path (default: anchor_registry_phase4_enriched.json).")
    args = parser.parse_args()

    input_file  = Path(args.registry) if args.registry else INPUT_FILE
    output_file = Path(args.output)   if args.output   else OUTPUT_FILE
    audit_file  = output_file.with_name(output_file.stem + "_audit.json")

    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    with input_file.open(encoding="utf-8") as f:
        anchors = json.load(f)

    if args.limit:
        anchors = anchors[: args.limit]
    print(f"→ classifying {len(anchors)} anchors …")

    enriched = [enrich_one(a) for a in anchors]
    audit = build_audit(enriched)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Pretty-print the audit to stdout — this is the primary review surface
    print()
    print("=" * 72)
    print(f" GENRE + EMOTION ENRICHMENT — heuristic_v1")
    print("=" * 72)
    print(f" Total anchors        {audit['total_anchors']:>6}")
    print(f" Classified           {audit['coverage']['classified']:>6}"
          f"   ({100 * (1 - audit['coverage']['abstain_rate']):.1f}%)")
    print(f" Abstained            {audit['coverage']['abstained']:>6}"
          f"   ({100 * audit['coverage']['abstain_rate']:.1f}%)")
    print(f" Elapsed              {elapsed_ms:>6.1f} ms")
    print()
    print(" Genre distribution:")
    for g in GENRES:
        n = audit["genre_distribution"][g]
        bar = "█" * (n // 15)
        print(f"   {g:<10}  {n:>5}   {bar}")
    print()
    print(" Emotion distribution (multi-label):")
    for e in EMOTIONS:
        n = audit["emotion_distribution"][e]
        bar = "█" * (n // 15)
        print(f"   {e:<10}  {n:>5}   {bar}")
    print()
    print(" Confidence histogram:")
    for bucket, n in audit["confidence_histogram"].items():
        print(f"   {bucket:<10}  {n:>5}")
    print()
    print(f" Occasion-field hints used on {audit['occasion_hint_used_count']} anchors "
          f"(of {audit['verses_with_occasion_field']} with a non-empty occasion)")
    print()

    if args.dry_run:
        print("(dry-run — no files written)")
        return 0

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    with audit_file.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"✅  wrote {len(enriched)} enriched anchors → {output_file.name}")
    print(f"✅  wrote audit summary                  → {audit_file.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
