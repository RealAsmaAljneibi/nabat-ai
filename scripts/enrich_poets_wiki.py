"""
scripts/enrich_poets_wiki.py
==============================
Why this file exists: 480 of the 510 distinct poet names in the registry
have no entry in poets_bio.json. This script auto-enriches them via
Wikipedia (AR first, EN fallback) and Brave Search (if BRAVE_API_KEY is set),
then writes a merged bio file.

The original poets_bio.json is NEVER overwritten. The output is:
    data/ground_truth/poets_bio_wiki_enriched.json

That file is a union of:
  - All 509 entries from poets_bio.json (original, higher quality)
  - New entries fetched from Wikipedia / Brave for the remaining 480 poets

Usage:
    # Full run (all missing poets):
    PYTHONPATH=src python scripts/enrich_poets_wiki.py

    # Preview without any network calls:
    PYTHONPATH=src python scripts/enrich_poets_wiki.py --dry-run

    # Limit to the first N missing poets (useful for testing quota):
    PYTHONPATH=src python scripts/enrich_poets_wiki.py --limit 20

    # Re-enrich poets already in poets_bio.json (merge with new sources):
    PYTHONPATH=src python scripts/enrich_poets_wiki.py --force

    # Use Brave Search as primary (needs BRAVE_API_KEY in .env):
    BRAVE_API_KEY=<key> PYTHONPATH=src python scripts/enrich_poets_wiki.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

from al_nassikh.poet_enrichment import PoetEnrichmentClient, is_valid_poet_name

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_DATA_DIR    = _REPO_ROOT / "data" / "ground_truth"
_REGISTRY    = _DATA_DIR / "anchor_registry_full_enriched.json"
_BIO_ORIG    = _DATA_DIR / "poets_bio.json"
_BIO_OUT     = _DATA_DIR / "poets_bio_wiki_enriched.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_registry_poets() -> list[str]:
    """Return all distinct non-empty poet names from the registry, sorted."""
    with open(_REGISTRY, encoding="utf-8") as fh:
        entries = json.load(fh)
    names = {e.get("poet_name", "").strip() for e in entries}
    names.discard("")
    return sorted(names)


def _load_bio_index(bio_path: Path) -> dict[str, dict]:
    """Return poets_bio.json as a dict keyed by poet_name."""
    if not bio_path.exists():
        return {}
    with open(bio_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    return {e["poet_name"]: e for e in entries}


def _report_row(status: str, name: str, detail: str = "") -> None:
    icon = {"found": "✅", "skipped": "⏭ ", "noise": "🗑 ", "missing": "❌"}.get(status, "  ")
    detail_str = f"  ({detail})" if detail else ""
    logger.info("%s  %-40s%s", icon, name, detail_str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich poets_bio.json with Wikipedia/Brave data.")
    parser.add_argument("--dry-run",  action="store_true", help="Print plan without making API calls.")
    parser.add_argument("--limit",    type=int, default=None, help="Max number of poets to enrich.")
    parser.add_argument("--force",    action="store_true", help="Re-enrich poets already in poets_bio.json.")
    args = parser.parse_args()

    # ── Load existing data ────────────────────────────────────────────────────
    all_poets   = _load_registry_poets()
    bio_index   = _load_bio_index(_BIO_ORIG)
    merged      = dict(bio_index)   # start with all originals (preserved intact)

    logger.info("Registry poets   : %d", len(all_poets))
    logger.info("Existing bio file: %d entries", len(bio_index))

    client = PoetEnrichmentClient()
    if not client.is_available():
        logger.error("httpx is not installed. Run: pip install httpx")
        sys.exit(1)
    if client.has_brave():
        logger.info("Brave Search     : enabled (BRAVE_API_KEY set)")
    else:
        logger.info("Brave Search     : disabled (no BRAVE_API_KEY) — Wikipedia only")

    # ── Build work list ───────────────────────────────────────────────────────
    to_enrich: list[str] = []
    noise_count   = 0
    already_count = 0

    for name in all_poets:
        if not is_valid_poet_name(name):
            noise_count += 1
            _report_row("noise", name, "OCR artifact — skipped")
            continue
        if name in bio_index and not args.force:
            already_count += 1
            continue
        to_enrich.append(name)

    if args.limit:
        to_enrich = to_enrich[: args.limit]

    logger.info("Already have bio : %d", already_count)
    logger.info("OCR noise skipped: %d", noise_count)
    logger.info("To enrich        : %d%s", len(to_enrich),
                f"  (capped at --limit {args.limit})" if args.limit else "")

    if args.dry_run:
        logger.info("── DRY RUN — no network calls will be made ──")
        for name in to_enrich[:30]:
            logger.info("  would enrich: %s", name)
        if len(to_enrich) > 30:
            logger.info("  … and %d more", len(to_enrich) - 30)
        return

    # ── Enrich loop ───────────────────────────────────────────────────────────
    found_count   = 0
    missing_count = 0

    for i, name in enumerate(to_enrich, start=1):
        logger.info("[%d/%d] Enriching: %s", i, len(to_enrich), name)
        result = client.enrich(name)

        if result:
            found_count += 1
            via = result.get("enriched_via", "?")
            _report_row("found", name, via)
            # Merge: if existing entry in bio_index, prefer the original bio
            # text but add new metadata fields (region, birth_year_approx)
            if name in merged:
                existing = dict(merged[name])
                for field in ("birth_year_approx", "region", "enriched_via", "enriched_at"):
                    if field in result and field not in existing:
                        existing[field] = result[field]
                if not existing.get("sources"):
                    existing["sources"] = result.get("sources", [])
                merged[name] = existing
            else:
                merged[name] = result
        else:
            missing_count += 1
            _report_row("missing", name)

        # Brief pause to avoid hitting rate limits on batch runs
        time.sleep(0.2)

    # ── Write output ──────────────────────────────────────────────────────────
    output = sorted(merged.values(), key=lambda e: e.get("poet_name", ""))

    with open(_BIO_OUT, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════")
    logger.info("Output file     : %s", _BIO_OUT.relative_to(_REPO_ROOT))
    logger.info("Total bio entries: %d  (was %d)", len(output), len(bio_index))
    logger.info("Newly enriched  : %d", found_count)
    logger.info("Not found       : %d", missing_count)
    logger.info("Cache size      : %d entries", client.cache_size())
    logger.info("══════════════════════════════════════════")
    logger.info("")
    logger.info("Next step: review poets_bio_wiki_enriched.json, then rename it")
    logger.info("to poets_bio.json when satisfied with the quality.")


if __name__ == "__main__":
    main()
