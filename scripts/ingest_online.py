#!/usr/bin/env python3
"""
scripts/ingest_online.py
========================
Why this script exists: runs the online corpus ingest pipeline (Worker 1
extension) and writes data/online_corpus/online_anchor_registry.json.
After running, rebuild the Qdrant index with rebuild_index.py so the
new entries become searchable.

Usage:
    python scripts/ingest_online.py
    python scripts/ingest_online.py --no-aldiwan      # skip aldiwan.net
    python scripts/ingest_online.py --no-4byt         # skip 4byt.com
    python scripts/ingest_online.py --max-per-poet 5  # fewer poems per poet
    python scripts/ingest_online.py --pages 2         # fewer 4byt pages

Requires: requests, beautifulsoup4 (both in requirements.txt).
Rate-limited to 1.2 s per request — a full run takes ~5–10 minutes.

After ingest, re-run the index builder:
    python scripts/rebuild_index.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch online Khaleeji/Nabati poetry corpus and write to data/online_corpus/."
    )
    parser.add_argument("--no-aldiwan",  action="store_true", help="Skip aldiwan.net scrape")
    parser.add_argument("--no-4byt",     action="store_true", help="Skip 4byt.com scrape")
    parser.add_argument("--no-ecssr",    action="store_true", help="Skip ECSSR scrape")
    parser.add_argument("--max-per-poet", type=int, default=15,
                        help="Max poems per poet from aldiwan.net (default 15)")
    parser.add_argument("--pages",       type=int, default=5,
                        help="Category pages to scrape from 4byt.com (default 5)")
    args = parser.parse_args()

    from al_nassikh.online_ingest import ingest_online_corpus

    print("Starting online corpus ingest…")
    entries = ingest_online_corpus(
        include_aldiwan=not args.no_aldiwan,
        include_4byt=not args.no_4byt,
        include_ecssr=not args.no_ecssr,
        max_poems_per_poet=args.max_per_poet,
        pages_4byt=args.pages,
    )

    print(f"\nIngested {len(entries)} online entries.")
    print("Next step: rebuild the Qdrant index:")
    print("  python scripts/rebuild_index.py --force")


if __name__ == "__main__":
    main()
