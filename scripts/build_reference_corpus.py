#!/usr/bin/env python3
"""
scripts/build_reference_corpus.py
==================================
One-shot OCR + chunking pass over the scholarly PDFs in `manuscripts/`. Writes
`data/ground_truth/reference_corpus.json`.

Usage:
    PYTHONPATH=src python scripts/build_reference_corpus.py
    PYTHONPATH=src python scripts/build_reference_corpus.py --only ibn_khaldoun_abt_nabatipoetry

Prereqs:
    brew install tesseract tesseract-lang     # macOS
    pip install pymupdf pdf2image pytesseract # already in requirements.txt path

Why a script (not always-on): OCR'ing ~140 pages takes ~3-5 minutes. We do this
once when the PDFs change and treat the resulting JSON as the cached input to
scripts/rebuild_index.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from al_nassikh.reference_ingest import (  # noqa: E402
    BOOK_METADATA,
    build_reference_corpus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("build_reference_corpus")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Limit ingestion to these PDF basenames (without .pdf). May be passed multiple times.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=REPO_ROOT / "manuscripts",
        help="Directory containing the source PDFs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "ground_truth" / "reference_corpus.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if args.only:
        unknown = [n for n in args.only if n not in BOOK_METADATA]
        if unknown:
            logger.error("Unknown book short_keys: %s. Known: %s",
                         unknown, list(BOOK_METADATA))
            return 2
        names = [f"{n}.pdf" for n in args.only]
    else:
        names = None

    logger.info("Building reference corpus from PDFs in %s", args.pdf_dir)
    chunks = build_reference_corpus(args.pdf_dir, args.out, pdf_names=names)
    logger.info("Done — %d chunks written to %s", len(chunks), args.out)

    # Quick distribution summary
    by_book: dict[str, int] = {}
    for c in chunks:
        by_book[c["book_short_key"]] = by_book.get(c["book_short_key"], 0) + 1
    print("\n── Chunk distribution by book ──")
    for k, v in sorted(by_book.items()):
        print(f"  {k:<20} {v:>4} chunks")
    print(f"  {'TOTAL':<20} {len(chunks):>4} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
