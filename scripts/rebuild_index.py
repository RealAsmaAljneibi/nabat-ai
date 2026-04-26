#!/usr/bin/env python3
"""
scripts/rebuild_index.py
=========================
Why this script exists: the index build step is expensive (AraBERT encoding
over 4,500 chunks) and should only run when the registry data changes.
This CLI wraps fatat_al_arab.index.build_index() with progress reporting and
a --force flag so the archivist can trigger a rebuild without touching Python.

Usage:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --force
    python scripts/rebuild_index.py --registry data/ground_truth/anchor_registry_phase4.json
    python scripts/rebuild_index.py --qdrant data/qdrant --force

The script prints a summary table after build completes, including chunk
counts per level and the embedding matrix shape.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sure src/ is on the path when run from repo root
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from fatat_al_arab.index import (
    DEFAULT_REGISTRY,
    DEFAULT_QDRANT,
    build_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the NABAT-AI file-backed Qdrant index from the anchor registry."
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help=f"Path to anchor_registry_phase4.json (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--qdrant",
        default=str(DEFAULT_QDRANT),
        help=f"Output directory for Qdrant storage (default: {DEFAULT_QDRANT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-build even if index already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32).",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    qdrant_path   = Path(args.qdrant)

    if not registry_path.exists():
        print(f"ERROR: Registry file not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Registry : {registry_path}")
    print(f"Qdrant   : {qdrant_path}")
    print(f"Force    : {args.force}")
    print()

    t0 = time.perf_counter()
    bundle = build_index(
        registry_path=registry_path,
        qdrant_path=qdrant_path,
        force_rebuild=args.force,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - t0

    s = bundle.stats
    print("─" * 45)
    print(f"{'Registry entries':<28} {s.get('registry_entries', '—'):>8}")
    print(f"{'Total chunks':<28} {s['total_chunks']:>8}")
    print(f"  {'verse (بيت)':<26} {s.get('verse_chunks', 0):>8}")
    print(f"  {'group (مجموعة)':<26} {s.get('group_chunks', 0):>8}")
    print(f"  {'poem (قصيدة)':<26} {s.get('poem_chunks', 0):>8}")
    print(f"  {'manuscript (مخطوطة)':<26} {s.get('manuscript_chunks', 0):>8}")
    print(f"  {'poet (شاعر)':<26} {s.get('poet_chunks', 0):>8}")
    print(f"  {'era (حقبة)':<26} {s.get('era_chunks', 0):>8}")
    print(f"  {'genre (نوع)':<26} {s.get('genre_chunks', 0):>8}")
    print(f"  {'emotion (مشاعر)':<26} {s.get('emotion_chunks', 0):>8}")
    print(f"{'Embedding shape':<28} {str(s['embedding_shape']):>8}")
    if s.get("loaded_from_cache"):
        print("  (loaded from cache — use --force to re-encode)")
    print(f"{'Elapsed':<28} {elapsed:>6.1f}s")
    print("─" * 45)
    print("✓ Index ready.")


if __name__ == "__main__":
    main()
