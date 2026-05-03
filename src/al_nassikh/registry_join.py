"""
src/al_nassikh/registry_join.py
================================
Why this file exists: §M1 — anchor_registry_phase4.json stores raw filenames
('manuscript06', '601-782') but the UI and citation builder need human-readable
names ("Al-Daoud Manuscript", "Ibn Yahya Manuscript (601-842)"). This one-shot
script enriches every anchor with three new fields by joining against
manuscript_registry.json via manuscript_filename_map.json.
Where it's called: Standalone build step
Purpose: Enriched the registry with human-readable manuscript names — already ran

Why a separate script (not done inside phase4_merger): the merger runs once per
eScriptorium export and is owned by Al-Nassikh's ETL pipeline. The registry join
is a M1-onward post-process that must be re-runnable whenever manuscript_registry
or manuscript_filename_map changes, without touching the merger.

Output: enriches anchor_registry_phase4.json in-place (with backup).
Audit:  writes data/ground_truth/registry_join_audit.json listing any anchor
        whose source_volume had no mapping in manuscript_filename_map.json.

Run with:
    python -m src.al_nassikh.registry_join      # from repo root
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import sys as _sys, os as _os
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from al_nassikh.registry import by_filename, list_all

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ── Path resolution ───────────────────────────────────────────────────────────

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path(os.getcwd())


REPO_ROOT        = _repo_root()
GROUND_TRUTH     = REPO_ROOT / "data" / "ground_truth"
REGISTRY_IN      = GROUND_TRUTH / "anchor_registry_phase4.json"
AUDIT_OUT        = GROUND_TRUTH / "registry_join_audit.json"


# ── Join logic ────────────────────────────────────────────────────────────────

def run_registry_join(
    registry_path: Path = REGISTRY_IN,
    audit_path: Path = AUDIT_OUT,
    dry_run: bool = False,
) -> dict:
    """
    Enrich every anchor in anchor_registry_phase4.json with:
      manuscript_short_key    — pipeline identifier (e.g., 'al_daoud')
      manuscript_arabic_name  — Arabic display name (e.g., 'مخطوطة الداود')
      manuscript_english_name — English display name (e.g., 'Al-Daoud Manuscript')

    Returns a summary dict with counts.
    """
    with registry_path.open(encoding="utf-8") as f:
        anchors: list[dict] = json.load(f)

    logger.info("Loaded %d anchors from %s", len(anchors), registry_path)

    enriched: list[dict] = []
    unmatched_stems: set[str] = set()
    matched_count   = 0
    unmatched_count = 0

    for anchor in anchors:
        # Prefer source_volume (cleaner); fall back to source_image_path
        source = anchor.get("source_volume") or anchor.get("source_image_path", "")
        ms_entry = by_filename(source) if source else None

        if ms_entry:
            anchor["manuscript_short_key"]    = ms_entry["short_key"]
            anchor["manuscript_arabic_name"]  = ms_entry["arabic_name"]
            anchor["manuscript_english_name"] = ms_entry["english_name"]
            matched_count += 1
        else:
            # Leave fields absent rather than setting None — downstream checks
            # can distinguish "never ran" from "ran but no match" by presence.
            from pathlib import Path as _P
            stem = _P(source).stem.split("_p")[0] if source else "<empty>"
            unmatched_stems.add(stem)
            unmatched_count += 1

        enriched.append(anchor)

    # ── Audit file ────────────────────────────────────────────────────────────
    audit: dict = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_anchors": len(anchors),
        "matched":        matched_count,
        "unmatched":      unmatched_count,
        "unmatched_stems": sorted(unmatched_stems),
        "acceptance_gate_passes": unmatched_count == 0,
    }
    if not dry_run:
        with audit_path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
        logger.info("Audit written → %s", audit_path)

    # ── Write enriched registry (with backup) ────────────────────────────────
    if not dry_run:
        backup_path = registry_path.with_suffix(
            f".backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(registry_path, backup_path)
        logger.info("Backup: %s", backup_path)

        with registry_path.open("w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        logger.info(
            "Enriched registry written → %s  (%d matched, %d unmatched)",
            registry_path, matched_count, unmatched_count,
        )

    return audit


def main() -> None:
    logger.info("Starting registry_join (M1)…")
    result = run_registry_join()

    print("\n── M1 Registry Join Summary ────────────────────────────────────")
    print(f"  Total anchors   : {result['total_anchors']}")
    print(f"  Matched         : {result['matched']}")
    print(f"  Unmatched       : {result['unmatched']}")
    if result["unmatched_stems"]:
        print(f"  Unmatched stems : {result['unmatched_stems']}")
        print("  ⚠️  Add these stems to manuscript_filename_map.json to clear M1 gate.")
    gate = "✅ PASS" if result["acceptance_gate_passes"] else "❌ FAIL"
    print(f"  Acceptance gate (0 unmatched): {gate}")
    print("────────────────────────────────────────────────────────────────\n")

    if not result["acceptance_gate_passes"]:
        raise SystemExit(
            "registry_join: unmatched anchors detected — M1 acceptance gate FAIL. "
            "Fix manuscript_filename_map.json, then re-run."
        )


if __name__ == "__main__":
    main()
