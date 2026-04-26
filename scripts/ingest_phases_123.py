#!/usr/bin/env python3
"""
scripts/ingest_phases_123.py
============================
Why this script exists: Phase 1, 2, and 3 ground-truth exports live as PAGE-XML
files in manuscripts/Ground_Truth_Exports but were never converted into the anchor
registry format. This script parses every Phase 1/2/3 PAGE-XML, pairs verse
hemistichs (sadr + ajuz) by their Y-baseline proximity, creates anchor entries in
the same schema as anchor_registry_phase4_enriched.json, and merges them into a
new combined registry file ready for rebuild_index.py.

Phase 4 XMLs are intentionally SKIPPED — anchor_registry_phase4.json was already
built from that data. The merged output is:
  data/ground_truth/anchor_registry_full.json       ← combined Phase 1-4
  data/ground_truth/anchor_registry_full_enriched.json ← after genre enrichment

Usage:
    python scripts/ingest_phases_123.py [--dry-run]
    python scripts/ingest_phases_123.py --output data/ground_truth/anchor_registry_full.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# ── Paths ─────────────────────────────────────────────────────────────────────

GT_DIR   = _REPO / "manuscripts" / "Ground_Truth_Exports"
DATA_DIR = _REPO / "data" / "ground_truth"

PHASE1_DIR = GT_DIR / "export_doc5_phase_1_pagexml_20260414140737"
PHASE2_DIR = GT_DIR / "export_doc6_phase_2_pagexml_20260419125859"
PHASE3_DIR = GT_DIR / "export_doc7_phase_3_pagexml_20260419125602"

EXISTING_ENRICHED  = DATA_DIR / "anchor_registry_phase4_enriched.json"
FILENAME_MAP       = DATA_DIR / "manuscript_filename_map.json"
MANUSCRIPT_REGISTRY = DATA_DIR / "manuscript_registry.json"

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Y-distance threshold (pixels) for pairing right/left hemistichs of one verse
PAIR_Y_THRESHOLD = 80


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_filename_map() -> dict[str, str]:
    """Return {filename_stem → short_key} dict from manuscript_filename_map.json."""
    with open(FILENAME_MAP, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["filename_stem"]: e["short_key"] for e in entries}


def _load_manuscript_names() -> dict[str, dict[str, Any]]:
    """
    Return {short_key → {ar, en, circa_date_start, circa_date_end, region, collector}}
    loaded from manuscript_registry.json — the authoritative source for all
    manuscript metadata. Never falls back to hardcoded strings.
    """
    with open(MANUSCRIPT_REGISTRY, encoding="utf-8") as f:
        entries = json.load(f)
    return {
        e["short_key"]: {
            "ar":               e.get("arabic_name", e["short_key"]),
            "en":               e.get("english_name", e["short_key"]),
            "circa_date_start": e.get("circa_date_start"),
            "circa_date_end":   e.get("circa_date_end"),
            "region":           e.get("region_of_origin", ""),
            "collector":        e.get("collector", ""),
        }
        for e in entries
    }


def _baseline_y(line_elem: ET.Element) -> float:
    """Extract the mean Y value of a Baseline element, or fallback to Coords."""
    ns = PAGE_NS
    baseline = line_elem.find(f"{{{ns}}}Baseline")
    if baseline is not None:
        pts_str = baseline.get("points", "")
        ys = [int(p.split(",")[1]) for p in pts_str.split() if "," in p]
        if ys:
            return sum(ys) / len(ys)
    # Fallback: Coords bounding-box centre Y
    coords = line_elem.find(f"{{{ns}}}Coords")
    if coords is not None:
        pts_str = coords.get("points", "")
        ys = [int(p.split(",")[1]) for p in pts_str.split() if "," in p]
        if ys:
            return (min(ys) + max(ys)) / 2
    return 0.0


def _baseline_x_centre(line_elem: ET.Element) -> float:
    """Return the mean X of the Baseline (or Coords) to determine left vs right."""
    ns = PAGE_NS
    baseline = line_elem.find(f"{{{ns}}}Baseline")
    if baseline is not None:
        pts_str = baseline.get("points", "")
        xs = [int(p.split(",")[0]) for p in pts_str.split() if "," in p]
        if xs:
            return sum(xs) / len(xs)
    coords = line_elem.find(f"{{{ns}}}Coords")
    if coords is not None:
        pts_str = coords.get("points", "")
        xs = [int(p.split(",")[0]) for p in pts_str.split() if "," in p]
        if xs:
            return sum(xs) / len(xs)
    return 0.0


def _bbox_from_coords(line_elem: ET.Element) -> list[int]:
    """Return [x_min, y_min, x_max, y_max] bounding box from Coords points."""
    ns = PAGE_NS
    coords = line_elem.find(f"{{{ns}}}Coords")
    if coords is None:
        return [0, 0, 0, 0]
    pts_str = coords.get("points", "")
    pts = [p.split(",") for p in pts_str.split() if "," in p]
    if not pts:
        return [0, 0, 0, 0]
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _get_text(line_elem: ET.Element) -> str:
    """Extract Unicode text from a TextLine element."""
    ns = PAGE_NS
    te = line_elem.find(f"{{{ns}}}TextEquiv")
    if te is None:
        return ""
    uc = te.find(f"{{{ns}}}Unicode")
    if uc is None or uc.text is None:
        return ""
    return uc.text.strip()


def _page_width(page_elem: ET.Element) -> int:
    """Return the page width from the Page element attribute."""
    try:
        return int(page_elem.get("imageWidth", "2000"))
    except ValueError:
        return 2000


def _parse_pagexml(xml_path: Path) -> list[dict[str, Any]]:
    """
    Parse one PAGE-XML file and return a list of verse dicts.

    Each verse dict:
        {
          "sadr":   str,   # right hemistich (first half)
          "ajuz":   str,   # left hemistich (second half)
          "full":   str,   # sadr + " || " + ajuz  (or just sadr if no pair)
          "bbox":   [x1,y1,x2,y2],
          "seq":    int,   # verse sequence on this page (1-based)
        }

    Pairing strategy:
      1. Collect all TextLines with non-empty text.
      2. Sort by Y baseline.
      3. Greedily group lines whose Y is within PAIR_Y_THRESHOLD of a group centre.
      4. Within each group, split into right-half (sadr) and left-half (ajuz) by
         comparing X centre to the page midpoint.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"  WARNING: could not parse {xml_path.name}: {exc}")
        return []

    root = tree.getroot()
    ns   = PAGE_NS

    page_elem = root.find(f"{{{ns}}}Page")
    if page_elem is None:
        return []

    page_mid = _page_width(page_elem) / 2

    # Gather all TextLines across all TextRegions
    lines: list[tuple[float, float, ET.Element]] = []
    for region in page_elem.iter(f"{{{ns}}}TextRegion"):
        for line in region.iter(f"{{{ns}}}TextLine"):
            text = _get_text(line)
            if not text:
                continue
            y = _baseline_y(line)
            x = _baseline_x_centre(line)
            lines.append((y, x, line))

    if not lines:
        return []

    # Sort top-to-bottom
    lines.sort(key=lambda t: t[0])

    # Greedy Y-grouping into verse rows
    groups: list[list[tuple[float, float, ET.Element]]] = []
    for y, x, elem in lines:
        placed = False
        for g in groups:
            g_y_mean = sum(t[0] for t in g) / len(g)
            if abs(y - g_y_mean) <= PAIR_Y_THRESHOLD:
                g.append((y, x, elem))
                placed = True
                break
        if not placed:
            groups.append([(y, x, elem)])

    # Build verse list from groups
    verses: list[dict[str, Any]] = []
    for seq, group in enumerate(groups, start=1):
        # Split by left vs right half
        right_lines = [t for t in group if t[1] >= page_mid]
        left_lines  = [t for t in group if t[1] < page_mid]

        # Concatenate all text in each side (multiple words/line fragments)
        sadr = " ".join(_get_text(t[2]) for t in right_lines).strip()
        ajuz = " ".join(_get_text(t[2]) for t in left_lines).strip()

        # Full verse text
        if sadr and ajuz:
            full = f"{sadr} || {ajuz}"
        elif sadr:
            full = sadr
        else:
            full = ajuz

        if not full.strip():
            continue

        # Bounding box: union of all lines in group
        all_bboxes = [_bbox_from_coords(t[2]) for t in group]
        x1 = min(b[0] for b in all_bboxes)
        y1 = min(b[1] for b in all_bboxes)
        x2 = max(b[2] for b in all_bboxes)
        y2 = max(b[3] for b in all_bboxes)

        verses.append({
            "sadr":  sadr,
            "ajuz":  ajuz,
            "full":  full,
            "bbox":  [x1, y1, x2, y2],
            "seq":   seq,
        })

    return verses


def _parse_page_num(filename_stem: str) -> int:
    """Extract page number from stems like 'manuscript07_p20' → 20."""
    m = re.search(r"_p(\d+)$", filename_stem)
    return int(m.group(1)) if m else 0


def _parse_ms_stem(filename_stem: str) -> str:
    """Extract manuscript stem from stems like 'manuscript07_p20' → 'manuscript07'."""
    return re.sub(r"_p\d+$", "", filename_stem)


def ingest_phase_dir(
    phase_dir: Path,
    filename_map: dict[str, str],
    ms_names: dict[str, dict[str, Any]],
    phase_label: str,
) -> list[dict[str, Any]]:
    """
    Parse all PAGE-XML files in one phase folder and return new anchor entries.

    Each XML → multiple verse entries, one per verse pair.
    """
    xml_files = sorted(phase_dir.glob("*.xml"))
    # Skip METS.xml (it's structural, not page content)
    xml_files = [f for f in xml_files if f.name != "METS.xml"]

    entries: list[dict[str, Any]] = []

    for xml_path in xml_files:
        stem         = xml_path.stem                      # e.g. "manuscript07_p20"
        ms_stem      = _parse_ms_stem(stem)               # e.g. "manuscript07"
        page_num     = _parse_page_num(stem)              # e.g. 20
        short_key    = filename_map.get(ms_stem, ms_stem) # e.g. "al_shuraiti"
        # Why use ms_names from manuscript_registry.json: it is the single
        # authoritative source for Arabic/English names and provenance data.
        names        = ms_names.get(short_key, {"ar": short_key, "en": short_key})

        # Derive source_image_path relative to repo root
        img_path = xml_path.with_suffix(".png")
        rel_img  = str(img_path.relative_to(_REPO)) if img_path.exists() else ""

        verses = _parse_pagexml(xml_path)
        if not verses:
            print(f"  [skip] {xml_path.name}: no verses extracted")
            continue

        for v in verses:
            row_id = f"{stem}_v{v['seq']:03d}"
            entry: dict[str, Any] = {
                # Content
                "matla_text":              v["full"],
                "verse_sadr":              v["sadr"],
                "verse_ajuz":              v["ajuz"],
                "verse_count":             None,
                "sequence_num":            v["seq"],
                # Source provenance
                "source_row_id":           row_id,
                "source_page_key":         stem,
                "source_volume":           ms_stem,
                "source_toc_page":         page_num,
                "source_image_path":       rel_img,
                "page_number":             page_num,
                "page_number_raw":         str(page_num),
                "bbox":                    v["bbox"],
                "layout_type":             f"{phase_label}_verse",
                # Manuscript metadata
                "manuscript_short_key":    short_key,
                "manuscript_arabic_name":  names["ar"],
                "manuscript_english_name": names["en"],
                # Poet (unknown from PAGE-XML alone)
                "poet_name":               "غير محدد",
                "occasion":                None,
                # Genre/emotions — filled later by enrich_genre_heuristic
                "genre":                   "غير_محدد",
                "genre_confidence":        0.0,
                "genre_source":            "pending",
                "genre_evidence":          [],
                "occasion_hint_used":      False,
                "emotions":                [],
                # Traceability
                "content_phase":           phase_label,
                "source_per_field":        {"_row_winners": ["pagexml"]},
            }
            entries.append(entry)

        print(f"  {xml_path.name}: {len(verses)} verses → short_key={short_key}")

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Phase 1/2/3 PAGE-XML files into the anchor registry."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and count without writing output files.")
    parser.add_argument("--output", default=str(DATA_DIR / "anchor_registry_full.json"),
                        help="Output path for merged registry (default: anchor_registry_full.json)")
    args = parser.parse_args()

    filename_map = _load_filename_map()
    ms_names     = _load_manuscript_names()
    print(f"Loaded {len(filename_map)} filename mappings, {len(ms_names)} manuscript registry entries.")

    # ── Load existing enriched registry ───────────────────────────────────────
    print(f"Loading existing registry from {EXISTING_ENRICHED.name} …")
    with open(EXISTING_ENRICHED, encoding="utf-8") as f:
        existing: list[dict] = json.load(f)
    print(f"  Existing entries: {len(existing)}")

    # Build a set of (short_key, page_num, seq) already present to detect duplicates
    existing_keys: set[str] = set()
    for e in existing:
        sk   = e.get("manuscript_short_key", "")
        pg   = str(e.get("page_number", e.get("source_toc_page", "")))
        seq  = str(e.get("sequence_num", ""))
        row  = e.get("source_row_id", "")
        existing_keys.add(row)
        existing_keys.add(f"{sk}|{pg}|{seq}")

    # ── Ingest each phase ─────────────────────────────────────────────────────
    new_entries: list[dict] = []

    for phase_dir, label in [
        (PHASE1_DIR, "phase1"),
        (PHASE2_DIR, "phase2"),
        (PHASE3_DIR, "phase3"),
    ]:
        if not phase_dir.exists():
            print(f"WARNING: {phase_dir} not found — skipping {label}")
            continue
        print(f"\nParsing {label} from {phase_dir.name} …")
        phase_entries = ingest_phase_dir(phase_dir, filename_map, ms_names, label)

        # Deduplicate against existing registry
        added = 0
        for e in phase_entries:
            row  = e["source_row_id"]
            key2 = f"{e['manuscript_short_key']}|{e['page_number']}|{e['sequence_num']}"
            if row in existing_keys or key2 in existing_keys:
                continue
            new_entries.append(e)
            existing_keys.add(row)
            existing_keys.add(key2)
            added += 1

        print(f"  → {len(phase_entries)} parsed, {added} new (non-duplicate) entries added")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    by_ms: Counter = Counter(e["manuscript_short_key"] for e in new_entries)
    print(f"\n{'─'*50}")
    print(f"New entries by manuscript:")
    for k, v in sorted(by_ms.items(), key=lambda x: -x[1]):
        print(f"  {k:<35} {v:>5}")
    print(f"{'─'*50}")
    print(f"Existing registry entries : {len(existing):>5}")
    print(f"New Phase 1/2/3 entries   : {len(new_entries):>5}")
    print(f"Combined total            : {len(existing) + len(new_entries):>5}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Merge and write ───────────────────────────────────────────────────────
    merged = existing + new_entries
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Wrote {len(merged)} entries to {out_path}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/enrich_genre_heuristic.py --registry {out_path} --output {out_path.with_name(out_path.stem + '_enriched.json')}")
    print(f"  2. python scripts/rebuild_index.py --registry {out_path.with_name(out_path.stem + '_enriched.json')} --force")


if __name__ == "__main__":
    main()
