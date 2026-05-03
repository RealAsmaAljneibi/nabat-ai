#!/usr/bin/env python3
"""
scripts/build_unified_registry.py
===================================
Single entry point for building the unified NABAT-AI corpus registry.

All four source types are normalised to the same 8-field schema:

  anchor_id    — poem #         unique identifier across all sources
  poet_ar      — poet (ar)      Arabic poet name
  poet_en      — poet (en)      English poet name
  matla        — poem matla     opening verse / first hemistich
  text         — poem           full poem or verse text
  year_approx  — year           approximate year (e.g. "c. 1850–1920")
  genre        — genre          Arabic genre label (غير_محدد = unclassified)
  data_tier    — source (flag)  "primary" (manuscript) | "secondary" (oral/online)

Additional fields for the Qdrant pipeline (index.py compatibility):
  poet_name, source_volume, manuscript_short_key, source_page,
  source_image_path, source_type, genre_confidence, genre_source,
  emotions, is_secondary_source, source_collection

Sources (in merge order — primary first so dedup keeps manuscripts):
  1. Manuscripts    data/ground_truth/anchor_registry_full_enriched.json  (always)
  2. MAAI7103       ~/poetry/data/processed/master_dataset.csv            (--maai7103)
  3. Online         data/online_corpus/online_anchor_registry.json        (--online)
  4. Oral/audio     data/oral_tradition/oral_anchor_registry.json         (--oral, skipped
                    automatically when --maai7103 is active: same 106 poems)

Usage:
    PYTHONPATH=src python scripts/build_unified_registry.py             # manuscripts only
    PYTHONPATH=src python scripts/build_unified_registry.py --maai7103  # + oral tradition
    PYTHONPATH=src python scripts/build_unified_registry.py --online    # + fetch online
    PYTHONPATH=src python scripts/build_unified_registry.py --all       # everything
    PYTHONPATH=src python scripts/build_unified_registry.py --all --dry-run

Output: data/unified_registry.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT        = Path(__file__).resolve().parent.parent
_PRIMARY_REGISTRY = _REPO_ROOT / "data" / "ground_truth" / "anchor_registry_full_enriched.json"
_MS_REGISTRY      = _REPO_ROOT / "data" / "ground_truth" / "manuscript_registry.json"
_POETS_BIO        = _REPO_ROOT / "data" / "ground_truth" / "poets_bio.json"
_ONLINE_REGISTRY  = _REPO_ROOT / "data" / "online_corpus"  / "online_anchor_registry.json"
_ORAL_REGISTRY    = _REPO_ROOT / "data" / "oral_tradition" / "oral_anchor_registry.json"
_MAAI7103_CSV     = _REPO_ROOT / "data" / "master_dataset_full.xlsx"
_OUTPUT           = _REPO_ROOT / "data" / "unified_registry.json"


# ── Lookup tables (built once at module load) ─────────────────────────────────

def _build_poet_en_lookup() -> dict[str, str]:
    """
    Build Arabic poet name → English name lookup from poets_bio.json.
    English name is extracted from the opening of bio_en ("Name was/is/born...").
    """
    if not _POETS_BIO.exists():
        return {}
    bios = json.loads(_POETS_BIO.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for b in bios:
        ar = (b.get("poet_name") or "").strip()
        bio_en = (b.get("bio_en") or "").strip()
        if not ar or not bio_en:
            continue
        # Extract the name before "was", "is", "born", or "("
        m = re.match(r'^([A-Z][^(]*?)(?:\s+(?:was|is|born|\())', bio_en)
        en = m.group(1).strip() if m else ""
        if en:
            lookup[ar] = en
    return lookup


def _build_ms_date_lookup() -> dict[str, str]:
    """
    Build manuscript_short_key → year_approx string from manuscript_registry.json.
    Format: "c. YYYY–YYYY" using the circa start/end dates.
    """
    if not _MS_REGISTRY.exists():
        return {}
    mss = json.loads(_MS_REGISTRY.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for m in mss:
        key   = (m.get("short_key") or "").strip()
        start = m.get("circa_date_start")
        end   = m.get("circa_date_end")
        if key and start and end:
            lookup[key] = f"c. {start}–{end}"
        elif key and start:
            lookup[key] = f"c. {start}"
    return lookup


_POET_EN:  dict[str, str] = _build_poet_en_lookup()
_MS_DATES: dict[str, str] = _build_ms_date_lookup()


# ── Unified schema builder ────────────────────────────────────────────────────

def _entry(
    anchor_id:   str,
    poet_ar:     str,
    poet_en:     str,
    matla:       str,
    text:        str,
    year_approx: str,
    genre:       str,
    data_tier:   str,
    *,
    source_type:          str   = "",
    source_volume:        str   = "",
    source_page:          str   = "",
    source_image_path:    str   = "",
    manuscript_short_key: str   = "",
    genre_confidence:     float = 0.0,
    genre_source:         str   = "",
    emotions:             list  = None,
    source_collection:    str   = "",
    extra:                dict  = None,
) -> dict:
    base = {
        # ── 8 canonical user-visible fields ────────────────────────────────
        "anchor_id":   anchor_id,
        "poet_ar":     poet_ar,
        "poet_en":     poet_en,
        "matla":       matla,
        "text":        text,
        "year_approx": year_approx,
        "genre":       genre,
        "data_tier":   data_tier,
        # ── poem-parent link ───────────────────────────────────────────────
        # parent_poem_id groups all bayts that belong to the same poem.
        # poem_matla is the opening verse (matla) of that parent poem.
        # Set by each loader; stamped in a post-processing pass for manuscripts.
        "parent_poem_id": "",   # filled in by loader post-processing
        "poem_matla":     matla,  # default = own matla; overridden for non-first bayts
        # ── transparency ───────────────────────────────────────────────────
        "is_secondary_source": data_tier == "secondary",
        # ── pipeline fields (index.py) ─────────────────────────────────────
        "poet_name":            poet_ar,
        "source_type":          source_type or ("manuscript" if data_tier == "primary" else ""),
        "source_volume":        source_volume,
        "source_page":          source_page,
        "source_image_path":    source_image_path,
        "manuscript_short_key": manuscript_short_key,
        "genre_confidence":     genre_confidence,
        "genre_source":         genre_source,
        "emotions":             emotions or [],
        "source_collection":    source_collection,
    }
    if extra:
        base.update(extra)
    return base


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text[:80] if text else ""


# ── Source 1: Primary manuscripts ────────────────────────────────────────────

def load_manuscripts() -> list[dict]:
    if not _PRIMARY_REGISTRY.exists():
        logger.error(f"Primary registry not found: {_PRIMARY_REGISTRY}")
        return []

    raw = json.loads(_PRIMARY_REGISTRY.read_text(encoding="utf-8"))
    entries = []
    for row in raw:
        text = (row.get("matla_text") or "").strip()
        if not text:
            continue
        poet_ar = (row.get("poet_name") or "").strip()
        ms_key  = (row.get("manuscript_short_key") or "").strip()
        entries.append(_entry(
            anchor_id            = row.get("source_row_id") or "",
            poet_ar              = poet_ar,
            poet_en              = _POET_EN.get(poet_ar, ""),
            matla                = text,
            text                 = text,
            year_approx          = _MS_DATES.get(ms_key, ""),
            genre                = row.get("genre") or "غير_محدد",
            data_tier            = "primary",
            source_type          = "manuscript",
            source_volume        = (row.get("source_volume") or "").strip(),
            source_page          = str(row.get("page_number") or ""),
            source_image_path    = (row.get("source_image_path") or "").strip(),
            manuscript_short_key = ms_key,
            genre_confidence     = float(row.get("genre_confidence") or 0.0),
            genre_source         = (row.get("genre_source") or "").strip(),
            emotions             = row.get("emotions") or [],
            source_collection    = "NABAT-AI manuscripts",
            extra                = {k: v for k, v in row.items()
                                    if k not in ("poet_name", "matla_text", "source_row_id",
                                                 "genre", "genre_confidence", "genre_source",
                                                 "emotions", "source_volume", "source_image_path",
                                                 "manuscript_short_key")},
        ))

    # Post-processing: stamp parent_poem_id and poem_matla on every entry.
    # Group key = source_page_key for multi-bayt phase 1-3 poems; anchor_id for
    # single-bayt TOC entries (each TOC entry IS the poem reference).
    poem_groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        pid = (e.get("source_page_key") or "").strip() or e["anchor_id"]
        poem_groups[pid].append(e)

    for pid, group in poem_groups.items():
        sorted_group = sorted(group, key=lambda x: x.get("sequence_num") or 0)
        opening_verse = sorted_group[0]["text"]   # seq=1 (or only) bayt = matla
        for e in sorted_group:
            e["parent_poem_id"] = pid
            e["poem_matla"]     = opening_verse

    logger.info(f"Manuscripts (primary): {len(entries)} bayts across {len(poem_groups)} poems  "
                f"({sum(1 for e in entries if e['poet_en'])} with English poet name, "
                f"{sum(1 for e in entries if e['year_approx'])} with year)")
    return entries


# ── Source 2: MAAI7103 oral tradition corpus ──────────────────────────────────

_GENRE_EN_TO_AR: dict[str, str] = {
    "ghazal":          "غزل",
    "pride":           "فخر",
    "boasting":        "فخر",
    "praise":          "مديح",
    "eulogy":          "مديح",
    "elegy":           "رثاء",
    "lament":          "رثاء",
    "wisdom":          "حكمة",
    "self-reflection": "حكمة",
    "satire":          "هجاء",
    "complaint":       "شكوى",
    "longing":         "شوق",
    "nature":          "وصف_الطبيعة",
    "description":     "وصف_الطبيعة",
    "travel":          "غزو_ورحلات",
    "tribal":          "قبلي_اجتماعي",
    "social":          "قبلي_اجتماعي",
    "religious":       "ديني",
    "spiritual":       "ديني",
}

_EMOTION_EN_TO_AR: dict[str, str] = {
    "pride":         "فخر",
    "contemplation": "تأمل",
    "admiration":    "إعجاب",
    "sadness":       "حزن",
    "longing":       "شوق",
    "joy":           "فرح",
    "anger":         "غضب",
    "defiance":      "تحدي",
    "love":          "حب",
    "nostalgia":     "حنين",
    "wisdom":        "حكمة",
    "humor":         "طرفة",
    "religious":     "تعبد",
    "hope":          "أمل",
}

def _map_genre(genre_en: str) -> str:
    key = genre_en.lower().strip()
    for pattern, ar in _GENRE_EN_TO_AR.items():
        if pattern in key:
            return ar
    return "غير_محدد"

def _map_emotion(emotion_en: str) -> str:
    key = emotion_en.lower().strip()
    for pattern, ar in _EMOTION_EN_TO_AR.items():
        if pattern in key:
            return ar
    return ""


def load_maai7103(csv_path: Path = _MAAI7103_CSV, min_quality: str = "") -> list[dict]:
    if not csv_path.exists():
        logger.warning(f"MAAI7103 dataset not found at {csv_path} — skipping")
        return []

    # Support both CSV and Excel (.xlsx) — same column schema
    raw_rows: list[dict] = []
    if csv_path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd  # type: ignore[import]
        df = pd.read_excel(csv_path, dtype=str).fillna("")
        raw_rows = df.to_dict(orient="records")
    else:
        with open(csv_path, encoding="utf-8-sig") as f:
            raw_rows = list(csv.DictReader(f))

    poem_groups: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        quality = (row.get("audio_quality") or "").strip().lower()
        if min_quality == "clean" and quality not in ("clean", ""):
            continue
        pid = (row.get("source_poem") or "").strip()
        if pid:
            poem_groups[pid].append(row)

    entries = []
    for poem_id, rows in poem_groups.items():
        if not rows:
            continue
        first      = rows[0]
        poet_ar    = (first.get("poet_ar") or "").strip()
        poet_en    = (first.get("poet_en") or "").strip()
        genre_en   = (first.get("genre_en") or "").strip()
        genre_ar   = _map_genre(genre_en)
        year       = (first.get("poem_date") or "").strip()
        title      = (first.get("poem_title") or "").strip()

        # Collect all corrected verses in timestamp order, skip empty/noise rows
        verse_texts: list[str] = []
        translations: list[str] = []
        imagery_tags: list[str] = []
        emotions_ar:  list[str] = []
        audio_files:  list[str] = []

        for r in rows:
            text = (r.get("text_corrected") or "").strip()
            if not text or len(text) < 5:
                continue
            verse_texts.append(text)
            tr = (r.get("translation_en") or "").strip()
            if tr:
                translations.append(tr)
            im = (r.get("imagery_tags_en") or "").strip()
            if im:
                imagery_tags.append(im)
            em = _map_emotion((r.get("emotion_text") or "").strip())
            if em and em not in emotions_ar:
                emotions_ar.append(em)
            af = (r.get("audio_filename") or "").strip()
            if af:
                audio_files.append(af)

        if not verse_texts:
            continue

        matla_text = verse_texts[0]
        # Full poem = all corrected verses joined with newlines
        full_poem_text = "\n".join(verse_texts)

        e = _entry(
            anchor_id            = f"maai7103_{poem_id}",
            poet_ar              = poet_ar,
            poet_en              = poet_en,
            matla                = matla_text,
            text                 = full_poem_text,
            year_approx          = year,
            genre                = genre_ar,
            data_tier            = "secondary",
            source_type          = "oral_tradition",
            source_volume        = poem_id,
            source_page          = poem_id,
            source_image_path    = audio_files[0] if audio_files else "",
            manuscript_short_key = "maai7103",
            genre_confidence     = 0.9,
            genre_source         = "maai7103_human_annotation",
            emotions             = emotions_ar,
            source_collection    = "MAAI7103 oral tradition",
            extra                = {
                "title":             title,
                "verse_count":       len(verse_texts),
                "translation_en":    " | ".join(translations),
                "imagery_tags_en":   " | ".join(dict.fromkeys(imagery_tags)),  # deduped
                "khaleeji_value_ar": (first.get("khaleeji_value_ar") or "").strip(),
                "audio_files":       audio_files,
                "maai7103_poem_id":  poem_id,
            },
        )
        e["parent_poem_id"] = poem_id
        e["poem_matla"]     = matla_text
        entries.append(e)

    logger.info(f"MAAI7103 (secondary): {len(entries)} full-poem entries  "
                f"({sum(1 for e in entries if e['poet_en'])} with English poet name, "
                f"{sum(1 for e in entries if e['year_approx'])} with year)")
    return entries


# ── Source 3: Online digitised poems ─────────────────────────────────────────

def load_online(fetch: bool = False) -> list[dict]:
    if fetch and not _ONLINE_REGISTRY.exists():
        logger.info("Fetching online poems (this may take several minutes)...")
        try:
            sys.path.insert(0, str(_REPO_ROOT / "src"))
            from al_nassikh.online_ingest import ingest_online_corpus
            ingest_online_corpus(output_path=_ONLINE_REGISTRY)
        except Exception as exc:
            logger.warning(f"Online fetch failed: {exc}")
            return []

    if not _ONLINE_REGISTRY.exists():
        logger.info("Online registry not found — run with --online to fetch")
        return []

    raw = json.loads(_ONLINE_REGISTRY.read_text(encoding="utf-8"))
    entries = []
    for row in raw:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        poet_ar = (row.get("poet_name") or "").strip()
        site    = row.get("source_site") or row.get("source_collection") or "online"
        entries.append(_entry(
            anchor_id            = row.get("anchor_id") or "",
            poet_ar              = poet_ar,
            poet_en              = _POET_EN.get(poet_ar, ""),
            matla                = _first_line(text),
            text                 = text,
            year_approx          = "",  # online sources carry no publication date
            genre                = row.get("genre") or "غير_محدد",
            data_tier            = "secondary",
            source_type          = "online_digitized",
            source_volume        = row.get("source_volume") or "",
            source_page          = row.get("source_page") or "",
            source_image_path    = "",
            manuscript_short_key = row.get("manuscript_short_key") or "",
            genre_confidence     = float(row.get("genre_confidence") or 0.0),
            genre_source         = row.get("genre_source") or f"heuristic_v1|{site}",
            emotions             = row.get("emotions") or [],
            source_collection    = site,
            extra                = {
                "title":      row.get("title") or "",
                "source_url": row.get("source_url") or "",
            },
        ))

    logger.info(f"Online (secondary): {len(entries)} entries from {_ONLINE_REGISTRY.name}")
    return entries


# ── Source 4: Generic oral tradition (audio) ─────────────────────────────────
# Note: when MAAI7103 is loaded, this is skipped automatically — the
# oral_anchor_registry.json covers the same 106 poems. Loading both would
# create near-duplicate entries with different anchor_ids.

def load_oral() -> list[dict]:
    if not _ORAL_REGISTRY.exists():
        return []

    raw = json.loads(_ORAL_REGISTRY.read_text(encoding="utf-8"))
    entries = []
    for row in raw:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        poet_ar = (row.get("poet_name") or "").strip()
        entries.append(_entry(
            anchor_id            = row.get("anchor_id") or "",
            poet_ar              = poet_ar,
            poet_en              = _POET_EN.get(poet_ar, ""),
            matla                = _first_line(text),
            text                 = text,
            year_approx          = "",
            genre                = row.get("genre") or "غير_محدد",
            data_tier            = "secondary",
            source_type          = "oral_tradition",
            source_volume        = row.get("source_volume") or "",
            source_page          = row.get("source_page") or "",
            source_image_path    = row.get("audio_file") or "",
            manuscript_short_key = row.get("manuscript_short_key") or "oral",
            genre_confidence     = float(row.get("genre_confidence") or 0.0),
            genre_source         = row.get("genre_source") or "oral_tradition_metadata",
            emotions             = row.get("emotions") or [],
            source_collection    = "oral tradition",
        ))

    logger.info(f"Oral/audio (secondary): {len(entries)} entries")
    return entries


# ── Merge ─────────────────────────────────────────────────────────────────────

def build(
    include_maai7103: bool  = False,
    include_online:   bool  = False,
    fetch_online:     bool  = False,
    dry_run:          bool  = False,
    maai7103_csv:     Path  = _MAAI7103_CSV,
) -> list[dict]:
    all_entries: list[dict] = []

    all_entries.extend(load_manuscripts())

    if include_maai7103:
        all_entries.extend(load_maai7103(csv_path=maai7103_csv))
        # Oral registry covers the same 106 poems as MAAI7103 — skip to avoid duplicates
    else:
        all_entries.extend(load_oral())

    if include_online or fetch_online:
        all_entries.extend(load_online(fetch=fetch_online))

    # Safety net: every entry must have parent_poem_id and poem_matla.
    # Loaders set these explicitly for multi-bayt sources (manuscripts, MAAI7103).
    # Single-poem entries (online, oral) fall back to self-reference here.
    for e in all_entries:
        if not e.get("parent_poem_id"):
            e["parent_poem_id"] = e["anchor_id"]
        if not e.get("poem_matla"):
            e["poem_matla"] = e.get("matla") or _first_line(e.get("text") or "")

    # Deduplicate: first occurrence wins (manuscripts take priority)
    seen:   set[str]   = set()
    unique: list[dict] = []
    for e in all_entries:
        aid = e.get("anchor_id", "")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        unique.append(e)

    n_primary   = sum(1 for e in unique if e["data_tier"] == "primary")
    n_secondary = len(unique) - n_primary
    by_source:  dict[str, int] = {}
    by_genre:   dict[str, int] = {}
    n_has_poet_en  = sum(1 for e in unique if e.get("poet_en"))
    n_has_year     = sum(1 for e in unique if e.get("year_approx"))

    # Count unique full poems per source (source_volume = poem_id for oral entries)
    oral_entries   = [e for e in unique if e.get("source_type") == "oral_tradition"]
    n_oral_poems   = len(set(e.get("source_volume", "") for e in oral_entries if e.get("source_volume")))

    for e in unique:
        sc = e.get("source_collection") or e.get("source_type") or "unknown"
        by_source[sc] = by_source.get(sc, 0) + 1
        g = e.get("genre") or "غير_محدد"
        by_genre[g] = by_genre.get(g, 0) + 1

    print(f"\n{'='*62}")
    print(f"  Unified Registry — NABAT-AI")
    print(f"{'='*62}")
    print(f"  {'Primary manuscripts (bayts):':<36} {n_primary:>6}")
    if n_oral_poems:
        print(f"  {'Oral tradition (bayts from MAAI7103):':<36} {len(oral_entries):>6}  ← {n_oral_poems} full poems")
        n_other_secondary = n_secondary - len(oral_entries)
        if n_other_secondary:
            print(f"  {'Other secondary (online bayts):':<36} {n_other_secondary:>6}")
    else:
        print(f"  {'Secondary (oral + online bayts):':<36} {n_secondary:>6}")
    print(f"  {'Total bayts indexed:':<36} {len(unique):>6}")
    print(f"  {'Bayts with poet_en:':<36} {n_has_poet_en:>6}")
    print(f"  {'Bayts with year_approx:':<36} {n_has_year:>6}")
    print(f"\n  By collection (bayts):")
    for coll, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        icon = "📜" if "manuscript" in coll.lower() else ("🎙️" if "oral" in coll.lower() or "maai" in coll.lower() else "🌐")
        print(f"    {icon}  {coll:<40} {cnt:>5} bayts")
    print(f"\n  Top genres:")
    for g, cnt in sorted(by_genre.items(), key=lambda x: -x[1])[:8]:
        print(f"    {g:<28} {cnt:>5}")
    print(f"{'='*62}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return unique

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
    poem_note = f" — {n_oral_poems} oral poems × their bayts" if n_oral_poems else ""
    print(f"\nWritten → {_OUTPUT}  ({len(unique)} bayts{poem_note})")
    print("\nNext: rebuild the Qdrant index")
    print(f"  python scripts/rebuild_index.py --registry {_OUTPUT} --force")
    return unique


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified NABAT-AI registry from all corpus sources"
    )
    parser.add_argument("--maai7103",     action="store_true",
                        help="Include MAAI7103 oral tradition CSV")
    parser.add_argument("--online",       action="store_true",
                        help="Include online corpus (fetches if not yet cached)")
    parser.add_argument("--all",          action="store_true",
                        help="Include all sources: manuscripts + MAAI7103 + online")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print stats without writing files")
    parser.add_argument("--maai7103-csv", default=str(_MAAI7103_CSV),
                        help="Path to master_dataset.csv")
    args = parser.parse_args()

    include_maai7103 = args.maai7103 or args.all
    include_online   = args.online   or args.all
    fetch_online     = include_online and not _ONLINE_REGISTRY.exists()

    entries = build(
        include_maai7103 = include_maai7103,
        include_online   = include_online,
        fetch_online     = fetch_online,
        dry_run          = args.dry_run,
        maai7103_csv     = Path(args.maai7103_csv),
    )

    if not entries and not args.dry_run:
        logger.error("No entries produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
