"""
src/fatat_al_arab/index.py
===========================
Why this file exists: M3 Stage 2 — anchor_registry_phase4.json holds 1,502
poetry entries. This module explodes each entry into chunk levels,
encodes them with AraBERT, and stores everything in a file-backed Qdrant
collection so any laptop can do approximate nearest-neighbour search.

Stanza-aware chunking (§2.4.1 of the architecture doc):
  Level 1 — verse (بيت):       each entry = one matla (opening verse).
                                Finest granularity; best for exact phrase retrieval.
  Level 2 — group (مجموعة):    sliding window of 3 consecutive verses by the same
                                poet+volume. Good for thematic span queries.
  Level 3 — poem (قصيدة):      all verses by same poet+volume concatenated.
                                Best for broad biographical queries.
  Level 4 — manuscript (مخطوطة): all verses in the same manuscript concatenated
                                with a header (Arabic+English name, date range,
                                region). Best for "what does ms X contain?" queries.
  Level 5 — poet (شاعر):       all verses across ALL volumes by the same poet.
                                Best for poet-level queries ("what themes does
                                poet X explore?").
  Level 6 — era (حقبة):        all verses whose manuscript's circa_date range
                                overlaps a 50-year era bucket (e.g. 1800–1849).
                                Best for "show me 19th-century poems" queries.
  Level 7 — genre (نوع):       all verses sharing the same genre label, grouped
                                by genre+manuscript. Best for "show me all elegies
                                in ms X" or broad genre-facet queries.
  Level 8 — emotion (مشاعر):   all verses sharing at least one emotion label,
                                grouped by emotion+manuscript. Best for "show me
                                longing poems" facet queries.

Why eight levels (not just verse): retrieval precision vs recall trade-off.
A verse-level hit is precise but may miss context; coarser levels provide
richer context for multi-hop or facet queries. RRF fusion at query time
lets the best granularity win per query.

Qdrant payload per chunk:
  chunk_id, level, text, anchor_id, poet_name, source_volume, source_page,
  source_image_path, manuscript_short_key, dense_embedding (stored separately)

The module builds:
  - An in-memory list of ScoredChunk (for BM25 and DenseRetriever)
  - A Qdrant collection at QDRANT_PATH for ANN lookups

Public API:
  build_index(registry_path, qdrant_path, force_rebuild) -> IndexBundle
  load_index(qdrant_path)  -> IndexBundle (fast path — no re-encoding)
  IndexBundle.retrievers   -> (BM25Retriever, DenseRetriever, ColBERTRetriever)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fatat_al_arab.embed import get_dense_embeddings_batch, normalise_arabic
from fatat_al_arab.rrf import ScoredChunk
from fatat_al_arab.retrievers.bm25 import BM25Retriever
from fatat_al_arab.retrievers.dense import DenseRetriever
from fatat_al_arab.retrievers.colbert import ColBERTRetriever

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent          # …/handwritten-poems
# Registry preference order (most complete → least complete):
#   1. anchor_registry_full_enriched.json  — Phase 1-4 combined, genre-tagged (2,222 entries)
#   2. anchor_registry_full.json           — Phase 1-4 combined, not yet genre-tagged
#   3. anchor_registry_phase4_enriched.json — Phase 4 only, genre-tagged (1,502 entries)
#   4. anchor_registry_phase4.json         — Phase 4 only, no genre tags
# Why this order: ingest_phases_123.py + enrich_genre_heuristic.py produce (1).
# Falling back ensures old setups and CI still work without re-running the pipeline.
_GT = _REPO_ROOT / "data" / "ground_truth"
_FULL_ENRICHED    = _GT / "anchor_registry_full_enriched.json"
_FULL_PLAIN       = _GT / "anchor_registry_full.json"
_ENRICHED_REGISTRY = _GT / "anchor_registry_phase4_enriched.json"
_PLAIN_REGISTRY    = _GT / "anchor_registry_phase4.json"
_MS_REGISTRY_PATH  = _GT / "manuscript_registry.json"
# Reference corpus = scholarly PDFs OCR'd into paragraph chunks (level="reference").
# Built by scripts/build_reference_corpus.py. Optional — if missing, the index
# is built from the manuscript registry alone (graceful degradation).
_REFERENCE_CORPUS  = _GT / "reference_corpus.json"
DEFAULT_REGISTRY = (
    _FULL_ENRICHED    if _FULL_ENRICHED.exists()    else
    _FULL_PLAIN       if _FULL_PLAIN.exists()       else
    _ENRICHED_REGISTRY if _ENRICHED_REGISTRY.exists() else
    _PLAIN_REGISTRY
)
DEFAULT_QDRANT   = _REPO_ROOT / "data" / "qdrant"
COLLECTION_NAME  = "nabat_ai_chunks"
CHUNK_META_FILE  = "chunks_meta.json"     # parallel JSON to Qdrant, stores payload

# Era bucket width in years. 50 years keeps buckets meaningful without being too sparse.
_ERA_BUCKET = 50


def _load_ms_registry() -> dict[str, dict]:
    """
    Load manuscript_registry.json as a dict keyed by short_key.
    Returns empty dict on failure (graceful — coarser chunks just lack date info).
    """
    try:
        with open(_MS_REGISTRY_PATH, encoding="utf-8") as f:
            entries = json.load(f)
        return {e["short_key"]: e for e in entries if "short_key" in e}
    except Exception:
        return {}


# ── Chunking ──────────────────────────────────────────────────────────────────

def _make_chunk_id(anchor_id: str, level: str, seq: int) -> str:
    return f"{anchor_id}__{level}__{seq:04d}"


def _entry_text(entry: dict) -> str:
    """
    Primary text for an entry — handles both registry schemas:
      - Primary manuscripts: matla_text (original field)
      - Unified registry / secondary sources: text (canonical field)
    Always returns normalised Arabic.
    """
    return normalise_arabic(entry.get("matla_text") or entry.get("text") or "")


def _entry_anchor_id(entry: dict) -> str:
    """
    Unique ID for an entry — handles both schemas:
      - Primary manuscripts: source_row_id
      - Unified registry / secondary sources: anchor_id
    """
    return entry.get("anchor_id") or entry.get("source_row_id") or ""


def _entry_source_page(entry: dict) -> str:
    """
    Source page — handles both schemas:
      - Primary manuscripts: page_number (int)
      - Secondary sources: source_page (string, e.g. "0-3200ms")
    Always returns a string.
    """
    sp = entry.get("source_page")
    if sp:
        return str(sp)
    pn = entry.get("page_number")
    return str(pn) if pn else ""


def _verse_text(entry: dict) -> str:
    return _entry_text(entry)


def _group_text(entries: list[dict]) -> str:
    return " | ".join(_entry_text(e) for e in entries)


def _poem_text(entries: list[dict]) -> str:
    return " | ".join(_entry_text(e) for e in entries)


def _manuscript_text(ms_info: dict, entries: list[dict]) -> str:
    """
    Build a manuscript-level chunk text: a descriptive header followed by
    all indexed matla verses it contains (empty list = header only).

    Why include the full header even for empty manuscripts: a query like
    "tell me about the Huber manuscript" or "ما هي مخطوطة سوسين الأولى؟"
    should match and return provenance information even before any poems from
    that manuscript have been transcribed. The source_note field gives the
    retrieval system useful prose to embed.
    """
    ar_name     = ms_info.get("arabic_name", "")
    en_name     = ms_info.get("english_name", "")
    region      = ms_info.get("region_of_origin", "")
    date_s      = ms_info.get("circa_date_start", "")
    date_e      = ms_info.get("circa_date_end", "")
    collector   = ms_info.get("collector", "")
    source_note = ms_info.get("source_note", "")

    header_parts = []
    if ar_name:
        header_parts.append(ar_name)
    if en_name:
        header_parts.append(en_name)
    if region:
        header_parts.append(f"المنطقة: {region}")
    if date_s and date_e:
        header_parts.append(f"التاريخ التقريبي: {date_s}–{date_e}")
    elif date_s:
        header_parts.append(f"التاريخ التقريبي: بعد {date_s}")
    if collector:
        header_parts.append(f"الجامع: {collector}")
    if source_note:
        header_parts.append(source_note)

    header = " | ".join(header_parts)

    if entries:
        verses = " | ".join(_entry_text(e) for e in entries)
        return f"{header} || {verses}" if header else verses
    else:
        # No indexed verses yet — header-only chunk. Still searchable by name/provenance.
        return header


def _poet_text(poet_name: str, entries: list[dict]) -> str:
    """
    Build a poet-level chunk: poet name header + all their matla verses
    across every manuscript in the corpus.

    Why the name prefix: queries like "شعر ابن سبيل" need the poet name
    in the chunk text so BM25 can match it.
    """
    header = normalise_arabic(poet_name)
    verses  = " | ".join(_entry_text(e) for e in entries)
    return f"{header} || {verses}" if header else verses


def _era_bucket(date_start: int | None, date_end: int | None) -> str | None:
    """
    Map a manuscript's circa date range to a 50-year bucket label.
    Uses the midpoint of the range. Returns None if no date data.

    Examples: (1800, 1884) → "1850–1899" (midpoint 1842 → bucket start 1850)
    Wait, use floor: mid 1842 → bucket 1800–1849.
    """
    if date_start is None and date_end is None:
        return None
    mid = ((date_start or date_end) + (date_end or date_start)) // 2  # type: ignore[operator]
    bucket_start = (mid // _ERA_BUCKET) * _ERA_BUCKET
    bucket_end   = bucket_start + _ERA_BUCKET - 1
    return f"{bucket_start}–{bucket_end}"


def _era_text(era_label: str, ms_entries: list[tuple[dict, list[dict]]]) -> str:
    """
    Build an era-level chunk: era header + all manuscripts in that era
    with their verses.

    ms_entries: list of (ms_info, [entry, ...]) tuples for this era.
    """
    header = f"حقبة {era_label}"
    parts  = []
    for ms_info, entries in ms_entries:
        ms_name = ms_info.get("arabic_name") or ms_info.get("short_key", "")
        verses  = " | ".join(_entry_text(e) for e in entries)
        parts.append(f"{ms_name}: {verses}")
    return f"{header} || " + " ||| ".join(parts)


def _genre_chunk_text(genre: str, entries: list[dict]) -> str:
    """
    Build a genre-level chunk: genre label header + all matla verses
    with that genre tag in the same manuscript.

    Why per-manuscript: a genre chunk that mixes all manuscripts would be
    enormous and semantically noisy. Per-manuscript keeps it coherent.
    """
    header = f"نوع: {genre}"
    verses  = " | ".join(_entry_text(e) for e in entries)
    return f"{header} || {verses}"


def _emotion_chunk_text(emotion: str, entries: list[dict]) -> str:
    """
    Build an emotion-level chunk: emotion label header + all matla verses
    carrying that emotion in the same manuscript.
    """
    header = f"مشاعر: {emotion}"
    verses  = " | ".join(_entry_text(e) for e in entries)
    return f"{header} || {verses}"


def _chunk_payload(entry: dict) -> dict:
    """
    Extract the citation-and-routing fields we store in every chunk.

    M3: genre/emotion fields come from the M2d silver-baseline enrichment.
    They default to abstention values ("غير_محدد", 0.0, []) for entries
    that were not enriched (e.g. if building from the plain registry).
    """
    return {
        "anchor_id":           _entry_anchor_id(entry),
        "poet_name":           entry.get("poet_name")     or "",
        "source_volume":       entry.get("source_volume") or "",
        "source_page":         _entry_source_page(entry),
        "source_image_path":   entry.get("source_image_path") or "",
        "manuscript_short_key": entry.get("manuscript_short_key") or "",
        # M3 — silver-baseline genre enrichment (absent = abstained)
        "genre":               entry.get("genre")            or "غير_محدد",
        "genre_confidence":    float(entry.get("genre_confidence") or 0.0),
        "genre_source":        entry.get("genre_source")     or "",
        "emotions":            list(entry.get("emotions")    or []),
        # Source transparency — populated by unified registry
        "source_type":         entry.get("source_type")      or "manuscript",
        "data_tier":           entry.get("data_tier")         or "primary",
        "is_secondary_source": bool(entry.get("is_secondary_source", False)),
        # Poem-parent link — groups all bayts of the same poem
        "parent_poem_id":      entry.get("parent_poem_id")   or "",
        "poem_matla":          entry.get("poem_matla")        or entry.get("matla") or "",
    }


def build_chunks(registry: list[dict]) -> list[ScoredChunk]:
    """
    Explode the registry into 8 chunk levels.
    Returns a flat list of ScoredChunk (rrf_score=0.0 — not yet retrieved).

    Levels built:
      1. verse      — one per registry entry (finest granularity)
      2. group      — sliding window of 3 verses, same poet+volume
      3. poem       — all verses by same poet+volume
      4. manuscript — all verses in the same manuscript (+ ms header)
      5. poet       — all verses by the same poet across all manuscripts
      6. era        — all verses whose manuscript's midpoint date falls in
                      the same 50-year bucket
      7. genre      — all verses sharing the same genre, per manuscript
      8. emotion    — all verses sharing an emotion label, per manuscript
    """
    from collections import defaultdict
    chunks: list[ScoredChunk] = []
    ms_registry = _load_ms_registry()

    # ── Grouping indices built once, used by multiple levels ──────────────────
    # poet+volume → entries (for levels 2 & 3)
    poet_vol_groups: dict[str, list[dict]] = defaultdict(list)
    # manuscript_short_key → entries (for levels 4, 7, 8)
    ms_groups: dict[str, list[dict]] = defaultdict(list)
    # poet_name → entries across all volumes (for level 5)
    poet_groups: dict[str, list[dict]] = defaultdict(list)
    # era_label → list of (ms_info, entries) (for level 6)
    era_groups: dict[str, list[tuple[dict, list[dict]]]] = defaultdict(list)

    for entry in registry:
        pv_key  = f"{entry.get('poet_name', '')}::{entry.get('source_volume', '')}"
        ms_key  = entry.get("manuscript_short_key", "") or ""
        poet    = entry.get("poet_name", "") or ""
        poet_vol_groups[pv_key].append(entry)
        ms_groups[ms_key].append(entry)
        poet_groups[poet].append(entry)

    # Build era groups: one entry per (ms_key, era) combination
    for ms_key, entries in ms_groups.items():
        ms_info   = ms_registry.get(ms_key, {"short_key": ms_key})
        era_label = _era_bucket(
            ms_info.get("circa_date_start"),
            ms_info.get("circa_date_end"),
        )
        if era_label:
            era_groups[era_label].append((ms_info, entries))

    seq_counter = 0

    # ── Level 1: verse — one per registry entry ────────────────────────────────
    for entry in registry:
        text = _verse_text(entry)
        if not text:
            continue
        p = _chunk_payload(entry)
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(p["anchor_id"], "verse", seq_counter),
            rrf_score=0.0, text=text, level="verse",
            anchor_id=p["anchor_id"], poet_name=p["poet_name"],
            source_volume=p["source_volume"], source_page=p["source_page"],
            source_image_path=p["source_image_path"],
            manuscript_short_key=p["manuscript_short_key"],
            genre=p["genre"], genre_confidence=p["genre_confidence"],
            genre_source=p["genre_source"], emotions=p["emotions"],
        ))
        seq_counter += 1

    # ── Level 2: group (sliding window=3) per poet+volume ─────────────────────
    # Why inherit genre from the first entry in the window: the first verse
    # (matla) sets the poem's tone and is the most genre-representative.
    for group_entries in poet_vol_groups.values():
        if len(group_entries) < 2:
            continue
        window = 3
        for i in range(len(group_entries) - window + 1):
            we = group_entries[i: i + window]
            text = _group_text(we)
            if not text:
                continue
            p = _chunk_payload(we[0])
            chunks.append(ScoredChunk(
                chunk_id=_make_chunk_id(p["anchor_id"], "group", seq_counter),
                rrf_score=0.0, text=text, level="group",
                anchor_id=p["anchor_id"], poet_name=p["poet_name"],
                source_volume=p["source_volume"], source_page=p["source_page"],
                source_image_path=p["source_image_path"],
                manuscript_short_key=p["manuscript_short_key"],
                genre=p["genre"], genre_confidence=p["genre_confidence"],
                genre_source=p["genre_source"], emotions=p["emotions"],
            ))
            seq_counter += 1

    # ── Level 3: poem — one per poet+volume group ──────────────────────────────
    for group_entries in poet_vol_groups.values():
        if len(group_entries) < 2:
            continue
        text = _poem_text(group_entries)
        if not text:
            continue
        p = _chunk_payload(group_entries[0])
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(p["anchor_id"], "poem", seq_counter),
            rrf_score=0.0, text=text, level="poem",
            anchor_id=p["anchor_id"], poet_name=p["poet_name"],
            source_volume=p["source_volume"], source_page=p["source_page"],
            source_image_path=p["source_image_path"],
            manuscript_short_key=p["manuscript_short_key"],
            genre=p["genre"], genre_confidence=p["genre_confidence"],
            genre_source=p["genre_source"], emotions=p["emotions"],
        ))
        seq_counter += 1

    # ── Level 4: manuscript — one per manuscript in the registry ─────────────
    # Why cover ALL registered manuscripts, not just those with indexed verses:
    # a researcher can ask "what is the Huber 2 manuscript?" even before its
    # poems are transcribed. Manuscripts with no verses yet get a header-only
    # chunk (metadata description); once poems are added and the index is
    # rebuilt, those verses appear in the same chunk automatically.
    for ms_key, ms_info in ms_registry.items():
        entries    = ms_groups.get(ms_key, [])   # [] for unindexed manuscripts
        text       = _manuscript_text(ms_info, entries)
        if not text:
            continue
        # For manuscripts with no indexed verses, anchor fields are empty
        anchor_id    = _chunk_payload(entries[0])["anchor_id"] if entries else ""
        source_vol   = _chunk_payload(entries[0])["source_volume"] if entries else ""
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(ms_key, "manuscript", seq_counter),
            rrf_score=0.0, text=text, level="manuscript",
            anchor_id=anchor_id, poet_name="",
            source_volume=source_vol, source_page=0,
            source_image_path="",
            manuscript_short_key=ms_key,
            genre="غير_محدد", genre_confidence=0.0, genre_source="",
            emotions=[],
            extra={
                "manuscript_arabic_name":  ms_info.get("arabic_name", ""),
                "manuscript_english_name": ms_info.get("english_name", ""),
                "circa_date_start":        ms_info.get("circa_date_start"),
                "circa_date_end":          ms_info.get("circa_date_end"),
                "region_of_origin":        ms_info.get("region_of_origin", ""),
                "collector":               ms_info.get("collector", ""),
                "source_note":             ms_info.get("source_note", ""),
                "entry_count":             len(entries),
                "indexed":                 len(entries) > 0,
            },
        ))
        seq_counter += 1

    # ── Level 5: poet — one per unique poet name across all volumes ───────────
    # Why cross-volume: a poet's work spans multiple manuscripts/volumes.
    # A query like "ما مواضيع شعر ابن سبيل؟" wants all his verses, not just
    # one volume's worth.
    for poet_name, entries in poet_groups.items():
        if not poet_name or len(entries) < 2:
            continue
        text = _poet_text(poet_name, entries)
        if not text:
            continue
        p = _chunk_payload(entries[0])
        # Collect all manuscripts this poet appears in
        ms_keys = list(dict.fromkeys(e.get("manuscript_short_key", "") for e in entries))
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(
                normalise_arabic(poet_name).replace(" ", "_")[:40], "poet", seq_counter
            ),
            rrf_score=0.0, text=text, level="poet",
            anchor_id=p["anchor_id"], poet_name=poet_name,
            source_volume="", source_page=0,
            source_image_path="",
            manuscript_short_key=ms_keys[0] if ms_keys else "",
            genre="غير_محدد", genre_confidence=0.0, genre_source="",
            emotions=[],
            extra={
                "all_manuscript_keys": ms_keys,
                "verse_count":         len(entries),
            },
        ))
        seq_counter += 1

    # ── Level 6: era — one per 50-year bucket ─────────────────────────────────
    # Why 50 years: fine enough to distinguish early 19th-century from late
    # 19th-century poetry (different political eras in Arabian Peninsula), but
    # broad enough that each bucket has meaningful coverage.
    for era_label, ms_entry_pairs in era_groups.items():
        text = _era_text(era_label, ms_entry_pairs)
        if not text:
            continue
        all_entries = [e for _, ents in ms_entry_pairs for e in ents]
        p = _chunk_payload(all_entries[0])
        ms_keys = [mi.get("short_key", "") for mi, _ in ms_entry_pairs]
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(era_label.replace("–", "_"), "era", seq_counter),
            rrf_score=0.0, text=text, level="era",
            anchor_id=p["anchor_id"], poet_name="",
            source_volume="", source_page=0,
            source_image_path="",
            manuscript_short_key="",
            genre="غير_محدد", genre_confidence=0.0, genre_source="",
            emotions=[],
            extra={
                "era_label":           era_label,
                "manuscript_keys":     ms_keys,
                "verse_count":         len(all_entries),
            },
        ))
        seq_counter += 1

    # ── Level 7: genre — one per (genre, manuscript) pair ─────────────────────
    # Why per-manuscript rather than global: a genre chunk that pools all 1,502
    # verses into one string is too large and loses manuscript provenance.
    # Per-manuscript genre chunks let a query like "elegies in Huber ms" land
    # precisely.
    genre_ms_groups: dict[str, list[dict]] = defaultdict(list)
    for ms_key, entries in ms_groups.items():
        for entry in entries:
            genre = entry.get("genre") or "غير_محدد"
            if genre == "غير_محدد":
                continue   # don't make a chunk for undecided genre
            gm_key = f"{genre}::{ms_key}"
            genre_ms_groups[gm_key].append(entry)

    for gm_key, entries in genre_ms_groups.items():
        genre, ms_key = gm_key.split("::", 1)
        if len(entries) < 1:
            continue
        text = _genre_chunk_text(genre, entries)
        if not text:
            continue
        p = _chunk_payload(entries[0])
        ms_info = ms_registry.get(ms_key, {"short_key": ms_key})
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(
                f"{genre}_{ms_key}"[:50], "genre", seq_counter
            ),
            rrf_score=0.0, text=text, level="genre",
            anchor_id=p["anchor_id"], poet_name="",
            source_volume=p["source_volume"], source_page=0,
            source_image_path="",
            manuscript_short_key=ms_key,
            genre=genre, genre_confidence=0.0, genre_source="",
            emotions=[],
            extra={
                "manuscript_arabic_name":  ms_info.get("arabic_name", ""),
                "manuscript_english_name": ms_info.get("english_name", ""),
                "verse_count":             len(entries),
            },
        ))
        seq_counter += 1

    # ── Level 8: emotion — one per (emotion, manuscript) pair ─────────────────
    # Same rationale as genre: per-manuscript keeps provenance intact.
    # An entry can contribute to multiple emotion chunks (multi-label).
    emotion_ms_groups: dict[str, list[dict]] = defaultdict(list)
    for ms_key, entries in ms_groups.items():
        for entry in entries:
            for emotion in (entry.get("emotions") or []):
                em_key = f"{emotion}::{ms_key}"
                emotion_ms_groups[em_key].append(entry)

    for em_key, entries in emotion_ms_groups.items():
        emotion, ms_key = em_key.split("::", 1)
        if len(entries) < 1:
            continue
        text = _emotion_chunk_text(emotion, entries)
        if not text:
            continue
        p = _chunk_payload(entries[0])
        ms_info = ms_registry.get(ms_key, {"short_key": ms_key})
        chunks.append(ScoredChunk(
            chunk_id=_make_chunk_id(
                f"{emotion}_{ms_key}"[:50], "emotion", seq_counter
            ),
            rrf_score=0.0, text=text, level="emotion",
            anchor_id=p["anchor_id"], poet_name="",
            source_volume=p["source_volume"], source_page=0,
            source_image_path="",
            manuscript_short_key=ms_key,
            genre="غير_محدد", genre_confidence=0.0, genre_source="",
            emotions=[emotion],
            extra={
                "emotion_label":           emotion,
                "manuscript_arabic_name":  ms_info.get("arabic_name", ""),
                "manuscript_english_name": ms_info.get("english_name", ""),
                "verse_count":             len(entries),
            },
        ))
        seq_counter += 1

    # ── Level 9: reference — scholarly PDF chunks (Ibn Khaldun, desert culture,
    #     suloom al-arab, semiotics). Optional layer; lifted from
    #     data/ground_truth/reference_corpus.json if it exists.
    # Why integrated here, not in a parallel index: the retriever and RRF fusion
    # work over a single vector space — a unified index lets a query that asks
    # for "Ibn Khaldun's view on Bedouin poetry" pull the relevant paragraph
    # alongside any matching manuscript matla, ranked together by RRF. The UI
    # later distinguishes them with a 📚 reference badge so users always know
    # whether a citation is primary (handwritten manuscript) or secondary
    # (scholarly source).
    chunks.extend(_build_reference_chunks(_REFERENCE_CORPUS, seq_counter))

    return chunks


def _build_reference_chunks(
    reference_corpus_path: Path,
    starting_seq: int,
) -> list[ScoredChunk]:
    """
    Load the OCR'd reference corpus JSON and convert each entry into a
    reference-level ScoredChunk. Returns [] if the file is missing.

    Why store everything in `extra`: book_title_ar/en, topic_ar/en, paragraph
    index, and source_pdf are reference-specific fields. Keeping them in
    `extra` (a dict) avoids polluting the ScoredChunk dataclass with fields
    that are empty for the 8 manuscript levels.
    """
    if not reference_corpus_path.exists():
        return []
    try:
        with open(reference_corpus_path, encoding="utf-8") as f:
            entries: list[dict] = json.load(f)
    except Exception as exc:
        # Fail-soft: a malformed reference corpus shouldn't block the whole index.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "index: failed to load reference corpus %s: %s — skipping.",
            reference_corpus_path, exc,
        )
        return []

    out: list[ScoredChunk] = []
    seq = starting_seq
    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        chunk_id = entry.get("chunk_id") or f"reference_unknown_{seq:04d}"
        out.append(ScoredChunk(
            chunk_id=chunk_id,
            rrf_score=0.0,
            text=text,
            level="reference",
            # anchor_id == chunk_id so the chunk can act as its own citation
            # target; guardrails treat reference-level chunks as resolvable.
            anchor_id=chunk_id,
            poet_name="",
            # Repurpose source_volume to carry the bilingual book title — the
            # synthesise prompt and UI can render this directly.
            source_volume=entry.get("book_title_ar") or entry.get("book_short_key", ""),
            source_page=int(entry.get("page") or 0),
            source_image_path=entry.get("source_pdf", ""),
            manuscript_short_key="",
            genre="غير_محدد",
            genre_confidence=0.0,
            genre_source="reference",
            emotions=[],
            extra={
                "book_short_key":  entry.get("book_short_key", ""),
                "book_title_ar":   entry.get("book_title_ar", ""),
                "book_title_en":   entry.get("book_title_en", ""),
                "topic_ar":        entry.get("topic_ar", ""),
                "topic_en":        entry.get("topic_en", ""),
                "paragraph_index": int(entry.get("paragraph_index") or 0),
                "source_pdf":      entry.get("source_pdf", ""),
            },
        ))
        seq += 1
    return out


# ── IndexBundle ───────────────────────────────────────────────────────────────

@dataclass
class IndexBundle:
    """
    Holds the built index artefacts.
    - chunks:       flat list of ScoredChunk (same order as embeddings)
    - embeddings:   float32 ndarray, shape (len(chunks), 768)
    - bm25:         BM25Retriever
    - dense:        DenseRetriever
    - colbert:      ColBERTRetriever (stub)
    - stats:        build statistics dict
    """
    chunks:     list[ScoredChunk]
    embeddings: np.ndarray
    bm25:       BM25Retriever
    dense:      DenseRetriever
    colbert:    ColBERTRetriever
    stats:      dict[str, Any]

    @property
    def retrievers(self) -> tuple[BM25Retriever, DenseRetriever, ColBERTRetriever]:
        return self.bm25, self.dense, self.colbert


# ── Build & load ──────────────────────────────────────────────────────────────

def build_index(
    registry_path: str | Path = DEFAULT_REGISTRY,
    qdrant_path:   str | Path = DEFAULT_QDRANT,
    force_rebuild: bool = False,
    batch_size:    int  = 32,
) -> IndexBundle:
    """
    Build the file-backed Qdrant index from anchor_registry_phase4.json.

    Steps:
      1. Load registry JSON
      2. Explode into 3-level chunks
      3. Encode with AraBERT (batch)
      4. Persist chunks metadata + embeddings to qdrant_path
      5. (Optionally) upsert into Qdrant collection for ANN queries
      6. Return IndexBundle

    If qdrant_path/chunks_meta.json already exists and force_rebuild=False,
    load_index() is called instead (fast path).
    """
    registry_path = Path(registry_path)
    qdrant_path   = Path(qdrant_path)
    meta_file     = qdrant_path / CHUNK_META_FILE
    emb_file      = qdrant_path / "embeddings.npy"

    if meta_file.exists() and emb_file.exists() and not force_rebuild:
        return load_index(qdrant_path)

    # Step 1: load registry
    with open(registry_path, encoding="utf-8") as f:
        registry: list[dict] = json.load(f)

    # Step 2: chunk
    chunks = build_chunks(registry)

    # Step 3: encode
    texts = [c.text for c in chunks]
    embeddings_list = get_dense_embeddings_batch(texts, batch_size=batch_size)
    embeddings = np.array(embeddings_list, dtype=np.float32)

    # Step 4: persist
    qdrant_path.mkdir(parents=True, exist_ok=True)
    np.save(emb_file, embeddings)
    chunk_dicts = [c.to_dict() for c in chunks]
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

    # Step 5: upsert into Qdrant (best-effort — skip if qdrant_client unavailable)
    _try_qdrant_upsert(qdrant_path, chunks, embeddings)

    # Step 6: return bundle
    level_counts = {}
    for c in chunks:
        level_counts[c.level] = level_counts.get(c.level, 0) + 1
    stats = {
        "registry_entries":    len(registry),
        "total_chunks":        len(chunks),
        "verse_chunks":        level_counts.get("verse",      0),
        "group_chunks":        level_counts.get("group",      0),
        "poem_chunks":         level_counts.get("poem",       0),
        "manuscript_chunks":   level_counts.get("manuscript", 0),
        "poet_chunks":         level_counts.get("poet",       0),
        "era_chunks":          level_counts.get("era",        0),
        "genre_chunks":        level_counts.get("genre",      0),
        "emotion_chunks":      level_counts.get("emotion",    0),
        "reference_chunks":    level_counts.get("reference",  0),
        "embedding_shape":     list(embeddings.shape),
    }

    bm25   = BM25Retriever(chunks)
    dense  = DenseRetriever(chunks, embeddings)
    colbert = ColBERTRetriever(chunks)
    return IndexBundle(chunks=chunks, embeddings=embeddings,
                       bm25=bm25, dense=dense, colbert=colbert, stats=stats)


def load_index(
    qdrant_path: str | Path = DEFAULT_QDRANT,
) -> IndexBundle:
    """
    Fast-path load from persisted chunks_meta.json + embeddings.npy.
    Skips re-encoding.
    """
    qdrant_path = Path(qdrant_path)
    meta_file = qdrant_path / CHUNK_META_FILE
    emb_file  = qdrant_path / "embeddings.npy"

    if not meta_file.exists() or not emb_file.exists():
        raise FileNotFoundError(
            f"Index not found at {qdrant_path}. "
            "Run `python scripts/rebuild_index.py` first."
        )

    with open(meta_file, encoding="utf-8") as f:
        chunk_dicts: list[dict] = json.load(f)

    # Known core fields — everything else goes into extra for forward-compat.
    _CORE_FIELDS = {
        "chunk_id", "rrf_score", "text", "level", "anchor_id", "poet_name",
        "source_volume", "source_page", "source_image_path",
        "manuscript_short_key", "genre", "genre_confidence", "genre_source", "emotions",
    }
    chunks = [
        ScoredChunk(
            chunk_id=d["chunk_id"],
            rrf_score=0.0,
            text=d["text"],
            level=d["level"],
            anchor_id=d["anchor_id"],
            poet_name=d["poet_name"],
            source_volume=d["source_volume"],
            source_page=d["source_page"],
            source_image_path=d["source_image_path"],
            manuscript_short_key=d["manuscript_short_key"],
            # M3: genre/emotion fields — absent in pre-M3 caches → default gracefully
            genre=d.get("genre", "غير_محدد"),
            genre_confidence=float(d.get("genre_confidence", 0.0)),
            genre_source=d.get("genre_source", ""),
            emotions=list(d.get("emotions") or []),
            # Pass through any extra fields (manuscript, poet, era, genre, emotion levels)
            extra={k: v for k, v in d.items() if k not in _CORE_FIELDS},
        )
        for d in chunk_dicts
    ]

    embeddings = np.load(emb_file)

    level_counts = {}
    for c in chunks:
        level_counts[c.level] = level_counts.get(c.level, 0) + 1
    stats = {
        "total_chunks":      len(chunks),
        "verse_chunks":      level_counts.get("verse",      0),
        "group_chunks":      level_counts.get("group",      0),
        "poem_chunks":       level_counts.get("poem",       0),
        "manuscript_chunks": level_counts.get("manuscript", 0),
        "poet_chunks":       level_counts.get("poet",       0),
        "era_chunks":        level_counts.get("era",        0),
        "genre_chunks":      level_counts.get("genre",      0),
        "emotion_chunks":    level_counts.get("emotion",    0),
        "embedding_shape":   list(embeddings.shape),
        "loaded_from_cache": True,
    }

    bm25    = BM25Retriever(chunks)
    dense   = DenseRetriever(chunks, embeddings)
    colbert = ColBERTRetriever(chunks)
    return IndexBundle(chunks=chunks, embeddings=embeddings,
                       bm25=bm25, dense=dense, colbert=colbert, stats=stats)


def _try_qdrant_upsert(
    qdrant_path: Path,
    chunks: list[ScoredChunk],
    embeddings: np.ndarray,
) -> None:
    """
    Best-effort Qdrant upsert. If qdrant-client is unavailable, log and
    return silently — the numpy/JSON fallback still supports BM25+Dense.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct,
        )
    except ImportError:
        return  # qdrant_client not installed — numpy fallback only

    try:
        client = QdrantClient(path=str(qdrant_path / "qdrant_storage"))
        dim = embeddings.shape[1]

        # Recreate collection (force_rebuild path already decided above)
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # Batch upsert in groups of 100
        BATCH = 100
        for start in range(0, len(chunks), BATCH):
            end = min(start + BATCH, len(chunks))
            points = [
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload={
                        "chunk_id":   chunks[i].chunk_id,
                        "text":       chunks[i].text,
                        "level":      chunks[i].level,
                        "anchor_id":  chunks[i].anchor_id,
                        "poet_name":  chunks[i].poet_name,
                        "source_volume": chunks[i].source_volume,
                        "source_page":   chunks[i].source_page,
                        "source_image_path": chunks[i].source_image_path,
                        "manuscript_short_key": chunks[i].manuscript_short_key,
                        # M3: genre/emotion fields for Qdrant payload filtering
                        "genre":             chunks[i].genre,
                        "genre_confidence":  chunks[i].genre_confidence,
                        "genre_source":      chunks[i].genre_source,
                        "emotions":          chunks[i].emotions,
                    },
                )
                for i in range(start, end)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception:
        pass   # Qdrant upsert is a best-effort enhancement; numpy always works
