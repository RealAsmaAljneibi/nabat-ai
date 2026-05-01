"""
src/al_nassikh/corpus_stats.py
==============================
Why this file exists: The RAG pipeline (Agent 2, Stage 4) is built to *find*
poems, not to *count* them. When a user asks "how many poems are in this
corpus?" or "how old are these manuscripts?", semantic retrieval returns a
noisy set of unrelated verses because no chunk in the Qdrant index contains a
count or a date — so CRAG marks the passages "Incorrect" and the pipeline
fires the refusal template. That is exactly what happened when Asma ran the
app on 2026-04-23.

The fix is a deterministic answer path: a tiny aggregations module that
computes the answer once, at import time, from the two ground-truth JSON
files (manuscript_registry.json + anchor_registry_phase4.json). The
intent_router (Agent 1, Stage 0.5) calls this module's helpers directly
for "counting" and "corpus metadata" intents, bypassing retrieval entirely.

Architecture ref: §2.4 new Stage 0.5 (Intent Router) — see
doc/IMPLEMENTATION_PLAN.md M2c. §2.9 guardrails still apply: every answer
emitted by this module cites the underlying file + field, so the provenance
trail is preserved.

TERMINOLOGY (enforced throughout this module):
    bayt (بيت, pl. أبيات) — a single verse couplet; the atomic unit stored in
        the registry. Each registry entry = one bayt.
    full poem (قصيدة) — a complete multi-bayt poem. Phase 1–3 pages are
        transcribed bayt-by-bayt (~39 full poems, 720 bayts). Phase 4 TOC
        entries record only the matla (opening bayt) of 1,502 distinct poems —
        those poems are *known* but not fully transcribed.
    Never conflate bayt count with poem count.

Public API:
    count_bayts() -> int              # total registry entries (= bayts indexed)
    count_toc_poems() -> int          # distinct poems known from TOC (matla only)
    count_full_poems() -> int         # fully transcribed poems (Phase 1-3 pages)
    count_poets() -> int
    count_manuscripts() -> int
    count_pages() -> int
    manuscript_age_range() -> dict
    regions_of_origin() -> list[str]
    corpus_summary() -> dict
    poet_counts_top_n(n: int) -> list[tuple[str, int]]
    manuscripts_overview() -> list[dict]
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


# ── Path resolution (mirrors registry.py) ─────────────────────────────────────
# Why copy the _repo_root() helper instead of importing it from registry.py:
# keeping corpus_stats.py free of intra-package imports means it can be called
# from anywhere in the pipeline, including at very early bootstrap time, without
# worrying about module import order.

def _repo_root() -> Path:
    """Walk up from this file until we find requirements.txt (repo root)."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path(os.getcwd())


_GROUND_TRUTH = _repo_root() / "data" / "ground_truth"
_REGISTRY_PATH = _GROUND_TRUTH / "manuscript_registry.json"
# Registry preference order — mirrors fatat_al_arab/index.py DEFAULT_REGISTRY:
#   1. anchor_registry_full_enriched.json  — Phase 1-4, genre-tagged (2,222 entries) ← preferred
#   2. anchor_registry_full.json           — Phase 1-4, no genre tags
#   3. anchor_registry_phase4_enriched.json — Phase 4 only, genre-tagged (1,502 entries)
#   4. anchor_registry_phase4.json         — Phase 4 only, no genre tags
# Genre-aware counting (count_poems_by_genre) requires an enriched file; falling
# back to un-enriched returns 0 for every genre without crashing.
_FULL_ENRICHED    = _GROUND_TRUTH / "anchor_registry_full_enriched.json"
_FULL_PLAIN       = _GROUND_TRUTH / "anchor_registry_full.json"
_ANCHORS_ENRICHED = _GROUND_TRUTH / "anchor_registry_phase4_enriched.json"
_ANCHORS_BASE     = _GROUND_TRUTH / "anchor_registry_phase4.json"
_ANCHORS_PATH = (
    _FULL_ENRICHED    if _FULL_ENRICHED.exists()    else
    _FULL_PLAIN       if _FULL_PLAIN.exists()       else
    _ANCHORS_ENRICHED if _ANCHORS_ENRICHED.exists() else
    _ANCHORS_BASE
)


# ── Load at import time ───────────────────────────────────────────────────────
# Why at import time: the intent router is called on every query. Opening and
# parsing two JSON files per query adds ~30ms of latency that is trivially
# avoided by caching the parsed structures in module globals. The two files
# change only when the pipeline is rebuilt, so staleness is not a concern at
# request time.

def _load_json(path: Path) -> Any:
    """Load a JSON file or return None if absent (defensive for first-run)."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


_REGISTRY: list[dict] = _load_json(_REGISTRY_PATH) or []
_ANCHORS:  list[dict] = _load_json(_ANCHORS_PATH)  or []


# ── Counting helpers ──────────────────────────────────────────────────────────

def count_bayts() -> int:
    """
    Total number of bayts (verse couplets) in the registry.

    Each registry entry = one bayt. This includes:
      - Phase 1-3: 720 bayts from ~39 fully transcribed poem pages
      - Phase 4:   1,502 matla bayts (opening verse of each TOC poem)
    Total: 2,222 bayts. NOT 2,222 poems.
    """
    return len(_ANCHORS)


def count_toc_poems() -> int:
    """
    Distinct poems *known* from the TOC — represented by their matla bayt only.

    Phase 4 entries have layout_type starting with 'type_'. Each is the
    opening bayt of one poem; the body was not transcribed. So 1,502 poems
    are named and citable, but only one bayt of each is in the corpus.
    """
    return sum(1 for a in _ANCHORS if (a.get("layout_type") or "").startswith("type_"))


def count_full_poems() -> int:
    """
    Fully transcribed poems — Phase 1-3 page scans where every bayt was
    captured. Grouped by source_page_key: each page ≈ one poem.

    These are the only poems where the complete text is retrievable.
    """
    pages: set[str] = set()
    for a in _ANCHORS:
        lt = a.get("layout_type") or ""
        if lt.startswith("phase"):
            pk = a.get("source_page_key") or ""
            if pk:
                pages.add(pk)
    return len(pages)


# Backward-compat alias — callers that used count_poems() still work.
# Do NOT use this in new code; use count_bayts() or count_toc_poems() instead.
count_poems = count_bayts


def count_poets() -> int:
    """
    How many distinct poets.

    Why we normalise on `normalised_poet` if present, else `poet_name`:
    the Phase-4 merger writes a normalised field when it can, and different
    spellings of the same poet collapse to one normalised key. Falling back
    to the raw name means we don't crash on anchors that pre-date the
    normaliser.
    """
    seen: set[str] = set()
    for a in _ANCHORS:
        key = a.get("normalised_poet") or a.get("poet_name") or ""
        key = key.strip()
        if key:
            seen.add(key)
    return len(seen)


def count_manuscripts() -> int:
    """
    How many canonical manuscripts.

    Why we read the registry rather than counting unique manuscript_short_key
    values on anchors: the registry is authoritative (25 entries). The anchor
    registry only references manuscripts whose TOC was legible — asking "how
    many manuscripts" should honestly reflect the full corpus shape, not just
    the retrievable subset.
    """
    return len(_REGISTRY)


def count_manuscripts_with_toc() -> int:
    """
    How many manuscripts contributed a legible TOC to the Phase-4 index.
    Useful when a reviewer asks "how many manuscripts have searchable metadata"
    vs. the total corpus count above.
    """
    seen: set[str] = set()
    for a in _ANCHORS:
        key = a.get("manuscript_short_key") or ""
        if key:
            seen.add(key)
    return len(seen)


def count_pages() -> int:
    """
    How many distinct source pages are cited across the Phase-4 anchors.
    Why useful: answers questions like "how many pages have we indexed?" —
    a direct proxy for retrieval coverage.
    """
    seen: set[str] = set()
    for a in _ANCHORS:
        img = a.get("source_image_path") or ""
        if img:
            seen.add(img)
    return len(seen)


# ── Metadata helpers ──────────────────────────────────────────────────────────

def manuscript_age_range() -> dict:
    """
    Oldest / newest manuscript dates across the registry.

    Why a dict rather than a tuple: the Streamlit UI renders this as a
    three-line block (oldest, newest, span), so returning a dict keeps the
    call-site one line of code. Missing dates are skipped rather than
    defaulted — a missing date should not artificially widen the span.
    """
    starts = [m["circa_date_start"] for m in _REGISTRY if m.get("circa_date_start")]
    ends   = [m["circa_date_end"]   for m in _REGISTRY if m.get("circa_date_end")]
    if not starts or not ends:
        return {"oldest_start": None, "newest_end": None, "span_years": None,
                "coverage": "no dates available yet — registry needs enrichment"}
    oldest = min(starts)
    newest = max(ends)
    return {
        "oldest_start": oldest,
        "newest_end":   newest,
        "span_years":   newest - oldest,
        "coverage":     f"{len(starts)}/{len(_REGISTRY)} manuscripts have dated provenance",
    }


def regions_of_origin() -> list[str]:
    """
    Distinct regions represented in the registry (for "where are they from?").
    Returned sorted so the UI renders a stable list.
    """
    seen: set[str] = set()
    for m in _REGISTRY:
        r = (m.get("region_of_origin") or "").strip()
        if r:
            seen.add(r)
    return sorted(seen)


def collectors() -> list[str]:
    """
    Distinct collectors/scribes across the registry (useful for Persona-1
    provenance questions).
    """
    seen: set[str] = set()
    for m in _REGISTRY:
        c = (m.get("collector") or "").strip()
        if c and c.lower() != "unknown":
            seen.add(c)
    return sorted(seen)


# ── Compound summaries ────────────────────────────────────────────────────────

def corpus_summary() -> dict:
    """
    One-shot summary used by the Streamlit sidebar's "Corpus Inventory"
    expander and by the deterministic answer node when the user asks an
    open-ended "tell me about this corpus" question.

    Why a single helper: the sidebar calls this once per rerender. Keeping
    all counts in one dict means one call site, one source of truth for the
    number-of-poems shown next to the number-of-poets.
    """
    age = manuscript_age_range()
    return {
        # Bayt counts (verse-level entries) — use these for "how many bayts"
        "bayts":                  count_bayts(),
        # Poem-level counts — use these for "how many poems"
        "toc_poems":              count_toc_poems(),    # named but only matla stored
        "full_poems_transcribed": count_full_poems(),   # complete bayts available
        # Corpus structure
        "poets":                  count_poets(),
        "manuscripts":            count_manuscripts(),
        "manuscripts_with_toc":   count_manuscripts_with_toc(),
        "pages_indexed":          count_pages(),
        "oldest_start":           age["oldest_start"],
        "newest_end":             age["newest_end"],
        "span_years":             age["span_years"],
        "regions":                regions_of_origin(),
        "collectors":             collectors(),
        "source_files": {
            "registry": str(_REGISTRY_PATH.name),
            "anchors":  str(_ANCHORS_PATH.name),
        },
    }


def poet_counts_top_n(n: int = 10) -> list[tuple[str, int]]:
    """
    Top-N poets by bayt count (poet_name, bayt_count).
    "Bayt count" = number of registry entries attributed to that poet.
    For Phase 4 poets this equals the number of poems they have in the TOC
    (one matla bayt each). For Phase 1-3 poets it equals the number of
    fully transcribed bayts. Do NOT present these counts as "poem counts".
    """
    counter: Counter[str] = Counter()
    for a in _ANCHORS:
        key = (a.get("normalised_poet") or a.get("poet_name") or "").strip()
        if key:
            counter[key] += 1
    return counter.most_common(n)


def manuscripts_overview() -> list[dict]:
    """
    One row per manuscript, joined with Phase-4 poem counts.
    Returned sorted by registry number so the Streamlit sidebar table is stable.
    """
    # Count poems per manuscript_short_key
    per_ms: Counter[str] = Counter()
    for a in _ANCHORS:
        key = a.get("manuscript_short_key") or ""
        if key:
            per_ms[key] += 1

    rows: list[dict] = []
    for m in sorted(_REGISTRY, key=lambda e: e["number"]):
        key = m["short_key"]
        rows.append({
            "number":        m["number"],
            "short_key":     key,
            "arabic_name":   m["arabic_name"],
            "english_name":  m["english_name"],
            "bayts_indexed": per_ms.get(key, 0),  # bayts, not poems
            "circa_date":    _format_date(m),
            "region":        m.get("region_of_origin") or "—",
            "collector":     m.get("collector") or "—",
        })
    return rows


def _format_date(m: dict) -> str:
    """Human-readable date string, e.g., 'c. 1800-1884' or '—' if unknown."""
    s = m.get("circa_date_start")
    e = m.get("circa_date_end")
    if s and e:
        return f"c. {s}-{e}"
    if s:
        return f"c. {s}"
    return "—"


# ── Filtered registry lookups ────────────────────────────────────────────────
# These support Persona-1 (cultural institution) and Persona-2 (researcher)
# queries that need more specific answers than the top-level corpus_summary.

def count_poems_by_poet(name: str) -> int:
    """
    Why useful: "how many poems did Al-Hazani write?" needs a per-poet count,
    not a total. Checks both normalised_poet and raw poet_name fields.
    """
    name_lower = name.strip().lower()
    count = 0
    for a in _ANCHORS:
        raw = (a.get("normalised_poet") or a.get("poet_name") or "").strip().lower()
        if name_lower in raw or raw in name_lower:
            count += 1
    return count


def count_poems_by_genre(genre: str) -> int:
    """
    How many anchors carry the given genre label (M2d silver-baseline tag).

    Why useful: "how many love poems" needs an exact filtered count, not the
    1,502-poem total. Reads the `genre` field on each anchor — present only
    when the enriched registry is loaded. Returns 0 if no anchor matches
    (including when the enriched file is absent), which the caller can detect
    and frame as "not yet classified" rather than "none in corpus".
    """
    if not genre:
        return 0
    return sum(1 for a in _ANCHORS if a.get("genre") == genre)


def genre_distribution() -> dict[str, int]:
    """
    Full {genre_label: count} map across all anchors.

    Why useful: the LLM prose pass for genre-counting answers can mention the
    relative ranking ("the largest classified genre") without inventing
    numbers. غير_محدد is included so the caller can compute classified-vs-
    unclassified shares and surface coverage honestly.
    """
    out: dict[str, int] = {}
    for a in _ANCHORS:
        g = (a.get("genre") or "").strip()
        if g:
            out[g] = out.get(g, 0) + 1
    return out


def sample_poets_by_genre(genre: str, n: int = 3) -> list[str]:
    """
    Up to N distinct poet names that contributed anchors of this genre.

    Why useful: when the LLM frames the genre count in prose, naming a few
    representative poets keeps the answer concrete without forcing a full
    retrieval pass. Returned in poem-count order (most prolific first) so the
    sample is the most informative subset, not arbitrary.
    """
    if not genre:
        return []
    counter: Counter[str] = Counter()
    for a in _ANCHORS:
        if a.get("genre") != genre:
            continue
        name = (a.get("normalised_poet") or a.get("poet_name") or "").strip()
        if name and name.lower() != "unknown":
            counter[name] += 1
    return [name for name, _ in counter.most_common(n)]


_CHILD_GRIEF_CHILD_TERMS = (
    "ولد", "ولدي", "ولده", "ابن", "ابني", "بني", "بنت", "بنتي",
    "ابنة", "أبناء", "ابناء", "طفل", "طفله", "عيال",
)
_CHILD_GRIEF_GRIEF_TERMS = (
    "رثاء", "مرثية", "موت", "وفاة", "مات", "الميت", "فقد",
    "حزن", "بكا", "يبكي", "دمع", "مصاب", "عزاء", "نعي",
)


def child_grief_poem_matches() -> list[dict]:
    """
    Return anchors that plausibly concern grief over a son/daughter/child.

    Why heuristic: the current registry has genre and occasion fields, but not
    a human-labelled "bereaved parent" theme. This gives a conservative,
    auditable demo answer by requiring both a child term and either a grief term
    or the silver-baseline رثاء genre.
    """
    matches: list[dict] = []
    for a in _ANCHORS:
        text = " ".join(
            str(a.get(k) or "")
            for k in ("matla_text", "occasion", "genre", "poet_name")
        )
        has_child = any(term in text for term in _CHILD_GRIEF_CHILD_TERMS)
        has_grief = any(term in text for term in _CHILD_GRIEF_GRIEF_TERMS) or a.get("genre") == "رثاء"
        if has_child and has_grief:
            matches.append(a)
    return matches


def count_child_grief_poems() -> int:
    """Count plausible child-grief poems in the current indexed registry."""
    return len(child_grief_poem_matches())


def sample_poets_child_grief(n: int = 3) -> list[str]:
    """Representative poets for child-grief matches, ordered by match count."""
    counter: Counter[str] = Counter()
    for a in child_grief_poem_matches():
        name = (a.get("normalised_poet") or a.get("poet_name") or "").strip()
        if name and name.lower() not in {"unknown", "مجهول"}:
            counter[name] += 1
    return [name for name, _ in counter.most_common(n)]


def manuscripts_by_region(region: str) -> list[dict]:
    """
    Why useful: "which manuscripts came from Al-Ahsa?" — returns the subset of
    registry entries whose region_of_origin matches the query.
    Returns a list of registry dicts (subset of _REGISTRY).
    """
    region_lower = region.strip().lower()
    return [
        m for m in _REGISTRY
        if region_lower in (m.get("region_of_origin") or "").strip().lower()
    ]


def manuscripts_by_collector(name: str) -> list[dict]:
    """
    Why useful: "what did Huber collect?" — returns manuscripts whose collector
    field matches. Useful for Persona-1 provenance questions.
    """
    name_lower = name.strip().lower()
    return [
        m for m in _REGISTRY
        if name_lower in (m.get("collector") or "").strip().lower()
    ]


def poet_in_corpus(name: str) -> dict:
    """
    Why useful: "is my ancestor represented here?" (Persona-4 family-history use).
    Returns {"found": bool, "poem_count": int, "manuscripts": list[str]}.
    """
    name_lower = name.strip().lower()
    mss: set[str] = set()
    count = 0
    for a in _ANCHORS:
        raw = (a.get("normalised_poet") or a.get("poet_name") or "").strip().lower()
        if name_lower in raw or raw in name_lower:
            count += 1
            key = a.get("manuscript_short_key") or ""
            if key:
                mss.add(key)
    return {"found": count > 0, "bayt_count": count, "manuscripts": sorted(mss)}


# ── CLI smoke test ────────────────────────────────────────────────────────────
# Why a __main__: lets Asma run `python -m al_nassikh.corpus_stats` to get a
# one-shot summary without launching the Streamlit app — useful for debugging
# the data layer in isolation from the agent pipeline.

if __name__ == "__main__":
    from pprint import pprint
    print("=== NABAT-AI corpus summary ===")
    pprint(corpus_summary())
    print()
    print("Top-10 poets:")
    for name, n in poet_counts_top_n(10):
        print(f"  {n:4d}  {name}")
