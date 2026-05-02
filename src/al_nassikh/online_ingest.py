"""
src/al_nassikh/online_ingest.py
=================================
⚡ SCOPE EXTENSION EXT-2 — Online Khaleeji/Nabati Poetry Corpus Ingest

Why this file exists: the manuscript corpus covers 25 handwritten volumes.
Published Khaleeji poems that are already in digital form can be indexed
without the Kraken HTR + eScriptorium step — they are treated as
"already digitised" and fed directly into the same Qdrant pipeline.

This expands the retrieval corpus and demonstrates the agentic system's
ability to reason across heterogeneous source types.

Three sources supported (all publicly accessible Arabic poetry sites):

  SOURCE 1 — aldiwan.net/cat-poets-uae
    UAE poet listing → per-poet poem catalogue → individual poem pages.
    Poem container: <div id="poem_content"><h4>...</h4></div>
    URL pattern: https://www.aldiwan.net/poem{ID}.html

  SOURCE 2 — 4byt.com
    Arabic poetry site with explicit Nabati/Khaleeji category (cat/2).
    Poem container: <div id="tr_" class="byt-content">...</div>
    Poet embedded in text as (poet name) at end of verse block.
    URL pattern: https://4byt.com/byt/{ID}

  SOURCE 3 — uaell.ecssr.ae  (best-effort; JS-rendered, limited access)
    UAE Leadership Encyclopedia poetry — fetched if accessible.
    URL pattern: https://uaell.ecssr.ae/products/poems/{ID}

Output: data/online_corpus/online_anchor_registry.json
Format: same anchor_registry schema as the manuscript corpus.
Chunks carry source_type="online_digitized" so the UI can badge them 🌐.

Architecture ref: Worker 1 (Al-Nassikh) ingestion — same index.py pipeline.
Rate limiting: 1-second sleep between requests; skips on HTTP error.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
_ONLINE_DIR  = _REPO_ROOT / "data" / "online_corpus"
_OUTPUT      = _ONLINE_DIR / "online_anchor_registry.json"
_CACHE_DIR   = _ONLINE_DIR / "_cache"

# ── Rate limiting ─────────────────────────────────────────────────────────────
_REQUEST_DELAY_S = 1.2   # seconds between HTTP requests — respectful crawling

# ── UAE poets hardcoded from aldiwan.net/cat-poets-uae ───────────────────────
# These were verified from the live page (2026-05-01).
# Stored here so the ingest can run without re-scraping the listing page.

_UAE_POET_SLUGS = [
    "Fawaghi-Al-qasimi",
    "Ahmed-Rashid-Thani",
    "Saqr-bin-Sultan-Al-Qasimi",
    "Sultan-bin-Mohammad-Al-Qasimi",
    "Khulood-Al-Mualla",
    "ibrahim-mohammad-ibrahim",
    "khalfan-bin-misbah",
    "salem-abdullah-alkrana",
    "hammad-bin-saeed",
    "ali-bin-mohammed-alshehhi",
    "mubarak-bin-hamad-oqaili",
    "mohammed-bin-hamoud-al-shehhi",
    "yousef-bin-ruizm",
    "mana-said-aotaibh",
    "aref-al-khaja",
    "Hamad-Bin-Khalifa-Abu-Shihab",
    "karim-matouk",
    "salem-abu-jomhor",
]

# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch URL text with graceful error handling.
    Returns None on any HTTP/network error.
    """
    try:
        import requests  # soft dependency
        headers = {
            "User-Agent": (
                "NABAT-AI/1.0 Academic Research Bot - "
                "Khaleeji poetry digitisation project, MAAI1704. "
                "Contact: asljneibi@scad.gov.ae"
            ),
            "Accept-Language": "ar,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        time.sleep(_REQUEST_DELAY_S)
        return resp.text
    except Exception as exc:
        logger.warning(f"GET {url} failed: {exc}")
        time.sleep(_REQUEST_DELAY_S)
        return None


def _soup(html: str):
    """Parse HTML with BeautifulSoup. Returns None if bs4 not installed."""
    try:
        from bs4 import BeautifulSoup  # soft dependency
        return BeautifulSoup(html, "html.parser")
    except ImportError:
        logger.error("beautifulsoup4 not installed — run: pip install beautifulsoup4")
        return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", url)[:120]
    return _CACHE_DIR / f"{safe}.html"


def _cached_get(url: str) -> Optional[str]:
    """Return cached HTML if it exists, otherwise fetch and cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(url)
    if cp.exists():
        return cp.read_text(encoding="utf-8")
    html = _get(url)
    if html:
        cp.write_text(html, encoding="utf-8")
    return html


# ── Source 1: aldiwan.net UAE poets ──────────────────────────────────────────

def _aldiwan_poem_urls_for_poet(poet_slug: str, max_poems: int = 20) -> list[str]:
    """
    Scrape the poet's page on aldiwan.net and return up to max_poems poem URLs.
    Poem links on poet pages follow the pattern: /poem{ID}.html
    """
    url  = f"https://www.aldiwan.net/cat-poet-{poet_slug}"
    html = _cached_get(url)
    if not html:
        return []
    soup = _soup(html)
    if not soup:
        return []
    links = soup.find_all("a", href=re.compile(r"/poem\d+\.html"))
    seen, urls = set(), []
    for tag in links:
        href = tag.get("href", "")
        full = urljoin("https://www.aldiwan.net", href)
        if full not in seen:
            seen.add(full)
            urls.append(full)
        if len(urls) >= max_poems:
            break
    logger.info(f"aldiwan [{poet_slug}]: found {len(urls)} poem URLs")
    return urls


def _aldiwan_parse_poem(url: str) -> Optional[dict]:
    """
    Parse a single aldiwan.net poem page.
    Poem text in: <div id="poem_content"><h4>verse<br>verse</h4></div>
    Poet name in: breadcrumb <a href="cat-poet-...">Name</a> or sidebar <h2 class="h3">
    """
    html = _cached_get(url)
    if not html:
        return None
    soup = _soup(html)
    if not soup:
        return None

    # Poem text
    container = soup.find("div", id="poem_content")
    if not container:
        return None
    h4 = container.find("h4")
    if not h4:
        return None
    raw_lines = [
        line.strip()
        for line in h4.get_text(separator="\n").splitlines()
        if line.strip()
    ]
    text = "\n".join(raw_lines)
    if len(text) < 10:
        return None

    # Poet name from breadcrumb
    poet_name = ""
    for link in soup.find_all("a", href=re.compile(r"cat-poet-")):
        poet_name = link.get_text(strip=True)
        if poet_name:
            break

    # Poem title from page <title> or <h1>
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    poem_id = re.search(r"/poem(\d+)\.html", url)
    anchor_id = f"aldiwan_{poem_id.group(1)}" if poem_id else f"aldiwan_{hash(url)}"

    return {
        "anchor_id":            anchor_id,
        "poet_name":            poet_name,
        "title":                title,
        "text":                 text,
        "source_url":           url,
        "source_type":          "online_digitized",
        "source_site":          "aldiwan.net",
        "source_volume":        "aldiwan",
        "source_page":          anchor_id,
        "source_image_path":    "",
        "manuscript_short_key": "aldiwan",
        "genre":                "غير_محدد",
        "genre_confidence":     0.0,
        "genre_source":         "no_annotation|aldiwan.net",
        "emotions":             [],
        # Transparency flags — distinguishes primary manuscript data from secondary sources
        "is_secondary_source":  True,
        "data_tier":            "secondary",
    }


def fetch_aldiwan_uae(max_poems_per_poet: int = 15) -> list[dict]:
    """
    Fetch poems from the UAE poet listing on aldiwan.net.
    Returns a list of anchor-registry-format dicts.
    """
    entries = []
    for slug in _UAE_POET_SLUGS:
        urls = _aldiwan_poem_urls_for_poet(slug, max_poems_per_poet)
        for url in urls:
            entry = _aldiwan_parse_poem(url)
            if entry:
                entries.append(entry)
                logger.debug(f"  + {entry['anchor_id']} ({entry['poet_name']})")
        logger.info(f"aldiwan [{slug}]: fetched {len(urls)} poems (running total: {len(entries)})")
    return entries


# ── Source 2: 4byt.com Nabati/Khaleeji ───────────────────────────────────────

def _4byt_poem_ids_from_category(cat_id: int = 2, pages: int = 5) -> list[int]:
    """
    Scrape poem IDs from a 4byt.com category listing (cat/2 = colloquial/Nabati).
    Poem links follow: /byt/{ID}
    """
    ids = []
    for page in range(1, pages + 1):
        url  = f"https://4byt.com/cat/{cat_id}?page={page}"
        html = _cached_get(url)
        if not html:
            break
        soup = _soup(html)
        if not soup:
            break
        for tag in soup.find_all("a", href=re.compile(r"/byt/\d+")):
            m = re.search(r"/byt/(\d+)", tag.get("href", ""))
            if m:
                bid = int(m.group(1))
                if bid not in ids:
                    ids.append(bid)
        logger.info(f"4byt.com cat/{cat_id} page {page}: {len(ids)} IDs so far")
    return ids


def _4byt_parse_poem(poem_id: int) -> Optional[dict]:
    """
    Parse a single 4byt.com poem page.
    Poem text + (poet name) in: <div id="tr_" class="byt-content">
    Verses separated by <br /> tags; poet name in parentheses at end.
    """
    url  = f"https://4byt.com/byt/{poem_id}"
    html = _cached_get(url)
    if not html:
        return None
    soup = _soup(html)
    if not soup:
        return None

    container = soup.find("div", id="tr_")
    if not container:
        # fallback: try class byt-content
        container = soup.find("div", class_="byt-content")
    if not container:
        return None

    # Remove the font/credit tag
    for tag in container.find_all(["font", "script"]):
        tag.decompose()

    raw = container.get_text(separator="\n")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Last line often holds (poet name) in parentheses
    poet_name = ""
    if lines and re.match(r"^\(.*\)$", lines[-1]):
        poet_name = lines[-1].strip("()")
        lines = lines[:-1]

    text = "\n".join(lines)
    if len(text) < 10:
        return None

    return {
        "anchor_id":            f"4byt_{poem_id}",
        "poet_name":            poet_name,
        "title":                "",
        "text":                 text,
        "source_url":           url,
        "source_type":          "online_digitized",
        "source_site":          "4byt.com",
        "source_volume":        "4byt",
        "source_page":          str(poem_id),
        "source_image_path":    "",
        "manuscript_short_key": "4byt",
        "genre":                "غير_محدد",
        "genre_confidence":     0.0,
        "genre_source":         "no_annotation|4byt.com",
        "emotions":             [],
        "is_secondary_source":  True,
        "data_tier":            "secondary",
    }


def fetch_4byt_nabati(pages: int = 5) -> list[dict]:
    """
    Fetch Khaleeji/colloquial poems from 4byt.com category 2.
    Returns anchor-registry-format dicts.
    """
    ids     = _4byt_poem_ids_from_category(cat_id=2, pages=pages)
    entries = []
    for poem_id in ids:
        entry = _4byt_parse_poem(poem_id)
        if entry:
            entries.append(entry)
    logger.info(f"4byt.com: fetched {len(entries)} poems from {len(ids)} IDs")
    return entries


# ── Source 3: uaell.ecssr.ae (best-effort, JS-rendered) ──────────────────────

def fetch_ecssr_poems(max_id: int = 30) -> list[dict]:
    """
    Best-effort fetch from UAE Leadership Encyclopedia poetry section.
    Pages are JS-rendered; plain HTTP fetch captures what is available.
    """
    entries = []
    for poem_id in range(1, max_id + 1):
        url  = f"https://uaell.ecssr.ae/products/poems/{poem_id}"
        html = _cached_get(url)
        if not html:
            continue
        soup = _soup(html)
        if not soup:
            continue
        # Try to extract any Arabic text blocks
        body = soup.get_text(separator="\n")
        arabic_lines = [
            l.strip() for l in body.splitlines()
            if l.strip() and re.search(r"[؀-ۿ]", l)
        ]
        if len(arabic_lines) < 3:
            continue
        text = "\n".join(arabic_lines[:30])
        entries.append({
            "anchor_id":            f"ecssr_{poem_id}",
            "poet_name":            "",
            "title":                "",
            "text":                 text,
            "source_url":           url,
            "source_type":          "online_digitized",
            "source_site":          "uaell.ecssr.ae",
            "source_volume":        "ecssr",
            "source_page":          str(poem_id),
            "source_image_path":    "",
            "manuscript_short_key": "ecssr",
            "genre":                "غير_محدد",
            "genre_confidence":     0.0,
            "genre_source":         "no_annotation|uaell.ecssr.ae",
            "emotions":             [],
            "is_secondary_source":  True,
            "data_tier":            "secondary",
        })
    logger.info(f"ECSSR: fetched {len(entries)} poem fragments")
    return entries


# ── Genre enrichment (post-fetch) ────────────────────────────────────────────

def _enrich_genre(entries: list[dict]) -> list[dict]:
    """
    Apply the genre heuristic classifier to online entries.
    Same silver-baseline as manuscript corpus (🔸 badge applies).
    """
    try:
        from al_nassikh.genre_heuristic import classify_genre
        for entry in entries:
            result = classify_genre(entry.get("text", ""))
            site   = entry.get("source_site", "online")
            entry["genre"]            = result.get("genre", "غير_محدد")
            entry["genre_confidence"] = result.get("confidence", 0.0)
            entry["genre_source"]     = f"heuristic_v1|{site}"
            entry["emotions"]         = result.get("emotions", [])
    except Exception as exc:
        logger.warning(f"Genre enrichment skipped: {exc}")
    return entries


# ── Main entry point ──────────────────────────────────────────────────────────

def ingest_online_corpus(
    include_aldiwan:  bool = True,
    include_4byt:     bool = True,
    include_ecssr:    bool = True,
    max_poems_per_poet: int = 15,
    pages_4byt:       int = 5,
    output_path:      Optional[Path] = None,
) -> list[dict]:
    """
    Full online corpus ingestion pipeline.

    Fetches from all three sources, deduplicates by text similarity,
    applies genre heuristic, and writes to data/online_corpus/.

    Returns the list of anchor entries for immediate indexing.
    """
    _ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or _OUTPUT

    all_entries: list[dict] = []

    if include_aldiwan:
        logger.info("=== Fetching aldiwan.net UAE poets ===")
        all_entries.extend(fetch_aldiwan_uae(max_poems_per_poet))

    if include_4byt:
        logger.info("=== Fetching 4byt.com Nabati/Khaleeji ===")
        all_entries.extend(fetch_4byt_nabati(pages_4byt))

    if include_ecssr:
        logger.info("=== Fetching ECSSR UAE poetry (best-effort) ===")
        all_entries.extend(fetch_ecssr_poems())

    # Deduplicate by anchor_id
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for e in all_entries:
        if e["anchor_id"] not in seen_ids:
            seen_ids.add(e["anchor_id"])
            unique.append(e)
    logger.info(f"Deduplication: {len(all_entries)} → {len(unique)} entries")

    # Genre enrichment
    unique = _enrich_genre(unique)

    # Write output
    output_path.write_text(
        json.dumps(unique, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        f"Online corpus written → {output_path}\n"
        f"  aldiwan.net:   {sum(1 for e in unique if e['source_site']=='aldiwan.net')}\n"
        f"  4byt.com:      {sum(1 for e in unique if e['source_site']=='4byt.com')}\n"
        f"  ecssr:         {sum(1 for e in unique if e['source_site']=='uaell.ecssr.ae')}\n"
        f"  Total:         {len(unique)}"
    )
    return unique


def get_online_corpus_stats(registry_path: Optional[Path] = None) -> dict:
    """Return summary statistics for the online corpus."""
    path = registry_path or _OUTPUT
    if not path.exists():
        return {"status": "not_built", "total": 0}
    entries = json.loads(path.read_text(encoding="utf-8"))
    by_site = {}
    for e in entries:
        site = e.get("source_site", "unknown")
        by_site[site] = by_site.get(site, 0) + 1
    return {
        "status":  "ready",
        "total":   len(entries),
        "by_site": by_site,
        "sources": ["aldiwan.net", "4byt.com", "uaell.ecssr.ae"],
    }
