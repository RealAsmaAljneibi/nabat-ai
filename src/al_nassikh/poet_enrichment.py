"""
src/al_nassikh/poet_enrichment.py
==================================
Why this file exists: poets_bio.json covers 509 poets, but anchor_registry
has 510+ distinct names and 480 of them have no bio entry. When a user asks
"من هو X?" and X is not in the bio file, the RAG pipeline returns a degraded
answer with no biographical grounding. This module bridges that gap via:

  1. Arabic Wikipedia (free, no key, best structured data)
  2. English Wikipedia (fallback for poets with EN coverage)
  3. Brave Search (fallback when Wikipedia has no page; requires BRAVE_API_KEY)

Where it's called: nabat_mcp_server.py (live), offline enrich script.
Purpose: Fetches missing poet bios from Wikipedia / Brave Search

Results are cached to data/ground_truth/poets_bio_web_cache.json so the
API is never hit twice for the same name. The cache schema mirrors poets_bio.json
so the enrichment script can merge both files without format translation.

Public API:
    PoetEnrichmentClient.enrich(poet_name) -> dict | None
    PoetEnrichmentClient.is_available() -> bool
    PoetEnrichmentClient.has_brave() -> bool
    is_valid_poet_name(name) -> bool   # filter OCR artifacts before calling enrich
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Optional HTTP dependency ───────────────────────────────────────────────────
# Why lazy import: keeps the module loadable in offline / test environments
# where httpx is not installed. All public methods degrade gracefully.
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    logger.debug("poet_enrichment: httpx not installed — web lookups disabled.")

# ── Constants ──────────────────────────────────────────────────────────────────
_WIKI_AR_URL = "https://ar.wikipedia.org/w/api.php"
_WIKI_EN_URL = "https://en.wikipedia.org/w/api.php"
_BRAVE_URL   = "https://api.search.brave.com/res/v1/web/search"

REQUEST_TIMEOUT   = 10   # seconds — must not stall the query pipeline
RATE_LIMIT_DELAY  = 0.5  # seconds between requests (polite client)
MIN_EXTRACT_CHARS = 40   # discard trivially short Wikipedia extracts

# Why a custom User-Agent: Wikipedia's API etiquette requires it and blocks
# default httpx/requests agents. Format: <tool>/<version> (<contact>).
_USER_AGENT = "NABAT-AI/1.0 (Khaleeji Nabati poetry research; contact: research)"

# Gulf region keywords used for structured region extraction from bio text
_REGION_KEYWORDS_AR = [
    "نجد", "الكويت", "البحرين", "قطر", "الإمارات", "السعودية",
    "عُمان", "عمان", "اليمن", "الحجاز", "الأحساء",
    "المنطقة الشرقية", "الرياض", "مكة", "المدينة",
    "أبوظبي", "دبي", "الشارقة",
]

# OCR noise patterns — anchored to full string (used with .match())
_NOISE_FULL_MATCH = [
    re.compile(r"^[\d\s٠-٩]+$"),      # pure Arabic or Western numerals
    re.compile(r"^[a-zA-Z\d\s]+$"),   # pure ASCII (transliteration noise)
]
# Any-position patterns — used with .search()
_NOISE_DIGIT_RE = re.compile(r"[0-9٠-٩]")   # digit anywhere → page-number fragment

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_cache_path() -> Path:
    return _REPO_ROOT / "data" / "ground_truth" / "poets_bio_web_cache.json"


# ── Name validation ────────────────────────────────────────────────────────────

def is_valid_poet_name(name: str) -> bool:
    """
    Why: The registry contains OCR artifacts like '١٣٤', 'دها 1', 'ون 1'.
    Calling Wikipedia for these wastes quota and returns garbage. This guard
    filters them out before any network call is made.
    """
    if not name or len(name.strip()) < 4:
        return False
    stripped = name.strip()
    for pattern in _NOISE_FULL_MATCH:
        if pattern.match(stripped):
            return False
    # Any digit anywhere → OCR page-number fragment (e.g. 'دها 1', 'ون 1')
    if _NOISE_DIGIT_RE.search(stripped):
        return False
    # Must contain at least one Arabic letter
    if not re.search(r"[؀-ۿ]", stripped):
        return False
    return True


# ── Client ─────────────────────────────────────────────────────────────────────

class PoetEnrichmentClient:
    """
    Why: Wraps Wikipedia (AR → EN) + Brave Search in a single fallback chain
    with persistent disk caching. One instance is safe to reuse across the
    lifetime of the enrichment script or a Streamlit session.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        brave_api_key: str | None = None,
    ) -> None:
        self._cache_path = cache_path or _default_cache_path()
        self._brave_key  = brave_api_key or os.getenv("BRAVE_API_KEY", "")
        self._cache: dict[str, dict] = self._load_cache()

    # ── Public API ─────────────────────────────────────────────────────────────

    def enrich(self, poet_name: str) -> dict | None:
        """
        Enrichment chain: local cache → Arabic Wikipedia → English Wikipedia
        → Brave Search (only if BRAVE_API_KEY is set).

        Returns a dict matching the poets_bio.json schema, or None if all
        sources fail or httpx is not installed.
        """
        if not _HTTPX_AVAILABLE:
            logger.debug("poet_enrichment.enrich: httpx unavailable — skipping.")
            return None

        if not is_valid_poet_name(poet_name):
            logger.debug("poet_enrichment.enrich: %r looks like OCR noise — skipping.", poet_name)
            return None

        if poet_name in self._cache:
            logger.debug("poet_enrichment.enrich: cache hit for %r", poet_name)
            return self._cache[poet_name]

        result = (
            self._wikipedia(poet_name, lang="ar")
            or self._wikipedia(poet_name, lang="en")
            or (self._brave(poet_name) if self._brave_key else None)
        )

        if result:
            self._cache[poet_name] = result
            self._save_cache()
            logger.info(
                "poet_enrichment: enriched %r via %s",
                poet_name, result.get("enriched_via"),
            )

        return result

    def is_available(self) -> bool:
        """True if at least one enrichment source (Wikipedia) is usable."""
        return _HTTPX_AVAILABLE

    def has_brave(self) -> bool:
        """True if Brave Search is configured and httpx is available."""
        return bool(self._brave_key and _HTTPX_AVAILABLE)

    def cache_size(self) -> int:
        return len(self._cache)

    # ── Wikipedia ──────────────────────────────────────────────────────────────

    def _wikipedia(self, poet_name: str, lang: str) -> dict | None:
        """
        MediaWiki API strategy:
          1. Search for the closest page title.
          2. Fetch the intro extract (plain text, no HTML).
          3. Parse birth year and Gulf region from the extract.

        No API key required. Rate-limited by RATE_LIMIT_DELAY.
        """
        base_url = _WIKI_AR_URL if lang == "ar" else _WIKI_EN_URL

        _wiki_headers = {"User-Agent": _USER_AGENT}

        # Step 1: search
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT, headers=_wiki_headers) as client:
                search_resp = client.get(base_url, params={
                    "action":      "query",
                    "list":        "search",
                    "srsearch":    poet_name,
                    "srnamespace": "0",
                    "srlimit":     "3",
                    "format":      "json",
                    "utf8":        "1",
                })
                search_resp.raise_for_status()
                hits = (search_resp.json().get("query") or {}).get("search") or []
        except Exception as exc:
            logger.debug("_wikipedia(%r, %s): search failed — %s", poet_name, lang, exc)
            return None

        if not hits:
            return None

        title = hits[0]["title"]
        time.sleep(RATE_LIMIT_DELAY)

        # Step 2: fetch intro extract
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT, headers=_wiki_headers) as client:
                ext_resp = client.get(base_url, params={
                    "action":      "query",
                    "titles":      title,
                    "prop":        "extracts",
                    "exintro":     "true",
                    "explaintext": "true",
                    "format":      "json",
                    "utf8":        "1",
                })
                ext_resp.raise_for_status()
                pages = (ext_resp.json().get("query") or {}).get("pages") or {}
        except Exception as exc:
            logger.debug("_wikipedia(%r, %s): extract failed — %s", poet_name, lang, exc)
            return None

        page    = next(iter(pages.values()), {})
        extract = (page.get("extract") or "").strip()

        if len(extract) < MIN_EXTRACT_CHARS:
            return None

        bio_text   = _first_sentences(extract, n=4)
        birth_year = _extract_year(bio_text)
        region     = _extract_region(bio_text)

        result: dict = {
            "poet_name":       poet_name,
            "normalised_name": poet_name,
            "bio_ar":          bio_text if lang == "ar" else "",
            "bio_en":          bio_text if lang == "en" else "",
            "sources":         [f"Wikipedia ({lang}) — {title}"],
            "enriched_via":    f"wikipedia_{lang}",
            "enriched_at":     _today(),
        }
        if birth_year:
            result["birth_year_approx"] = birth_year
        if region:
            result["region"] = region

        return result

    # ── Brave Search ───────────────────────────────────────────────────────────

    def _brave(self, poet_name: str) -> dict | None:
        """
        Query Brave Search for '{poet_name} شاعر نبطي خليجي'.
        Takes the top result's description as a bio snippet.
        Requires BRAVE_API_KEY in the environment.
        """
        if not self._brave_key:
            return None

        query = f"{poet_name} شاعر نبطي خليجي"
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(
                    _BRAVE_URL,
                    params={
                        "q":           query,
                        "count":       "3",
                        "search_lang": "ar",
                    },
                    headers={
                        "Accept":              "application/json",
                        "X-Subscription-Token": self._brave_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("_brave(%r): request failed — %s", poet_name, exc)
            return None

        results = ((data.get("web") or {}).get("results") or [])
        if not results:
            return None

        top         = results[0]
        description = (top.get("description") or "").strip()
        title       = top.get("title", "")
        url         = top.get("url", "")

        if len(description) < 20:
            return None

        bio_text   = _first_sentences(description, n=3)
        birth_year = _extract_year(bio_text)
        region     = _extract_region(bio_text)

        result: dict = {
            "poet_name":       poet_name,
            "normalised_name": poet_name,
            "bio_ar":          bio_text,
            "bio_en":          "",
            "sources":         [f"Brave Search — {title} ({url})"],
            "enriched_via":    "brave_search",
            "enriched_at":     _today(),
        }
        if birth_year:
            result["birth_year_approx"] = birth_year
        if region:
            result["region"] = region

        return result

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("poet_enrichment: cache load failed — %s", exc)
        return {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("poet_enrichment: cache save failed — %s", exc)


# ── Text utilities ─────────────────────────────────────────────────────────────

def _first_sentences(text: str, n: int = 4) -> str:
    """Return the first n sentences, splitting on . ! ? ؟ — handles Arabic punctuation."""
    sentences = re.split(r"(?<=[.!?؟])\s+", text.strip())
    return " ".join(sentences[:n]).strip()


def _extract_year(text: str) -> str | None:
    """
    Extract a CE year from text. Arabic Wikipedia writes CE years as '1790م'
    (miladi) and Hijri years as '1204هـ'. We prefer the CE year.
    """
    # Prefer explicit CE marker: digits followed by م (miladi)
    match = re.search(r"(?<!\d)(1[0-9]{3})م", text)
    if match:
        return match.group(1)
    # Fall back: any 4-digit run starting with 1 not adjacent to another digit
    match = re.search(r"(?<!\d)(1[0-9]{3})(?!\d)", text)
    return match.group(1) if match else None


def _extract_region(text: str) -> str | None:
    """Return the first Gulf region keyword found in the text."""
    for keyword in _REGION_KEYWORDS_AR:
        if keyword in text:
            return keyword
    return None


def _today() -> str:
    from datetime import date
    return date.today().isoformat()
