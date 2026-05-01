"""
src/al_nassikh/khaleeji_lexicon.py
====================================
⚡ SCOPE EXTENSION EXT-1 — Khaleeji/Emirati Dialect Lexicon

Why this file exists: Khaleeji Nabati poetry uses vocabulary, negation patterns,
and question markers that differ significantly from Modern Standard Arabic (MSA).
A query typed in Khaleeji dialect (e.g. "وين الغيث") will fail BM25 matching
against manuscripts indexed with MSA normalisation (e.g. "أين المطر").

This module provides:
  1. A curated bilingual lexicon: Khaleeji term → MSA equivalent → English gloss.
  2. khaleeji_to_msa(text) — normalises Khaleeji text to its closest MSA form,
     improving BM25 recall for dialect queries.
  3. detect_khaleeji_terms(text) — returns all Khaleeji tokens found in the text,
     used by the intent router to set detected_dialect = "khaleeji".
  4. A registered tool: khaleeji_dialect_bridge(term) → dict — callable by Agent 1
     via the tool registry when it detects a dialect term worth expanding.

Architecture ref: §2.9 (extensible tool registry), §2.4 Stage 2b (bilingual
expansion uses this to add Khaleeji variant queries), al_nassikh/ (Worker 1
normalisation applies khaleeji_to_msa before embedding).
"""

from __future__ import annotations

import re
from typing import Optional

# ── Lexicon ───────────────────────────────────────────────────────────────────
# Format: khaleeji_term → {msa: str, en: str, domain: str, notes: str}
# domain: "grammar" | "poetry" | "daily" | "nature" | "emotion"

KHALEEJI_LEXICON: dict[str, dict] = {
    # ── Question markers ──────────────────────────────────────────────────────
    "شنو":    {"msa": "ماذا",    "en": "what",         "domain": "grammar",  "notes": "Emirati/Gulf question marker"},
    "شو":     {"msa": "ماذا",    "en": "what",         "domain": "grammar",  "notes": "Common Khaleeji"},
    "وين":    {"msa": "أين",     "en": "where",        "domain": "grammar",  "notes": "Khaleeji/Iraqi — 'أين' in MSA"},
    "ليش":    {"msa": "لماذا",   "en": "why",          "domain": "grammar",  "notes": "Khaleeji dialect"},
    "شلون":   {"msa": "كيف",     "en": "how",          "domain": "grammar",  "notes": "Gulf/Levantine"},
    "كيفك":   {"msa": "كيف حالك","en": "how are you",  "domain": "grammar",  "notes": ""},
    "متى":    {"msa": "متى",     "en": "when",         "domain": "grammar",  "notes": "Same in both"},

    # ── Negation ─────────────────────────────────────────────────────────────
    "مو":     {"msa": "ليس",     "en": "is not",       "domain": "grammar",  "notes": "Core Khaleeji negation"},
    "ماب":    {"msa": "ليس",     "en": "is not",       "domain": "grammar",  "notes": "Emirati: ما + ب"},
    "مو ذا":  {"msa": "ليس هذا", "en": "not this",     "domain": "grammar",  "notes": ""},
    "ما راح": {"msa": "لن يذهب", "en": "will not go",  "domain": "grammar",  "notes": "Future negation"},

    # ── Pronouns ──────────────────────────────────────────────────────────────
    "اني":    {"msa": "أنا",     "en": "I",            "domain": "grammar",  "notes": "Gulf first person"},
    "انا":    {"msa": "أنا",     "en": "I",            "domain": "grammar",  "notes": "Also MSA"},
    "انتي":   {"msa": "أنتِ",    "en": "you (f)",      "domain": "grammar",  "notes": "Feminine second person"},
    "انتو":   {"msa": "أنتم",    "en": "you (pl)",     "domain": "grammar",  "notes": "Plural second person"},
    "هو":     {"msa": "هو",      "en": "he",           "domain": "grammar",  "notes": "Same"},
    "هي":     {"msa": "هي",      "en": "she",          "domain": "grammar",  "notes": "Same"},
    "احنا":   {"msa": "نحن",     "en": "we",           "domain": "grammar",  "notes": "Gulf/Levantine"},
    "هم":     {"msa": "هم",      "en": "they",         "domain": "grammar",  "notes": "Same"},

    # ── Common words ──────────────────────────────────────────────────────────
    "جان":    {"msa": "كان",     "en": "was/were",     "domain": "grammar",  "notes": "Gulf past tense form of كان"},
    "يبه":    {"msa": "يا أبي",  "en": "oh father",    "domain": "emotion",  "notes": "Affectionate address, common in Nabati poetry"},
    "فدوة":   {"msa": "فداء",    "en": "sacrifice/devotion","domain": "emotion","notes": "Khaleeji expression of love/loyalty"},
    "بعد":    {"msa": "أيضاً",   "en": "also/still",   "domain": "grammar",  "notes": "As discourse marker; not the preposition 'after'"},
    "حيل":    {"msa": "جداً",    "en": "very/much",    "domain": "grammar",  "notes": "Intensifier in Gulf dialect"},
    "زين":    {"msa": "حسن",     "en": "good/fine",    "domain": "daily",    "notes": "Universal Gulf approval"},
    "خوش":    {"msa": "جيد",     "en": "good",         "domain": "daily",    "notes": "Common in UAE/Kuwait"},
    "عشان":   {"msa": "لأجل",    "en": "because/for",  "domain": "grammar",  "notes": "Causal connector"},
    "هالـ":   {"msa": "هذا الـ", "en": "this",         "domain": "grammar",  "notes": "Demonstrative contraction"},
    "ذاك":    {"msa": "ذلك",     "en": "that",         "domain": "grammar",  "notes": "Gulf demonstrative"},
    "عيل":    {"msa": "طفل",     "en": "child",        "domain": "daily",    "notes": "Singular; 'عيال' for plural"},
    "عيال":   {"msa": "أطفال",   "en": "children",     "domain": "daily",    "notes": ""},
    "ربع":    {"msa": "أصدقاء",  "en": "friends/companions","domain": "daily","notes": "Very Khaleeji — close companions"},
    "صاحب":   {"msa": "صديق",    "en": "friend",       "domain": "daily",    "notes": "Also MSA"},

    # ── Nature / Desert / Sea ─────────────────────────────────────────────────
    "يم":     {"msa": "بحر",     "en": "sea",          "domain": "nature",   "notes": "Gulf word for the sea — very common in Nabati poetry"},
    "غيث":    {"msa": "مطر",     "en": "rain",         "domain": "nature",   "notes": "Life-giving rain; highly symbolic in desert poetry"},
    "فلا":    {"msa": "صحراء",   "en": "desert/open land","domain": "nature","notes": "The open desert as spiritual space in Nabati poetry"},
    "قفر":    {"msa": "أرض قاحلة","en": "barren land", "domain": "nature",   "notes": "Emptiness, loneliness"},
    "نجد":    {"msa": "المرتفعات","en": "Najd highlands","domain": "nature",  "notes": "Geographic/poetic reference"},
    "بيداء":  {"msa": "صحراء",   "en": "vast desert",  "domain": "nature",   "notes": "Poetic desert imagery"},
    "سرب":    {"msa": "قطيع",    "en": "flock/group",  "domain": "nature",   "notes": "Birds or camels; migratory imagery"},
    "ظعن":    {"msa": "رحل",     "en": "journey/departure","domain": "nature","notes": "The caravan departure — central Nabati theme"},

    # ── Emotional / Poetic vocabulary ─────────────────────────────────────────
    "هجران":  {"msa": "هجر",     "en": "abandonment",  "domain": "emotion",  "notes": "The pain of being left behind — core Nabati theme"},
    "شوق":    {"msa": "شوق",     "en": "longing",      "domain": "emotion",  "notes": "Same in both; core Nabati theme"},
    "حنين":   {"msa": "حنين",    "en": "yearning/nostalgia","domain": "emotion","notes": "Longing for homeland or beloved"},
    "غرام":   {"msa": "غرام",    "en": "passionate love","domain": "emotion", "notes": ""},
    "هوى":    {"msa": "هوى",     "en": "desire/love",  "domain": "emotion",  "notes": "Romantic longing"},
    "لوعة":   {"msa": "ألم",     "en": "heartache",    "domain": "emotion",  "notes": "Burning grief"},
    "دموع":   {"msa": "دموع",    "en": "tears",        "domain": "emotion",  "notes": "Same"},
    "صبر":    {"msa": "صبر",     "en": "patience",     "domain": "emotion",  "notes": "Same"},
    "وجد":    {"msa": "حزن",     "en": "grief/passion","domain": "emotion",   "notes": "Intense emotional suffering"},
    "لهفة":   {"msa": "شوق",     "en": "eager longing","domain": "emotion",   "notes": ""},

    # ── Praise / Tribal / Honor ───────────────────────────────────────────────
    "قبيلة":  {"msa": "قبيلة",   "en": "tribe",        "domain": "poetry",   "notes": "Tribal identity central to مديح"},
    "كرم":    {"msa": "كرم",     "en": "generosity",   "domain": "poetry",   "notes": "Core virtue in Nabati مديح"},
    "شجاعة":  {"msa": "شجاعة",   "en": "courage",      "domain": "poetry",   "notes": "Heroic virtue"},
    "وفاء":   {"msa": "وفاء",    "en": "loyalty",      "domain": "poetry",   "notes": ""},
    "عزّ":    {"msa": "عزة",     "en": "glory/dignity","domain": "poetry",   "notes": ""},
    "سخاء":   {"msa": "سخاء",    "en": "open-handedness","domain": "poetry",  "notes": "Praised in مديح"},

    # ── Poetic / Prosody terms ────────────────────────────────────────────────
    "مطلع":   {"msa": "البيت الأول","en": "opening verse","domain": "poetry", "notes": "First verse/couplet of a Nabati poem"},
    "عجز":    {"msa": "الشطر الثاني","en": "second hemistich","domain": "poetry","notes": "Second half of a verse"},
    "صدر":    {"msa": "الشطر الأول","en": "first hemistich","domain": "poetry","notes": "First half of a verse"},
    "قافية":  {"msa": "قافية",   "en": "rhyme scheme", "domain": "poetry",   "notes": "End rhyme pattern"},
    "وزن":    {"msa": "وزن",     "en": "poetic meter", "domain": "poetry",   "notes": "Al-Khalil's 16 metres"},
    "شطر":    {"msa": "مصراع",   "en": "hemistich",    "domain": "poetry",   "notes": "Half a verse line"},
    "نبطي":   {"msa": "شعر عامي خليجي","en": "Nabati dialect poetry","domain": "poetry","notes": "The genre itself"},
    "خليجي":  {"msa": "خليجي",   "en": "Khaleeji/Gulf","domain": "poetry",   "notes": ""},

    # ── Emirati-specific ──────────────────────────────────────────────────────
    "بالكيف": {"msa": "بالرضا",  "en": "contentedly",  "domain": "daily",    "notes": "UAE expression of satisfaction"},
    "يلا":    {"msa": "هيا",     "en": "let's go / come on","domain": "daily","notes": "Universal Gulf"},
    "والله":  {"msa": "والله",   "en": "by God / truly","domain": "grammar",  "notes": "Oath/emphasis, same"},
    "إن شاء الله":{"msa": "إن شاء الله","en": "God willing","domain": "grammar","notes": "Same"},
    "ما شاء الله":{"msa": "ما شاء الله","en": "how wonderful","domain": "daily","notes": "Admiration"},
}

# ── Phonological substitution patterns ───────────────────────────────────────
# Why these: Khaleeji speech often realises certain consonants differently.
# Applying these before BM25 indexing/querying improves cross-dialect match.
# Pattern: (khaleeji_pattern, msa_replacement)

_PHONOLOGICAL_SUBS: list[tuple[str, str]] = [
    # ق → ج in many Gulf dialects (e.g. جلب vs قلب)
    # Note: we do NOT apply this universally — only in known stopwords
    # because it would corrupt poet names.
    (r"\bجلب\b", "قلب"),
    (r"\bجيل\b", "قيل"),
    # ث → ت (common in Gulf)
    (r"\bثلاثة\b", "ثلاثة"),  # keep as-is — same in manuscripts
    # Alef maqsura / ya variants
    (r"ى$", "ي"),
    # Remove tatweel (kashida)
    ("ـ", ""),
]


# ── Public API ────────────────────────────────────────────────────────────────

def khaleeji_to_msa(text: str) -> str:
    """
    Normalise Khaleeji dialect text to MSA equivalents.
    Used in embed.py before BM25 tokenisation and dense encoding.

    Only substitutes whole-word matches to avoid corrupting proper nouns.
    Returns the normalised text.
    """
    tokens = text.split()
    normalised = []
    for token in tokens:
        clean = token.strip("،.,؟?!()[]")
        entry = KHALEEJI_LEXICON.get(clean)
        if entry and entry["domain"] in ("grammar", "daily"):
            # Only bridge grammar/daily words — keep poetry/emotion/nature vocabulary
            # because those are what we want to FIND in the corpus, not replace.
            normalised.append(entry["msa"])
        else:
            normalised.append(token)
    result = " ".join(normalised)
    for pattern, replacement in _PHONOLOGICAL_SUBS:
        result = re.sub(pattern, replacement, result)
    return result


def detect_khaleeji_terms(text: str) -> list[dict]:
    """
    Return a list of Khaleeji terms found in the text, with their MSA equivalents.
    Used by intent_router to set detected_dialect = "khaleeji".
    """
    found = []
    tokens = text.split()
    for token in tokens:
        clean = token.strip("،.,؟?!()[]")
        entry = KHALEEJI_LEXICON.get(clean)
        if entry:
            found.append({
                "khaleeji": clean,
                "msa":      entry["msa"],
                "en":       entry["en"],
                "domain":   entry["domain"],
            })
    return found


def khaleeji_dialect_bridge(term: str) -> dict:
    """
    Tool callable by Agent 1 (§2.9 tool registry extension EXT-1).
    Given a Khaleeji term, returns its MSA equivalent, English gloss,
    domain classification, and usage notes.

    Returns a structured dict; returns {"found": False} if term not in lexicon.
    """
    entry = KHALEEJI_LEXICON.get(term.strip())
    if entry is None:
        return {
            "found":    False,
            "term":     term,
            "message":  f"'{term}' not in Khaleeji lexicon — may be MSA or a name",
        }
    return {
        "found":    True,
        "term":     term,
        "msa":      entry["msa"],
        "en":       entry["en"],
        "domain":   entry["domain"],
        "notes":    entry["notes"],
        "bridge":   f"Searching corpus for both '{term}' and '{entry['msa']}'",
    }


def list_poetry_vocabulary() -> list[dict]:
    """Return all poetry-domain entries — useful for UI display and discovery mode."""
    return [
        {"term": k, **v}
        for k, v in KHALEEJI_LEXICON.items()
        if v["domain"] in ("poetry", "emotion", "nature")
    ]


def lexicon_size() -> int:
    """Return the number of entries in the lexicon."""
    return len(KHALEEJI_LEXICON)
