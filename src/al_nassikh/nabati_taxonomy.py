"""
src/al_nassikh/nabati_taxonomy.py
==================================
Why this file exists: The genre + emotion enrichment pass (M2d) classifies
1,502 verses into a fixed label space. Without a frozen taxonomy, every
classifier invocation invents its own labels (حماسة vs. فخر vs. بطولة all mean
roughly the same thing), and the retrieval facet UI ends up with 40
near-duplicate buttons instead of 10 clean ones. Freezing the label space
*before* any classifier runs is the single biggest lever on output quality.

The taxonomy below is grounded in three sources:
  1. Sowayan's own classification of Najdi Nabati (7 genres + war narrative)
  2. The 137 non-empty `occasion` fields in the Phase-4 anchor registry,
     which revealed that a huge fraction of this corpus is war/raid poetry
     (بذبحة X, يرد على Y) — a category missing from the original taxonomy.
  3. Standard Arabic literary classification (غزل, رثاء, مديح, حكمة, وصف, فخر,
     هجاء, حماسة, دينية) — established for 1,400 years so we are not
     inventing labels, we are picking the subset that fits Nabati.

Why include GHAZW (war narrative) as its own class: the corpus is heavily
dominated by battle/raid poems (Sowayan's own collection is named for this).
Folding them into فخر (boasting) or حماسة (valour) loses the distinction
between "I am brave" (فخر) and "here is what happened when tribe X raided
tribe Y" (غزو). The latter is narrative, the former is lyric.

Why emotions are English keys while genres are Arabic:
  - Genre labels are *Arabic literary terms* with precise technical meanings
    — there is no stable English translation of حكمة that a reviewer would
    recognise. Keep the Arabic.
  - Emotion labels are *psychological primitives* the UI surfaces as facets.
    Bilingual users of both Streamlit tabs will parse "longing" and "grief"
    in either language; keeping them English avoids the Ekman-vs-Plutchik
    translation debate in Arabic emotional vocabulary.

Consumers of this module:
  - genre_heuristic.py           — classifier (this milestone)
  - scripts/enrich_genre_heuristic.py — batch driver
  - fatat_al_arab/embed.py       — Qdrant payload schema
  - agent1_query_understanding/nodes/self_query.py — filter extraction schema
  - app/tabs/scholar_workbench.py — facet sidebar
"""

from __future__ import annotations

# ── Genre space (Arabic literary terms) ───────────────────────────────────────
# Ordering matters: the UI facet sidebar renders in this order, so common
# genres come first. غير_محدد is LAST so it visually reads as "everything else".

GENRES: tuple[str, ...] = (
    "غزل",      # love / romantic
    "رثاء",     # elegy / lament for the dead
    "مديح",     # praise (tribe, patron, ruler, prophet)
    "فخر",      # boasting / self-praise
    "غزو",      # raid / war narrative — heavy in this corpus, see Sowayan
    "حماسة",    # valour / war exhortation (lyric, not narrative)
    "هجاء",     # satire / invective
    "حكمة",     # wisdom / gnomic / proverbial
    "وصف",      # descriptive (nature, camel, horse, landscape)
    "دينية",    # religious / devotional
    "غير_محدد", # UNSPECIFIED — explicit abstention, never silent default
)

# One-line human-readable definitions. Keep short: the Streamlit hover card
# renders these as tooltip text; the LLM enrichment pass (if we add it later)
# uses them as the system-prompt taxonomy.
GENRE_DEFINITIONS: dict[str, str] = {
    "غزل":      "Love and romantic verse — beloved, longing, separation, reunion.",
    "رثاء":     "Elegy — lament for the dead, eulogy of a fallen kinsman or leader.",
    "مديح":     "Praise poetry — tribe, ruler, patron, the Prophet; generosity and nobility.",
    "فخر":      "Boasting / self-praise — lyric assertion of one's own or one's tribe's worth.",
    "غزو":      "Raid / war narrative — specific battles, retaliations, named adversaries.",
    "حماسة":    "Valour / war exhortation — courage as a general theme, not a named battle.",
    "هجاء":     "Satire / invective — ridicule of a named rival or rival tribe.",
    "حكمة":     "Wisdom / gnomic — proverbs, moral reflections, life lessons.",
    "وصف":      "Descriptive — nature, camels, horses, landscape, weather as primary subject.",
    "دينية":    "Religious / devotional — God, prayer, piety, pilgrimage.",
    "غير_محدد": "Unclassified — classifier abstained or evidence was insufficient.",
}

# ── Emotion space (English keys, Plutchik-inspired) ───────────────────────────
# Why 10 not 8 or 20: fewer than 8 collapses distinct Nabati registers (pride
# vs. awe vs. defiance all become "positive"); more than 12 exceeds what a
# keyword classifier can reliably distinguish without an LLM. Ten is where
# precision holds.

EMOTIONS: tuple[str, ...] = (
    "longing",    # شوق, حنين — the signature Nabati emotion
    "grief",      # حزن, أسى, دموع
    "joy",        # فرح, سعادة, بشر
    "pride",      # عزة, فخر, كبرياء
    "anger",      # غضب, غيظ, سخط
    "love",       # حب, ود, مودة
    "awe",        # هيبة, إجلال — especially in praise/religious genres
    "nostalgia",  # ذكرى, ماضٍ — recall of places and people
    "hope",       # رجاء, أمل, وعد
    "fear",       # خوف, وجل, هلع
)

EMOTION_DEFINITIONS: dict[str, str] = {
    "longing":   "Yearning for an absent person or place; the central emotion of غزل.",
    "grief":     "Sorrow, mourning, weeping; dominates رثاء.",
    "joy":       "Happiness, celebration, delight; often tied to reunion or victory.",
    "pride":     "Honour, dignity, tribal assertion; central to فخر and مديح.",
    "anger":     "Wrath, indignation, hostility; powers هجاء and parts of غزو.",
    "love":      "Affection or devotion — broader than longing; includes ود/مودة.",
    "awe":       "Reverence for a greater power or figure; central to دينية and مديح.",
    "nostalgia": "Memory of a past place or time; distinct from longing (which targets a person).",
    "hope":      "Expectation of future good; promise, trust, anticipation.",
    "fear":      "Apprehension, dread, vulnerability; often paired with pride in war verse.",
}


# ── Validators (called by the classifier and the enrichment script) ───────────
# Why these live here and not in genre_heuristic.py: any consumer should be
# able to import `validate_genre` without pulling in the entire classifier,
# and having the validator next to the frozen lists means they cannot drift.

def validate_genre(label: str) -> str:
    """Return the label if valid, else 'غير_محدد'. Never raises — drift-safe."""
    return label if label in GENRES else "غير_محدد"


def validate_emotions(labels: list[str]) -> list[str]:
    """Filter to valid emotion labels. Drops invalids silently; preserves order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in labels:
        if x in EMOTIONS and x not in seen:
            out.append(x)
            seen.add(x)
    return out


__all__ = [
    "GENRES", "GENRE_DEFINITIONS",
    "EMOTIONS", "EMOTION_DEFINITIONS",
    "validate_genre", "validate_emotions",
]
