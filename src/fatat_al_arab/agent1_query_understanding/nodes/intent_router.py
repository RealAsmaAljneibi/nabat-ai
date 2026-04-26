"""
agent1_query_understanding/nodes/intent_router.py
==================================================
Why this file exists: On 2026-04-23 Asma ran the app and asked
"how many poems do I have?" — the query went straight into the semantic
retrieval pipeline, which is built to find poems, not count them. CRAG
marked the retrieved verses "Incorrect" (none of them contained a count)
and the orchestrator fired the refusal template. This node is the fix.

It sits as Stage 0.5 of Agent 1 — between state initialisation and the
bilingual_analyzer LLM call — and detects three deterministic question
classes the corpus_stats module can answer without an LLM or a vector
search:

  1. counting   — "how many poems / poets / manuscripts / pages"
  2. age        — "how old are the manuscripts / from what century"
  3. provenance — "where are they from / which regions / who collected them"

When one of these is detected with confidence ≥ 0.7 the node sets
`query_context.answer_source = "registry_lookup"` + a routing flag read by
the orchestrator, which then calls the deterministic_answer_node instead
of HyDE + retrieval. When confidence is < 0.7 the node does nothing and
the full RAG pipeline runs normally — we never short-circuit on an
ambiguous query.

Design principle: pure regex, no LLM. The patterns are bilingual (English
+ Arabic, including Khaleeji variants like "كم" / "شو عدد"). A single LLM
call here would cost ~400ms and defeat the point of a fast-path router.

Architecture ref: doc/IMPLEMENTATION_PLAN.md M2c — "Intent Router +
Deterministic Answer Path". §2.4 new Stage 0.5.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ...state import AgentState, make_query_context

logger = logging.getLogger(__name__)


# ── Intent pattern library ────────────────────────────────────────────────────
# Why two-pass (intent → slot): we first decide *which* class of question this
# is, then separately extract *which slot* the user is asking about (poems vs.
# poets vs. manuscripts). That way adding a new slot (e.g. "how many pages")
# means extending one list, not writing a new regex for the whole question.
#
# Confidence scoring:
#   +0.6  strong counting/age/provenance cue matches
#   +0.3  slot word matches (poems, poets, manuscripts, …)
#   +0.1  bonus if question ends with '?'
# Capped at 1.0. The router fires only if total ≥ 0.7.

_COUNTING_CUES = [
    # English
    r"\bhow many\b",
    r"\bcount of\b",
    r"\bnumber of\b",
    r"\btotal (number of|count of)?\b",
    r"\bsize of (the )?(corpus|collection|dataset)\b",
    # Arabic (MSA + Khaleeji)
    r"كم(\s+عدد)?",          # كم / كم عدد
    r"شو\s+عدد",             # شو عدد (Khaleeji)
    r"كمية",                 # كمية
    r"عدد\s+",               # عدد X
    r"مجموع\s+",             # مجموع X
]

_AGE_CUES = [
    r"\bhow old\b",
    r"\bwhat (century|era|period|year)\b",
    r"\bage of (the )?(manuscripts?|poems?|collection)\b",
    r"\bdate(d)? (from|to)\b",
    r"\btime period\b",
    r"\bwhen were .* (written|collected|compiled)\b",
    # Arabic
    r"كم\s+عمر",
    r"ما\s+(عمر|تاريخ)",
    r"في\s+أي\s+(قرن|عصر|سنة)",
    r"من\s+أي\s+(قرن|عصر|حقبة)",
    r"متى\s+(كُتبت|جُمعت|تم\s+تأليف)",
]

_PROVENANCE_CUES = [
    r"\bwhere are (they|these|the manuscripts?) from\b",
    r"\bwhere (did|do) .* come from\b",
    r"\bwhat (region|country|area|place)s?\b",
    r"\bregions? of origin\b",
    r"\bwho (collected|compiled|gathered)\b",
    r"\bcollectors?\b",
    # Arabic
    r"من\s+أين",
    r"(أين|وين)\s+من",
    r"ما\s+(منطقة|أصل|مكان)",
    r"أي\s+(منطقة|بلد)",
    r"من\s+(جمع|ألّف|دوّن)",
]

# Queries asking to name/list poets are corpus metadata questions, not verse
# retrieval questions. Without this explicit cue, "who are the poets..." falls
# into RAG and the guardrails refuse a noisy generated list.
_LIST_POETS_CUES = [
    r"\bwho are (the )?(poets|authors)\b",
    r"\blist (the )?(poets|authors)\b",
    r"\bshow (me )?(the )?(poets|authors)\b",
    r"\bwhich (poets|authors)\b",
    r"من\s+هم\s+(الشعراء|الشاعر)",
    r"اذكر\s+(الشعراء|أسماء\s+الشعراء)",
    r"اعرض\s+(الشعراء|أسماء\s+الشعراء)",
    r"قائمة\s+(الشعراء|أسماء\s+الشعراء)",
]

_CHILD_GRIEF_CUES = [
    r"\b(griev\w*|mourn\w*|lament\w*|eleg(y|ies)|ritha|rithaa)\b.*\b(son|daughter|child|children)\b",
    r"\b(son|daughter|child|children)\b.*\b(griev\w*|mourn\w*|lament\w*|eleg(y|ies)|ritha|rithaa)\b",
    r"رثاء.*(ابن|بنت|ولد|ولده|ولده|طفل|أبناء|ابناء)",
    r"(ابن|بنت|ولد|ولده|ولده|طفل|أبناء|ابناء).*(رثاء|حزن|فقد|موت|وفاة|مرثية)",
]

# ── Capabilities & instructor-debug fast-path patterns ───────────────────────
# Why in the regex router (not just semantic_router): when the LLM call times
# out or fails (e.g. bad API key), Stage 0.5b defaults to poetic_rag. These
# patterns give a zero-LLM fallback so "how can you help me" never goes to RAG.
_CAPABILITIES_CUES = [
    r"\bhow (can|do) (you|u|i|we)\b",        # "how can you/u help me"
    r"\bwhat can (you|u) (do|search|find|cover|help)\b",
    r"\bwhat (are your|do you) (features?|capabilities|functions?)\b",
    r"\bwhat (does|do) (this|the) (system|tool|app|assistant)\b",
    r"\bwhat (topics?|subjects?) (do you|does this|can you)\b",
    r"\btell me (about|what) (you|this system)\b",
    r"\bhow (do|does) (this|the) (system|tool|assistant) work\b",
    r"\bintroduc(e|tion)\b",
    r"\bhelp me understand what (you|this)\b",
    r"\bwhat (is|are) (this|your) (purpose|goal|function)\b",
    # Arabic — broad patterns for "what can you do / how can you help"
    r"كيف (يمكنك|تستطيع|يمكن لك)",
    r"ماذا (تستطيع|يمكنك|تفعل)",
    r"ما\s+(الذي|يمكن)\s+(تستطيع|يمكنك|تقدر|تفعل)",  # ما الذي تستطيع فعله
    r"(تستطيع|تقدر|يمكنك)\s+(فعل|عمل|القيام|مساعدة|البحث)",
    r"ما (هي|هو) (إمكانياتك|وظائفك|قدراتك|ميزاتك)",
    r"كيف (يعمل|يشتغل) (هذا|النظام)",
    r"عرّفني (على|بـ)",
    r"ما\s+(الذي|هو)\s+(يستطيع|يمكن).*النظام",   # ما الذي يستطيع هذا النظام
    r"(أخبرني|قل لي)\s+(ما|عن|ماذا)\s+(تفعل|تستطيع|يمكنك)",
]

_INSTRUCTOR_DEBUG_CUES = [
    r"\b(crag|self.?rag|rrf|hyde|self.?query)\b",
    r"\b(debug|pipeline|verdict|fallback|introspect)\b",
    r"\blast turn\b",
    r"\bexplain (the|your|how)\b.*(retriev|pipeline|filter|fusion|rank)",
    r"\bshow (me )?(the )?(last|previous|debug|pipeline)\b",
    r"\bwhat (filters?|did you extract|self.?query)\b",
]

# ── Image-grounded provenance ────────────────────────────────────────────────
# Why a separate cue list: when the user uploads an image of a verse and asks
# a meta-question ("who wrote this?", "what poem is this?"), the typed text
# alone has zero retrieval signal — the *image* carries the search key. This
# router check fires only when state["input_image_path"] is set AND the typed
# question matches one of these patterns; the dedicated image_provenance node
# then runs verse-level retrieval and produces a *specific* answer (cited
# source if found, "image read but verse not in corpus" if not) instead of
# the generic refusal that previously came back identically for both cases.

_IMAGE_PROVENANCE_CUES = [
    # English meta-questions about what the image shows
    r"\bwho\s+(wrote|composed|authored|is\s+the\s+(author|poet)\s+of)\s+(this|it)\b",
    r"\bwhat\s+(poem|verse|qasida|line|matla)\s+is\s+this\b",
    r"\bwhich\s+(poem|verse|poet|manuscript)\b",
    r"\bis\s+this\s+(in|from)\s+(the\s+)?(corpus|archive|collection|manuscript)",
    r"\bidentify\s+(this|the)\s+(poem|verse|poet|line)",
    r"\bwhere\s+(is|does)\s+(this|it)\s+(come\s+from|appear)",
    # Arabic — MSA + Khaleeji
    r"من\s+(كتب|قال|نظم|ألف|كاتب)\s+(هذا|هذه|هذي|ذي)",
    r"ما\s+(هذه|هذي|ذي)\s+(القصيدة|الأبيات|البيت)",
    r"أيّ\s+(قصيدة|شاعر|مخطوطة)",
    r"(هل|أ)\s*(هذا|هذه)\s+في\s+(المجموعة|الأرشيف|المخطوطات)",
    r"عرّف\s+(هذا|هذه)\s+(البيت|القصيدة|الأبيات)",
    r"شو\s+هاي?\s+(القصيدة|الأبيات)",   # Khaleeji "what is this poem"
    r"مين\s+(كتب|قال)\s+(هاي?|هذي?)",  # Khaleeji "who wrote this"
]

# Slot vocabularies — what is being counted / asked about
_SLOT_POEMS = [
    r"\bpoems?\b", r"\bqasidas?\b", r"\bverses?\b",
    r"قصائد", r"قصيدة", r"أبيات", r"بيت",
]
_SLOT_POETS = [
    r"\bpoets?\b", r"\bauthors?\b",
    r"شعراء", r"شاعر",
]
_SLOT_MANUSCRIPTS = [
    r"\bmanuscripts?\b", r"\bcodex(es)?\b", r"\bcodices\b", r"\bvolumes?\b",
    r"مخطوطات", r"مخطوطة", r"مجلدات", r"مجلد",
]
_SLOT_PAGES = [
    r"\bpages?\b", r"\bfolios?\b",
    r"صفحات", r"صفحة", r"أوراق", r"ورقة",
]
_SLOT_CORPUS = [
    r"\bcorpus\b", r"\bcollection\b", r"\bdataset\b", r"\barchive\b", r"\blibrary\b",
    r"المجموعة", r"الأرشيف", r"المكتبة", r"المخزون",
]

# ── Genre slot detection ──────────────────────────────────────────────────────
# Why a separate slot dimension: "how many love poems" is a counting question
# *narrowed by genre*. Without this map the router strips the qualifier and
# answers with the unfiltered total — exactly what Asma reported on 2026-04-25.
# Maps English / Arabic / Khaleeji surface forms → frozen taxonomy label from
# nabati_taxonomy.GENRES so the deterministic answer node and the LLM prose
# pass speak the same Arabic label the corpus is tagged with.

_GENRE_SLOT_MAP: list[tuple[str, list[str]]] = [
    ("غزل", [
        r"\blove\b", r"\bromantic\b", r"\bromance\b", r"\bbeloved\b",
        r"\blonging\b", r"\bghazal\b", r"\bnasib\b",
        r"غزل", r"غزلية", r"حب", r"عشق",
    ]),
    ("رثاء", [
        r"\belegy\b", r"\belegies\b", r"\blament\b", r"\bmourn(ing)?\b",
        r"\beulogy\b", r"\bmarthiya\b", r"\britha?\b",
        r"رثاء", r"مرثية", r"عزاء",
    ]),
    ("مديح", [
        r"\bpraise\b", r"\bpanegyric\b", r"\bmadih\b", r"\beulogi", r"\bextol",
        r"مديح", r"مدح", r"مدائح",
    ]),
    ("فخر", [
        r"\bboast(ing)?\b", r"\bself.?praise\b", r"\bfakhr\b",
        r"فخر", r"تفاخر",
    ]),
    ("غزو", [
        r"\braid\b", r"\braiding\b", r"\bbattle\b", r"\bbattles\b", r"\bwar\b(?!.*exhortation)",
        r"\bghazw\b", r"\bghaazw\b", r"\bwar.?narrative\b",
        r"غزو", r"غزوات", r"حروب", r"حرب", r"معارك", r"معركة",
    ]),
    ("حماسة", [
        r"\bvalou?r\b", r"\bbravery\b", r"\bcourage\b", r"\bhamasa\b",
        r"\bwar.?exhortation\b",
        r"حماسة", r"شجاعة", r"بسالة",
    ]),
    ("هجاء", [
        r"\bsatire\b", r"\binvective\b", r"\bridicule\b", r"\bhija(\'|a)?\b",
        r"هجاء", r"تهكم",
    ]),
    ("حكمة", [
        r"\bwisdom\b", r"\bgnomic\b", r"\bproverb(s|ial)?\b", r"\bhikma\b",
        r"حكمة", r"حكم", r"أمثال",
    ]),
    ("وصف", [
        r"\bdescriptive\b", r"\bdescription\b", r"\bwasf\b",
        r"وصف", r"وصفية",
    ]),
    ("دينية", [
        r"\breligious\b", r"\bdevotional\b", r"\bspiritual\b", r"\bpiety\b",
        r"\bsacred\b", r"\bdiniyya\b",
        r"دينية", r"ديني", r"تعبدي",
    ]),
]


def _detect_genre_slot(q: str) -> Optional[str]:
    """
    Return the canonical Arabic genre label if the query mentions a genre,
    else None. First match wins; the map is ordered by intuitive query frequency
    (love > elegy > praise > …) so the common case resolves on the first scan.
    """
    for label, patterns in _GENRE_SLOT_MAP:
        if _any_match(patterns, q):
            return label
    return None

# ── Unsupported-dimension allow-list ────────────────────────────────────────
# Why a separate allow-list: these queries ask about annotation fields that are
# NOT indexed in the corpus (tribal marks, marginalia, secondary scribes, etc.).
# Routing them to the RAG pipeline returns 0 useful results. Instead we catch
# them here and route to the educational "unsupported_dim" answer path so the
# user gets a helpful bilingual note + general context, not a confusing refusal.
#
# Format: (dim_name, pattern_list). dim_name becomes the subintent key.

_UNSUPPORTED_DIMS: list[tuple[str, list[str]]] = [
    ("wasm", [
        r"\bwasm\b", r"\bwusoom\b", r"\bwasum\b", r"\btribal[\s-]mark",
        r"\btribal[\s-]brand", r"وسم", r"وسوم",
    ]),
    ("marginalia", [
        r"\bmarginali[ae]\b", r"\bmargin[\s-]notes?\b", r"\bhashiya\b",
        r"\bgloss(es)?\b", r"\bannotation[s]?\b.*margin",
        r"هامش", r"حواشي", r"تعليقات\s+هامشية",
    ]),
    ("secondary_scribe", [
        r"\bsecondary\s+scrib", r"\bsecond\s+hand\b", r"\blater\s+hand\b",
        r"\banother\s+scrib", r"\bنسّاخ\s+ثانٍ", r"\bخط\s+ثانٍ",
    ]),
    ("ink_bleed", [
        r"\bink[\s-]bleed", r"\bbleed[\s-]through", r"\bshow[\s-]through",
        r"\bghost[\s-]ink", r"\bnazif\s+al[\s-]hibr\b",
        r"نزيف\s+الحبر", r"تسرب\s+الحبر",
    ]),
    ("library_stamp", [
        r"\blibrary[\s-]stamp", r"\bstamp[\s-]coord", r"\bseals?\s+(?:location|bbox)",
        r"\bbounding\s+box.*stamp", r"\bختم\b.*إحداثيات", r"إحداثيات\s+الختم",
    ]),
    ("handwriting_style", [
        r"\b(naskh|nasakh|naskhi|ruqah|ruq[`']ah|diwani|thuluth)\b",
        r"\bhandwriting\s+style\b", r"\bscript\s+style\b",
        r"\bwriting\s+style\b", r"\bcalligraph",
        r"\bscribal\s+hand\b", r"\bscribe\s+style\b",
        r"خط\s+(نسخ|رقعة|ديواني|ثلث|نسخي)",
        r"أسلوب\s+(الكتابة|الخط)",
        r"نمط\s+الخط",
    ]),
]


def _any_match(patterns: list[str], text: str) -> bool:
    """Case-insensitive, Unicode-aware scan."""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


# ── Thematic "about X" detector ──────────────────────────────────────────────
# Why this exists: "how many poems about horse" matches _COUNTING_CUES +
# _SLOT_POEMS and would be classified as count_poems — returning the 1,502
# total instead of something useful. "Horse" is a theme/topic, not an indexed
# genre. Counting poems by theme requires semantic retrieval, not a registry
# lookup. This pattern detects the "poems about/on X" construction so the
# classifier can fall through to the RAG pipeline when X is not in our genre map.

_THEMATIC_ABOUT_EN = re.compile(
    r"\b(?:poems?|verses?|qasidas?|abyat)\s+(?:about|on|regarding|concerning|dealing\s+with)\s+\w",
    re.IGNORECASE,
)
_THEMATIC_ABOUT_AR = re.compile(
    r"(?:قصائد|أبيات|قصيدة|بيت)\s+(?:عن|حول|في|تتحدث\s+عن|تتناول)\s+\S",
)


def _has_thematic_about(q: str) -> bool:
    """Return True when the query asks to count poems *about* a topic (not a genre)."""
    return bool(_THEMATIC_ABOUT_EN.search(q) or _THEMATIC_ABOUT_AR.search(q))


def _check_unsupported_dim(q: str) -> Optional[str]:
    """
    Scan for unsupported-dimension queries. Returns the dim_name if matched,
    else None. Unsupported dimensions are annotation fields not indexed in the
    corpus (wasm, marginalia, secondary scribe, ink-bleed, library-stamp coords).
    """
    for dim_name, patterns in _UNSUPPORTED_DIMS:
        if _any_match(patterns, q):
            return dim_name
    return None


# ── Intent classifier ────────────────────────────────────────────────────────

def _classify(raw_query: str) -> dict:
    """
    Return {intent, slot, confidence, genre}. intent is one of:
        "count_poems" | "count_poems_by_genre"
        | "count_poets" | "count_manuscripts" | "count_pages"
        | "corpus_overview" | "age" | "provenance" | None

    The `genre` field is the canonical Arabic taxonomy label (غزل / رثاء / …)
    when a counting question carries a genre qualifier; None otherwise.
    """
    q = raw_query.strip()
    conf = 0.0
    cues_hit = []

    is_counting    = _any_match(_COUNTING_CUES, q)
    is_age         = _any_match(_AGE_CUES, q)
    is_provenance  = _any_match(_PROVENANCE_CUES, q)
    is_list_poets  = _any_match(_LIST_POETS_CUES, q)
    is_child_grief = _any_match(_CHILD_GRIEF_CUES, q)
    genre_label    = _detect_genre_slot(q)

    # Age and provenance have dedicated cue vocabularies that are unambiguous
    # ("how old", "where from", "who collected") — counting-style questions
    # typically need a slot word to be specific. So age/provenance carry a
    # stronger prior (0.7) and counting stays at 0.6 until a slot pushes it
    # over the 0.7 firing threshold.
    if is_counting:
        conf += 0.6
        cues_hit.append("counting")
    if is_age:
        conf += 0.7
        cues_hit.append("age")
    if is_provenance:
        conf += 0.7
        cues_hit.append("provenance")
    if is_list_poets:
        conf += 0.7
        cues_hit.append("list_poets")
    if is_child_grief:
        conf += 0.7
        cues_hit.append("child_grief")

    if q.endswith("?") or q.endswith("؟"):
        conf += 0.1

    # Slot detection — which noun are we aggregating over?
    slot: Optional[str] = None
    if _any_match(_SLOT_POEMS, q):
        slot = "poems";       conf += 0.3
    elif _any_match(_SLOT_POETS, q):
        slot = "poets";       conf += 0.3
    elif _any_match(_SLOT_MANUSCRIPTS, q):
        slot = "manuscripts"; conf += 0.3
    elif _any_match(_SLOT_PAGES, q):
        slot = "pages";       conf += 0.3
    elif _any_match(_SLOT_CORPUS, q):
        slot = "corpus";      conf += 0.2

    # Map (intent class, slot) → final intent key the answer node understands.
    # Priority order: age and provenance beat counting when both fire. Example:
    # "كم عمر هذه المخطوطات" ("how old are these manuscripts") contains "كم"
    # (a counting cue in MSA) but its *meaning* is age — the classifier has to
    # honour the more specific cue, not the first one it matched.
    intent: Optional[str] = None
    if is_child_grief and is_counting:
        intent = "count_child_grief_poems"
    elif is_list_poets:
        intent = "list_poets"
    elif is_age:
        intent = "age"
    elif is_provenance:
        intent = "provenance"
    elif is_counting:
        # Genre takes priority when both counting + a genre label fire — "how
        # many love poems" must route to the filtered count, not the unfiltered
        # one. Compatible with poems / corpus slots; ignored for poets/pages
        # because "how many love poets" or "how many love pages" don't have
        # well-defined genre-filtered answers in the registry.
        if genre_label and slot in ("poems", "corpus", None):
            intent = "count_poems_by_genre"
            cues_hit.append(f"genre:{genre_label}")
            conf += 0.2
        elif slot == "poems":
            # If the query says "poems about X" where X is not a genre (genre_label
            # is None), counting by theme requires semantic retrieval — fall through
            # to the RAG pipeline instead of returning the unfiltered total.
            if _has_thematic_about(q):
                intent = None
            else:
                intent = "count_poems"
        elif slot == "poets":         intent = "count_poets"
        elif slot == "manuscripts":   intent = "count_manuscripts"
        elif slot == "pages":         intent = "count_pages"
        elif slot == "corpus":        intent = "corpus_overview"
        else:                         intent = "count_poems"  # sensible default

    conf = min(conf, 1.0)

    return {
        "intent":     intent,
        "slot":       slot,
        "confidence": conf,
        "cues_hit":   cues_hit,
        "genre":      genre_label,
    }


# ── Node callable ────────────────────────────────────────────────────────────

def intent_router_node(state: AgentState) -> AgentState:
    """
    Stage 0.5a — lightweight deterministic fast-path.
    Checks:
      1. Unsupported-dimension queries → educational answer (new)
      2. Counting / age / provenance questions → registry_lookup fast path

    All regex hits set track="registry_lookup" per the architecture contract:
    every non-poetic_rag track requires answer_source="registry_lookup" so
    the orchestrator routes to deterministic_answer_node (not the RAG pipeline).

    When neither matches, falls through to Stage 0.5b (semantic_router_node).
    """
    raw_query = state.get("raw_query", "")
    if not raw_query:
        return state

    qc = state.get("query_context") or {}
    has_arabic = any("\u0600" <= c <= "\u06ff" for c in raw_query)

    # ── Check 1: unsupported dimension ────────────────────────────────────────
    dim = _check_unsupported_dim(raw_query)
    if dim is not None:
        logger.info("intent_router: unsupported_dim=%s — routing to educational answer.", dim)
        new_qc = {
            **qc,
            "query_lang":           "ar" if has_arabic else "en",
            "query_ar":             raw_query if has_arabic else "",
            "query_en":             "" if has_arabic else raw_query,
            "detected_intent":      "factual_deterministic",
            "detected_dialect":     "msa" if has_arabic else "unknown",
            "intent_confidence":    0.95,
            "answer_source":        "registry_lookup",
            "deterministic_intent": f"unsupported_dim_{dim}",
            "track":                "registry_lookup",
            "router_source":        "regex",
            "router_cues":          [f"unsupported_dim_{dim}"],
        }
        state["query_context"] = new_qc  # type: ignore[assignment]
        return state

    # ── Check 1b: image-grounded provenance ───────────────────────────────────
    # Fires only when an image accompanies a meta-question. The image carries
    # the verse text (the search key); the typed text only conveys *what to do
    # with it* (identify the poet, confirm corpus membership, …). Routing here
    # bypasses the standard RAG path so the answer can be a *specific* hit-or-
    # miss verdict against verse-level retrieval rather than the generic
    # refusal both cases used to share.
    has_image = bool(state.get("input_image_path"))
    if has_image and _any_match(_IMAGE_PROVENANCE_CUES, raw_query):
        logger.info("intent_router: image_grounded_provenance — routing to image_provenance node.")
        new_qc = {
            **qc,
            "query_lang":           "ar" if has_arabic else "en",
            "query_ar":             raw_query if has_arabic else "",
            "query_en":             "" if has_arabic else raw_query,
            "detected_intent":      "factual_deterministic",
            "detected_dialect":     "msa" if has_arabic else "unknown",
            "intent_confidence":    0.95,
            "answer_source":        "registry_lookup",
            "deterministic_intent": "image_grounded_provenance",
            "track":                "image_grounded_provenance",
            "router_source":        "regex",
            "router_cues":          ["image_grounded_provenance"],
        }
        state["query_context"] = new_qc  # type: ignore[assignment]
        return state

    # ── Check 2: capabilities (zero-LLM fast-path) ───────────────────────────
    if _any_match(_CAPABILITIES_CUES, raw_query):
        logger.info("intent_router: capabilities query — routing to deterministic answer.")
        new_qc = {
            **qc,
            "query_lang":           "ar" if has_arabic else "en",
            "query_ar":             raw_query if has_arabic else "",
            "query_en":             "" if has_arabic else raw_query,
            "detected_intent":      "factual_deterministic",
            "detected_dialect":     "msa" if has_arabic else "unknown",
            "intent_confidence":    0.95,
            "answer_source":        "registry_lookup",
            "deterministic_intent": "capabilities",
            "track":                "capabilities",
            "router_source":        "regex",
            "router_cues":          ["capabilities"],
        }
        state["query_context"] = new_qc  # type: ignore[assignment]
        return state

    # ── Check 3: instructor debug (zero-LLM fast-path) ───────────────────────
    if _any_match(_INSTRUCTOR_DEBUG_CUES, raw_query):
        logger.info("intent_router: instructor_debug query — routing to deterministic answer.")
        new_qc = {
            **qc,
            "query_lang":           "ar" if has_arabic else "en",
            "query_ar":             raw_query if has_arabic else "",
            "query_en":             "" if has_arabic else raw_query,
            "detected_intent":      "factual_deterministic",
            "detected_dialect":     "msa" if has_arabic else "unknown",
            "intent_confidence":    0.90,
            "answer_source":        "registry_lookup",
            "deterministic_intent": "instructor_debug",
            "track":                "instructor_debug",
            "router_source":        "regex",
            "router_cues":          ["instructor_debug"],
        }
        state["query_context"] = new_qc  # type: ignore[assignment]
        return state

    # ── Check 4: counting / age / provenance ──────────────────────────────────
    verdict = _classify(raw_query)
    intent = verdict["intent"]
    conf   = verdict["confidence"]

    if intent is not None and conf >= 0.7:
        logger.info(
            "intent_router: matched intent=%s confidence=%.2f cues=%s — routing to registry.",
            intent, conf, verdict["cues_hit"],
        )
        new_qc = {
            **qc,
            "query_lang":           "ar" if has_arabic else "en",
            "query_ar":             raw_query if has_arabic else "",
            "query_en":             "" if has_arabic else raw_query,
            "detected_intent":      "factual_deterministic",
            "detected_dialect":     "msa" if has_arabic else "unknown",
            "intent_confidence":    conf,
            "answer_source":        "registry_lookup",
            "deterministic_intent": intent,
            "track":                "registry_lookup",
            "router_source":        "regex",
            "router_cues":          verdict["cues_hit"],
            # Carries the canonical genre label (غزل / رثاء / …) when the
            # router resolved a "how many <genre> poems" intent. The
            # deterministic_answer node reads this to call
            # corpus_stats.count_poems_by_genre and to ground the LLM prose pass.
            "intent_genre":         verdict.get("genre"),
        }
        state["query_context"] = new_qc  # type: ignore[assignment]
        return state

    # ── Check 5: Tier 2 — prototype embedding similarity ─────────────────────
    # Why this is here, not in semantic_router.py: the LLM call costs ~400 ms
    # of latency and a token-budget hit. If the user is just paraphrasing a
    # known meta-question ("the names of the writers", "tally of authors")
    # we can short-circuit to the deterministic answer in <100 ms with no
    # LLM call. The prototype router uses AraBERT cosine if available and
    # falls back to TF-IDF char-n-gram cosine when sentence-transformers is
    # not installed (e.g. CI / fresh check-out). It NEVER fires unless the
    # nearest prototype clears a strict threshold AND beats the runner-up
    # intent by a margin — ambiguous queries deliberately fall through to the
    # full LLM router.
    try:
        from ..prototype_router import classify as _proto_classify  # lazy import
        proto = _proto_classify(raw_query)
    except Exception as exc:
        logger.debug("intent_router: prototype router error (%s) — skipping Tier 2.", exc)
        proto = {"fired": False}

    if proto.get("fired") and proto.get("intent"):
        proto_intent = proto["intent"]
        proto_conf   = float(proto.get("confidence", 0.0))
        logger.info(
            "intent_router: Tier 2 prototype match intent=%s conf=%.2f encoder=%s "
            "best=%r — routing to registry.",
            proto_intent, proto_conf, proto.get("encoder"),
            proto.get("best_prototype", "")[:60],
        )
        new_qc = {
            **qc,
            "query_lang":           "ar" if has_arabic else "en",
            "query_ar":             raw_query if has_arabic else "",
            "query_en":             "" if has_arabic else raw_query,
            "detected_intent":      "factual_deterministic",
            "detected_dialect":     "msa" if has_arabic else "unknown",
            "intent_confidence":    proto_conf,
            "answer_source":        "registry_lookup",
            "deterministic_intent": proto_intent,
            "track":                "registry_lookup",
            "router_source":        f"prototype_{proto.get('encoder', 'unknown')}",
            "router_cues":          [
                f"prototype:{proto_intent}",
                f"nearest:{proto.get('best_prototype','')[:60]}",
            ],
            # No genre filter from prototype router — the genre-counting
            # intents stay regex-only because they need exact taxonomy match.
            "intent_genre":         None,
        }
        state["query_context"] = new_qc  # type: ignore[assignment]
        return state

    # ── Fall-through — semantic router (Stage 0.5b) will classify ─────────
    logger.debug(
        "intent_router: no deterministic / prototype match (regex_conf=%.2f, "
        "proto_conf=%.2f) — semantic router will run.",
        conf, float(proto.get("confidence", 0.0)),
    )
    new_qc = {**qc, "answer_source": "rag_pipeline"}
    state["query_context"] = new_qc  # type: ignore[assignment]

    return state
