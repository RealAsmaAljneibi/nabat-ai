"""
agent1_query_understanding/prototype_router.py
================================================
Why this file exists: Tier 2 of the four-tier intent funnel.

The funnel (low → high cost):
    Tier 1  intent_router (regex)            ~0.5 ms   deterministic, brittle
    Tier 2  prototype_router (this file)     ~5–80 ms  zero-LLM, paraphrase-robust   ← here
    Tier 3  semantic_router  (LLM)           ~400 ms   most flexible, costs tokens
    Tier 4  full RAG pipeline                ~5–10 s   for genuine poetic queries

Tier 2 catches the wide class of paraphrases that regex always misses
("the names of the writers", "tally of authors", "give me a head-count of
the manuscripts") **without** spending an LLM call. The technique is the
same one used by ChatGPT's classifier-as-a-service tools and by the OpenAI
guardrails router: embed a small set of *prototype* questions per intent
once, then at query time embed the user's question and pick the closest
prototype if the cosine similarity clears a threshold.

Encoder chain (graceful fallback so this ships today):
    1. AraPoemBERT / AraBERTv2 via sentence-transformers (if installed).
       This is the production target — same encoder used by index.py, so
       the prototype embeddings live in the same semantic space as the
       corpus chunks.
    2. TF-IDF over character n-grams (sklearn). Robust to Arabic morphology
       and bilingual short queries. Cosine similarity on a fitted vocab is
       a well-known approximation of dense semantic similarity for short,
       lexically-overlapping texts — which prototype matching is by design.

Prototypes were drafted from five user personas:
    - Poetry researcher (academic, Arabic-fluent)
    - Cultural-institution archivist (Arabic-first, asset-focused)
    - Curious general user (plain English)
    - Student / educator (mixed AR/EN, simple language)
    - Grief / elegy scholar (specific genre vocabulary)

Architecture ref: §2.4 Stage 0.5b — Tier 2 paraphrase router (added 2026-04-26).
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Prototype library ────────────────────────────────────────────────────────
# Why a Python dict (not YAML): zero startup I/O, easy to add new intents,
# tests can import it directly to assert no intent has fewer than N prototypes.
#
# Each entry maps a deterministic_intent (string the answer node already
# understands) to a list of natural-language prototype questions. We mix
# Arabic + English prototypes so a query in either language has a fair shot
# at matching its intent's cluster.
#
# Coverage rule: every intent listed here MUST already be handled by
# deterministic_answer_node (see _SUPPORTED_DETERMINISTIC_INTENTS in that
# file). The prototype router is a *router*, not an answer source — it must
# never invent intents the answer node can't render.

PROTOTYPES: dict[str, list[str]] = {
    # ── how many poets are in the corpus ──────────────────────────────────
    "count_poets": [
        "how many poets are in the corpus",
        "how many distinct poets are represented",
        "tally of authors in the archive",
        "give me a count of unique poets",
        "number of writers across all manuscripts",
        "how many poetry writers do you have",
        "what's the total number of poets",
        "size of the poet roster",
        "كم عدد الشعراء في المجموعة",
        "كم شاعرا تم توثيقه",
        "ما عدد الشعراء الموجودين",
        "اعطني عدد الشعراء",
        "عدد المؤلفين في الأرشيف",
        "كم كاتبا في هذا المتن",
        "ما إجمالي الشعراء",
    ],

    # ── name the poets / list the poets ───────────────────────────────────
    "list_poets": [
        "who are the poets in this corpus",
        "list the poets",
        "tell me who wrote these poems",
        "the names of the writers",
        "which authors feature in the manuscripts",
        "show me the roster of poets",
        "name the most-represented poets",
        "give me the top poets",
        "who composed these qasidas",
        "main authors in this archive",
        "من هم الشعراء في هذا المتن",
        "اذكر أسماء الشعراء",
        "اعرض قائمة الشعراء",
        "من كتب هذه القصائد",
        "ما هي أسماء المؤلفين",
        "من أبرز الشعراء في المجموعة",
    ],

    # ── how many manuscripts / volumes / books ────────────────────────────
    "count_manuscripts": [
        "how many manuscripts are in your repo",
        "how many manuscripts do you have",
        "count of bound volumes in the archive",
        "how many books are in here",
        "size of the manuscript collection",
        "how many codices have you digitised",
        "number of source volumes",
        "how many physical manuscripts",
        "كم عدد المخطوطات",
        "ما عدد المخطوطات في الأرشيف",
        "كم مخطوطة في هذا المتن",
        "كم مجلدا تم رقمنته",
        "ما إجمالي المخطوطات",
        "عدد الكتب الأصلية",
        "كم كتابا مخطوطا لديك",
    ],

    # ── how many poems total ──────────────────────────────────────────────
    "count_poems": [
        "how many poems are in the corpus",
        "size of the poetry collection",
        "total number of qasidas",
        "how many verses do you index",
        "give me the count of poems",
        "what is the corpus size in poems",
        "كم عدد القصائد في المتن",
        "كم قصيدة لديك",
        "ما عدد القصائد المفهرسة",
        "إجمالي عدد القصائد",
        "ما حجم مجموعة الشعر",
        "كم قصيدة في الأرشيف",
    ],

    # ── elegies for a son / daughter (rithaa al-abnaa) ────────────────────
    "count_child_grief_poems": [
        "how many poems are about grieving a son or daughter",
        "elegies mourning the death of a child",
        "poems lamenting a lost son",
        "rithaa for a daughter",
        "qasidas about losing a child",
        "verses on the death of one's offspring",
        "how many laments for sons and daughters",
        "poems mourning a young child",
        "كم قصيدة في رثاء الابن",
        "قصائد رثاء البنت",
        "مرثيات الأبناء",
        "كم مرثية لابن أو ابنة",
        "قصائد في موت الولد",
        "رثاء فقد الأبناء",
        "كم قصيدة حزن على ابن متوفى",
    ],

    # ── how old / which century / time period ─────────────────────────────
    "age": [
        "how old are these manuscripts",
        "from what century are the manuscripts",
        "what era do these poems belong to",
        "what time period does the corpus cover",
        "when were these texts written",
        "date range of the collection",
        "how far back does the archive go",
        "ما عمر هذه المخطوطات",
        "في أي قرن كُتبت",
        "إلى أي عصر تنتمي",
        "ما الحقبة الزمنية للمتن",
        "متى كُتبت هذه النصوص",
        "ما تاريخ المجموعة",
    ],

    # ── geographic origin / collectors ────────────────────────────────────
    "provenance": [
        "where are these manuscripts from",
        "what regions did the manuscripts come from",
        "geographic origin of the collection",
        "who collected these poems",
        "what families donated these manuscripts",
        "provenance of the archive",
        "what countries do these volumes represent",
        "من جمع هذه المخطوطات",
        "من أين أتت المخطوطات",
        "ما المناطق التي تمثلها",
        "أصل المجموعة من أين",
        "من جامع المتن",
        "ما البلدان المصدرة لهذه المخطوطات",
    ],

    # ── what can the system do ────────────────────────────────────────────
    "capabilities": [
        "what can you do",
        "what can this tool actually do",
        "show me what this system does",
        "tell me your skills",
        "how can you help me",
        "what kinds of questions can I ask",
        "what are your features",
        "ماذا يمكنك أن تفعل",
        "ما هي قدراتك",
        "كيف يمكنك مساعدتي",
        "ما الذي يستطيع هذا النظام فعله",
        "اشرح لي ميزاتك",
        "أي أسئلة يمكنني طرحها",
    ],

    # ── overall corpus summary ────────────────────────────────────────────
    "corpus_overview": [
        "give me a summary of the corpus",
        "describe the corpus",
        "high-level overview of the archive",
        "snapshot of the collection",
        "tell me about the corpus",
        "what's in this archive",
        "اعطني نظرة عامة عن المتن",
        "صف المجموعة",
        "ما محتوى الأرشيف",
        "ملخص عن المجموعة",
        "نبذة عن المتن",
    ],
}


# ── Threshold configuration ──────────────────────────────────────────────────
# Why two thresholds: dense AraBERT cosine and TF-IDF char-ngram cosine sit on
# different scales. AraBERT paraphrases typically score 0.65–0.85; TF-IDF char-
# ngram (3,5) paraphrases typically score 0.50–0.80. We tune each separately
# so the firing rate stays roughly the same regardless of which encoder loaded.
#
# Margin gate: even if the top score clears the threshold, it must beat the
# runner-up intent's best prototype by ≥ MARGIN. This rejects ambiguous queries
# that look equally similar to two intents (e.g. "tell me about your poets"
# is half list_poets and half count_poets — better to fall through to the LLM).
THRESHOLD_ARABERT = 0.65
THRESHOLD_TFIDF   = 0.55     # tuned 2026-04-26: 0.50 produced a false positive
                             # on "what does this verse mean?" matching
                             # capabilities prototypes ("tell me ...").
MARGIN            = 0.05

# Encoder timeout: if AraBERT model-load hangs for any reason we don't want
# the demo to freeze. 1.5 s is enough for cached weights; first-ever load
# downloads a 500 MB model so we let that fail fast and fall back to TF-IDF.
ENCODER_TIMEOUT_S = 1.5


# ── Module-level lazy state ──────────────────────────────────────────────────
# We never load the encoder at import-time so unit tests, scripts/rebuild_index
# and CI all stay snappy.
_encoder_kind: Optional[str] = None     # "arabert" | "tfidf" | "fallback" | None
_encoder_obj:  Any           = None     # the model / vectorizer
_proto_vecs:   Any           = None     # numpy array (n_protos, dim)
_proto_meta:   list[tuple[str, str]] = []  # parallel list of (intent, raw_prototype)
_load_lock                   = threading.Lock()


def _flatten_prototypes() -> tuple[list[str], list[tuple[str, str]]]:
    """Return (texts, [(intent, prototype), ...]) in stable order."""
    texts: list[str] = []
    meta:  list[tuple[str, str]] = []
    for intent in sorted(PROTOTYPES.keys()):
        for proto in PROTOTYPES[intent]:
            texts.append(proto)
            meta.append((intent, proto))
    return texts, meta


def _try_load_arabert() -> bool:
    """
    Attempt to load AraBERT/AraPoemBERT via sentence-transformers and embed
    every prototype. Returns True on success (and populates module globals),
    False on any failure. We use a thread-bounded load so a slow first-time
    HuggingFace download does not freeze the demo.
    """
    global _encoder_kind, _encoder_obj, _proto_vecs

    result: list[Optional[Any]] = [None]

    def _target() -> None:
        try:
            from ..embed import _load_model, _model, _model_available, normalise_arabic
            _load_model()
            if not _model_available or _model is None:
                return
            texts, _ = _flatten_prototypes()
            normalised = [normalise_arabic(t) for t in texts]
            vecs = _model.encode(
                normalised, normalize_embeddings=True, show_progress_bar=False
            )
            result[0] = (_model, vecs)
        except Exception as exc:
            logger.debug("prototype_router: arabert load failed: %s", exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=ENCODER_TIMEOUT_S)

    if t.is_alive() or result[0] is None:
        if t.is_alive():
            logger.info(
                "prototype_router: AraBERT load exceeded %.1fs — falling back to TF-IDF.",
                ENCODER_TIMEOUT_S,
            )
        return False

    import numpy as np
    model, vecs = result[0]
    _encoder_kind = "arabert"
    _encoder_obj  = model
    _proto_vecs   = np.asarray(vecs, dtype="float32")
    return True


def _try_load_tfidf() -> bool:
    """
    Fit a TF-IDF char-n-gram vectoriser on the prototype set and cache the
    transformed prototype matrix. Char n-grams (3,5) handle Arabic morphology
    (prefixes/suffixes) and short bilingual queries gracefully.

    Returns True on success.
    """
    global _encoder_kind, _encoder_obj, _proto_vecs

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        logger.error("prototype_router: sklearn unavailable (%s) — Tier 2 disabled.", exc)
        _encoder_kind = "fallback"
        return False

    texts, _ = _flatten_prototypes()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
        min_df=1,
    )
    matrix = vec.fit_transform(texts)
    _encoder_kind = "tfidf"
    _encoder_obj  = vec
    _proto_vecs   = matrix      # sparse matrix; we'll do sparse cosine
    return True


def _ensure_loaded() -> None:
    """Idempotent encoder + prototype-matrix loader."""
    global _proto_meta
    if _encoder_kind is not None:
        return
    with _load_lock:
        if _encoder_kind is not None:
            return
        _, _proto_meta = _flatten_prototypes()
        if _try_load_arabert():
            logger.info(
                "prototype_router: encoder=arabert prototypes=%d",
                len(_proto_meta),
            )
            return
        if _try_load_tfidf():
            logger.info(
                "prototype_router: encoder=tfidf prototypes=%d",
                len(_proto_meta),
            )
            return
        logger.warning("prototype_router: no encoder available — Tier 2 will always abstain.")


# ── Query encoding + similarity ──────────────────────────────────────────────

def _encode_query_arabert(text: str):
    from ..embed import normalise_arabic
    import numpy as np
    vec = _encoder_obj.encode(
        normalise_arabic(text), normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vec, dtype="float32")


def _encode_query_tfidf(text: str):
    return _encoder_obj.transform([text])


def _cosine_scores(query_vec) -> "list[float]":
    """Return cosine similarity between query_vec and every prototype vector."""
    if _encoder_kind == "arabert":
        # Both sides are L2-normalised, so dot product == cosine similarity.
        import numpy as np
        return (_proto_vecs @ query_vec).tolist()
    # TF-IDF: query_vec is a (1, V) sparse row, prototypes are (N, V) sparse.
    # Both are L2-normalised by TfidfVectorizer, so sparse dot == cosine.
    sims = (_proto_vecs @ query_vec.T).toarray().ravel()
    return sims.tolist()


# ── Public API ───────────────────────────────────────────────────────────────

def classify(query: str) -> dict:
    """
    Classify *query* against the prototype set.

    Returns a dict shaped like:
        {
          "intent":              "count_poets" | ... | None,
          "confidence":          0.0–1.0    (top cosine score),
          "best_prototype":      "...",
          "runner_up_intent":    "..." | None,
          "runner_up_confidence": 0.0–1.0,
          "encoder":             "arabert" | "tfidf" | "fallback",
          "fired":               bool,
        }

    `fired` is True when the intent should be accepted (top score ≥ threshold
    AND margin over runner-up ≥ MARGIN). When fired is False, the caller MUST
    fall through to the next tier (LLM router or full RAG).
    """
    if not query or not query.strip():
        return {
            "intent": None, "confidence": 0.0, "best_prototype": "",
            "runner_up_intent": None, "runner_up_confidence": 0.0,
            "encoder": "fallback", "fired": False,
        }

    _ensure_loaded()
    if _encoder_kind in (None, "fallback") or _proto_vecs is None:
        return {
            "intent": None, "confidence": 0.0, "best_prototype": "",
            "runner_up_intent": None, "runner_up_confidence": 0.0,
            "encoder": _encoder_kind or "fallback", "fired": False,
        }

    threshold = THRESHOLD_ARABERT if _encoder_kind == "arabert" else THRESHOLD_TFIDF

    if _encoder_kind == "arabert":
        qvec = _encode_query_arabert(query)
    else:
        qvec = _encode_query_tfidf(query)
    scores = _cosine_scores(qvec)

    # Find best score per intent (a query that paraphrases one prototype well
    # is much more meaningful than a query that vaguely overlaps every prototype
    # in the same intent — so we always take the *max* per intent, never the
    # mean).
    per_intent_best: dict[str, tuple[float, str]] = {}
    for (intent, proto), score in zip(_proto_meta, scores):
        prev = per_intent_best.get(intent)
        if prev is None or score > prev[0]:
            per_intent_best[intent] = (score, proto)

    # Rank intents by their best prototype score
    ranked = sorted(
        per_intent_best.items(), key=lambda kv: kv[1][0], reverse=True
    )
    top_intent, (top_score, top_proto) = ranked[0]
    if len(ranked) > 1:
        runner_intent = ranked[1][0]
        runner_score  = ranked[1][1][0]
    else:
        runner_intent = None
        runner_score  = 0.0

    fired = (
        top_score >= threshold
        and (top_score - runner_score) >= MARGIN
    )

    return {
        "intent":               top_intent if fired else None,
        "confidence":           float(top_score),
        "best_prototype":       top_proto,
        "runner_up_intent":     runner_intent,
        "runner_up_confidence": float(runner_score),
        "encoder":              _encoder_kind,
        "fired":                fired,
    }


def reset_for_tests() -> None:
    """Clear the cached encoder + prototype matrix (used by unit tests)."""
    global _encoder_kind, _encoder_obj, _proto_vecs, _proto_meta
    _encoder_kind = None
    _encoder_obj  = None
    _proto_vecs   = None
    _proto_meta   = []


def prototype_count() -> int:
    """Total number of prototypes across all intents — used by tests/eval."""
    return sum(len(v) for v in PROTOTYPES.values())


def intents() -> list[str]:
    """Stable sorted list of intents the router can route to."""
    return sorted(PROTOTYPES.keys())
