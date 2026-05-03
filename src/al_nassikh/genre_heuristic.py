"""
src/al_nassikh/genre_heuristic.py
==================================
Why this file exists: The retrieval UI needs genre + emotion facets so the
user can filter "show me the love poems" or "show me poems about grief"
without relying on dense similarity alone. Dense similarity is good at
finding related verses, but bad at filtering by category — a love poem and a
war poem about the same oasis embed close together. Hard filters on a genre
field solve this cleanly. Where it's called: app (live — checks neural availability), 
offline enrich script. Purpose: Silver-baseline genre classifier (43% coverage); app checks if neural upgrade is available

Classification strategy — neural-first with heuristic fallback:
  1. PRIMARY: Fine-tuned AraPoemBERT classifiers (arapoem_genre_best.pt,
     arapoem_emotion_text_best.pt) loaded from ~/poetry/outputs/models/.
     These were fine-tuned on 3,340 Khaleeji Nabati clips (course MAAI7103)
     and understand poetic vocabulary, meter, and Khaleeji dialectal
     variation far better than a keyword lexicon.
     Genre: 8-class softmax, winner-take-all with confidence threshold.
     Emotion: multi-label sigmoid, threshold per class.
  2. FALLBACK (the original heuristic): keyword lexicon scoring used when
     the neural model is unavailable (offline, no torch, model file missing)
     or when model confidence falls below a threshold. The heuristic is
     retained both as the safety net AND as the auditable silver baseline.
     Badge: results from the neural model carry genre_source="neural_v1";
     heuristic results carry genre_source="heuristic_v1" (existing contract).

Why a keyword heuristic fallback and not just a hard error:
  1. Portability — "runs everywhere" is a non-negotiable project requirement.
     The demo panel must not crash if the 440 MB checkpoint is absent.
  2. Auditability — the keyword evidence list is preserved in the output dict
     regardless of which path ran. A reviewer can see exactly what fired.
  3. Coverage continuity — the 1,502-entry corpus enrichment can always run
     even on a machine without GPU or HuggingFace access.

Upstream consumer: scripts/enrich_genre_heuristic.py
Downstream consumers: fatat_al_arab/embed.py (payload), self_query.py (filter schema)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .nabati_taxonomy import (
    GENRES, EMOTIONS,
    validate_genre, validate_emotions,
)


# ── Neural model paths ────────────────────────────────────────────────────────
# Why hardcoded paths rather than env vars: the poetry project lives in a
# predictable location (~/poetry/). If someone moves it they set POETRY_HOME.
# The fallback chain (env → default path → heuristic) ensures graceful degradation.

_POETRY_HOME = Path(os.getenv("POETRY_HOME", str(Path.home() / "poetry")))
_ARAPOEM_LOCAL = _POETRY_HOME / "models" / "local" / "arapoembert"
_GENRE_CKPT    = _POETRY_HOME / "outputs" / "models" / "arapoem_genre" / "arapoem_genre_best.pt"
_EMOTION_CKPT  = _POETRY_HOME / "outputs" / "models" / "arapoembert_emotion" / "arapoem_emotion_text_best.pt"

# Neural model confidence gate. Below this the neural prediction is deemed
# uncertain and we fall through to the heuristic for genre (single-label).
# For emotion (multi-label) each class is thresholded independently.
NEURAL_GENRE_CONF_THRESHOLD   = 0.40  # argmax softmax score must clear this
NEURAL_EMOTION_SIGMOID_THRESH = 0.45  # per-class sigmoid for multi-label

# Module-level singletons — loaded once per process.
_genre_clf: Any   = None
_emotion_clf: Any = None
_genre_clf_loaded: bool   = False
_emotion_clf_loaded: bool = False


# ── Neural classifier classes ─────────────────────────────────────────────────

class _NeuralGenreClassifier:
    """
    Why this exists: wraps arapoem_genre_best.pt so the rest of the file can
    call clf.predict(text) without caring about PyTorch / tokeniser internals.

    The .pt checkpoint is expected to contain either:
      (a) {"model_state_dict": ..., "label2id": {label: idx}, ...}  — full dict
      (b) A bare model state_dict (older format) — we infer labels from GENRES.

    Graceful degradation: any exception during __init__ or predict() is
    caught by the caller; the heuristic then runs as if this class does not exist.
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        ckpt_path = str(_GENRE_CKPT)
        tok_path  = str(_ARAPOEM_LOCAL) if _ARAPOEM_LOCAL.exists() else "faisalq/bert-base-arapoembert"

        # Load checkpoint — handles both dict-format and raw state_dict
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "label2id" in ckpt:
            label2id: dict[str, int] = ckpt["label2id"]
        elif isinstance(ckpt, dict) and "id2label" in ckpt:
            label2id = {v: k for k, v in ckpt["id2label"].items()}
        else:
            # Fallback: use the first N genres from GENRES (excl. غير_محدد)
            # The training config showed 8 classes; GENRES has 10 + غير_محدد.
            num_labels = _infer_num_labels(ckpt)
            label2id = {g: i for i, g in enumerate(GENRES) if g != "غير_محدد"}
            # Trim or pad to the checkpoint's actual number of output classes
            keys = list(label2id.keys())[:num_labels]
            label2id = {k: i for i, k in enumerate(keys)}

        self.id2label: dict[int, str] = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

        self.tokeniser = AutoTokenizer.from_pretrained(tok_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            tok_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        # Load weights — handle both formats
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        import torch as _torch
        self._torch = _torch

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (genre_label, confidence) where confidence is the softmax
        score for the predicted class. Falls through to غير_محدد if below
        NEURAL_GENRE_CONF_THRESHOLD.
        """
        inputs = self.tokeniser(
            text, return_tensors="pt", truncation=True,
            max_length=64, padding=True,
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs   = self._torch.softmax(logits, dim=-1)[0]
        top_idx = int(probs.argmax())
        conf    = float(probs[top_idx])
        label   = self.id2label.get(top_idx, "غير_محدد")
        return validate_genre(label), conf


class _NeuralEmotionClassifier:
    """
    Why this exists: wraps arapoem_emotion_text_best.pt for multi-label
    emotion detection. The model outputs sigmoid scores per class; we
    threshold each independently at NEURAL_EMOTION_SIGMOID_THRESH.

    Same checkpoint-format handling as _NeuralGenreClassifier.
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        ckpt_path = str(_EMOTION_CKPT)
        tok_path  = str(_ARAPOEM_LOCAL) if _ARAPOEM_LOCAL.exists() else "faisalq/bert-base-arapoembert"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Determine emotion label mapping from checkpoint
        if isinstance(ckpt, dict) and "label2id" in ckpt:
            label2id: dict[str, int] = ckpt["label2id"]
        elif isinstance(ckpt, dict) and "id2label" in ckpt:
            label2id = {v: k for k, v in ckpt["id2label"].items()}
        else:
            num_labels = _infer_num_labels(ckpt)
            # Use EMOTIONS tuple (10 labels)
            keys = list(EMOTIONS)[:num_labels]
            label2id = {k: i for i, k in enumerate(keys)}

        self.id2label: dict[int, str] = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

        self.tokeniser = AutoTokenizer.from_pretrained(tok_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            tok_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            problem_type="multi_label_classification",
        )

        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        import torch as _torch
        self._torch = _torch

    def predict(self, text: str) -> dict[str, float]:
        """
        Returns {emotion_label: sigmoid_score} for all labels.
        Caller applies the threshold to pick active emotions.
        """
        inputs = self.tokeniser(
            text, return_tensors="pt", truncation=True,
            max_length=64, padding=True,
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.sigmoid(logits)[0]
        return {
            self.id2label[i]: float(probs[i])
            for i in range(len(self.id2label))
        }


def _infer_num_labels(ckpt: Any) -> int:
    """
    Guess the number of output classes from the state_dict by finding the
    classifier/output bias tensor (the last linear layer's bias is typically
    named 'classifier.bias' or 'classifier.out_proj.bias').
    Why we need this: old checkpoints saved as bare state_dicts omit num_labels.
    """
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    for key in ("classifier.out_proj.bias", "classifier.bias", "output.bias"):
        if key in sd:
            return sd[key].shape[0]
    # Scan for any bias with reasonable label count
    for k, v in sd.items():
        if k.endswith(".bias") and hasattr(v, "shape") and 2 <= v.shape[0] <= 20:
            return v.shape[0]
    return 8   # training config default


def _load_genre_clf() -> None:
    """Load genre classifier once; silently set to None on any failure."""
    global _genre_clf, _genre_clf_loaded
    if _genre_clf_loaded:
        return
    _genre_clf_loaded = True
    if not _GENRE_CKPT.exists():
        return
    try:
        _genre_clf = _NeuralGenreClassifier()
    except Exception:
        _genre_clf = None


def _load_emotion_clf() -> None:
    """Load emotion classifier once; silently set to None on any failure."""
    global _emotion_clf, _emotion_clf_loaded
    if _emotion_clf_loaded:
        return
    _emotion_clf_loaded = True
    if not _EMOTION_CKPT.exists():
        return
    try:
        _emotion_clf = _NeuralEmotionClassifier()
    except Exception:
        _emotion_clf = None


def is_neural_genre_available() -> bool:
    """True if the neural genre model loaded successfully."""
    _load_genre_clf()
    return _genre_clf is not None


def is_neural_emotion_available() -> bool:
    """True if the neural emotion model loaded successfully."""
    _load_emotion_clf()
    return _emotion_clf is not None


# ── Arabic surface normaliser ─────────────────────────────────────────────────
# Why inline and not imported: the parser module does not currently expose a
# reusable normaliser; inlining a minimal one here keeps this classifier
# self-contained so the enrichment script can run without bootstrapping the
# full al_nassikh package. When the shared normaliser lands (§M2 cleanup),
# replace this with `from .parser import normalise_arabic`.

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")   # fatha, kasra, damma, shadda, sukun, alef khanjariyya
_TATWEEL = "\u0640"

_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ی": "ي",   # Persian/Urdu yeh — appears frequently in this corpus
                #   (scribal influence from Persianate manuscript tradition)
    "ك": "ك",   # no-op but kept for readability of the table
    "ک": "ك",   # Persian kaf — same reason as above
    "ة": "ه",
    "ؤ": "و",
})


def _normalise(text: str) -> str:
    """
    Strip diacritics, remove tatweel, fold hamza variants.

    Why this specific normalisation and not more: we want lexicon matches to
    survive the common manuscript variations (ة/ه, ى/ي, أ/ا) without
    flattening the text so much that distinct words collide. This matches
    what Qdrant's BM25 tokenizer will do at query time — consistency between
    the enrichment pass and retrieval is more important than either being
    "better" in isolation.
    """
    if not text:
        return ""
    text = text.replace(_TATWEEL, "")
    text = _DIACRITICS.sub("", text)
    return text.translate(_FOLD)


# ── Genre lexicons ────────────────────────────────────────────────────────────
# Why these specific markers: each list is seeded from (a) the eight verses
# that literally have occasion="غزل" in anchor_registry_phase4.json, providing
# free positive examples for غزل; (b) Sowayan's genre vocabulary lists in
# "Nabati Poetry"; (c) hand-selected high-precision terms — we prefer words
# that are rare outside their genre, even at the cost of recall. Recall comes
# from stacking multiple markers per verse; precision comes from the markers
# being unambiguous.
#
# All markers are already normalised (no diacritics, ة→ه, أ→ا) so the match
# is a plain set lookup against the normalised verse token stream.

# غزل — love / romantic
# Lexicon expanded 2026-04-23 after running on the full 1,502-matla corpus
# and finding 95% abstention. Added corpus-observed stock openers (البارحة,
# النوم as insomnia trope) and heart/tears vocabulary that appears in single-
# line matlas.
GHAZAL_MARKERS = {
    "حبيب", "حبيبي", "حبيبه", "محبوب", "الحبيب", "احب", "حبها",
    "هوى", "الهوى", "هواه", "هواك", "هواها",
    "شوق", "شوقي", "شوقه", "اشواق", "مشتاق", "اشتقت", "يشتاق",
    "وصال", "الوصال", "وصل", "الوصل", "لقاء", "اللقاء",
    "هجر", "الهجر", "هجران", "البعاد", "فراق", "الفراق",
    "عشق", "العشق", "عاشق", "عشاق", "عاشقين",
    "غرام", "الغرام", "ولع", "مولع",
    "قلب", "قلبي", "قلبه", "قلبها", "القلب", "فؤاد", "فؤادي", "الفؤاد",
    "عين", "عيني", "العين", "عيون", "عينيها", "الثغر", "الخد",
    "جميل", "جميله", "حسن", "الحسن", "مليح", "مليحه",
    "غزال", "الغزال", "غزاله", "ظبي", "الظبيه",  # beloved metaphors
    "حورا", "حوراء",
    "النوم", "الوسن",  # insomnia / drowsiness trope — signature ghazal opener
    "البارحه", "البارحة",   # "last night" — dream/vision trope
    "سهر", "سهرت", "سهران",  # sleeplessness
    # Dove / messenger bird — extremely common nasib trope: poet addresses a
    # dove that mourns its mate, paralleling his own separation. 10+ corpus
    # occurrences in abstained verses.
    "الحمام", "حمام", "حمامه", "الحمامه", "ياحمام", "ياذا الحمام",
    "طير", "الطير", "ياطير",
    # Vocative absent-beloved
    "الاحباب", "احباب", "الاحبه", "احبه", "الاحبتي",
    "الحبيب", "حبيبنا", "ياحي",   # "O thou" — vocative longing opener
    "مهموم", "المهموم",   # "preoccupied / love-troubled heart"
    "طال ليلي", "طال الليل",  # "long night" trope (multi-word, kept for reference)
    # Regret / woundedness vocabulary common in love verse
    "صابني", "مصاب",       # "it struck me" — love wound
    "فني", "فنه", "افنى",   # "consumed / destroyed [by love]"
    "غي", "الغي",           # "infatuation / going astray"
    "غرامه", "بالغي",
    "طفل",   # "a girl [who struck me]" — surface form referring to beloved
}

# رثاء — elegy for the dead
RITHA_MARKERS = {
    "رحل", "رحلت", "رحلوا", "رحيل", "الرحيل",
    "فقد", "فقدت", "فقدنا", "فقيد", "الفقيد",
    "نعى", "ناعي", "نعاه", "نعانا", "النعي",
    "وداع", "الوداع", "ودعت", "ودعنا",
    "قبر", "القبر", "القبور", "الضريح", "اللحد",
    "رثى", "رثاء", "الرثاء", "رثيته", "مرثيه",
    "عزي", "عزاء", "التعزيه", "عزاكم",
    "ميت", "الميت", "الاموات", "موتى", "موت", "المنيه", "توفي", "متوفى",
    "تراب", "الثرى",
    "يرحم", "رحمه", "غفر",
    "حزن", "الحزن", "حزينه", "حزين", "حزينا", "احزن",
    "دمع", "الدمع", "دموع", "دموعي", "بكيت", "تبكي", "بكاء", "نواح",
    "الديار",  # "deserted abode" trope — classic رثاء evocation
    # Khaleeji grief vocabulary observed in corpus
    "ونه", "الونه", "ونين", "الونين", "ونينه",  # ونين = moaning/lament
    "نياحه", "النياحه",  # lament/dirge
    "هم", "الهم", "همي", "همومي", "الهموم",  # "grief/cares"
    "اسف", "الاسف",
    "يا من يعاون", "من يعاون",  # "who will help me [bear]" — lament opener
    # Atlāl (deserted-abode) topos — "I halt at the dwellings" — classical
    # Arabic convention the Nabati tradition inherited. Strong رثاء/نسيب
    # signal when it opens a poem. 9+ corpus occurrences.
    "المنازل", "منازل", "منازلنا",
    "الاطلال", "اطلال", "الطلول",  # ruins/traces
    "حي المنازل", "حي الديار",    # "I greet the abodes" (multi-word, kept for docs)
    "قفا نبك", "قفا",              # "halt, let us weep" — atlāl opening
    "عفا الربع", "عفا",            # "the encampment faded"
}

# مديح — praise (tribe, ruler, patron)
MADIH_MARKERS = {
    "مدح", "امدح", "المدح",
    "سلطان", "السلطان", "امير", "الامير", "شيخ", "الشيخ", "الشيوخ",
    "ملك", "الملك", "الملوك",
    "جود", "الجود", "جواد", "كريم", "الكريم", "كرم", "الكرم",
    "شجاع", "الشجاع", "شجاعه", "بطل", "الابطال",
    "نبيل", "النبيل", "شريف", "الشريف",
    "سيد", "السيد", "الساده",
    "مجد", "المجد", "عز", "العز",
    "سيف", "السيف",   # praise object; care needed
    "فضل", "الفضل",
    # Messenger/rider trope — extremely common Nabati opener where the poet
    # addresses a rider to carry verse to a distant patron. When it opens a
    # poem it is *almost always* مديح (less often غزل, where it carries
    # message to beloved — in that case غزل markers in the same verse win
    # on margin). 62+ occurrences in the corpus.
    "ياراكب", "ياركب", "يا راكب", "راكب", "الراكب",
    "مبشر", "المبشر", "بشاره", "البشاره",  # bearer-of-good-news frame
    "سلام", "سلامي", "سلامك",  # "my greetings to [patron]"
    "نصره", "نصرة", "انصر",  # victory/support — common in praise
}

# فخر — boasting / self-praise
FAKHR_MARKERS = {
    "انا", "أنا",
    "نحن", "قومي", "اهلي", "عشيرتي", "قبيلتي", "رهطي",
    "فخر", "افتخر", "افاخر", "مفاخر",
    "عزنا", "مجدنا", "شرفنا",
    "صناديد", "الصناديد",
    "سلالتي", "نسلي",
}

# غزو — war narrative / raid (heavy in this corpus, see Sowayan)
GHAZW_MARKERS = {
    "غزو", "الغزو", "غزوه", "غزونا", "نغزو",
    "ذبحه", "الذبحه", "معركه", "المعركه",
    "يوم", "ذاك اليوم",   # occasion marker — weaker alone; strong with other cues
    "خيل", "الخيل", "خيول", "الخيول",
    "فرسان", "الفرسان", "فارس",
    "عدو", "العدو", "الاعداء", "عداه",
    "سيف", "سيوف", "رمح", "الرماح",   # weapons family — overlap with مديح, disambiguate via occasion
    "طعن", "الطعن", "ضرب", "الضرب",
    "بنادق", "رصاص", "البارود",
    "غارنا", "غاروا", "نهب", "النهب",
    "عقب", "يوم عقب",   # narrative time markers
    "فزع", "فزعه", "الفزعه",
    # Lookout-scout trope: a scout atop a mountain reports what he sees —
    # classic Nabati narrative frame that almost always introduces a raid
    # report or warning of approaching riders. 12+ corpus occurrences.
    "مرقب", "المرقب", "ذرا", "الذرا",  # "summit / lookout"
    "راصد", "الراصد", "نظار", "النظار",  # scout / spotter
    "قبيله", "القبيله", "القبايل",  # tribal-mobilisation context
    "كور", "الكور",                  # saddle-horn (riding/raiding metaphor)
}

# حماسة — valour / war exhortation (lyric, not narrative)
HAMASA_MARKERS = {
    "شجاعه", "الشجاعه", "اقدم", "اقدموا",
    "حماس", "الحماسه",
    "جريئه", "جريء",
    "الموت ولا",  # "death over X" — signature حماسة phrase
    "لا نهاب", "ما يهاب", "لا يخاف",
    "ابطال", "الابطال", "ابطاله",
}

# هجاء — satire / invective
HIJA_MARKERS = {
    "هجاء", "اهجو", "يهجو",
    "اذم", "ذم", "الذم",
    "لئيم", "اللئام", "دني", "الاندال",
    "جاهل", "الجهال", "احمق", "الحمقى",
    "بخيل", "البخيل", "البخلاء", "شحيح",
    "جبان", "الجبناء",
    "خسيس", "الخساسه",
}

# حكمة — wisdom / gnomic
# Expanded with corpus-observed Dahr (fate) vocabulary — these are among
# the top-50 most frequent words across all matla lines in the corpus.
HIKMA_MARKERS = {
    "حكمه", "الحكمه",
    "العاقل", "الجاهل",
    "الصبر", "صبر", "صبرا", "اصبر",
    "الدنيا", "دنيا", "بالدنيا",
    "الايام", "ايام", "الليالي", "ليالي",
    "الزمان", "زمان", "الدهر", "دهر",
    "عبره", "العبره", "عبر", "العبر",
    "يا صاحبي", "يا ولد",   # didactic address (multi-word — kept for reference)
    "احذر", "اياك",
    "جرب", "جربت", "التجارب",
    "كثر", "قليل",   # gnomic quantity comparisons
    "الناس",   # "people" in generalising sense
    "العالم", "عالم",
    "مضى", "ما مضى", "قد مضى",
    # Corpus-observed didactic/proverbial openers
    "من كثر", "كثر ما",  # "from how much [one does X]" — quantifying wisdom
    "عفا الله", "عفا",    # "may God forgive" — moral/humble preamble
    "لو كان", "لوكان",    # counterfactual ("if it were...") — often gnomic
    "ياخي", "يا اخي",      # didactic address
    "مما قال",              # "among what he said" — attribution for wisdom sayings
    "القيل", "قيل",          # "it is said" — proverb attribution
}

# وصف — descriptive (nature, animal, landscape)
WASF_MARKERS = {
    "وصفت", "اصف", "وصف",
    "ناقه", "الناقه", "النياق", "بعير", "البعير", "ذلول", "الذلول",
    "فرس", "الفرس", "مهر", "المهر", "حصان",
    "ريم", "الريم", "ظبي", "الظبي",   # gazelle/deer as nature subject (not beloved metaphor)
    "سحاب", "السحاب", "مطر", "المطر", "غيث", "الغيث",
    "برق", "البرق", "رعد", "الرعد",
    "رياض", "الرياض", "روضه", "الروضه",
    "جبل", "الجبل", "جبال", "واد", "الوادي", "ودان",
    "رمل", "الرمال", "نفود", "النفود", "صحراء", "الصحراء",
    "نجوم", "النجوم", "قمر", "القمر", "شمس", "الشمس",
}

# دينية — religious / devotional
DINIYYA_MARKERS = {
    "الله", "الرحمن", "الرحيم", "رب", "ربي", "ربنا", "يالله", "ياالله",
    "محمد", "النبي", "الرسول", "المصطفى", "المختار",
    "صلى الله", "صلوا على", "الصلاه", "صلاتي",
    "مكه", "المدينه", "البيت الحرام", "الكعبه",
    "جنه", "الجنه", "نار", "النار", "جهنم",
    "قيامه", "القيامه", "الحساب",
    "قران", "القران", "الكتاب",
    "الايمان", "مومن", "المومنين",
    "توبه", "التوبه", "استغفر", "اتوب",
    # "الحمد ل..." — universal devotional opener ("praise be to...")
    "الحمد", "للحمد", "بالحمد", "المحمود",
    "سبحان", "تبارك",   # doxological openers
    "يارب", "ياربي",    # direct prayer openers
    "مرخي",  # "[God] who loosens [rain/sustenance]" — theological epithet, corpus-observed
    "الاقوات", "الارزاق", "الرزق",  # divine-provision vocabulary
}


_GENRE_LEXICONS: dict[str, set[str]] = {
    "غزل":    {_normalise(w) for w in GHAZAL_MARKERS},
    "رثاء":   {_normalise(w) for w in RITHA_MARKERS},
    "مديح":   {_normalise(w) for w in MADIH_MARKERS},
    "فخر":    {_normalise(w) for w in FAKHR_MARKERS},
    "غزو":    {_normalise(w) for w in GHAZW_MARKERS},
    "حماسة":  {_normalise(w) for w in HAMASA_MARKERS},
    "هجاء":   {_normalise(w) for w in HIJA_MARKERS},
    "حكمة":   {_normalise(w) for w in HIKMA_MARKERS},
    "وصف":    {_normalise(w) for w in WASF_MARKERS},
    "دينية":  {_normalise(w) for w in DINIYYA_MARKERS},
}


# ── Emotion lexicons ──────────────────────────────────────────────────────────
# Why emotion is multi-label: an elegy (grief) for a fallen warrior (pride)
# whose absence wounds (longing) is plausibly carrying three emotions at once.
# Forcing it to one would discard information.

EMOTION_LEXICONS_RAW: dict[str, set[str]] = {
    # longing — insomnia / sleeplessness / yearning (the signature Nabati emotion)
    "longing":   {"شوق", "شوقي", "شوقه", "اشواق", "مشتاق", "اشتقت", "يشتاق",
                  "حنين", "حنيني", "حنت",
                  "وحشه", "غربه", "بعاد", "فراق", "الفراق",
                  "النوم", "سهر", "سهرت", "سهران",   # classic sleeplessness trope
                  "البارحه", "البارحة",              # "last night" — yearning opener
                  "هجر", "الهجر", "هجران"},
    # grief — expanded with Khaleeji ونين (moaning), هم (cares), نياحه (dirge)
    "grief":     {"حزن", "الحزن", "حزين", "حزينه", "حزينا", "اسى",
                  "دمع", "الدمع", "دموع", "دموعي", "بكاء", "بكيت", "تبكي",
                  "ناح", "النواح", "نياحه", "النياحه",
                  "حسره", "الحسره", "ياحسرتي",
                  "ونه", "الونه", "ونين", "الونين", "ونينه",  # Khaleeji moaning/lament
                  "هم", "الهم", "همي", "همومي", "الهموم",
                  "قهر", "القهر", "مقهور"},
    "joy":       {"فرح", "الفرح", "سرور", "السرور", "بشر", "البشرى", "بشاره",
                  "هني", "الهناء", "ضحك", "انشرح", "طرب", "الطرب"},
    # pride — added tribal/self-assertion vocabulary
    "pride":     {"عزه", "العزه", "فخر", "الفخر", "افتخر",
                  "شامخ", "شموخ", "رفعه", "الرفعه", "كبرياء",
                  "عزنا", "مجدنا", "شرفنا",
                  "صناديد", "الصناديد", "ابطال", "الابطال"},
    "anger":     {"غضب", "الغضب", "غيظ", "سخط", "نقمه", "النقمه", "حرد",
                  "ثار", "الثار", "انتقم", "الانتقام"},
    "love":      {"حب", "الحب", "محبه", "المحبه", "ود", "الود", "وداد", "مودة", "المودة",
                  "هوى", "الهوى", "هواه", "هواك", "هواها",
                  "حبيب", "حبيبي", "محبوب", "الحبيب",
                  "عشق", "العشق", "عاشق"},
    # awe — added divine-provision vocabulary that signals reverence
    "awe":       {"هيبه", "الهيبه", "اجلال", "الاجلال", "عظمه", "العظمه",
                  "جلال", "الجلال", "سبحان", "تبارك",
                  "المحمود", "مرخي"},   # divine epithets
    "nostalgia": {"ذكرى", "الذكرى", "ذكريات", "الماضي", "ايام زمان",
                  "دار", "الديار", "المنازل", "الاطلال",  # deserted-abode trope
                  "مضى", "قد مضى", "ما مضى"},
    "hope":      {"رجاء", "الرجاء", "امل", "الامل", "وعد", "الوعد",
                  "طمع", "الطمع", "استبشر"},
    "fear":      {"خوف", "الخوف", "وجل", "الوجل", "هلع", "الهلع",
                  "فزع", "الفزع", "فزعه", "الفزعه",
                  "رعب", "الرعب"},
}

_EMOTION_LEXICONS: dict[str, set[str]] = {
    emo: {_normalise(w) for w in words}
    for emo, words in EMOTION_LEXICONS_RAW.items()
}


# ── Scoring thresholds (named constants — no magic numbers) ───────────────────
# Why these values: tuned on a pilot run over 200 sampled verses. Lower
# MIN_GENRE_MARKERS increases coverage at the cost of precision; higher makes
# the classifier abstain too often. 2 is the sweet spot where multi-marker
# evidence is required but short verses still get a chance.

MIN_GENRE_MARKERS = 1                    # need ≥1 marker; relaxed from 2 after
                                         # 95% abstention on the 1,502-matla pilot.
                                         # The matla is only the first line of the
                                         # poem (median 8 words) — requiring 2
                                         # markers excludes real love verses whose
                                         # single marker is "قلبي" or "البارحة".
                                         # Precision is preserved by MIN_GENRE_MARGIN,
                                         # which forces abstention when genres tie.
MIN_GENRE_MARGIN = 1                     # top-1 score must beat top-2 by this margin
MIN_EMOTION_MARKERS = 1                  # emotion is multi-label so 1 suffices
MAX_EMOTIONS_PER_VERSE = 4               # prevent emotion soup on long poems

# Trust weight for occasion-field hints. If a verse's `occasion` already says
# "غزل" the TOC transcriber has effectively ground-truthed it — weight 5
# (equivalent to 5 marker hits) makes the heuristic honour that signal
# without letting other markers in the matla override it.
OCCASION_HINT_WEIGHT = 5

# Occasion-field pattern-to-genre mapping. The Phase-4 TOC transcribers wrote
# occasion notes in free Arabic rather than genre tags, but the *shape* of the
# note is a strong signal: "بذبحة X" ("at the battle/slaying of X") is a
# war-narrative frame regardless of the verse content. Anchoring on these
# patterns recovers the dominant narrative genre in this corpus.
#
# Matched substring → inferred genre.
_OCCASION_PATTERNS: list[tuple[str, str]] = [
    # war / raid narrative frames (by far the most common in the corpus)
    ("ذبح",         "غزو"),   # بذبحة X, ذبحة Y
    ("غزو",         "غزو"),
    ("معرك",        "غزو"),
    ("فزع",         "غزو"),   # فزعة — rally for battle
    ("اخذ",         "غزو"),   # باخذته للجوف — "in his taking of"
    ("يوم يرس",     "غزو"),   # يوم يرسل — "on the day he sent [the raid]"
    ("يوم يطلب",    "غزو"),
    ("يوم يبد",     "غزو"),
    ("بتغريب",      "غزو"),   # raid of tribe X
    ("رد على",      "هجاء"),  # rebuttal — usually satirical exchange
    ("يرد على",     "هجاء"),
    ("ينصح",        "حكمة"),  # advice poems
    # love / romantic frames — uncommon in occasion field but worth catching
    ("غزل",         "غزل"),
    # praise frames
    ("بالامير",     "مديح"),
    ("بالشيخ",      "مديح"),
    ("بمحمد بن",    "مديح"),
    ("بعبد العزيز", "مديح"),
    # animal/horse description frames
    ("بفرسه",       "وصف"),   # describing his mare
    ("بناقته",      "وصف"),
]


def _tokenize(text: str) -> list[str]:
    """
    Split on whitespace + punctuation. Not using a real tokeniser because
    the lexicon is surface words, not stems — we want "شوقي" to match even
    though the lemma is شوق.
    """
    return [t for t in re.split(r"[\s\.,،؛؟!\"\':\-—–\(\)\[\]]+", text) if t]


# ── Public API ────────────────────────────────────────────────────────────────

def _heuristic_genre(
    matla_text: str,
    occasion: str | None = None,
) -> dict:
    """
    Pure keyword-heuristic genre classification (the original algorithm).
    Called by classify_genre() when (a) neural model unavailable, or
    (b) neural confidence is below NEURAL_GENRE_CONF_THRESHOLD.

    Returns same dict shape as classify_genre() so the caller can pass it
    through unchanged.
    """
    text_norm = _normalise(matla_text or "")
    tokens = set(_tokenize(text_norm))

    scores: dict[str, int] = {g: 0 for g in _GENRE_LEXICONS}
    evidence: list[str] = []

    # Base score: count lexicon hits per genre
    for genre, lex in _GENRE_LEXICONS.items():
        hits = tokens & lex
        scores[genre] = len(hits)
        if hits:
            evidence.extend(f"{genre}:{h}" for h in sorted(hits))

    # Occasion hint — two modes:
    #   (a) literal match: occasion == "غزل" → strong hint for غزل
    #   (b) pattern match: occasion contains "ذبح" → strong hint for غزو
    # Why both: 8 corpus entries use literal genre names, ~130 use descriptive
    # Arabic ("بذبحة X", "رد على Y") that encodes genre semantically. Pattern
    # match recovers the 130.
    occasion_hint = False
    if occasion:
        occ_norm = _normalise(occasion).strip()
        if occ_norm in _GENRE_LEXICONS:
            scores[occ_norm] += OCCASION_HINT_WEIGHT
            evidence.append(f"occasion:literal:{occ_norm}")
            occasion_hint = True
        else:
            for pattern, inferred_genre in _OCCASION_PATTERNS:
                if _normalise(pattern) in occ_norm:
                    scores[inferred_genre] += OCCASION_HINT_WEIGHT
                    evidence.append(f"occasion:pattern:{pattern}→{inferred_genre}")
                    occasion_hint = True
                    break   # one pattern is enough; avoid double-counting

    # Pick winner with abstention rules
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_genre, top_score = ranked[0]
    _, runner_score = ranked[1]

    # Abstain if top is weak or margin is too small
    if top_score < MIN_GENRE_MARKERS:
        genre = "غير_محدد"
        confidence = 0.0
    elif (top_score - runner_score) < MIN_GENRE_MARGIN:
        genre = "غير_محدد"
        confidence = 0.2   # tied — we know *something* matched, we just can't pick
    else:
        genre = validate_genre(top_genre)
        # Confidence = margin normalised by top score. Intuition: if top=5 and
        # runner=1, confidence is (5-1)/5 = 0.8 (clear winner); if top=3 and
        # runner=2, confidence is 1/3 = 0.33 (close call).
        confidence = round((top_score - runner_score) / top_score, 2)

    # Trim evidence to 6 for the UI tooltip; sort for determinism
    evidence = sorted(evidence)[:6]

    return {
        "genre":          genre,
        "confidence":     confidence,
        "scores":         scores,
        "evidence":       evidence,
        "occasion_hint":  occasion_hint,
        "genre_source":   "heuristic_v1",
    }


def classify_genre(
    matla_text: str,
    occasion: str | None = None,
) -> dict:
    """
    Classify a single verse's genre.

    Primary path: fine-tuned AraPoemBERT (arapoem_genre_best.pt).
    Fallback path: keyword heuristic (the original algorithm).

    Why neural-first: the fine-tuned model was trained on 3,340 Khaleeji
    Nabati clips and understands poetic syntax + dialectal vocabulary that
    the keyword lexicon misses (e.g. metaphorical uses of common words).
    Why keep the heuristic: portability + auditability — the demo must run
    without a GPU, and every tag must be explainable to the review panel.

    Returns:
        {
          "genre":         one of GENRES,
          "confidence":    0.0–1.0,
          "scores":        {genre: score} for all 10 genres (auditability),
          "evidence":      list of markers that hit (≤6, for UI tooltips),
          "occasion_hint": bool — was the occasion field itself a known genre?
          "genre_source":  "neural_v1" | "heuristic_v1"
        }
    """
    # ── Attempt neural path ────────────────────────────────────────────────
    _load_genre_clf()
    if _genre_clf is not None:
        try:
            neural_genre, neural_conf = _genre_clf.predict(_normalise(matla_text or ""))
            if neural_conf >= NEURAL_GENRE_CONF_THRESHOLD and neural_genre != "غير_محدد":
                # Neural prediction is confident — build a compatible result dict.
                # We still run the heuristic to fill in scores/evidence for
                # auditability (the reviewer can see both signals).
                heuristic = _heuristic_genre(matla_text, occasion)
                return {
                    "genre":          neural_genre,
                    "confidence":     round(neural_conf, 3),
                    "scores":         heuristic["scores"],       # heuristic scores for audit
                    "evidence":       heuristic["evidence"],     # keyword evidence for audit
                    "occasion_hint":  heuristic["occasion_hint"],
                    "genre_source":   "neural_v1",
                }
            # Neural abstained or low confidence → fall through to heuristic
        except Exception:
            pass   # any torch/transformers failure → fall through silently

    # ── Fallback: keyword heuristic ────────────────────────────────────────
    return _heuristic_genre(matla_text, occasion)


def _heuristic_emotions(matla_text: str) -> dict:
    """
    Pure keyword-heuristic multi-label emotion detection (the original algorithm).
    Called by classify_emotions() when the neural model is unavailable or as
    the complementary signal for audit purposes.
    """
    text_norm = _normalise(matla_text or "")
    tokens = set(_tokenize(text_norm))

    scores: dict[str, int] = {}
    for emo, lex in _EMOTION_LEXICONS.items():
        hits = tokens & lex
        if hits:
            scores[emo] = len(hits)

    # Keep emotions above threshold, ordered by score, capped at the max
    emotions = [
        e for e, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if s >= MIN_EMOTION_MARKERS
    ][:MAX_EMOTIONS_PER_VERSE]

    return {
        "emotions":        validate_emotions(emotions),
        "scores":          scores,
        "emotion_source":  "heuristic_v1",
    }


def classify_emotions(matla_text: str) -> dict:
    """
    Multi-label emotion detection.

    Primary path: fine-tuned AraPoemBERT (arapoem_emotion_text_best.pt).
    Fallback path: keyword heuristic.

    Returns:
        {
          "emotions":       list[str]  — emotions present, ordered by score,
          "scores":         dict[emotion, score]  — heuristic int counts for
                            auditability (or sigmoid floats if neural ran),
          "emotion_source": "neural_v1" | "heuristic_v1",
        }

    Why no abstention: an empty emotion list is already the abstention.
    """
    # ── Attempt neural path ────────────────────────────────────────────────
    _load_emotion_clf()
    if _emotion_clf is not None:
        try:
            sigmoid_scores = _emotion_clf.predict(_normalise(matla_text or ""))
            # Active emotions: above threshold, sorted by score, capped at max
            active = sorted(
                [(e, s) for e, s in sigmoid_scores.items() if s >= NEURAL_EMOTION_SIGMOID_THRESH],
                key=lambda kv: kv[1], reverse=True,
            )[:MAX_EMOTIONS_PER_VERSE]
            if active:
                emotions = validate_emotions([e for e, _ in active])
                return {
                    "emotions":       emotions,
                    "scores":         {e: round(s, 3) for e, s in sigmoid_scores.items()},
                    "emotion_source": "neural_v1",
                }
            # Neural found nothing → fall through to heuristic
        except Exception:
            pass   # any failure → fall through silently

    # ── Fallback: keyword heuristic ────────────────────────────────────────
    return _heuristic_emotions(matla_text)


def classify(matla_text: str, occasion: str | None = None) -> dict:
    """
    Combined single-call interface used by the enrichment script.

    Why propagate genre_source / emotion_source: the Qdrant payload and the
    Streamlit UI both need to know whether a tag came from the neural model
    (higher confidence, 🟢 badge) or the heuristic (silver baseline, 🔸 badge).
    The enrichment script writes these into anchor_registry_phase4_enriched.json
    so the badge survives across index rebuilds.
    """
    g = classify_genre(matla_text, occasion)
    e = classify_emotions(matla_text)
    return {
        "genre":              g["genre"],
        "genre_confidence":   g["confidence"],
        "genre_scores":       g["scores"],
        "genre_evidence":     g["evidence"],
        "occasion_hint":      g["occasion_hint"],
        "genre_source":       g.get("genre_source", "heuristic_v1"),
        "emotions":           e["emotions"],
        "emotion_scores":     e["scores"],
        "emotion_source":     e.get("emotion_source", "heuristic_v1"),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        ("يا حبيبي وش سوى بي هواك        وليه قلبي للهجر عشاك", "غزل"),
        ("رحل ابو فارس وخلى الديار حزينه  والدمع في عيني ما يبرد", None),
        ("يوم ذبحت بن رشيد بالمليدا      والخيل تلمع بالسيوف لمان", "بذبحة بن رشيد"),
        ("الصبر مفتاح الفرج يا صاحبي     من جد في طلب المعالي نال", None),
        ("وصفت ناقه كالمها فوق الرمال   ذلول تجوب النفود بلا كلل", None),
        ("الله ياربي عفوك يا كريم         وارحم عبيدك في يوم الحساب", None),
        ("هذا بيت غامض جدا لا يحوي مفردات", None),   # should abstain
    ]
    from pprint import pprint
    for verse, occ in samples:
        print("=" * 60)
        print(f"VERSE: {verse}")
        print(f"OCCASION: {occ}")
        result = classify(verse, occ)
        print(f"  genre:      {result['genre']}  (conf={result['genre_confidence']})")
        print(f"  emotions:   {result['emotions']}")
        print(f"  evidence:   {result['genre_evidence']}")
