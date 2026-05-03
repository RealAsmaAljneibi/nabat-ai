"""
src/fatat_al_arab/embed.py
==========================
Why this file exists: M3 Stage 1 — every RAG chunk needs a dense vector before
it can be stored in Qdrant. This module owns the embedding contract so the rest
of the pipeline imports a stable interface, not a model name.
When triggered: At first call to dense retrieval, prototype router, or index.py
build (lazy-loaded singleton).
Purpose: Loads AraBERT 768-dim sentence-transformer + Arabic normaliser; deterministic hash fallback for offline/CI

Two embeddings are produced per text:
  1. Dense: AraPoemBERT (locally fine-tuned on Khaleeji Nabati poetry) via
     sentence-transformers. 768-dimensional cosine-normalised float32 vector.
     Why AraPoemBERT over generic AraBERTv2: AraPoemBERT was pre-trained on
     2M+ Arabic poetry verses and then fine-tuned on 3,340 Khaleeji Nabati
     clips (~/poetry project, MAAI7103). Its embeddings are calibrated for
     poetic vocabulary, meter, and Khaleeji dialectal variation — outperforming
     MSA news-trained encoders for verse-level semantic retrieval.
     Fallback chain: local AraPoemBERT → env-var EMBED_MODEL → AraBERTv2 (HF).

  2. BM25 sparse: rank-bm25 BM25Okapi index over the corpus vocabulary.
     Sparse retrieval is the backstop for exact / rare token queries
     (poet names, volume numbers) where dense models fail.

The module does NOT manage the Qdrant collection — that is index.py.
It only returns (dense_vector: list[float], bm25_tokens: list[str]).

Graceful degradation: if sentence-transformers / the HuggingFace model is
unavailable (CI, offline), we fall back to a deterministic random vector seeded
from the text hash so tests can run without a GPU or network.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Why: local AraPoemBERT fine-tuned on 3,340 Khaleeji Nabati clips is the
# primary encoder. EMBED_MODEL env var overrides (set to a HF model name to
# use a hosted model). Fallback to AraBERTv2 if local path does not exist.
_LOCAL_ARAPOEM = Path.home() / "poetry" / "models" / "local" / "arapoembert"
_HF_FALLBACK   = "aubmindlab/bert-base-arabertv02"

ARABERT_MODEL   = os.getenv("EMBED_MODEL") or (
    str(_LOCAL_ARAPOEM) if _LOCAL_ARAPOEM.exists() else _HF_FALLBACK
)
DENSE_DIM       = 768
_FALLBACK_SEED  = 42          # only used when the real model is unavailable

# Module-level singleton so we only load the model once per process.
_model: Any = None
_model_loaded: bool = False   # True once we attempted a load (even if it failed)
_model_available: bool = False


def _load_model() -> None:
    """Load sentence-transformers model once; set _model_available flag."""
    global _model, _model_loaded, _model_available
    if _model_loaded:
        return
    _model_loaded = True
    try:
        from sentence_transformers import SentenceTransformer  # soft dependency
        _model = SentenceTransformer(ARABERT_MODEL)
        _model_available = True
    except Exception:
        _model = None
        _model_available = False


# ── Arabic text pre-processing ────────────────────────────────────────────────

_ALEF_VARIANTS  = re.compile(r"[آأإٱ]")
_TATWEEL        = re.compile(r"ـ")

# Comprehensive tashkeel pattern covering three Unicode bands:
#   U+0610–U+061A  Arabic combining marks (sign Sallallahou, etc.)
#   U+064B–U+065F  Standard harakat: fathatan, dammatan, kasratan, fatha, damma,
#                  kasra, shadda, sukun, maddah, hamza above/below, subscript alef
#   U+0670         Superscript alef (used with alef wasla)
#   U+06D6–U+06DC  Quranic small high marks (sajdah, rub el hizb, etc.)
#   U+06DF–U+06E4  Arabic small low/high marks
#   U+06E7–U+06E8  Small high ya / small high noon
#   U+06EA–U+06ED  Arabic poetic verse signs — appear in scanned manuscript OCR
# Why all three bands: manuscript scans from different eras use different encoding
# conventions; stripping only the base harakat band leaves poetic verse signs in
# the text, causing BM25 token mismatches between tashkeel-on and tashkeel-off copies.
_TASHKEEL = re.compile(
    r"[\u0610-\u061A"
    r"\u064B-\u065F"
    r"\u0670"
    r"\u06D6-\u06DC"
    r"\u06DF-\u06E4"
    r"\u06E7-\u06E8"
    r"\u06EA-\u06ED]"
)

_NON_ARABIC_SEP = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]")


def strip_tashkeel(text: str) -> str:
    """
    Remove all Arabic diacritical marks (tashkeel) from *text*.
    Covers standard harakat, shadda, sukun, and the extended Quranic/poetic
    mark bands that appear in scanned manuscript OCR output.
    Called by normalise_arabic (step 5) and usable standalone when only
    tashkeel removal is needed without full normalisation.
    """
    return _TASHKEEL.sub("", text)


def normalise_arabic(text: str) -> str:
    """
    Light normalisation before embedding/BM25 tokenisation.
    Why this (not camel-tools): portability. camel-tools is heavy; this covers
    the critical equivalences for Nabati Khaleeji (alef variants, tashkeel, ة→ه
    for dialectal variants, tatweel stripping, ى→ي for cache-key stability).

    Normalisation steps (order matters):
      1. Unify alef variants → ا
      2. ة (ta-marbuta) → ه  — dialectal equivalence
      3. ى (alef-maksura) → ي — prevents cache miss on "مكي" vs "مكى"
      4. Strip tatweel (ـ)
      5. Strip tashkeel — all three Unicode bands (standard + Quranic + poetic verse signs)
      6. Replace non-Arabic punctuation with space
    """
    if not text:
        return ""
    text = _ALEF_VARIANTS.sub("ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")   # unify ya / alef-maksura for cache-key stability
    text = _TATWEEL.sub("", text)
    text = strip_tashkeel(text)
    text = _NON_ARABIC_SEP.sub(" ", text)
    return " ".join(text.split())


def tokenise(text: str) -> list[str]:
    """Split normalised text into whitespace tokens (BM25 input)."""
    return normalise_arabic(text).split()


# ── Dense embedding ───────────────────────────────────────────────────────────

def _fallback_vector(text: str) -> list[float]:
    """
    Deterministic pseudo-random unit vector seeded from text hash.
    Used in CI / offline environments where the real model is unavailable.
    The vector is stable across runs for the same text, so tests can assert
    on retrieval ordering without loading a 500 MB model.
    """
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    v = rng.standard_normal(DENSE_DIM).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v.tolist()


def get_dense_embedding(text: str) -> list[float]:
    """
    Return a 768-d cosine-normalised float32 vector for *text*.
    Falls back to _fallback_vector() if the model cannot be loaded.
    """
    _load_model()
    if _model_available and _model is not None:
        vec = _model.encode(
            normalise_arabic(text),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()
    return _fallback_vector(text)


def get_dense_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Batch encode for index build — more efficient than one-by-one.
    Returns list of 768-d vectors in the same order as *texts*.
    """
    _load_model()
    if _model_available and _model is not None:
        normalised = [normalise_arabic(t) for t in texts]
        vecs = _model.encode(
            normalised,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [v.tolist() for v in vecs]
    return [_fallback_vector(t) for t in texts]


# ── BM25 sparse index builder ─────────────────────────────────────────────────

class BM25Index:
    """
    Thin wrapper around rank-bm25 BM25Okapi.
    Accepts a corpus of raw texts; tokenises internally.
    Why wrap rather than expose BM25Okapi directly: the wrapper normalises
    Arabic before tokenisation — calling code shouldn't have to remember to do that.
    """

    def __init__(self, corpus: list[str]) -> None:
        """
        Args:
            corpus: list of raw text strings (order must match chunk IDs used later).
        """
        from rank_bm25 import BM25Okapi  # soft dependency
        self._tokenised = [tokenise(t) for t in corpus]
        self._bm25 = BM25Okapi(self._tokenised)
        self._corpus = corpus

    def get_scores(self, query: str) -> np.ndarray:
        """Return BM25 score array (length = corpus size) for *query*."""
        q_tokens = tokenise(query)
        if not q_tokens:
            return np.zeros(len(self._corpus), dtype=np.float32)
        return self._bm25.get_scores(q_tokens).astype(np.float32)

    def get_top_n(self, query: str, n: int = 10) -> list[tuple[int, float]]:
        """Return list of (corpus_index, score) sorted descending, top *n*."""
        scores = self.get_scores(query)
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._corpus)


def is_model_available() -> bool:
    """
    Check whether the real AraBERT model can be loaded.
    Used by Streamlit to show a warning if only fallback vectors are active.
    """
    _load_model()
    return _model_available
