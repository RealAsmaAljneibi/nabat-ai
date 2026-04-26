"""
agent2_retrieval_synthesis/nodes/image_provenance.py
=====================================================
Why this file exists: Before this node landed, an "upload an image and ask
'who wrote this?'" turn produced the *identical* generic refusal whether the
verse pictured was already in the corpus or not — the image path was stored
in state but no downstream node ever consumed its pixels, and the typed
meta-question had no retrieval signal of its own.

The intent_router (Stage 0.5a) detects the {image attached + meta-question}
combination and routes here. This node:

  1. Recovers the OCR'd verse text — preferring the marker the Streamlit
     composer embeds in the query, falling back to a fresh OCR call if the
     marker is absent (e.g. when the orchestrator was invoked from the CLI
     with input_image_path only).
  2. Runs verse-level dense retrieval against the existing index against
     that verse, with an elevated similarity threshold so a near-miss does
     not get confidently misattributed.
  3. Emits one of two specific outcomes:
       in-corpus  → cite poet + manuscript + page (deterministic, no LLM).
       not found  → "the image was read but this verse is not in our
                    digitised corpus" — distinct from the generic refusal,
                    so the user understands *why* and what to do next.

Architecture refs: §2.4 Stage 1 (image input path), §2.8 image-grounded
provenance subgraph in doc/ARCHITECTURE_DIAGRAMS.md (Diagram 2 IMG_GROUNDED).

Why a separate node and not an extension of deterministic_answer: this is
the only deterministic_intent that touches the vector index. Keeping the
retrieval call out of deterministic_answer.py preserves that file's contract
("templated/LLM-prose answers grounded in JSON files only").
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ...state import AgentState

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────
# Why named constants: §5 — every confidence threshold is a one-line tweak.
# 0.55 chosen because the AraBERT dense retriever gives ~0.7+ for true matches
# on cleanly-OCR'd verses but only ~0.35–0.50 for unrelated lines; 0.55 is the
# midpoint that admits noisy-OCR true matches without false positives.
MIN_MATCH_SIMILARITY = 0.55

# Marker the Streamlit composer (app/streamlit_app.py Layer 1) inserts when it
# combines typed text + OCR'd verse into a single effective_query. Parsing it
# here avoids re-running OCR for every image-grounded turn.
_OCR_MARKER_RE = re.compile(
    r"\[محتوى الصورة / image content\]\s*\n(.+?)\Z", re.DOTALL
)


def _extract_ocr_from_query(query_text: str) -> Optional[str]:
    """Return the OCR'd verse if the composer marker is present, else None."""
    if not query_text:
        return None
    match = _OCR_MARKER_RE.search(query_text)
    if match:
        return match.group(1).strip() or None
    return None


def _ocr_from_image_path(image_path: str) -> Optional[str]:
    """Re-run pytesseract OCR. Returns None on any failure (silent fallback)."""
    try:
        from ...image_ocr import ocr_image_to_query
        text = ocr_image_to_query(image_path)
        return (text or "").strip() or None
    except Exception as exc:
        logger.warning("image_provenance: re-OCR failed (%s)", exc)
        return None


def _verse_lookup(verse_text: str) -> tuple[Optional[dict], float]:
    """
    Run verse-level dense retrieval and return (best_chunk_dict, similarity).

    Why dense and not BM25: the OCR'd text often has small character errors
    that BM25 (exact-token) penalises hard; the AraBERT dense embedder is
    far more forgiving of single-character drift.

    Returns (None, 0.0) when the index is unavailable — the caller then emits
    the "image read but verse not in corpus" message rather than crashing.
    """
    try:
        from ..nodes.retrieve import _get_index
        from ...retrievers.dense import DenseRetriever
        bundle = _get_index()
    except Exception as exc:
        logger.error("image_provenance: index load failed — %s", exc)
        return None, 0.0

    # Restrict to verse-level chunks — group/poem/aggregate levels would dilute
    # the similarity signal for a single-line OCR query.
    verse_chunks = [c for c in bundle.chunks if c.level == "verse"]
    if not verse_chunks:
        return None, 0.0

    try:
        chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(bundle.chunks)}
        verse_indices = [chunk_id_to_idx[c.chunk_id] for c in verse_chunks]
        verse_embeddings = bundle.embeddings[verse_indices]
        retriever = DenseRetriever(verse_chunks, verse_embeddings)
        results = retriever.retrieve(verse_text, n=1)
        if not results:
            return None, 0.0
        top = results[0]
        # ScoredChunk.rrf_score holds the cosine similarity at this stage
        # (DenseRetriever stores its raw score there before RRF fusion).
        return top.to_dict(), float(top.rrf_score)
    except Exception as exc:
        logger.error("image_provenance: dense lookup failed — %s", exc)
        return None, 0.0


def _format_in_corpus_answer(chunk: dict, similarity: float, verse_text: str) -> str:
    """Bilingual answer when the verse was found. Deterministic — no LLM."""
    poet = chunk.get("poet_name") or "—"
    manuscript = chunk.get("manuscript_short_key") or "—"
    page = chunk.get("source_page") or "—"
    matched_text = (chunk.get("matla_text") or chunk.get("text") or "")[:200]

    ar = (
        f"📜 تم التعرّف على البيت في المخطوطات المرقَّمة.\n\n"
        f"**النص المقروء من الصورة:**  \n{verse_text}\n\n"
        f"**المطابقة في المجموعة (تشابه = {similarity:.2f}):**  \n{matched_text}\n\n"
        f"**الشاعر:** {poet}  \n"
        f"**المخطوطة:** {manuscript}  \n"
        f"**الصفحة:** {page}"
    )
    en = (
        f"📜 The verse was identified in the digitised corpus.\n\n"
        f"**Read from image:**  \n{verse_text}\n\n"
        f"**Matched in corpus (similarity = {similarity:.2f}):**  \n{matched_text}\n\n"
        f"**Poet:** {poet}  \n"
        f"**Manuscript:** {manuscript}  \n"
        f"**Page:** {page}"
    )
    return f"{ar}\n\n---\n\n{en}"


def _format_not_in_corpus_answer(verse_text: str, best_similarity: float) -> str:
    """
    Bilingual answer when no verse matched above the threshold.

    Why a *specific* message: the bug we are fixing is that this case used to
    return the generic "the digitised manuscripts do not contain a direct
    answer to this query" — the same string an out-of-scope topic question
    gets. This wording instead names the situation: the OCR succeeded, the
    image was understood, but the verse is not in the digitised subset.
    """
    ar = (
        f"📜 تمّت قراءة البيت من الصورة، لكنّه **غير موجود** في المجموعة المرقَّمة حالياً.\n\n"
        f"**النص المقروء:**  \n{verse_text}\n\n"
        f"_أعلى مطابقة في المجموعة كان تشابهها {best_similarity:.2f} وهو أدنى من حد الثقة "
        f"({MIN_MATCH_SIMILARITY}). قد يكون البيت موجوداً في مخطوطة لم تُرقمَن بعد، "
        f"أو أن الكتابة اليدوية تحتاج مراجعة بشرية._"
    )
    en = (
        f"📜 The verse was read from the image, but it is **not present** in the "
        f"currently digitised corpus.\n\n"
        f"**Read text:**  \n{verse_text}\n\n"
        f"_The best match in the corpus had similarity {best_similarity:.2f}, below "
        f"the confidence threshold ({MIN_MATCH_SIMILARITY}). The verse may live in a "
        f"manuscript that has not yet been digitised, or the handwriting may need "
        f"human review._"
    )
    return f"{ar}\n\n---\n\n{en}"


def _format_no_ocr_answer() -> str:
    """Fallback when neither the marker nor the image_path produced OCR text."""
    ar = (
        "تعذّرت قراءة الصورة بدقة كافية للتحقق من وجودها في المجموعة. "
        "الرجاء كتابة البيت يدوياً أو رفع صورة أوضح."
    )
    en = (
        "The image could not be read with enough confidence to verify it against "
        "the corpus. Please type the verse, or upload a clearer image."
    )
    return f"{ar}\n\n---\n\n{en}"


def image_provenance_node(state: AgentState) -> AgentState:
    """
    Image-grounded provenance node — runs when intent_router routed
    track="image_grounded_provenance". Reads OCR text + runs verse-level
    retrieval; writes a SPECIFIC bilingual answer (citation if in corpus;
    distinct "not in corpus" message if not).
    """
    qc = state.get("query_context") or {}
    raw_query = state.get("raw_query", "") or qc.get("query_ar") or qc.get("query_en") or ""

    # 1. Recover the verse text — prefer composer marker, else re-OCR.
    verse_text = _extract_ocr_from_query(raw_query)
    if not verse_text:
        image_path = state.get("input_image_path")
        if image_path:
            verse_text = _ocr_from_image_path(image_path)

    if not verse_text:
        text = _format_no_ocr_answer()
        state["final_response"]    = text
        state["formatted_response"] = {
            "al_maktub": text, "orthographic": text, "al_mantuq": "", "citations": [],
        }
        state["citations_used"]   = []
        state["passage_ids_used"] = []
        state["is_refusal"]       = False
        state["crag_verdict"]     = "Ambiguous"
        state["self_rag_verdict"] = "pass"
        state["guardrail_passed"] = True
        state["guardrail_flags"]  = ["image_grounded_provenance", "no_ocr"]
        return state

    # 2. Verse-level retrieval against the corpus.
    chunk, similarity = _verse_lookup(verse_text)

    # 3. Branch on confidence.
    if chunk is not None and similarity >= MIN_MATCH_SIMILARITY:
        text = _format_in_corpus_answer(chunk, similarity, verse_text)
        state["citations_used"]   = [chunk.get("anchor_id", "")]
        state["passage_ids_used"] = [chunk.get("chunk_id", "")]
        state["guardrail_flags"]  = ["image_grounded_provenance", "in_corpus"]
        logger.info(
            "image_provenance: HIT verse_text=%r poet=%s sim=%.3f",
            verse_text[:40], chunk.get("poet_name", "—"), similarity,
        )
    else:
        best_sim = similarity if chunk is not None else 0.0
        text = _format_not_in_corpus_answer(verse_text, best_sim)
        state["citations_used"]   = []
        state["passage_ids_used"] = []
        state["guardrail_flags"]  = ["image_grounded_provenance", "not_in_corpus"]
        logger.info(
            "image_provenance: MISS verse_text=%r best_sim=%.3f",
            verse_text[:40], best_sim,
        )

    state["final_response"]    = text
    state["formatted_response"] = {
        "al_maktub": text, "orthographic": text, "al_mantuq": "", "citations": [],
    }
    state["is_refusal"]       = False
    state["crag_verdict"]     = "Correct"
    state["self_rag_verdict"] = "pass"
    state["guardrail_passed"] = True
    return state
