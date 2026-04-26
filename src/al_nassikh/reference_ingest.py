"""
src/al_nassikh/reference_ingest.py
====================================
Why this file exists: Asma added 4 scholarly Arabic PDFs to `manuscripts/` that
are *secondary references* about Khaleeji/Bedouin Nabati poetry — not the
primary handwritten corpus. They give the RAG broader cultural and theoretical
context (Ibn Khaldun's discussion of Bedouin poetry, desert culture, tribal
customary law, palm/camel symbolism). These PDFs use a private-use font
encoding so direct text extraction returns garbled bytes; we OCR them with
tesseract-ara via `pytesseract` to get clean Unicode Arabic.

Output: a flat JSON file (`data/ground_truth/reference_corpus.json`) with one
record per chunk that the index builder treats as a new chunk level
(`level="reference"`). Each record carries:

  chunk_id            stable id, e.g. `reference_ibn_khaldoun_p012_para02`
  level               always "reference"
  text                cleaned Arabic paragraph
  book_short_key      one of: ibn_khaldoun, desert_culture, suloom_alarab, semiotics
  book_title_ar       Arabic title of the source book
  book_title_en       English title of the source book
  page                1-indexed page number in the PDF
  paragraph_index     0-indexed paragraph within the page
  source_pdf          relative path to the PDF for clickable citation

Bilingual searchability: AraBERT is Arabic-only, but Agent 1's bilingual_expand
+ HyDE pipeline already translates EN queries into AR and generates AR
hypothetical passages before retrieval. So an English query like "what does
Ibn Khaldun say about Bedouin poetry?" hits the same Arabic vector space the
PDFs were embedded in — no second EN index needed.

Architecture refs: §2.4 (Agent 1 bilingual handling), §2.5 Stage 4 (retrieval
sees one unified Arabic vector space), §0 guideline 5 (silver-baseline 🔸 badge
must show on heuristic-only metadata — references inherit no genre tag, which
the UI signals with a 📚 reference badge instead of 🔸).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# ── Book metadata (source of truth for bilingual titles) ─────────────────────
# Why hand-curated: the PDFs themselves are anonymous excerpts from larger
# scholarly works. We bake the AR + EN titles here so the UI can render
# "📚 Ibn Khaldun on Bedouin Poetry — p.113" alongside the AR equivalent
# without depending on PDF metadata (which is empty for these files).
BOOK_METADATA: dict[str, dict[str, str]] = {
    "ibn_khaldoun_abt_nabatipoetry": {
        "short_key":  "ibn_khaldoun",
        "title_ar":   "الشعر البدوي في مقدمة ابن خلدون",
        "title_en":   "Bedouin Poetry in Ibn Khaldun's Muqaddimah",
        "topic_ar":   "تحليل تاريخي ولغوي لانتقال الشعر العربي من الفصيح إلى النبطي",
        "topic_en":   "Historical and linguistic analysis of the transition from Classical to Nabati Arabic poetry",
    },
    "desert_culture": {
        "short_key":  "desert_culture",
        "title_ar":   "ثقافة الصحراء",
        "title_en":   "Desert Culture",
        "topic_ar":   "قيم البداوة: الكرم والشجاعة والرحيل وعلاقة البدوي بالأرض",
        "topic_en":   "Bedouin values: generosity, courage, nomadism, and the desert's hold on Arab imagination",
    },
    "suloom_alarab": {
        "short_key":  "suloom_alarab",
        "title_ar":   "سلوم العرب",
        "title_en":   "The Customs of the Arabs (Sulum al-Arab)",
        "topic_ar":   "القانون العرفي القبلي: الغزو والاشتباك والعفو وحقن الدماء",
        "topic_en":   "Tribal customary law: raiding, combat conventions, pardon, and the shedding of blood",
    },
    "semiotics": {
        "short_key":  "semiotics",
        "title_ar":   "ثنائية الرموز (النخلة والإبل)",
        "title_en":   "Symbolic Duality: Palm and Camel in Bedouin/Urban Poetry",
        "topic_ar":   "رمزية النخلة والإبل في الشعر النبطي والفصيح والتمييز بين البداوة والحضارة",
        "topic_en":   "Palm-tree vs camel symbolism in Nabati and Classical poetry; nomadic versus settled identity",
    },
}


@dataclass
class ReferenceChunk:
    """One paragraph-sized record produced from a PDF page after OCR."""
    chunk_id:        str
    level:           str
    text:            str
    book_short_key:  str
    book_title_ar:   str
    book_title_en:   str
    topic_ar:        str
    topic_en:        str
    page:            int
    paragraph_index: int
    source_pdf:      str

    def to_dict(self) -> dict:
        return asdict(self)


# ── OCR + cleaning helpers ────────────────────────────────────────────────────

# Tesseract sometimes injects rare-glyph noise on margins (page numbers, footers,
# stray underline ligatures). These regexes keep the body text clean enough that
# AraBERT's tokeniser doesn't waste embedding budget on noise tokens.
_HARAKAT_RE      = re.compile(r"[ً-ٰٟ]")   # tashkeel + dagger alif
_TATWEEL_RE      = re.compile(r"ـ+")
_PAGE_HEADER_RE  = re.compile(r"^\s*[\d/\\٠-٩]+\s*[/\\]\s*[^\n]{0,40}$", re.MULTILINE)
_FOOTNOTE_NUM_RE = re.compile(r"^\s*\(\s*[\d٠-٩]+\s*\)\s*", re.MULTILINE)
_MULTI_WS_RE     = re.compile(r"[ \t]+")


def _clean_ocr_text(raw: str) -> str:
    """Normalise OCR output for consistent embedding quality."""
    s = raw
    s = _HARAKAT_RE.sub("", s)
    s = _TATWEEL_RE.sub("", s)
    s = _PAGE_HEADER_RE.sub("", s)
    s = _MULTI_WS_RE.sub(" ", s)
    # Collapse the inevitable trailing-blank-lines explosion
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_paragraphs(page_text: str, min_chars: int = 120, max_chars: int = 900) -> list[str]:
    """
    Split cleaned page text into paragraph-sized chunks.

    Why two thresholds: paragraphs in academic Arabic prose are long. We keep
    natural paragraphs intact when they fit under max_chars; oversize paragraphs
    are split on sentence boundaries (full-stop / Arabic punctuation). Tiny
    fragments (< min_chars) are merged into the previous paragraph so that
    poetry couplets and isolated lines don't become standalone chunks.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            if out and len(para) < min_chars:
                out[-1] = out[-1] + " " + para
            else:
                out.append(para)
            continue
        # Oversize — split on sentence boundary.
        sentences = re.split(r"(?<=[\.\!\?\؟])\s+", para)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= max_chars:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    out.append(buf)
                buf = s
        if buf:
            out.append(buf)
    # Final pass: merge any chunk shorter than min_chars into its neighbour.
    merged: list[str] = []
    for p in out:
        if merged and len(p) < min_chars:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


# ── PDF iteration ─────────────────────────────────────────────────────────────

def _ocr_pdf_pages(pdf_path: Path, dpi: int = 200, lang: str = "ara") -> Iterable[tuple[int, str]]:
    """
    Yield (page_number_1_indexed, raw_ocr_text) for every page of `pdf_path`.

    Lazy imports so the module can be loaded for reading existing JSON output
    without requiring pdf2image/pytesseract on every dev machine.
    """
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(pdf_path), dpi=dpi)
    for i, img in enumerate(images, 1):
        try:
            text = pytesseract.image_to_string(img, lang=lang, config="--psm 1")
        except Exception as exc:
            logger.warning("OCR failed for %s page %d: %s", pdf_path.name, i, exc)
            text = ""
        yield i, text


def chunks_from_pdf(pdf_path: Path) -> list[ReferenceChunk]:
    """
    OCR `pdf_path`, clean each page, split into paragraph chunks, and return
    a list of ReferenceChunk records ready to be merged into the index.
    """
    stem = pdf_path.stem
    meta = BOOK_METADATA.get(stem)
    if meta is None:
        raise ValueError(
            f"reference_ingest: no BOOK_METADATA entry for {stem!r}. "
            f"Add an entry to BOOK_METADATA before ingesting."
        )

    out: list[ReferenceChunk] = []
    for page_num, raw_text in _ocr_pdf_pages(pdf_path):
        cleaned = _clean_ocr_text(raw_text)
        if not cleaned:
            continue
        for para_idx, para in enumerate(_split_paragraphs(cleaned)):
            chunk_id = f"reference_{meta['short_key']}_p{page_num:03d}_para{para_idx:02d}"
            out.append(ReferenceChunk(
                chunk_id=chunk_id,
                level="reference",
                text=para,
                book_short_key=meta["short_key"],
                book_title_ar=meta["title_ar"],
                book_title_en=meta["title_en"],
                topic_ar=meta["topic_ar"],
                topic_en=meta["topic_en"],
                page=page_num,
                paragraph_index=para_idx,
                source_pdf=str(pdf_path.relative_to(pdf_path.parent.parent)),
            ))
    logger.info("reference_ingest: %s → %d chunks", pdf_path.name, len(out))
    return out


def build_reference_corpus(
    pdf_dir: Path,
    output_path: Path,
    pdf_names: list[str] | None = None,
) -> list[dict]:
    """
    Convert every recognised PDF in `pdf_dir` into reference chunks and write
    the merged list as JSON at `output_path`. Returns the list of chunk dicts.
    """
    if pdf_names is None:
        pdf_names = [f"{name}.pdf" for name in BOOK_METADATA]

    all_chunks: list[ReferenceChunk] = []
    for name in pdf_names:
        pdf_path = pdf_dir / name
        if not pdf_path.exists():
            logger.warning("reference_ingest: missing %s — skipping.", pdf_path)
            continue
        all_chunks.extend(chunks_from_pdf(pdf_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in all_chunks]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("reference_ingest: wrote %d chunks → %s", len(payload), output_path)
    return payload


def load_reference_corpus(path: Path) -> list[dict]:
    """Read a previously-built reference_corpus.json. [] if missing."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
