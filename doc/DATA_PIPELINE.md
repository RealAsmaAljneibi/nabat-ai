# NABAT-AI — Data Pipeline (Source → Ground Truth → RAG Index)

**Owner:** Asma Salem Mubarak Najem Aljneibi · **Course:** MAAI1704 · **Last updated:** 2026‑04‑19

This document is the authoritative map of *where data comes from*, *what phase it belongs to*,
and *what role it plays in the final Fatat Al Arab RAG system*. It replaces
`MVP_GroundTruth_50Pages_Plan.md` as the single source of truth for corpus questions.

---

## 1. Two corpora, two purposes

NABAT-AI draws on two conceptually different bodies of text. They are merged by
the RAG layer but curated on separate tracks.

| Corpus | Role | Size | Transcription strategy |
|---|---|---|---|
| **Showcase poems (Phases 1–3)** | Demonstrate that Al‑Nassikh can handle the 8 visual challenges across a deliberate easy → hard curve. Each phase is a proof-point, not an index. | ~45 pages of full poetry bodies across 4 manuscripts per phase | Full HITL transcription in eScriptorium with multi-variant (manuscript / MSA / dialectal) per line |
| **Dictionary / Index (Phase 4)** | The lookup table that lets Fatat Al Arab answer *"who wrote this?"*, *"on what page?"*, *"list poems by poet X"* across the entire library. | **36 merged TOC pages, 1,687 anchors, 1,502 RAG-ready entries — this is the complete dictionary.** Every readable TOC in the corpus has been gathered; manuscripts or volumes not represented here either have no TOC or their TOC was judged unreadable at transcription time. | HITL + Kraken export, deterministically merged line-by-line (Stage 9) |

**Why the split matters for the user experience.**
A researcher asking "شو قال محمد بن عبدالرحمن في الخيل؟" hits both corpora in a single
hybrid-retrieval call: Phase 4 supplies the poem-level anchor (poet, volume, page,
matla), Phases 1–3 supply the full text of the verses when they exist. When the full
text doesn't exist yet, Fatat Al Arab still answers the metadata part and cites the
original page image from Phase 4.

---

## 2. Source material inventory

### 2.1 Sowayan digital library (primary long-tail dictionary)

Four PDFs from <http://www.saadsowayan.info/Publications/Ar_Book4/B4C01.html>.
All are **facsimile scans** of handwritten Nabati manuscripts — they look like digital
PDFs, but `pdftotext` produces garbled glyph codes because they use a private cmap.
In practice they are treated as images, same as the manuscript PDFs.

| File | TOC contribution | Notes |
|---|---|---|
| `601-782.pdf` | p.178–182 (5 pages) | ✅ merged into Phase 4 — the only volume in the set whose TOC is cleanly legible at transcription quality |
| `401-600.pdf` | none | TOC pages absent or unreadable |
| `201-400.pdf` | none | TOC pages absent or unreadable |
| `001-200.pdf` | none | TOC pages absent or unreadable |

The three excluded Sowayan volumes remain available as **body-text sources** for the
showcase phases or for a future phase, but they do **not** contribute to the dictionary
layer because no legible TOC could be extracted.

### 2.2 Local manuscript PDFs (21 files)

Original scans held in `manuscripts/manuscript{01–23}.pdf` (02, 11, 20 absent from the
numbering). Every manuscript has been examined for TOC pages — TOCs appear at the
front of some manuscripts (ms10 p.1–3) and at the back of others (ms01 p.70–72,
ms05 p.301–305, ms12 p.377–379).

**All readable TOCs have been gathered into Phase 4.** The table below lists every
manuscript whose TOC contributes to the dictionary.

| Manuscript | Total pages | TOC pages | Placement |
|---|---:|---|---|
| manuscript01 | 73 | 70, 71, 72 | back |
| manuscript03 | 88 | 87, 88 | back |
| manuscript04 | 59 | 59 | back |
| manuscript05 | 307 | 301–305 | back |
| manuscript06 | 255 | 250–255 | back |
| manuscript08 | 173 | 2 | front |
| manuscript10 | 304 | 1, 2, 3 | front |
| manuscript12 | 383 | 377, 378, 379 | back |
| manuscript18 | 363 | 24, 25, 26 | front-ish |
| manuscript21 | 119 | 2, 3 | front |
| manuscript22 | 97 | 2, 3 | front |

**Not in the dictionary:** every other manuscript — ms02, ms07, ms09, ms13, ms14,
ms15, ms16, ms17, ms19, ms23 — either has no TOC, or the TOC pages are unreadable
(ink-faded, torn, or illegible at the resolution of the scan). These manuscripts
remain eligible sources for Phase 1–3 showcase selection where the body text is of
suitable quality.

### 2.3 Phase 1–3 showcase curation

Phase curation is *strategic*, not random. Each phase picks manuscripts/pages that
surface specific items from the architecture document's 8-challenge catalogue.

| Phase | Theme | Manuscripts | Pages | Challenges demonstrated |
|---|---|---|---:|---|
| **1 — Momentum** | Clean metered poetry on good paper | ms07, ms14, ms15, ms22 | 20 | baseline line segmentation; alef variants; digit normalisation |
| **2 — Core Sadr/Ajuz layout** | Mid-complexity two-column verse with side notes | ms04, ms05, ms19, ms21 | 15 | sadr/ajuz split; marginal glosses; variable x-height |
| **3 — Edge cases** | Hard pages: bleed-through, overwrites, faded ink | ms01, ms03, ms06, ms08 | 10 | reverse-page ink bleed; interlinear corrections; faded diacritics |
| **4 — Metadata TOC** | Poet/volume/page index | 601-782 + 13 manuscripts (above) | 48 images / 36 merged | 4 distinct TOC layout types (see §3.2) |

---

## 3. Pipeline architecture

### 3.1 Al‑Nassikh stages (Worker 1)

```
Raw PDF / scan
     │
     ▼
[Stage 1] Page extraction  → manuscripts/MVP_Ground_Truth_Images/Phase_N/
     │
     ▼
[Stage 2] eScriptorium HITL transcription (PAGE-XML, PRIMA 2019-07-15)
     │    Two parallel exports for Phase 4:
     │      (a) manual  = expert transcription
     │      (b) kraken  = automatic OCR with same line IDs
     ▼
[Stage 3] src/al_nassikh/parser.py
     │    - Parses Phases 1, 2, 3 into stanza-aware JSON
     │    - Multi-variant transcription per verse (manuscript / MSA / dialectal)
     │    - Anchor-based CER verification (HIGH/MEDIUM/LOW confidence)
     ▼
[Stage 9] src/al_nassikh/phase4_merger.py (TOC-specific)
     │    - Pairs manual vs kraken lines by shared XML IDs
     │    - Deterministic scoring (no LLM): Arabic-char density,
     │      numeric-cell structure bonus, Kraken-noise penalties
     │    - Geometric inference of layout type (type_1…type_4)
     │    - Field decomposition: poet | page | matla | occasion | verse_count
     ▼
data/ground_truth/
     phase1_poems.json                       (Phase 1 showcase)
     phase2_poems.json                       (Phase 2 showcase)
     phase3_poems.json                       (Phase 3 showcase)
     phase4_toc_anchors_merged.json          (TOC anchors, per page)
     phase4_anchor_registry_merged.json      (TOC anchors, flattened)
     phase4_merge_audit.json                 (every line decision with reason)
     anchor_registry_phase4.json             (RAG-ready, normalised fields)
     manuscripts/Ground_Truth_Exports/
          export_doc9_phase_4_pagexml_merged (winning XML per page)
```

### 3.2 Phase 4 layout taxonomy

Phase 4 is not uniform — the TOC pages fall into four visually distinct layouts
that require different field-extraction rules:

| Layout | Shape | Example source |
|---|---|---|
| **type_1** | Two-column metadata (page \| matla \| poet on one row) | Sowayan 601-782, ms18, ms22 |
| **type_2** | Extended poetry metadata with occasion column | ms05, ms06, ms12 |
| **type_3** | Single-row metadata with free-form matla text | ms10, ms21 |
| **type_4** | Inverted two-column (RTL swap of type_1) | ms01, ms03, ms04 |

Layout is inferred geometrically from line x-coordinates because the eScriptorium
export only labels every region as `structure {type:default}`. Distribution across the
36 merged pages: 14 type_2, 11 type_1, 6 type_4, 5 type_3.

### 3.3 Line-quality merge scoring (deterministic, no LLM)

Each paired line is scored independently on both sources; the higher score wins and
is stamped with provenance in the merged XML. Signal weights:

- Arabic character count × 1.0 (primary content signal)
- Token count × 0.5 (fragmentation penalty)
- Confidence × min(1, 0.5) × 1.5 (capped — prevents Kraken over-confidence from winning)
- Structured numeric cells with `/` separator: +6.0 if `NNN / NNN` pattern
- Manual bonus for short metadata cells (≤4 tokens, any Arabic content): +2.5
- Kraken hallucination penalty for Khaleeji-unlikely n-grams like `[بتثنير]م[يل]`: −2.0
- Garbage-pattern penalties (repeated-char runs, isolated diacritics, long non-Arabic tokens): up to −5.0

**Outcome on 4,129 paired lines:** 3,663 manual wins / 466 Kraken wins.
Kraken wins are concentrated on manuscript10 and manuscript21 type_3 pages —
exactly the pages where the manual export was known to have alignment gaps.

### 3.4 RAG-ready registry fields

`anchor_registry_phase4.json` is the final artefact Fatat Al Arab imports:

| Field | Fill rate (1,502 entries) | Example |
|---|---:|---|
| `poet_name` | 100 % | *محمد بن عبدالرحمن* |
| `matla_text` | 96.9 % | *يا طير يا اللي مشيت ولعت نار الشكوى* |
| `page_number` | 60.2 % | 523 |
| `verse_count` | 42.5 % | 14 |
| `occasion` | 9.1 % | *في مدح الشيوخ* |
| `layout_type` | 100 % | `type_1` |
| `source_row_id` | 100 % | `manuscript12_p378_r014c0` |
| `source_image_path` | 100 % | `manuscripts/MVP_Ground_Truth_Images/Phase_4_Metadata_TOC/manuscript12_p378.png` |
| `bbox` | 100 % | `{x:120, y:2340, w:560, h:38}` |

Every RAG citation derived from a Phase 4 anchor therefore has a guaranteed
pointer back to: the source page image, the exact bounding box on that image, the
manual/kraken provenance per field, and a stable `source_row_id` for auditing.

---

## 4. Known scope boundaries

The Phase 4 dictionary is **closed** — every readable TOC in the corpus has been
gathered. Fatat Al Arab may therefore answer "not in corpus" for poems housed in
Sowayan vols 001-200, 201-400, 401-600, or in manuscripts without a TOC; this is
a data-availability limit, not a system limitation, and is exposed honestly to the
user in the citation trail.

Remaining engineering work (not corpus work):

1. **Phase 1–3 body text is not cross-linked to Phase 4 anchors.** A poem parsed
   in `phase1_poems.json` does not yet carry an `anchor_id` pointing into the
   Phase 4 registry. A join step keyed on (poet, matla) fuzzy match is required
   for the "show me the full poem" retrieval path.
2. **Poet biographies are external to the corpus.** Dialect explanation and poet
   context answers will require either a small curated `poets_bio.json` or a
   grounded web retrieval step; neither exists yet.
3. **Bilingual handling** is defined in the architecture (Bilingual Query
   Analyzer, EN→AR translation, image OCR for uploaded questions) but the demo
   app does not yet exercise it end-to-end.

---

## 5. How to re-run the pipeline

```bash
cd nabat-ai
python -m src.al_nassikh.parser        # regenerates phase1/2/3/4 JSON outputs
python -m src.fatat_al_arab.rag        # builds embeddings + demo queries
```

The parser is idempotent — it reads the PAGE-XML in `Ground_Truth_Exports/`, so
adding new TOC exports only requires dropping them into the appropriate folder
and re-running.
