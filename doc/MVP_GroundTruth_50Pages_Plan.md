# MVP Ground Truth — 50-Page Selection Plan
### Al-Nassikh HTR Pipeline | Khaleeji Nabati Poetry Project

> **Governing Rule:** ms09 and any heavily degraded manuscripts are excluded — they route directly to specialist review per the architecture document.
> **Total Target:** 65 annotated pages across 4 phases in **~8 days** (~2 hours/day).

---

## Visual Challenge Reference (from Architecture Document)

| ID | Challenge Name | Description |
|---|---|---|
| C1 | Column Gap Detection | Two-column Sadr/Ajuz layout with varying gap widths |
| C2 | Bleed-through / Show-through | Ink from reverse side visible through thin paper |
| C3 | Calligraphic vs. Informal | Wide variation in handwriting style across poets |
| C4 | Mixed Prose/Poetry | Section headers, attributions (مما قال...) mixed with verse |
| C5 | Margin Annotations | Corrections, numbers, or notes written in margins |
| C6 | Compressed Naskh | Tightly packed lines with minimal inter-line spacing |
| C7 | Stamps / Labels / Seals | Library stamps, seals, or ownership marks on pages |
| C8 | Structured Metadata / TOC | Tabular index pages (Poet, Matla, Page, Verse count) |

---

## Phase 1 — Momentum Building (20 pages)
**Goal:** Establish baseline CER, train on authentic handwritten styles, build annotator confidence.
**Manuscripts:** ms22, ms15, ms14, ms07
**Timeline:** Days 1–2

| Phase | Manuscript | Suggested Pages | Visual Challenge(s) | Rationale |
|---|---|---|---|---|
| 1 | **ms22** ⭐ | pp. 5, 12, 20, 30, 40, 55, 70 (7 pages) | C3 — Authentic handwritten Naskh, C4 — Section headers | **Start here — your preferred manuscript.** Genuine handwritten naskh, single column, readable and consistent. Contains poet attribution headers (وله أيضاً) giving early C4 exposure at low difficulty. 7 pages builds a strong style baseline from day 1. |
| 1 | **ms15** | pp. 10, 20, 35, 50 (4 pages) | C3 — Calligraphic style | Exceptionally clean with wide line spacing. Use as a complementary style to ms22 — together they give the model two distinct "easy" handwriting modes to learn from. |
| 1 | **ms14** | pp. 10, 25, 40, 55 (4 pages) | C3 — Transitional style | Near-printed clarity in parts. Wide inter-line spacing. Ideal for the second half of Phase 1 when you need slightly more complexity without pain. |
| 1 | **ms07** | pp. 8, 20, 35, 50, 65 (5 pages) | C3 — Informal style, C5 — Margin annotations (light) | Clean A4 scans with occasional margin marks (ع symbols). Introduces C5 gently. Good handwriting style bridge toward Phase 2. |

**Phase 1 Subtotal: 20 pages**

---

## Phase 2 — The Core Sadr-Ajuz Layout (15 pages)
**Goal:** Master the two-column layout which is the defining feature of Nabati manuscript structure.
**Manuscripts:** ms19, ms05, ms04, ms21
**Timeline:** Days 3–4

| Phase | Manuscript | Suggested Pages | Visual Challenge(s) | Rationale |
|---|---|---|---|---|
| 2 | **ms19** | pp. 5, 15, 25, 40 (4 pages) | C1 — Column gap detection, C4 — Mixed prose/poetry | 71-page manuscript, manageable scope. Two-column layout is visible and well-separated. Good first exposure to Sadr/Ajuz structure before tackling the larger manuscripts. |
| 2 | **ms21** | pp. 8, 20, 35, 50 (4 pages) | C1 — Column gap detection, C3 — Mixed handwriting styles | Clear two-column layout, moderate density. The column gap is consistent, making it easier for Kraken to segment correctly. |
| 2 | **ms05** | pp. 15, 40, 70 (3 pages) | C1 — Column gap (variable), C4 — Mixed prose/poetry, C5 — Margin marks | The largest manuscript (307pp). Select pages from early, mid, and late sections to capture layout variation across the volume. C4 is prominent here — many poet attributions and section dividers. |
| 2 | **ms04** | pp. 10, 20, 30, 45 (4 pages) | C1 — Narrow column gap, C6 — Compressed Naskh (moderate) | Already in eScriptorium. Two-column layout with tighter line spacing than ms19. Bridges Phase 2 into Phase 3 difficulty. Keep these 4 pages to not waste work already done. |

**Phase 2 Subtotal: 15 pages**

---

## Phase 3 — Edge Cases & Stress Tests (10 pages)
**Goal:** Cover all remaining visual challenges. These pages will be harder to annotate but critical for model robustness.
**Manuscripts:** ms01, ms08, ms03, ms06
**Timeline:** Days 5–6

| Phase | Manuscript | Suggested Pages | Visual Challenge(s) | Rationale |
|---|---|---|---|---|
| 3 | **ms01** | pp. 12, 28, 45 (3 pages) | C2 — Bleed-through, C5 — Margin annotations, C7 — Stamps/labels | Landscape format (two pages per scan). Contains the most complex mix of challenges: ink bleed-through from reverse side, margin corrections, and occasional library stamps. Select pages 12, 28, 45 — these are mid-manuscript where degradation is moderate, not at its worst. |
| 3 | **ms08** | pp. 15, 35 (2 pages) | C2 — Bleed-through, C6 — Compressed Naskh (heavy) | Landscape A4, dense naskh with visible show-through from the opposite page. Two pages is enough to represent this challenge without over-investing time. |
| 3 | **ms06** | pp. 30, 50 (2 pages) | C1 — Variable column gap, C4 — Complex mixed layout, C5 — Margin annotations | Large manuscript (255pp) with the most complex mixed layout. Annotations appear in both margins. Select just 2 pages to cover this combination without exhausting annotators. **Note:** p.20 was excluded — it is a blank duplicate of p.18 ("صفحة مكررة من صفحة ١٨"). p.30 substituted: full two-column poetry with multiple poems and clear Sadr/Ajuz structure. |
| 3 | **ms03** | pp. 10, 30, 50 (3 pages) | C6 — Compressed Naskh (extreme), C3 — Difficult informal style | Small page size with extremely dense, compressed naskh. This is your hardest handwriting style. Annotating 3 pages here teaches the model what "hard" looks like. Do this last in Phase 3. |

**Phase 3 Subtotal: 10 pages**

---

## Phase 4 — Complete TOC Metadata (all manuscripts with TOC)
**Goal:** Build the complete structured metadata layer for the RAG system — capturing **all available metadata fields** across every manuscript that has a TOC. This is not a sample; it is the complete anchor registry. Every annotated TOC row becomes a structured record the RAG agent can query by poet, occasion, verse count, or first line.
**Manuscripts:** ms01, ms02, ms03, ms04, ms05, ms06, ms07, ms08, ms10, ms12, ms18, ms21, ms22, ms23, 601-782
**Timeline:** Days 7–10

> **Corpus TOC audit (definitive):** All 25 PDFs inspected. The following 15 sources contain TOC/index pages verified by direct inspection. ms09 excluded per architecture spec. ms13 excluded (high degradation). ms14–ms17 excluded (German/scholarly editions, not Nabati anthologies). ms19 confirmed no TOC.

### TOC Metadata Fields — Capture Wherever Present

| Field (Arabic) | Field (English) | Notes |
|---|---|---|
| الشاعر | Poet name | Present in all TOC manuscripts |
| مطلع القصيدة / اول سطر من القصيدة | First line (Matla) — primary RAG anchor | Present in all TOC manuscripts |
| ص / ع | Page number / Verse count | Present in all TOC manuscripts |
| المناسبة / غناسبة | Occasion / dedicatee / genre context | ms06 (fully present), ms05 (partial), ms21 (present) |
| رقم القصيدة | Poem number / sequence within volume | ms12, ms21 (where present) |
| المنطقة / القبيلة | Region / tribe of poet | ms22 (unique field) |

### Phase 4 Page Selection

| Phase | Manuscript | Total Pages | TOC Location | Pages Selected | Metadata Fields | Notes |
|---|---|---|---|---|---|---|
| 4 | **ms01** | 73 pp | pp. 70–73 | pp. 70–72 (3 pages) | الشاعر, مطلع القصيدة, ص/ع | Clean tabular layout: Poet \| Matla \| Page \| Verse Count. 3 of 4 TOC pages. |
| 4 | **ms02** | 59 pp | pp. 1–5 | pp. 1–5 (5 pages) | الشاعر, مطلع القصيدة, ص | Two-column فهرس at front of manuscript. Dense, structured index with poet names and page numbers. |
| 4 | **ms03** | ~65 pp | last ~2 pages | last 2 pages (2 pages) | الشاعر, مطلع القصيدة, ص/ع | Compact TOC labelled "مخطوطة هوبير/٣". |
| 4 | **ms04** | ~90 pp | last page | last 1 page (1 page) | الشاعر, مطلع القصيدة, ص/ع | Single-page TOC "مخطوطة عبيد الرشيد". Already in eScriptorium. |
| 4 | **ms05** | 307 pp | pp. 301–307 | pp. 301–305 (5 pages) | الشاعر, مطلع القصيدة, ص/ع, **المناسبة** (partial) | Largest manuscript. 7 TOC pages total; annotate 5 dense core pages. المناسبة column present but not always filled — record wherever available. |
| 4 | **ms06** | 255 pp | pp. 250–255 | pp. 250–255 (6 pages) | الشاعر, مطلع القصيدة, ص/ع, **المناسبة** | Richest handwritten TOC — all 4 fields fully populated. Poets: محمد العوني, حمود العبيد, عبيد بن رشيد and others. Annotate all 6 pages. |
| 4 | **ms07** | 130 pp | pp. 1–3 | pp. 1–3 (3 pages) | الشاعر, مطلع القصيدة, ص | Title + structured index at front. "مخطوطة لشعراء الجبل وشعراء من نجد". |
| 4 | **ms08** | 173 pp | pp. 1–3 | pp. 1–3 (3 pages) | الشاعر, ص (page range) | Front-matter index showing poet names with page ranges (e.g. 89–94, 205–204). Landscape format. |
| 4 | **ms10** | 304 pp | pp. 1–3 | pp. 1–3 (3 pages) | الشاعر, مطلع القصيدة, ص | Dense two-column فهرس spanning 3 pages; covers poems up to ~p.305. |
| 4 | **ms12** | ~383 pp | pp. 377–383 | pp. 377–379 (3 pages) | الشاعر, مطلع القصيدة, ص/ع, **رقم القصيدة** | Largest index. رقم القصيدة (poem sequence number) is unique to ms12 — critical for RAG ordering. |
| 4 | **ms18** | 363 pp | pp. 24–26 | pp. 24–26 (3 pages) | Dichtername (الشاعر), Anfang (مطلع), Metrum, Reim, Verszahl | German scholarly edition (Albert Socin). Structured catalog with Fol. / Poet / Incipit / Metre / Rhyme / Verse-count columns. Transliterated Arabic incipits — valuable cross-reference metadata for RAG. |
| 4 | **ms21** | ~120 pp | pp. 2–3 | pp. 2–3 (2 pages) | الشاعر, مطلع القصيدة, ص/ع, **غناسبة**, **رقم** | Rich فهرس with 5 columns: رقم / الشاعر / عدد البيت / غناسبة / مطلع. ~60 poems across 2 pages. غناسبة (occasion/genre) fully present. |
| 4 | **ms22** | ~100 pp | pp. 2–3 | pp. 2–3 (2 pages) | الشاعر, عدد القصائد, **المنطقة/القبيلة**, ص | Unique structure: p.2 = poet list with poem counts per region; p.3 = الشاعر + page range + tribe/region (من أهل القصيم, من أهل الربق, etc.). Adds geographical metadata unique in the corpus. |
| 4 | **ms23** | ~110 pp | pp. 2–3 | pp. 2–3 (2 pages) | الشاعر, مطلع القصيدة, ص | Degraded two-column فهرس with show-through. Poet names + first lines + page numbers visible. |
| 4 | **601-782** | 182 pp | pp. 178–182 | pp. 178–182 (5 pages) | الشاعر, مطلع القصيدة, ص, **القصيدة رقم** | Exceptional four-column TOC (two poems per spread): اسم الشاعر / القصيدة / اول سطر من القصيدة. 5 full pages, cleanly typeset. Covers poems 482–790+. |

**Phase 4 Subtotal: 43 pages**

---

## Master Summary Table

| Phase | Focus | Manuscripts Used | Page Count | Challenges Covered | Timeline |
|---|---|---|---|---|---|
| 1 — Momentum | Authentic handwritten baseline | ms22 ⭐, ms15, ms14, ms07 | 20 | C3, C4 (light), C5 (light) | Days 1–2 |
| 2 — Sadr/Ajuz | Two-column layout mastery | ms19, ms21, ms05, ms04 | 15 | C1, C4, C5, C6 (moderate) | Days 3–4 |
| 3 — Edge Cases | Stress test & robustness | ms01, ms08, ms06, ms03 | 10 | C2, C5, C6, C7 | Days 5–6 |
| 4 — Metadata (complete) | Full TOC — all 15 sources, all fields | ms01, ms02, ms03, ms04, ms05, ms06, ms07, ms08, ms10, ms12, ms18, ms21, ms22, ms23, 601-782 | 43 | C8 — all fields: الشاعر, مطلع, ص/ع, المناسبة, رقم, منطقة | Days 7–10 |
| **TOTAL** | | **17 manuscripts / sources** | **88 pages** | **All 8 challenges ✅** | **10 days** |

---

## Challenge Coverage Verification

| Challenge | First Covered In | Pages Covering It |
|---|---|---|
| C1 — Column Gap Detection | Phase 2 | ms19×4, ms21×4, ms05×3, ms04×4 = **15 pages** |
| C2 — Bleed-through | Phase 3 | ms01×3, ms08×2 = **5 pages** |
| C3 — Handwriting Style Variation | Phase 1 | ms22×7, ms15×4, ms14×4, ms07×5 = **20 pages** |
| C4 — Mixed Prose/Poetry | Phase 1–2 | ms22×7, ms05×3, ms06×2 = **12 pages** |
| C5 — Margin Annotations | Phase 1–3 | ms07×5, ms05×3, ms01×3, ms06×2 = **13 pages** |
| C6 — Compressed Naskh | Phase 2–3 | ms04×4, ms08×2, ms03×3 = **9 pages** |
| C7 — Stamps / Labels / Seals | Phase 3 | ms01×3 = **3 pages** |
| C8 — Structured Metadata | Phase 4 | ms01×3, ms02×5, ms03×2, ms04×1, ms05×5, ms06×6, ms07×3, ms08×3, ms10×3, ms12×3, ms18×3, ms21×2, ms22×2, ms23×2, 601-782×5 = **43 pages** — fields: الشاعر, مطلع القصيدة, ص/ع, المناسبة, رقم القصيدة, المنطقة |

> ✅ All 8 challenges from the Al-Nassikh Architecture Document are covered.
> ✅ All 15 confirmed TOC sources included in Phase 4.
> ❌ ms09 excluded — routes to specialist review per architecture spec.
> ❌ ms13 excluded — high degradation.
> ❌ ms14, ms15, ms16, ms17 excluded — German/scholarly works, not Nabati poetry anthologies (ms18 included as it contains a structured incipit catalog directly relevant to the corpus).
> ❌ ms19 — confirmed no TOC by direct inspection.

---

## Annotator Time Estimate — 10-Day Sprint

| Phase | Pages | Est. Minutes/Page | Est. Total Time | Day(s) |
|---|---|---|---|---|
| Phase 1 | 20 | 8–12 min | ~3–4 hours | Days 1–2 |
| Phase 2 | 15 | 15–20 min | ~4–5 hours | Days 3–4 |
| Phase 3 | 10 | 25–35 min | ~5–6 hours | Days 5–6 |
| Phase 4 | 43 | 5–8 min (structured TOC) | ~3.5–6 hours | Days 7–10 |
| **Total** | **88** | | **~16–21 hours** | **10 days** |

> **Total: ~16–21 hours.** Phase 4 is fast relative to page count — TOC pages are structured/tabular and quick to annotate:
> - At 2h/day → 10 days
> - At 4h/day → 6 days
> - At 6h/day → 4 days (aggressive sprint — fully achievable)
>
> **eScriptorium shortcuts that save ~30% time:** `Tab` = next line · `Ctrl+S` = save · `I` = reverse line direction · `Ctrl+A` = select all lines
