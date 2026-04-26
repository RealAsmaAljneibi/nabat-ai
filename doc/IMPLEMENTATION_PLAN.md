# NABAT-AI — Implementation Plan v2 (Stage 3 · Three-Worker)

**Owner:** Asma Salem Mubarak Najem Aljneibi · **Course:** MAAI1704 · **Last updated:** 2026-04-22

This plan replaces the v1 plan (2026-04-20). It is the build order that turns
`Deliverable_2_Architecture_Updated_v4.docx` + `DATA_PIPELINE.md` into running code,
and it now commits explicitly to the four user personas declared in §1.2 of the
architecture doc. Every milestone starts with a short "why this exists" line, then
the file-level changes, then the acceptance gate.

---

## 0 · Guiding decisions (constraints on everything below)

These four commitments are non-negotiable. They shape every downstream choice and
should be cited in code comments so the reason survives the author.

1. **LLM inference runs on a hosted API, not a local GPU.** The doctor / judge must
   be able to run the system with just `pip install -r requirements.txt` and a
   `.env` file containing an API key. Provider is Together.ai or Groq (both host
   Qwen2.5-7B and Mistral-7B). No CUDA, no Ollama, no vLLM in the MVP. Every LLM
   call funnels through `src/fatat_al_arab/llm.py`.

2. **The user-facing app is Streamlit.** One file to run, bilingual input and image
   upload out of the box, and crucially — Streamlit supports multi-tab layouts, so
   the persona design (§3 below) lands in the UI naturally. All UI work lands in
   `app/streamlit_app.py`.

3. **Three workers, not two.** The v4 architecture splits the old single Fatat Al
   Arab into two agents: Worker 2 (Query & Understanding) owns §2.4 stages 1-3 and
   emits a `QueryContext`; Worker 3 (Retrieval & Synthesis) consumes that context
   and runs §2.5 stages 4-10. This is the architectural split, so the **code layout
   mirrors it** — `src/fatat_al_arab/agent1_query_understanding/` and
   `src/fatat_al_arab/agent2_retrieval_synthesis/` are siblings, and a reviewer can
   open §2.4/§2.5 in the doc and the matching folder side-by-side.

4. **Persona coverage is a first-class deliverable.** The v4 doc names four user
   personas (Cultural Institution, Researcher, Student, Enthusiast) and §1.2 of
   this plan's §3 maps each one to a concrete UI surface and a specific
   architectural feature (HyDE, CRAG, Multi-Variant, Ancestral Mirror). The demo
   must let a reviewer click through every persona's user story end-to-end.

5. **eScriptorium is self-hosted and wired in via its REST API — not opened in
   a separate browser tab.** eScriptorium is open-source (GPL, mirrored at
   [UB-Mannheim/escriptorium](https://github.com/UB-Mannheim/escriptorium) from
   [the upstream GitLab](https://gitlab.com/scripta/escriptorium)), exposes a full
   REST API at `{base_url}/api/`, and has a maintained Python client
   [`escriptorium-connector`](https://pypi.org/project/escriptorium-connector/).
   The Archive Manager tab (§3.1, M2b + M9) therefore drives eScriptorium through
   this API rather than asking the annotator to alt-tab to a separate UI — the
   transcription heavy-lifting still happens in eScriptorium's native editor
   (it's purpose-built and we won't reinvent it), but image upload, Kraken
   segmentation runs, zone-label setup, and PAGE-XML pull-back are **one-click
   operations inside our Streamlit app**. This keeps the demo a single surface
   for the doctor while reusing a battle-tested open-source tool for the parts
   we'd otherwise build badly.

---

## 1 · Target code structure (end state)

*Why this shape matters: each Python module is the implementation of a named
stage from the architecture doc. A reviewer should be able to open §2.3 Stage 7
(Ground Truth Transcription) and look at `al_nassikh/operator/transcribe.py`;
open §2.4 Stage 2 (HyDE) and look at `agent1_query_understanding/nodes/hyde.py`.
Do not flatten this into a single `agent.py` — the 1:1 mapping is the defensibility.*

Files marked ✨ are new; 🔧 are modified; no mark means already exists and stays.

```
handwritten-poems/
├── .env.example                                     ✨ LLM_PROVIDER, LLM_API_KEY, EMBED_MODEL
├── requirements.txt                                 🔧 streamlit, langgraph, qdrant-client,
│                                                        sentence-transformers, rank-bm25,
│                                                        together|groq, python-dotenv,
│                                                        rapidfuzz, pytesseract, langdetect,
│                                                        lxml, Pillow, plotly,
│                                                        escriptorium-connector
├── infra/                                           ✨ local deployment
│   └── escriptorium/
│       ├── docker-compose.yml                       ✨ self-host eScriptorium (from UB-Mannheim)
│       ├── .env.example                             ✨ ESCR_ADMIN_USER, ESCR_ADMIN_PASS, DB vars
│       └── README.md                                ✨ one-command bring-up instructions
├── CLAUDE.md                                        🔧 updated for Stage 3 + three workers
├── doc/
│   ├── DATA_PIPELINE.md
│   ├── IMPLEMENTATION_PLAN.md                       ← this file
│   ├── Deliverable_2_Architecture_Poetry.docx       (legacy v3 — kept for reference)
│   └── Deliverable_2_Architecture_Updated_v4.docx   (authoritative)
├── src/
│   ├── al_nassikh/
│   │   ├── parser.py                                existing — PAGE-XML → stanza JSON
│   │   ├── phase4_merger.py                         existing — manual/kraken line merge
│   │   ├── cross_link.py                            ✨ Phase 1-3 ↔ Phase 4 fuzzy join
│   │   ├── registry.py                              ✨ loader + lookup for manuscript_registry.json
│   │   │                                               (short_key → arabic_name/english_name)
│   │   ├── registry_join.py                         ✨ joins anchor_registry_phase4 to registry,
│   │   │                                               writes registry_join_audit.json
│   │   ├── corpus_stats.py                          ✨ M2c — deterministic aggregations
│   │   │                                               (counts, age range, regions, collectors)
│   │   │                                               consumed by the intent router fast-path
│   │   ├── nabati_taxonomy.py                       ✨ M2d — frozen 10-genre + 10-emotion label
│   │   │                                               space with validators (Sowayan-grounded)
│   │   ├── genre_heuristic.py                       ✨ M2d — silver-baseline keyword classifier,
│   │   │                                               corpus-tuned lexicons, first-class abstention
│   │   ├── escriptorium_client.py                   ✨ REST wrapper over escriptorium-connector
│   │   │                                               (upload image, trigger Kraken, pull PAGE-XML)
│   │   ├── ingest/                                  ✨ ingestion pipeline for new manuscripts
│   │   │   ├── pdf_to_pages.py                      ✨ PDF → page PNG extractor
│   │   │   ├── kraken_draft.py                      ✨ drives eScriptorium's Kraken v5 BLLA endpoint
│   │   │   └── queue_for_review.py                  ✨ pushes pages into operator queue + eScr project
│   │   └── operator/                                ✨ 8-challenge-aware HITL helpers
│   │       ├── triage.py                            ✨ Stage 0 — quality gate (is_degraded)
│   │       ├── bleed_suppress.py                    ✨ Stage 1 — background-estimate filter
│   │       ├── standardise.py                       ✨ Stage 2 — 300 DPI grayscale + deskew
│   │       ├── intrusion_mask.py                    ✨ Stage 3 — stamp/label detection
│   │       ├── style_profile.py                     ✨ Stage 4 — per-ms handwriting tag
│   │       ├── zone_classify.py                     ✨ Stage 5 — 5-category zone stubs
│   │       └── qa_jury.py                           ✨ Stage 8 — confidence review helper
│   └── fatat_al_arab/
│       ├── __init__.py
│       ├── llm.py                                   ✨ provider adapter (Together / Groq / stub)
│       ├── state.py                                 ✨ QueryContext + AgentState TypedDict
│       ├── translate.py                             ✨ EN↔AR via llm.py
│       ├── image_ocr.py                             ✨ pytesseract + llm.py fallback
│       ├── guardrails.py                            ✨ citation-resolvability + paraphrase
│       ├── embed.py                                 ✨ GATE-AraBERT dense + BM25 sparse
│       ├── index.py                                 ✨ Qdrant builder (embedded, file-backed)
│       ├── rrf.py                                   ✨ RRF k=60
│       ├── retrievers/
│       │   ├── bm25.py                              ✨
│       │   ├── dense.py                             ✨ GATE-AraBERT-v1
│       │   └── colbert.py                           ✨ late-interaction (stub first)
│       ├── agent1_query_understanding/              ✨ Worker 2 — §2.4
│       │   ├── __init__.py
│       │   ├── graph.py                             ✨ LangGraph wiring, emits QueryContext
│       │   ├── nodes/
│       │   │   ├── intent_router.py                 ✨ M2c Stage 0.5 — regex fast-path
│       │   │   │                                        (counting / age / provenance, EN+AR)
│       │   │   ├── bilingual_analyzer.py            ✨ Stage 1 — lang/intent/dialect
│       │   │   ├── hyde.py                          ✨ Stage 2 — hypothetical verse + embed
│       │   │   ├── self_query.py                    ✨ Stage 3 — typed filter extraction
│       │   │   └── bilingual_expand.py              ✨ Stage 2 — 3-5 AR + 3-5 EN variants
│       │   └── tools.py                             ✨ registry from §2.9 (translate_query,
│       │                                                extract_filters, hyde_passage,
│       │                                                expand_bilingual) — nothing else
│       ├── agent2_retrieval_synthesis/              ✨ Worker 3 — §2.5
│       │   ├── __init__.py
│       │   ├── graph.py                             ✨ LangGraph wiring, consumes QueryContext
│       │   ├── nodes/
│       │   │   ├── deterministic_answer.py          ✨ M2c Stage 0.5 — registry-lookup answer
│       │   │   │                                        (bilingual templates + file+field citation)
│       │   │   ├── retrieve.py                      ✨ Stage 4 — triple hybrid (parallel)
│       │   │   ├── rrf_fuse.py                      ✨ Stage 5 — RRF top-5
│       │   │   ├── resolve_heritage.py              ✨ Stage 6 — MSA→Khaleeji swap
│       │   │   ├── crag_grader.py                   ✨ Stage 7 — Correct/Ambig/Incorrect
│       │   │   ├── synthesise.py                    ✨ Stage 8 — grounded with mandatory cite
│       │   │   ├── reflect.py                       ✨ Stage 9 — Self-RAG 3-axis critic
│       │   │   └── format_variants.py               ✨ Stage 10 — al-Maktub+ortho+al-Mantuq
│       │   └── tools.py                             ✨ registry from §2.9 (retrieve_triple,
│       │                                                rrf_fuse, resolve_heritage,
│       │                                                crag_grade, synthesise, reflect,
│       │                                                format_multi_variant) — no query tools
│       ├── orchestrator.py                          ✨ wires Agent 1 → QueryContext → Agent 2
│       └── rag.py                                   🔧 thin CLI on top of orchestrator.py
├── app/
│   ├── legacy_server.py                             🔧 renamed from server.py
│   ├── streamlit_app.py                             ✨ persona-tabbed UI (§3 below)
│   ├── tabs/                                        ✨ one module per persona tab
│   │   ├── archive_manager.py                       ✨ Operator Console (Persona 1)
│   │   ├── scholar_workbench.py                     ✨ Researcher / Student / Enthusiast
│   │   └── shared_components.py                     ✨ citation block, folio viewer, debug
│   └── index.html                                   kept — GitHub-Pages landing page
├── data/
│   └── ground_truth/
│       ├── phase1_poems.json                        existing
│       ├── phase2_poems.json                        existing
│       ├── phase3_poems.json                        existing
│       ├── anchor_registry_phase4.json              existing — 1,502 RAG-ready entries
│       ├── anchor_registry_phase4_enriched.json     ✨ M2d — same 1,502 entries + genre,
│       │                                               emotions, genre_confidence,
│       │                                               genre_evidence, genre_source
│       ├── genre_enrichment_audit.json              ✨ M2d — coverage/distribution summary
│       ├── phase4_merge_audit.json                  existing — every line decision
│       ├── manuscript_registry.json                 🔧 M2c — 25-entry canonical name map +
│       │                                               provenance (circa_date_start/_end,
│       │                                               region_of_origin, collector, source_note)
│       │                                               (Huber, Ibn Yahya vols, Fahad Al-Mark,
│       │                                                Albert Socin, Al-Ghuwainem, etc. —
│       │                                                was "Sowssan", renamed to correct
│       │                                                transliteration "Socin" on 2026-04-23)
│       ├── manuscript_filename_map.json             ✨ filename (manuscript01.pdf) → short_key
│       ├── registry_join_audit.json                 ✨ which anchors matched, which didn't
│       ├── verse_anchor_crosslink.json              ✨ output of cross_link.py
│       ├── poets_bio.json                           ✨ hand-curated top-30 poet bios
│       ├── operator_queue.json                      ✨ new-manuscript ingest queue
│       └── qdrant_store/                            ✨ file-backed Qdrant collection
├── scripts/
│   ├── evaluate.py                                  ✨ M10 harness — all metrics in one run
│   ├── run_ingestion.py                             ✨ CLI for al_nassikh.ingest
│   ├── rebuild_index.py                             ✨ CLI for fatat_al_arab.index
│   └── enrich_genre_heuristic.py                    ✨ M2d — batch-classify anchor_registry_phase4
│                                                       into *_enriched.json + audit
├── tests/                                           ✨ new package
│   ├── test_llm_adapter.py
│   ├── test_crosslink.py
│   ├── test_self_query_schema.py
│   ├── test_rrf.py
│   ├── test_guardrails.py
│   ├── test_operator_triage.py
│   ├── test_intent_router.py                      ✨ M2c — 20 cases, 13 positive + 7 negative
│   ├── test_deterministic_answer.py               ✨ M2c — end-to-end for 7 intents × EN/AR
│   ├── test_agent1_end_to_end.py
│   ├── test_agent2_end_to_end.py
│   ├── test_orchestrator.py
│   └── fixtures/
│       ├── scholar_queries.jsonl                    ✨ 20 in-corpus queries + gold citations
│       ├── out_of_corpus_queries.jsonl              ✨ 20 OOC queries (refusal must fire)
│       ├── bilingual_queries.jsonl                  ✨ 10 EN + 10 AR paired
│       └── persona_scenarios.jsonl                  ✨ one scenario per persona user story
└── notebooks/
    └── nabat_ai_demo.ipynb                          🔧 persona-by-persona walkthrough
```

---

## 2 · Stage → Milestone map (the architectural contract)

*Why this table exists: when a reviewer opens the architecture doc and points at a
stage, they need to find the exact milestone and file that implements it. This is
the single-page audit map; every row must have a code artefact by M10.*

| Architecture Section | Stage / Component | Milestone | File(s) |
|---|---|---|---|
| §2.3 Stage 0 | Al-Nassikh Page Quality Triage | M2 | `al_nassikh/operator/triage.py` |
| §2.3 Stage 1 | Bleed-Through Management (HITL Phase) | M2 | `al_nassikh/operator/bleed_suppress.py` |
| §2.3 Stage 2 | Image Standardisation + diacritics sharpening | M2 | `al_nassikh/operator/standardise.py` |
| §2.3 Stage 3 | Manual Intrusion Masking (stamps, labels) | M2 | `al_nassikh/operator/intrusion_mask.py` |
| §2.3 Stage 4 | Manual Style Profiling (NASKH/RUQAH/CALLIGRAPHIC) | M2 | `al_nassikh/operator/style_profile.py` |
| §2.3 Stage 5 | Manual Zone Classification (5 categories) | M2 | `al_nassikh/operator/zone_classify.py` |
| §2.3 Stage 6 | Corrected Segmentation (Kraken v5 BLLA draft) | M2 | `al_nassikh/ingest/kraken_draft.py` |
| §1.6.3 Platform | eScriptorium self-host + REST client | M2b | `infra/escriptorium/`, `al_nassikh/escriptorium_client.py` |
| §2.3 Stage 7 | Ground Truth Transcription (existing) | existing | `al_nassikh/parser.py` |
| §2.3 Stage 8 | QA "The Jury" | M2 | `al_nassikh/operator/qa_jury.py` |
| §2.3 Stage 9 | Phase-4 Selective Source Merge | existing | `al_nassikh/phase4_merger.py` |
| Phase 1-3 ↔ Phase 4 cross-link | Deterministic join | M1 | `al_nassikh/cross_link.py` |
| Canonical manuscript names (25 entries) | Registry loader + anchor join | M1 | `al_nassikh/registry.py`, `registry_join.py` |
| Deterministic corpus aggregations | Registry-backed counts / dates / regions | M2c | `al_nassikh/corpus_stats.py` |
| §2.4 Stage 0.5 | Intent Router (counting/age/provenance fast-path, Agent 1) | M2c | `agent1/nodes/intent_router.py` |
| §2.5 Stage 0.5 | Deterministic Answer Path (registry lookup, Agent 2) | M2c | `agent2/nodes/deterministic_answer.py` |
| Nabati genre + emotion taxonomy | Frozen label space (10 genres + 10 emotions) | M2d | `al_nassikh/nabati_taxonomy.py` |
| Genre + emotion silver-baseline classifier | Corpus-grounded keyword heuristic with abstention | M2d | `al_nassikh/genre_heuristic.py`, `scripts/enrich_genre_heuristic.py` |
| ScoredChunk genre/emotion fields | genre, genre_confidence, genre_source, emotions on every chunk | M3 | `fatat_al_arab/rrf.py` |
| Qdrant payload + index genre fields | Auto-select enriched registry; genre in Qdrant payload | M3 | `fatat_al_arab/index.py` |
| Self-Query genre/emotion extraction | LLM prompt lists valid labels; routes to hard/soft filters | M3 | `agent1/nodes/self_query.py` |
| Hard-filter genre + emotions_any | retrieve.py _apply_hard_filters with §5 safety net | M3 | `agent2/nodes/retrieve.py` |
| §2.4 Stage 1 | Bilingual Query Analysis (Agent 1) | M4 | `agent1/nodes/bilingual_analyzer.py` |
| §2.4 Stage 2 | Bilingual Multi-Query + HyDE (Agent 1) | M4 | `agent1/nodes/hyde.py`, `bilingual_expand.py` |
| §2.4 Stage 3 | Self-Query Extraction (Agent 1) | M4 | `agent1/nodes/self_query.py` |
| §2.5 Stage 4 | Triple Hybrid Retrieval (Agent 2) | M5 | `agent2/nodes/retrieve.py` |
| §2.5 Stage 5 | RRF Fusion k=60 (Agent 2) | M5 | `agent2/nodes/rrf_fuse.py` |
| §2.5 Stage 6 | Multi-Representation Resolution (Agent 2) | M5 | `agent2/nodes/resolve_heritage.py` |
| §2.5 Stage 7 | CRAG Grading + bounded re-query (Agent 2) | M6 | `agent2/nodes/crag_grader.py` |
| §2.5 Stage 8 | Grounded Synthesis + mandatory citation | M6 | `agent2/nodes/synthesise.py` |
| §2.5 Stage 9 | Self-RAG 3-axis reflection + 2 retries | M6 | `agent2/nodes/reflect.py` |
| §2.5 Stage 10 | Multi-Variant Formatter | M6 | `agent2/nodes/format_variants.py` |
| §2.6 LangGraph state contract | QueryContext + AgentState typed state | M0 + M7 | `state.py`, `orchestrator.py` |
| §2.8 Failure handling budgets | Loop caps, fallback LLM edge | M7 | `orchestrator.py` + per-node |
| §2.9 Agent Capability Surface | Tool registries + 3 guardrails | M0 + M6 | `agent{1,2}/tools.py`, `guardrails.py` |
| §7 Evaluation axes | Correctness / Robustness / Efficiency / Human | M10 | `scripts/evaluate.py` |

---

## 2.1 · §3 Component Specification compliance audit

*Why this section exists: §3 of the architecture doc lists 23 components across
three tables (3.1 Al-Nassikh MVP, 3.2 Agent 1, 3.3 Agent 2), each with a Role /
Input / Output / Tools / Memory / Strategy spec. A reviewer pointing at any
row must find a code artefact that matches all six columns — or a documented
reason it is deferred. This table is that audit.*

*Status key:* ✅ implemented · 🟡 partial (specific gap named) · 📋 planned for a later milestone · ⏸ deferred past MVP.

### §3.1 Al-Nassikh Pipeline (10 components)

| Component | Status | Code artefact | Notes |
|---|---|---|---|
| Page Quality Triage | 📋 M2 | `al_nassikh/operator/triage.py` | Produces NORMAL/DEGRADED flag; input = raw page images |
| Bleed Suppressor | 📋 M2 | `al_nassikh/operator/bleed_suppress.py` | Annotator visual filter during eScriptorium transcription |
| Preprocessor | 📋 M2 | `al_nassikh/operator/standardise.py` | OpenCV deskew + binarise; context-aware for Challenge 8 |
| Intrusion Detector | 📋 M2 | `al_nassikh/operator/intrusion_mask.py` | eScriptorium poly-masking, stamps/labels coords |
| Style Profiler | 📋 M2 | `al_nassikh/operator/style_profile.py` | Manual NASKH/RUQAH/CALLIGRAPHIC tag per manuscript |
| Zone Classifier | 📋 M2 | `al_nassikh/operator/zone_classify.py` | 5-category eScriptorium ontology |
| Line Segmenter | 📋 M2 + M2b | `al_nassikh/ingest/kraken_draft.py` + eScriptorium | Kraken v5 BLLA draft + manual correction |
| HTR Ensemble | ⏸ post-MVP | — | MVP = 100 % expert transcription (Phase 1–3 ground truth), no HTR ensemble yet |
| Confidence Router | ⏸ post-MVP | — | MVP is 100 % HITL; auto/jury split is a future deliverable |
| HITL Coordinator | 📋 M9 | Tab A Archive Manager Console | eScriptorium iframe from M2b; Streamlit wrapper from M9 |
| Gold Assembler | ✅ existing | `al_nassikh/parser.py` + `phase4_merger.py` | Verse structures + anchor links already built |

### §3.2 Fatat Al Arab — Agent 1 (4 components + the new router)

| Component | Role | Input | Output | Tools | Memory | Strategy | Status | Code artefact |
|---|---|---|---|---|---|---|---|---|
| Intent Router (Stage 0.5, added M2c) | Deterministic fast-path for counting/age/provenance | User query | `answer_source`, `deterministic_intent` | Pure regex (EN+AR, Khaleeji) | None | Confidence ≥ 0.7 → bypass LLM path | ✅ M2c | `agent1/nodes/intent_router.py` |
| Bilingual Query Analyzer | Lang + intent + dialect | User query | Lang, intent, AR translation, params | LangGraph + langdetect + Qwen2.5 | Conv. context | Detect → translate EN→AR → classify | 🟡 M4 | `agent1/nodes/bilingual_analyzer.py` — **Gap: conversational context is declared in `AgentState.conversation_id`/`turn_index` but not fed into the analyzer prompt yet. Tracked as M4b below.** |
| Bilingual Multi-Query + HyDE | Multi-variant expansion | Query + AR + intent | 3–5 AR+EN variants + HyDE embedding | Qwen2.5 + AraBERT | None | HyDE on Arabic; RRF merge | ✅ M4 | `agent1/nodes/bilingual_expand.py` + `hyde.py` |
| Self-Query Extractor | Metadata filter extraction | Query | Filters (poet, ms, theme) | LLM function calling | None | Hard Qdrant pre-filters; confidence 0.7 | ✅ M4 | `agent1/nodes/self_query.py` |
| QueryContext Builder | State handoff package | All Agent 1 outputs | QueryContext object | LangGraph state | None | Serialize and pass via shared state | ✅ implicit | `fatat_al_arab/state.py` `make_query_context()` — every Agent 1 node writes directly into the TypedDict, so the "Builder" is the sum of their writes rather than a dedicated node. Documented as a design choice; no dedicated file. |

### §3.3 Fatat Al Arab — Agent 2 (9 components + the new deterministic path)

| Component | Role | Input | Output | Tools | Memory | Strategy | Status | Code artefact |
|---|---|---|---|---|---|---|---|---|
| Deterministic Answer Path (Stage 0.5, added M2c) | Registry-lookup answer for counting/age/provenance | QueryContext with `answer_source=registry_lookup` | Bilingual answer + file+field citation | `corpus_stats` module + templates | None | Bypass retrieval entirely | ✅ M2c | `agent2/nodes/deterministic_answer.py` |
| Sparse Retriever | Keyword search | Params + filters | BM25 top-20 | Qdrant BM25 | None | Khaleeji surface overlap | 📋 M5 | `retrievers/bm25.py` + `agent2/nodes/retrieve.py` |
| Dense Retriever | Semantic search | HyDE embedding + filters | Vector top-20 | GATE-AraBERT + Qdrant HNSW | **Embed cache** | Cosine over MSA summaries | 🟡 M5 | `retrievers/dense.py` — **Gap: the §3.3 "Embed cache" column is not yet wired. Tracked as M5b below.** |
| ColBERT Retriever | Token-level search | Query tokens + filters | Token-scored top-20 | ColBERT index | None | Late interaction | 📋 M5 | `retrievers/colbert.py` (stub first) |
| RRF Fusion | Score merge | 3 ranked lists | Top-5 | RRF k=60 | None | sum(1/(k+rank)) | 📋 M5 | `agent2/nodes/rrf_fuse.py` |
| Multi-Rep Resolver | Heritage text swap | Top-5 + parent_chunk_id | Original Khaleeji passages | Qdrant payload lookup | None | Swap MSA summary → heritage parent | 📋 M5 | `agent2/nodes/resolve_heritage.py` |
| CRAG Grader | Relevance check | Top-5 + query | Graded passages | LLM classifier | None | All-Incorrect → re-retrieve (≤1×) | 📋 M6 | `agent2/nodes/crag_grader.py` |
| Synthesiser | Grounded generation | CRAG-approved + history | Response + citations | Qwen2.5-7B (primary) / Mistral-7B (fallback) | **Last 5 turns** | Mandatory citations; reject if unsupported | 🟡 M6 | `agent2/nodes/synthesise.py` — **Gap: "last 5 turns" history channel is declared in state but not piped into the synthesis prompt. Tracked as M6b below.** |
| Self-RAG Reflector | Hallucination check | Response + sources | Validated or re-trigger | LLM self-eval | None | Faithfulness + relevance + completeness; ≤2 retries | 📋 M6 | `agent2/nodes/reflect.py` |
| Multi-Variant Formatter | Response formatting | Validated response + anchors | al-Maktub + orthographic + al-Mantuq + citations | Deterministic formatter | None | Folio image links | 📋 M6 | `agent2/nodes/format_variants.py` |

### Follow-up milestones to close the three 🟡 gaps

The audit surfaces three small gaps where the code's *shape* is present but the
architecture's *Memory* column is not wired end-to-end. Each is a one-file
fix that should land before M10 (evaluation harness) so the §3 contract is
fully covered.

- **M4b · Conversational context in Bilingual Query Analyzer (¼ day)** —
  The analyzer currently sees only the current raw query. §3.2 says its
  Memory column is "Conv. context". Plumb `state["conversation_id"]` and
  the last 2 user turns into the analyzer prompt so follow-up queries like
  "and how old are those?" resolve against the prior turn. File:
  `agent1/nodes/bilingual_analyzer.py` + a thin `session_store.py` that
  keeps an in-memory ring buffer per `conversation_id`. Acceptance:
  follow-up pronoun queries resolve correctly in the 10-turn fixture.

- **M5b · Dense retriever embed cache (¼ day)** —
  §3.3 says the Dense Retriever's Memory column is "Embed cache". Add a
  module-level `functools.lru_cache(maxsize=512)` around the query-side
  GATE-AraBERT encode call in `retrievers/dense.py`. Why: the same HyDE
  embedding is re-used on every CRAG re-query retry; caching avoids the
  ~180 ms re-encode. Acceptance: `pytest tests/test_dense_cache.py` shows
  a 2nd identical call returns < 5 ms.

- **M6b · Last-5-turns history in Synthesiser (¼ day)** —
  §3.3 says the Synthesiser's Memory column is "Last 5 turns". Extend the
  synthesis prompt to include the last 5 `(user, assistant)` turns from the
  `session_store` introduced in M4b, capped by a token budget (≈ 1200
  tokens) so long prior answers don't push passages out of the context
  window. Acceptance: a multi-turn chat (Q1 factual, Q2 follow-up) cites
  the same passages Q1 used without the user restating the poet name.

Closing these three gaps fully satisfies every Memory column in §3 and turns
the entire §3 table into a row-by-row ✅ before M10.

---

## 3 · Persona-to-feature map (the UX contract)

*Why this section exists: you told me the four personas have different second-order
desires — institutional immortality, computational philology, cultural fluency,
identity validation. A Streamlit tab for each means every persona has a visible
landing place that demonstrates their user story. Some personas will not go through
Worker 1 at all; they consume the already-transcribed corpus. This map encodes who
touches what.*

### 3.1 Tab A — Archive Manager Console (Persona 1: Cultural Institutions)

*Audience:* Poetry Academy / ADHA leadership who need to digitise new manuscripts.
*Second-order desire:* "Institutional Immortality" — proving cultural continuity
with zero-hallucination provenance.
*Uses Workers:* **Worker 1 (Al-Nassikh) only.** No RAG, no LLM generation.
*UI surface:* a single Streamlit tab that **orchestrates a locally-running
eScriptorium instance through its REST API** so the annotator gets a single
integrated surface instead of alt-tabbing between tools. Our tab adds our own
pre-processing helpers on top (triage, bleed-suppress, etc.), delegates the
actual manuscript editor to eScriptorium (which is purpose-built for this), and
pulls results back automatically.

- **Triage** — upload PDF or page images into our Streamlit; `triage.py`
  assesses each for DEGRADED status and flags pages that should go to a
  specialist track (Stage 0, ms09 pattern). NORMAL pages are forwarded to
  eScriptorium via `escriptorium_client.upload_pages(document_id, images)`.
- **Scalpel launcher (Zone & Baseline Correction)** — one button triggers
  `escriptorium_client.run_kraken_segmentation(document_id)` which calls
  eScriptorium's `POST /api/documents/{id}/segment/` with the BLLA model. When
  it completes, the tab embeds eScriptorium's own editor view in an `<iframe>`
  pre-seeded with our 5 zone labels (TWO_COLUMN_POETRY, PROSE_ATTRIBUTION,
  ORNAMENTAL_DIVIDER, MARGIN_PERPENDICULAR, INTERLINEAR_ANNOTATION) via the
  ontology API. The annotator does the manual baseline / zone work *inside
  eScriptorium's interface* — which is what it was built for.
- **Provenance & Merge** — when the annotator marks the document as complete,
  the tab calls `escriptorium_client.export_pagexml(document_id)` to pull the
  PAGE-XML back, then runs `phase4_merger.py` and `cross_link.py`, showing
  which lines won (manual vs Kraken) and flagging anchors with confidence
  < 0.85 for re-review (Stages 8, 9).

*Persona 1 acceptance:* an archivist can take a fresh PDF, upload it through
Streamlit, triage pages, click "Open in eScriptorium" to do the manual zone +
transcription work, click "Pull & Merge" to bring the PAGE-XML back, and see
the new rows appear in `anchor_registry_phase4.json` without ever opening a
separate browser window outside the Streamlit app.

### 3.2 Tab B — Scholar Workbench (Personas 2, 3, 4: query-side users)

*Audience:* Researchers, Students, Enthusiasts. None of them transcribe; they
consume the corpus Worker 1 produced.
*Uses Workers:* **Workers 2 + 3 (both Fatat Al Arab agents).**
*UI surface:* one chat input, three response-mode toggles that reveal
persona-specific affordances:

| Mode (radio) | Persona | What the mode does in the UI | Architecture feature it showcases |
|---|---|---|---|
| **Philology view** | Researcher (P2) | Shows HyDE hypothetical verse, the 3 retriever score ladders side-by-side, CRAG grades, ColBERT token-heatmap on the winning passage | §2.4 Stage 2 (HyDE), §2.5 Stage 4 (triple hybrid), Stage 7 (CRAG) |
| **Three-Layer Reader** | Student (P3) | Response rendered as 3 stacked cards: al-Maktub (original) / orthographic MSA / al-Mantuq (phonetic). English translation toggle. | §2.5 Stage 10 Multi-Variant Formatter + Agent 1 EN→AR translation |
| **Ancestral Mirror** | Enthusiast (P4) | Search by manuscript ID or family poet name; response returns a single passage with a large folio thumbnail, bounding-box overlay, and a "not in corpus" refusal when CRAG grades all-Incorrect | §2.5 Stage 6 (Multi-Representation Resolution) + §2.9 guardrail (c) refusal |

A fourth **Debug** toggle (for the doctor / judge) exposes the full LangGraph
state after each stage — the same thing the notebook walkthrough shows.

*Persona 2 acceptance:* a researcher can ask "metaphors of the night" in English,
see HyDE generate an Arabic verse, trigger bilingual retrieval, and read the
ColBERT top-token evidence on the winning passage.
*Persona 3 acceptance:* a student can search for poet المهادي in English, get
back three-layer output with phonetic guide, and flip to "Computational Philology"
mode if curious.
*Persona 4 acceptance:* an enthusiast can search a specific manuscript by its
**canonical name** (e.g. "مخطوطة ابن يحي 401-600" or "Huber Manuscript 2" —
not `manuscript03`), get back a single cited response with the manuscript's
Arabic+English name displayed prominently, a clickable folio-image + bounding
box, and an honest "not in corpus" refusal for verses outside the Phase-4
dictionary.

### 3.3 Cross-persona guardrails (always on, never mentioned to the user)

- Every factual sentence carries a citation resolvable to `anchor_registry_phase4`
  (§2.9 guardrail a).
- No verse text appears in output that is not verbatim in the approved passage set
  (§2.9 guardrail b).
- When CRAG returns empty, the system returns the scoped "not found in corpus"
  template — never speculates (§2.9 guardrail c). This is what powers the
  Enthusiast's "Ancestral Mirror" protection against hallucinated family history.

---

## 4 · Implementation order (hand this to Claude Code)

*Why the order: each milestone produces an artefact that the next one consumes.
Nothing is "done" until its acceptance check passes. Milestones with no
dependency on each other can be parallelised, but the default is sequential.*

### M0 · Foundation (½ day)

*Why first: without the LLM adapter and the typed `QueryContext` state, every later
milestone would hardcode a provider or invent its own state shape, and portability
would be lost immediately.*

1. `.env.example` with `LLM_PROVIDER=together`, `LLM_API_KEY=`,
   `EMBED_MODEL=aubmindlab/bert-base-arabertv02`, `QDRANT_PATH=data/ground_truth/qdrant_store`.
2. `requirements.txt` — add the packages listed in §1.
3. `src/fatat_al_arab/llm.py` — single public function
   `chat(prompt, system=None, json_schema=None, max_tokens=512, model="qwen2.5-7b")`.
   Implementations: `together`, `groq`, and a deterministic `stub` (returns canned
   JSON for offline tests). Exponential-backoff retry built in (the CRAG+Self-RAG
   loops fire up to 8 LLM calls per query — without backoff, rate limits will kill
   the demo).
4. `src/fatat_al_arab/state.py` — `QueryContext` (Agent 1 output) and `AgentState`
   (full graph state) TypedDicts mirroring §2.5 exactly. Every field from §2.4 Step 6
   and §2.5 "Agent 2 state" is a key here.
5. `src/fatat_al_arab/guardrails.py` scaffold — three functions stubbed out matching
   §2.9 (a), (b), (c); full implementation lands in M6.

**Acceptance:** `pytest tests/test_llm_adapter.py` passes against the `stub`
provider with no network; `python -c "from fatat_al_arab.state import QueryContext; QueryContext(query_lang='ar', ...)"` round-trips without type errors.

### M1 · Cross-link + poet bios + manuscript registry wiring (¾ day)

*Why second: every retrieval call joins the Phase-1/2/3 verse to its Phase-4 anchor,
its poet bio, AND its canonical manuscript identity (Huber, Ibn Yahya vol 3, etc.).
Building these three artefacts before the retriever means we never have to rewrite
the retriever to cope with schema drift. The manuscript registry is especially
load-bearing: a Cultural-Institution reviewer will see "Ibn Yahya Manuscript
(401-600)" — not `manuscript03` — so the join has to happen at index time.*

1. `src/al_nassikh/cross_link.py` — loads `phase{1,2,3}_poems.json` and
   `anchor_registry_phase4.json`; uses `rapidfuzz` on normalised `(poet, matla)`
   keys; writes `data/ground_truth/verse_anchor_crosslink.json` with rows
   `{verse_id, anchor_id, confidence, decision}`. Decisions: `linked` (≥0.85),
   `review` (0.70-0.85), `unlinked` (<0.70).
2. `data/ground_truth/poets_bio.json` — hand-curated short bios (≤120 words) for
   the 30 most-cited poets in the registry. Schema:
   `{poet_name, normalised_name, bio_ar, bio_en, sources:[…]}`. Sources are
   citations to Sowayan volumes only — **no web retrieval**, this keeps provenance
   clean for the Cultural Institution persona.
3. `src/al_nassikh/registry.py` — loads `manuscript_registry.json` once per
   process; exposes `by_short_key(key)`, `by_number(n)`, `by_filename(path)`,
   and a `list_all()` helper the Streamlit sidebar uses to show the corpus
   inventory. Idempotent — safe to import from any module.
4. `data/ground_truth/manuscript_filename_map.json` — hand-curated one-shot
   mapping from the historical filename stems (`manuscript01`…`manuscript23`
   plus the four Sowayan PDFs) to the 25 `short_key` values in the registry.
   **Why this file is separate from the registry:** the registry is the
   canonical list Asma maintains; the filename map is a lossy join that should
   evolve as files are re-organised. Keeping them apart means the registry
   never has to know about filesystem paths.
5. `src/al_nassikh/registry_join.py` — one-shot script that:
   (a) reads `anchor_registry_phase4.json`,
   (b) for every anchor, derives `manuscript_short_key` from the
       `source_image_path` via `manuscript_filename_map.json`,
   (c) writes back an enriched anchor registry with three new fields —
       `manuscript_short_key`, `manuscript_arabic_name`, `manuscript_english_name`,
   (d) emits `registry_join_audit.json` listing any anchor whose filename had
       no mapping (these are fixed by editing `manuscript_filename_map.json`,
       not by silently dropping rows).

**Acceptance:** `pytest tests/test_crosslink.py` confirms ≥ 70% of Phase-1 verses
crosslink at confidence ≥ 0.85; `poets_bio.json` validates against its schema and
covers ≥ 25 unique poets; `pytest tests/test_registry_join.py` confirms
**100%** of rows in `anchor_registry_phase4.json` resolve to a registry entry
(zero unmatched anchors — if any appear in `registry_join_audit.json`, M1 does
not pass until `manuscript_filename_map.json` is fixed).

### M2 · Al-Nassikh Operator Helpers (1 day)

*Why here: Persona 1 (Cultural Institutions) needs the 8-challenge-aware helpers
as callable modules before the Operator Console UI (M9) can wire them up. These
are **HITL helpers** — tools the human annotator uses, not an automated pipeline.
They handle image pre-processing and expose structured stubs the Streamlit tab
renders as buttons.*

Every helper corresponds to one architecture stage (§2.3). Each is a small pure
function taking an image or PAGE-XML snippet and returning a decision + a
suggested action — the annotator stays in the loop.

1. `operator/triage.py` — Stage 0. `is_degraded(image) -> {status, reason, score}`;
   threshold on pixel-intensity histogram + edge coherence. DEGRADED pages return
   a specialist-track recommendation.
2. `operator/bleed_suppress.py` — Stage 1. Morphological background estimate +
   Gaussian DoG filter, applied only when bleed-through is detected by a simple
   std-dev check.
3. `operator/standardise.py` — Stage 2. Resize to 300 DPI grayscale, deskew via
   Hough-line dominant angle, small-kernel unsharp for diacritics (Challenge 8).
4. `operator/intrusion_mask.py` — Stage 3. Circular Hough transform for stamps,
   rectangular-label detection for commercial labels (Challenge 7). Returns mask
   polygons that the annotator confirms or adjusts.
5. `operator/style_profile.py` — Stage 4. Tags a manuscript as
   `NASKH_COMPRESSED | RUQAH_OPEN | CALLIGRAPHIC | HURR` based on inter-line
   spacing + baseline variance + x-height distribution.
6. `operator/zone_classify.py` — Stage 5. Geometric heuristics that suggest zone
   labels for the 5 categories (`TWO_COLUMN_POETRY`, `PROSE_ATTRIBUTION`,
   `ORNAMENTAL_DIVIDER`, `MARGIN_PERPENDICULAR`, `INTERLINEAR_ANNOTATION`).
7. `operator/qa_jury.py` — Stage 8. Takes the final PAGE-XML and surfaces any line
   with annotator-assigned confidence < 1.0 for re-review.
8. `src/al_nassikh/ingest/pdf_to_pages.py` — takes a PDF, emits one PNG per page
   at 300 DPI into `manuscripts/MVP_Ground_Truth_Images/Phase_{N}/`.
9. `src/al_nassikh/ingest/kraken_draft.py` — thin wrapper that calls
   `escriptorium_client.run_kraken_segmentation()` (M2b). Kraken runs *inside*
   eScriptorium's Celery worker, not as a separate binary; this removes the
   Kraken CLI as a local dependency and keeps model versions in one place.
10. `src/al_nassikh/ingest/queue_for_review.py` — writes `operator_queue.json` so
    the Operator Console knows which pages are pending.

**Acceptance:** `pytest tests/test_operator_triage.py` confirms `triage.py` flags
ms09 pages as DEGRADED and ms14 pages as NORMAL on the known test images; the
ingestion CLI `python scripts/run_ingestion.py --pdf manuscripts/manuscript07.pdf`
produces page PNGs, a Kraken draft XML, and a queue entry.

### M2b · eScriptorium integration (1 day)

*Why its own milestone: eScriptorium is a real Django app with Postgres +
Redis + Celery workers, not a library. It has to be brought up, configured,
and its REST API exercised before the Operator Console (M9) can drive it. If
this is buried inside M2 or M9 it either rushes the Docker setup or delays the
tab that showcases Persona 1 — neither is acceptable. Pulling it out also
means Claude Code has an explicit gate to prove eScriptorium is reachable
before wiring the UI.*

1. `infra/escriptorium/docker-compose.yml` — vendored from the upstream
   [`UB-Mannheim/escriptorium`](https://github.com/UB-Mannheim/escriptorium)
   deployment (GPL-licensed; we redistribute the compose file, not the code).
   Services: `app` (Django), `db` (Postgres 13), `redis`, `nginx`, one Celery
   worker for Kraken jobs. Volume mounts point at `data/escriptorium/` so the
   uploaded images and trained baselines survive container restarts.
2. `infra/escriptorium/README.md` — one-command bring-up:
   ```
   cd infra/escriptorium && docker compose up -d && docker compose exec app \
     python manage.py createsuperuser
   ```
   Plus: how to pre-seed the 5 zone labels via the `escriptorium_client` module.
3. `src/al_nassikh/escriptorium_client.py` — thin wrapper around the
   [`escriptorium-connector`](https://pypi.org/project/escriptorium-connector/)
   PyPI package. Exposes exactly the verbs our Operator Console needs:
   - `ensure_project(name)` — idempotent create/find of a project.
   - `create_document(project_id, name, metadata)` — one document per manuscript.
   - `upload_pages(document_id, images)` — bulk upload; returns part IDs.
   - `run_kraken_segmentation(document_id, model="blla")` — triggers Stage 6
     (Corrected Segmentation). Blocks until Celery reports completion or
     times out at 300 s.
   - `set_ontology(zone_types=[TWO_COLUMN_POETRY, …])` — pre-seeds the 5 zone
     labels from §2.3 Stage 5 so the annotator doesn't have to recreate them
     per document.
   - `export_pagexml(document_id, format="pagexml_alto")` — pulls PAGE-XML
     back into `manuscripts/Ground_Truth_Exports/`.
   - `embed_editor_iframe_url(document_id)` — returns the `{base}/document/{id}/`
     URL that Streamlit iframes into Tab A for the manual work.
4. Environment wiring: `.env.example` gains `ESCR_BASE_URL=http://localhost:8080`,
   `ESCR_API_TOKEN=` (obtained via `docker compose exec app python manage.py
   drf_create_token admin`). `llm.py`-style fallback: if the API is unreachable,
   the Operator Console shows a clear "eScriptorium is not running — start it
   with `cd infra/escriptorium && docker compose up`" message rather than
   failing silently.
5. `tests/test_escriptorium_client.py` — records VCR cassettes against a live
   local instance on first run; subsequent test runs replay without network.
   This keeps CI offline-safe while still exercising the real API shape.

**Acceptance:** `docker compose up -d` in `infra/escriptorium/` produces a
reachable app at `http://localhost:8080`; `python -c "from al_nassikh.escriptorium_client
import ensure_project; print(ensure_project('nabat-ai-mvp'))"` returns a
project ID; uploading one known page (`manuscripts/manuscript07_p1.png`) and
running Kraken returns ≥ 1 text region; `export_pagexml` round-trips valid XML.

**Portability note for the doctor:** eScriptorium is optional for the retrieval
demo (Tab B). The whole RAG side (Tabs B + the notebook walkthrough) runs
without it. Tab A's eScriptorium features degrade to "eScriptorium not
running — download PDF output only" messages when the container is down, so
the doctor can still see and grade the retrieval system without standing up
Docker.

### M2c · Intent Router + Deterministic Answer Path (½ day) — added 2026-04-23

*Why this milestone exists: On 2026-04-23 Asma ran the app and asked "how
many poems do I have?" — a question the RAG pipeline cannot answer because
no chunk in the Qdrant index contains a count. The query went through HyDE,
retrieval, and CRAG; CRAG correctly marked the passages "Incorrect" (they
were unrelated verses), and the orchestrator fired the INSUFFICIENT_PASSAGES
refusal template. The user experience: a poetry question succeeded, a
simple inventory question refused. M2c is the fix — a deterministic
fast-path that intercepts counting / age / provenance questions before they
touch retrieval and answers them directly from the ground-truth JSON.*

*Why it lives between M2b and M3: the router depends on the manuscript
registry and the anchor registry (both produced by M1/M2), but it must be
wired into Agent 1's graph BEFORE the Qdrant index (M3) is built, because
the whole point is that the router never calls the retriever. Putting it
here also keeps the stage→milestone map honest: Stage 0.5 is not part of
the retrieval pipeline, so its milestone comes before the index milestone.*

1. **Registry enrichment (data side, already done)** —
   `data/ground_truth/manuscript_registry.json` gained four provenance
   fields on every entry: `circa_date_start`, `circa_date_end`,
   `region_of_origin`, `collector`, plus a `source_note` paragraph.
   Provenance was pulled from http://www.saadsowayan.info/index.html and
   cross-checked against public records on the named collectors (Charles
   Huber 1847-1884, Albert Socin's *Diwan aus Centralarabien* 1900-1901,
   Saad Sowayan, Abd al-Karim al-Juhayman, Khalid Ibn Yahya). The
   `albert_sowssan_{1,2,3}` short_keys were renamed to `albert_socin_{1,2,3}`
   (the original transliteration was phonetic, the correct Latin form is
   "Socin"); this rename cascaded into `anchor_registry_phase4.json` (63
   occurrences) and `manuscript_filename_map.json` (3 occurrences).
   **Why this matters for M2c:** without dated provenance the router could
   detect an "age" question but have nothing to return; now `manuscript_age_range()`
   yields `1800-1970 CE / 170 year span / 25-of-25 manuscripts dated`.

2. `src/al_nassikh/corpus_stats.py` — **NEW**. Deterministic aggregations
   module. Loads the two registries at import time into module globals so
   there is zero per-query I/O cost. Public API: `count_poems()`,
   `count_poets()`, `count_manuscripts()`, `count_manuscripts_with_toc()`,
   `count_pages()`, `manuscript_age_range()`, `regions_of_origin()`,
   `collectors()`, `corpus_summary()`, `poet_counts_top_n(n)`,
   `manuscripts_overview()`. Has a `__main__` block so Asma can run
   `python -m al_nassikh.corpus_stats` for a CLI sanity check without
   launching Streamlit. **Why this sits in al_nassikh (not fatat_al_arab):**
   the data layer owns aggregations; the agent layer only consumes them.
   Keeping the split honest means Worker 2 never touches a JSON file
   directly.

3. `src/fatat_al_arab/agent1_query_understanding/nodes/intent_router.py` —
   **NEW**. Pure-regex classifier with three bilingual pattern libraries
   (`_COUNTING_CUES`, `_AGE_CUES`, `_PROVENANCE_CUES`) and five slot
   vocabularies (`_SLOT_POEMS`, `_SLOT_POETS`, `_SLOT_MANUSCRIPTS`,
   `_SLOT_PAGES`, `_SLOT_CORPUS`). Covers English + MSA + Khaleeji
   variants (`كم`, `شو عدد`, `كم عمر`, `من أين`, `وين من`). Confidence
   scoring: counting cue +0.6, age cue +0.7, provenance cue +0.7, slot
   word +0.3, trailing `?`/`؟` +0.1; cap 1.0; fire at ≥ 0.7. Intent
   priority: **age > provenance > counting** (more specific beats more
   generic) — this is what fixes "كم عمر هذه المخطوطات" ("how old are
   these manuscripts") being misclassified as `count_manuscripts` just
   because "كم" matched the counting cue. When a match fires, the node
   writes to `query_context`:
   - `answer_source = "registry_lookup"`
   - `deterministic_intent ∈ {count_poems, count_poets, count_manuscripts, count_pages, corpus_overview, age, provenance}`
   - `intent_confidence`, `router_cues`, and a heuristic `query_lang`.
   When confidence < 0.7 the node sets `answer_source = "rag_pipeline"`
   and the rest of Agent 1 runs normally. **Why pure regex, no LLM:** an
   LLM call here would cost ~400 ms and defeat the point of a fast path.
   The regex library is small enough (< 40 patterns) that a misclassification
   is easy to diagnose by grepping the cues hit.

4. `src/fatat_al_arab/agent2_retrieval_synthesis/nodes/deterministic_answer.py` —
   **NEW**. Short-circuit answer path. Reads `corpus_stats.corpus_summary()`
   once, renders a bilingual template (Arabic + English, separated by
   `---`) for the detected intent, and emits a citation block that points
   to the JSON file + field (e.g., `count_poems → (anchor_registry_phase4.json, len(entries))`).
   Sets `crag_verdict="Correct"`, `self_rag_verdict="pass"`,
   `guardrail_passed=True`, `is_refusal=False`, and
   `guardrail_flags=["registry_lookup"]` so the Streamlit renderer knows
   to badge the answer as "deterministic". **Why templates, not LLM text:**
   a counting answer is fully determined by the number — an LLM would
   either hallucinate a different number (the exact bug we are fixing) or
   add waffle. Templates keep it exact and auditable.

5. `src/fatat_al_arab/agent1_query_understanding/graph.py` — **EDITED**.
   Added `NODE_ROUTER = "intent_router"` constant; imported the new
   `intent_router_node`; added a new `_after_router()` conditional edge
   function that routes to `END` when `qc.answer_source == "registry_lookup"`;
   changed the graph entry point from `NODE_ANALYZE` to `NODE_ROUTER`.
   **Why END and not a sentinel node:** Agent 1's contract is "enrich
   `query_context` and return"; the deterministic answer is produced by
   Agent 2, so Agent 1 is literally done once the router has set its
   flags. The orchestrator reads those flags and dispatches to the right
   Agent 2 node.

6. `src/fatat_al_arab/orchestrator.py` — **EDITED**. Two changes:
   - `_import_agent1_nodes()` gained an `intent_router` entry.
   - `_import_agent2_nodes()` gained a `deterministic_answer` entry.
   - In the direct-node path of `_run_agent1`, the router runs first; if
     it sets `answer_source = "registry_lookup"`, we return early and
     skip Stages 1-3 entirely.
   - `_run_agent2` inspects `qc.answer_source` at entry; if it is
     `registry_lookup`, the orchestrator calls `deterministic_answer_node`
     directly and bypasses the full Agent 2 graph (no retrieval, no CRAG,
     no Self-RAG, no synthesis). **Why orchestrator-level and not Agent 2
     graph-level:** keeping the shortcut at the orchestrator means the
     Agent 2 graph stays a clean RAG pipeline; the orchestrator owns
     "which pipeline do I run?", the graph owns "how does that pipeline
     flow?".

**Seven intents supported** (mapped to `corpus_stats` helpers):
| Intent | `corpus_stats` call | Example EN | Example AR |
|---|---|---|---|
| `count_poems` | `count_poems()` | "how many poems do I have?" | "كم قصيدة عندي؟" |
| `count_poets` | `count_poets()` | "how many poets?" | "كم عدد الشعراء؟" |
| `count_manuscripts` | `count_manuscripts()` | "how many manuscripts?" | "كم مخطوطة؟" |
| `count_pages` | `count_pages()` | "how many pages indexed?" | "كم صفحة مفهرسة؟" |
| `corpus_overview` | `corpus_summary()` | "tell me about this corpus" | "أخبرني عن المجموعة" |
| `age` | `manuscript_age_range()` | "how old are the manuscripts?" | "كم عمر المخطوطات؟" |
| `provenance` | `regions_of_origin()` + `collectors()` | "where are they from?" | "من أين هذه المخطوطات؟" |

**Acceptance (verified 2026-04-23 end-to-end):**
- `python -m al_nassikh.corpus_stats` prints `poems=1502, poets=509, manuscripts=25, manuscripts_with_toc=11, pages=35, oldest=1800, newest=1970, span=170`.
- `pytest tests/test_intent_router.py` — 20/20 cases pass (13 positive
  routes to 7 distinct intents + 7 negatives that correctly fall through
  to `rag_pipeline` — e.g., "describe the poems about love" stays on the
  RAG path because it has no counting/age/provenance cue).
- End-to-end via orchestrator:
  - "how many poems do I have?" → **1,502** in 158 ms (first call; LLM-free), citation → `anchor_registry_phase4.json / len(entries)`.
  - "how old are the manuscripts?" → **1800-1970 CE (170 years)** in 0.9 ms, citation → `manuscript_registry.json / min(circa_date_start), max(circa_date_end)`.
  - "where are they from?" → **6 regions + 16 collectors** in 0.6 ms, citation → `manuscript_registry.json / region_of_origin, collector`.
- All three responses carry `crag_verdict="Correct"`, `is_refusal=False`,
  `guardrail_flags=["registry_lookup"]`.
- A control query — "show me verses about love in the desert" — still
  runs the full RAG pipeline (router confidence = 0.0, `answer_source =
  "rag_pipeline"`, no short-circuit).

**Non-goals for M2c** (intentionally left to the RAG pipeline):
"show me a poem about X", "who wrote the matla Y?", "what does this verse
mean?", any question with a specific poet name or a thematic content
request. These need retrieval + synthesis and are not degraded by M2c.

### M2d · Genre + Emotion Enrichment (Silver Baseline, ½ day) — added 2026-04-23

*Why this milestone exists: the retrieval UI (Scholar Workbench Tab B) needs
genre + emotion **facets** so users can filter "show me the love poems" or
"show me poems about grief" without relying on dense similarity alone. Dense
embeddings are good at finding semantically related verses but bad at
filtering by category — a love poem and a war poem about the same oasis
embed close together. Hard filters on typed fields solve this cleanly. The
Self-Query extractor (Stage 3) also needs a typed filter schema to emit
against — with no fields to filter on, its output degenerates to free-text
search.*

*Why a keyword heuristic and not an LLM:* three reasons, the same ones that
justify the regex router in M2c. (1) Cost: 1,502 LLM calls per retagging
pass burns the Together.ai free tier. (2) Auditability: a reviewer can open
`genre_heuristic.py` and see exactly why a verse was tagged غزل — "it
contained البارحه, الحمام, and النوم". An LLM decision is a black box.
(3) Baseline honesty: we do NOT have a human-labelled gold set, so we cannot
claim accuracy. What we *can* claim is coverage (how many verses got a
confident tag) and auditability (every tag carries evidence tokens). When
an LLM enrichment pass lands later, this heuristic is the baseline it must
beat.

*Why "silver" baseline: "gold" = human-labelled; "silver" = automated,
inspectable. The UI must surface the distinction (🔸 badge + "heuristic
classifier" tooltip) so reviewers do not mistake machine output for
ground truth.*

**Build plan:**

1. **Freeze the taxonomy first.** `src/al_nassikh/nabati_taxonomy.py` — ten
   Arabic genres (غزل, رثاء, مديح, فخر, **غزو**, حماسة, هجاء, حكمة, وصف,
   دينية) plus `غير_محدد` (explicit abstention). Ten Plutchik-inspired
   emotions (longing, grief, joy, pride, anger, love, awe, nostalgia,
   hope, fear) with English keys. Genre is Arabic because the terms are
   1,400-year-old literary categories; emotion is English because the
   Streamlit facet UI is bilingual and "longing" reads in either language.
   **Why غزو:** the corpus is heavily dominated by war-narrative poems
   (Sowayan's own collection is named for this); folding them into فخر
   or حماسة loses the narrative/lyric distinction.

2. **Build the classifier.** `src/al_nassikh/genre_heuristic.py` — pure
   regex + set-lookup; no models. Arabic surface normaliser (strip
   diacritics, fold hamza, fold Persian yeh ی→ي and Persian kaf ک→ك —
   these appear in the corpus due to Persianate scribal tradition).
   Ten corpus-grounded genre lexicons + ten emotion lexicons. Single-
   label genre (argmax with margin-based confidence); multi-label
   emotion (threshold per class, capped at 4). First-class abstention
   via غير_محدد when the top genre has fewer than `MIN_GENRE_MARKERS`
   hits OR ties with the runner-up within `MIN_GENRE_MARGIN`.

3. **Pattern-based occasion hints.** The Phase-4 TOC transcribers wrote
   `occasion` fields in free Arabic ("بذبحة بن رشيد", "يرد على فلان"),
   not genre tags. An `_OCCASION_PATTERNS` table maps substrings → inferred
   genre ("ذبح" → غزو, "يرد على" → هجاء, "بفرسه" → وصف). When matched, the
   hint gets `OCCASION_HINT_WEIGHT=5` — strong enough to dominate weak
   matla-only evidence without overriding clear lyric content.

4. **Batch driver + audit.** `scripts/enrich_genre_heuristic.py` walks
   `anchor_registry_phase4.json`, classifies each entry, writes
   `anchor_registry_phase4_enriched.json` (original fields preserved +
   `genre`, `genre_confidence`, `genre_source="heuristic_v1"`,
   `genre_evidence`, `occasion_hint_used`, `emotions`) alongside
   `genre_enrichment_audit.json` (coverage, distribution, confidence
   histogram). **Why alongside and not in-place:** the original Phase-4
   file is human TOC ground truth; overwriting it conflates transcription
   with heuristic output. Downstream consumers (`registry_join.py`,
   `embed.py`) prefer the enriched file when present, fall back
   otherwise.

5. **Wire into retrieval schema.**
   - `fatat_al_arab/embed.py` — add `genre`, `emotions`, `genre_confidence`,
     `genre_source` to the Qdrant payload (Stage 4 consumers).
   - `agent1_query_understanding/nodes/self_query.py` — add `genre` and
     `emotions` to the typed-filter extraction schema so the Self-Query
     LLM can emit them as structured filters.

6. **UI surface.** `app/tabs/scholar_workbench.py` — facet sidebar with
   genre multi-select + emotion multi-select. Every heuristic-derived
   tag renders with a 🔸 badge and a hover tooltip: "Heuristic classifier
   — keyword-based, not human-reviewed". This is non-negotiable for
   academic integrity.

**Tuning notes (from the pilot run on 1,502 matlas):**
- `MIN_GENRE_MARKERS` was initially 2; lowered to 1 after the first pass
  showed 95% abstention. The matla field is only the first line of each
  poem (median 8 words) — requiring two markers excluded real love verses
  whose single marker was "قلبي" or "البارحة". Precision is preserved by
  the margin rule rather than the marker-count floor.
- Lexicons were expanded iteratively by inspecting the top unigrams and
  bigrams in the abstained pile after each pass. Each expansion cited
  the corpus evidence that motivated it (see inline comments).
- Persian yeh/kaf folding added after frequency analysis revealed "علی"
  (with Persian yeh) in the top-10 corpus words — common Persianate
  scribal variant in Khaleeji manuscripts.

**Acceptance gate:**
- Coverage ≥ 40% on the full 1,502-matla corpus (effective coverage
  ≥ 45% when the 153 very-short matla artifacts are excluded).
- Every classified verse carries `genre_evidence` (≥ 1 token).
- Every tag carries `genre_source = "heuristic_v1"` — no silent upgrades.
- `غير_محدد` is the largest bucket (confirms first-class abstention, not
  a hidden default).
- `--dry-run` mode prints an audit summary without writing files;
  `--limit N` mode processes the first N anchors for CI smoke tests.

**Pilot result (2026-04-23):**
1,502 anchors classified in ~25 ms. Coverage 43.1% (647 classified, 855
abstained). Genre distribution: غزل 157, دينية 125, مديح 106, غزو 82,
حكمة 78, رثاء 49, فخر 18, وصف 18, هجاء 14, حماسة 0. Emotion multi-label
coverage: longing 66, grief 47, nostalgia 30, love 14, joy 13, awe 7,
fear 4, anger 2, pride 1, hope 1. The heavy abstention is expected and
intentional — Nabati matla lines are often neutral narrative frames
("قال ابن غيث...") that only reveal genre in later verses. Abstaining
explicitly is a feature.

**Non-goals for M2d:**
- **Accuracy claims.** Without a human-labelled gold set we report
  coverage and distribution only. Any accuracy-style statement ("the
  classifier is 85% accurate") would be unsupported and will not appear
  in documentation or UI.
- **Full-poem enrichment.** Only the matla is classified here. A
  full-poem pass would require the Phase 2/3 body transcriptions, which
  are post-MVP.
- **LLM refinement pass.** A targeted LLM pass over the abstention pile
  is a natural follow-up (M2e?) but out of scope — this milestone is the
  silver baseline an LLM would have to beat.

### M3 · Embedding + Qdrant index + Genre/Emotion Retrieval Filters (½ day) — ✅ COMPLETE 2026-04-23

*Why now: the three Agent-2 retrievers are meaningless without a populated index.
Building the index before the retriever nodes lets every retriever test run
against real data, not mocks. It also surfaces embedding-quality issues early.*

**M3 also wires the M2d silver-baseline genre+emotion tags into the retrieval
pipeline** — so queries like "show me love poems" (غزل) or "poems with grief"
(رثاء + emotion=grief) filter at retrieval time rather than relying on dense
similarity to do the genre work implicitly.

#### What was built (2026-04-23)

**1. `src/fatat_al_arab/rrf.py` — ScoredChunk extended with genre fields**
- Added `genre`, `genre_confidence`, `genre_source`, `emotions` to `ScoredChunk`.
- `to_dict()` serialises all four; `fuse()` passes them through unchanged.
- Why on ScoredChunk rather than in `extra`: the retrieve and filter nodes need
  to access these fields directly (no dict-key juggling); typed attributes also
  make the schema explicit in code review.

**2. `src/fatat_al_arab/index.py` — enriched registry + Qdrant payload**
- `DEFAULT_REGISTRY` auto-selects `anchor_registry_phase4_enriched.json` when it
  exists, falls back to the plain registry. No flag needed; safe for existing setups.
- `_chunk_payload()` now extracts genre/emotions into the chunk payload dict.
- `build_chunks()` passes all four fields into the `ScoredChunk` constructor at all
  three levels (verse, group, poem). Group and poem chunks inherit genre from the
  first (matla) entry — the matla sets the poem's tone.
- `load_index()` reads genre fields from the cached `chunks_meta.json` with graceful
  defaults (`غير_محدد`, 0.0, []) for pre-M3 caches.
- Qdrant upsert payload includes genre/genre_confidence/genre_source/emotions for
  native Qdrant payload filtering in future.

**3. `agent1_query_understanding/nodes/self_query.py` — genre/emotion extraction**
- LLM prompt now lists all 10 valid genre labels and 10 valid emotion labels with
  examples ("show me love poems" → genre: "غزل"; "war poems" → genre: "غزو").
- `_build_filters()` routes `genre` (if not غير_محدد) and `emotions_any` (if non-empty)
  into hard_filters (confidence ≥ 0.7) or soft_filters (< 0.7) like existing fields.

**4. `agent2_retrieval_synthesis/nodes/retrieve.py` — genre/emotion hard filters**
- `_apply_hard_filters()` now handles `genre` (exact match) and `emotions_any`
  (any-element intersection with chunk.emotions).
- Both have the standard §5 safety net: if the filter drops all results, it logs a
  warning and skips the filter rather than returning an empty result set.

**Acceptance smoke tests run and passed:**
- ScoredChunk genre defaults and `to_dict()` output ✓
- `DEFAULT_REGISTRY` auto-selects enriched file ✓
- `_chunk_payload` reads genre from enriched entry ✓
- `build_chunks(50 entries)` → 35 classified verse chunks with correct genre ✓
- `_apply_hard_filters` genre exact match, emotions_any, combined, safety-net fallback ✓
- `_build_filters` high/low confidence routing, غير_محدد suppression, empty emotion suppression ✓

**Index rebuild note:** the existing `data/qdrant/` cache (if any) was built without genre
fields. Run `python scripts/rebuild_index.py --force` to regenerate with genre in the
Qdrant payload. The numpy/JSON fallback path (`chunks_meta.json`) will be regenerated
automatically.

1. `src/fatat_al_arab/embed.py` — GATE-AraBERT-v1 dense embed via
   `sentence-transformers`; BM25 token stats from `rank_bm25`. Shares the Arabic
   normaliser from `al_nassikh.parser`.
2. `src/fatat_al_arab/index.py` — file-backed Qdrant collection with two vector
   fields (`dense`, `sparse`) and the payload shape required by Stage 6
   (Multi-Representation Resolution): `{chunk_id, text_khaleeji, text_msa_summary,
    anchor_id, parent_chunk_id, poet_name, normalised_poet,
    manuscript_short_key, manuscript_arabic_name, manuscript_english_name,
    manuscript_id_legacy, page_number, bbox, source_image_path, level,
    meter_pattern, genre, genre_confidence, genre_source, emotions}`.
   `level ∈ {verse, group, poem}` — 3-level stanza chunking from §1.5 of the
   architecture. **Why both `short_key` and `arabic_name` in the payload:** the
   short_key is the filterable identifier (`eq`-filter on Qdrant), the arabic/english
   names are what the UI renders — denormalising means the Multi-Variant Formatter
   and the Ancestral-Mirror mode never have to re-join the registry at response time.
3. Ingestion: for each Phase-1/2/3 verse, attach its Phase-4 `anchor_id` via the
   M1 crosslink; for each Phase-4 anchor with no body text yet, index the matla +
   metadata only.

**Acceptance:** `python scripts/rebuild_index.py` produces
`data/ground_truth/qdrant_store/` with ≥ 1,500 points; a smoke query
`python -m src.fatat_al_arab.index --query "الخيل"` returns ≥ 3 hits, at least one
of which links back to a Phase-4 anchor with a `source_image_path`.

### M4 · Agent 1 — Query & Understanding (1 day)

*Why here: Agent 1's only output is the `QueryContext`. If Agent 1's output shape
is stable, Agent 2 can be built and tested against a frozen-context fixture,
avoiding coupled debugging.*

Every node corresponds to one stage in §2.4.

1. `src/fatat_al_arab/translate.py` — thin wrapper over `llm.py`; returns
   `{ar, en}` regardless of input language.
2. `src/fatat_al_arab/image_ocr.py` — `pytesseract` with the Arabic language pack
   + LLM cleanup via `llm.py` for low-confidence OCR. Returns the question as
   text, then hands off to `translate.py`.
3. `agent1_query_understanding/nodes/bilingual_analyzer.py` — Stage 1. Uses
   `langdetect` for lang ID, then a Qwen2.5 function call for intent
   (`factual | semantic | interpretive`) and Khaleeji-dialect-feature detection.
   Populates `query_lang`, `query_ar`, `query_en`, `detected_intent`,
   `detected_dialect`.
4. `agent1_query_understanding/nodes/bilingual_expand.py` — part of Stage 2.
   Generates 3-5 AR + 3-5 EN paraphrases; merged via RRF downstream.
5. `agent1_query_understanding/nodes/hyde.py` — Stage 2 continued. Prompts the
   LLM to hallucinate a single plausible Nabati verse answering the query (Arabic
   only, bounded to one attempt + 3s cap per §2.8 failure budget); embeds it;
   stores as `hyde_embedding`.
6. `agent1_query_understanding/nodes/self_query.py` — Stage 3. Typed-schema
   extraction into `{poet?, manuscript_short_key?, theme?, verse_min/max?,
   page_min/max?, confidences}`. Manuscript detection is **registry-aware**:
   when the query mentions "Ibn Yahya vol 3" / "الجزء الثالث من ابن يحي" /
   "Huber manuscript", the LLM returns the registry `short_key` directly
   (e.g. `ibn_yahya_401_600`, `huber_1`); `registry.by_arabic_name(...)` and
   `.by_english_name(...)` provide fuzzy fallbacks for partial matches.
   Confidence ≥ 0.7 → hard Qdrant pre-filter on `manuscript_short_key`;
   < 0.7 → soft boost.
7. `agent1_query_understanding/graph.py` — LangGraph `StateGraph(QueryContext)`
   wiring the four nodes; enforces the §2.9 Agent 1 tool registry (no retrieval
   tools available).
8. `agent1_query_understanding/tools.py` — registers only `translate_query`,
   `extract_filters`, `hyde_passage`, `expand_bilingual`. Registry enforcement
   at import time; any other tool raises.

**Acceptance:** `pytest tests/test_agent1_end_to_end.py` validates 30
hand-labelled queries with Self-Query precision ≥ 0.85, recall ≥ 0.75 (the §3.2
gate); for every query the output `QueryContext` validates against the TypedDict
from M0.

### M5 · Agent 2 Stages 4-6 — Retrieval + heritage resolution (1 day)

*Why grouped: these three stages form one unit — take a `QueryContext` + filters,
return approved Khaleeji passages. They share test fixtures and never make sense
apart.*

1. `retrievers/bm25.py`, `retrievers/dense.py`, `retrievers/colbert.py`.
   ColBERT starts as a **working stub** (returns dense top-20) so the pipeline is
   never short-circuited; real ColBERT-v2 late interaction is upgraded post-MVP.
2. `rrf.py` — standard RRF with k=60 and a cross-retriever agreement bonus
   (if a chunk appears in all 3 lists, its RRF score gets +0.1).
3. `agent2_retrieval_synthesis/nodes/retrieve.py` — Stage 4. Fires the three
   retrievers in parallel with an 800 ms timeout each; drops slow retrievers and
   logs the drop (per §2.8 failure budget). Applies Self-Query hard filters from
   the `QueryContext`.
4. `agent2_retrieval_synthesis/nodes/rrf_fuse.py` — Stage 5. Merges lists; emits
   top-5.
5. `agent2_retrieval_synthesis/nodes/resolve_heritage.py` — Stage 6. For every
   MSA-summary hit, swaps the payload `text_msa_summary` → `text_khaleeji` via
   `parent_chunk_id`. Drops rows where `parent_chunk_id` is missing rather than
   silently returning MSA as if it were heritage text.

**Acceptance:** `pytest tests/test_rrf.py` confirms Recall@5 ≥ 0.75 on the 20-query
scholar set (`tests/fixtures/scholar_queries.jsonl`); a bilingual smoke test
confirms EN queries return the same top-5 as their AR equivalents (HyDE
embedding match).

### M6 · Agent 2 Stages 7-10 — Grading, synthesis, reflection, formatting (1 day)

*Why last in the node work: these nodes consume everything upstream. Building them
with retrieval already green means every failure is localised to generation.*

1. `crag_grader.py` — Stage 7. Qwen2.5 grader returning
   `[{chunk_id, label ∈ {Correct, Ambiguous, Incorrect}, confidence, rationale}]`
   per passage. All-Incorrect triggers one bounded re-query loop (§2.8); if the
   re-query also returns all-Incorrect, the flow transitions to the scoped
   "not-found-in-corpus" template (§2.9 guardrail c).
2. `synthesise.py` — Stage 8. Grounded synthesis under a system prompt that
   hard-requires a citation tag `[anchor_id]` after every factual claim, and
   forbids any verse text not verbatim in the approved passage set. Returns
   `{draft, citations, passage_ids_used}`.
3. `reflect.py` — Stage 9. Self-RAG three-axis critic: **faithfulness**
   (no claim outside passages), **relevance** (answers the actual question),
   **completeness** (covers every sub-part). Pass requires all three ≥ 0.8;
   failure triggers up to 2 retries with axis-specific correction prompts.
4. `format_variants.py` — Stage 10. **Deterministic formatter, no LLM.** Lays
   al-Maktub + orthographic + al-Mantuq side by side and attaches a citation
   block with `source_image_path` + `bbox` per citation.
5. `guardrails.py` (full impl) — runs **after** Stage 10 and blocks emission if:
   (a) any factual sentence lacks a resolvable citation; (b) any verse text is
   not verbatim in the approved passage set (paraphrase detection on Arabic
   surface tokens); (c) the output is a scoped-refusal template that was
   nonetheless decorated with unverified detail.
6. `agent2_retrieval_synthesis/graph.py` — LangGraph wiring with two conditional
   edges: CRAG re-query (Stage 7 → Stage 4 once) and Self-RAG retry (Stage 9 →
   Stage 8 up to twice).
7. `agent2_retrieval_synthesis/tools.py` — registers only `retrieve_triple`,
   `rrf_fuse`, `resolve_heritage`, `crag_grade`, `synthesise`, `reflect`,
   `format_multi_variant`. No query tools permitted (§2.9).

**Acceptance:** `pytest tests/test_guardrails.py` confirms all three guardrails
fire on known-bad fixtures; `pytest tests/test_agent2_end_to_end.py` runs
stages 4-10 on a seed `QueryContext` and emits a citation-resolvable response;
a known OOC query triggers the scoped-refusal template.

### M7 · Orchestrator — wire Agents 1 and 2 (½ day)

*Why this is its own milestone: the QueryContext handoff and the failure-handling
contract in §2.8 (fallback-LLM edge after 2 Qwen failures) are the single most
scrutinised part of the architecture. They need dedicated coverage, not just an
integration pass.*

1. `src/fatat_al_arab/orchestrator.py` — calls Agent 1 graph, serialises
   `QueryContext`, passes it into Agent 2 graph. Enforces:
   - **Factual vs interpretive routing.** If `self_query` extracts `poet` or
     `page` with confidence ≥ 0.9, short-circuit to a PostgreSQL-style metadata
     lookup in `anchor_registry_phase4` before invoking Agent 2.
   - **Fallback LLM edge.** Two consecutive Qwen2.5 failures (timeout or
     rate-limit) switch to Mistral-7B via `llm.py` for the remainder of the
     query; logged for evaluation.
   - **Clarification path.** Ambiguous queries (Agent 1 intent confidence < 0.5)
     return a clarifying question to the user rather than running retrieval.
2. `src/fatat_al_arab/rag.py` — thin CLI wrapping `orchestrator.run(query)`.

**Acceptance:** `python -m src.fatat_al_arab.rag --query "شو قال محمد بن عبدالرحمن في الخيل؟"`
returns a response with ≥ 1 resolvable citation + a `source_image_path`;
`pytest tests/test_orchestrator.py` exercises both conditional paths
(factual short-circuit + fallback-LLM edge).

### M8 · Streamlit — Tab B Scholar Workbench (1 day)

*Why Tab B first: Personas 2, 3, 4 share the same retrieval back-end (Agents 1 +
2); only the response-presentation layer differs. Building the shared chat once
and parameterising it with response-mode toggles is cheaper than three tabs.*

1. `app/streamlit_app.py` — top-level shell with `st.tabs(["Archive Manager",
   "Scholar Workbench"])` and a debug toggle (doctor / judge only).
2. `app/tabs/scholar_workbench.py` — single chat with:
   - Language toggle (AR / EN) + `st.chat_input` for text + `st.file_uploader`
     for image questions (routes through `image_ocr.py`).
   - Response-mode radio: **Philology view / Three-Layer Reader / Ancestral
     Mirror** (§3.2 above).
   - Three-panel response: (i) the mode-specific answer, (ii) clickable citation
     list with folio-image preview, (iii) a "show me the retrieved passages"
     expander (debug mode only).
3. `app/tabs/shared_components.py` — citation block renderer, folio viewer with
   bounding-box overlay (PIL draw on the source image), three-layer card layout.
4. Sidebar always shows:
   - The corpus-coverage note ("this system knows 1,502 poems from the Phase-4
     dictionary; poems outside that scope will return 'not in corpus'").
   - A **Corpus Inventory** expander listing all 25 manuscripts from
     `manuscript_registry.json` with their Arabic+English names and the count
     of indexed anchors each contributes; the student-persona user can click
     any entry to seed a search scoped to that manuscript.
   - A link to the evaluation report once M10 is done.

**Acceptance:** `streamlit run app/streamlit_app.py` with a populated `.env`
runs on any laptop with only `pip install -r requirements.txt`; Personas 2, 3, 4
each have a walkable scenario (per `tests/fixtures/persona_scenarios.jsonl`)
that surfaces the right mode-specific panel.

### M9 · Streamlit — Tab A Archive Manager Console (1 day)

*Why after Tab B: the retrieval side earns the grade; the operator side is the
extra polish for Persona 1 that the v4 architecture now commits to. Keeping it
last means if time gets tight, the core RAG demo still ships.*

1. `app/tabs/archive_manager.py` — three sub-sections corresponding to §3.1,
   each leaning on the M2b `escriptorium_client` for the actual work:
   - **Triage** — file uploader (PDF or folder of images) first asks the
     annotator to pick the manuscript from `manuscript_registry.json` via a
     dropdown (25 canonical entries: Huber 1/2, Al-Hassawi, Ibn Yahya vols
     1-4, …). New manuscripts that are not yet in the registry trigger an
     "Add to registry" sub-form that appends a new entry before ingestion can
     proceed — this keeps the registry authoritative. Once the manuscript is
     selected, `triage.py` verdicts (NORMAL / DEGRADED / review) are rendered
     on page thumbnails. NORMAL pages are pushed into an eScriptorium document
     named by `short_key` via `escriptorium_client.upload_pages()`.
   - **Scalpel launcher** — one button that:
     (a) triggers `escriptorium_client.run_kraken_segmentation()`,
     (b) waits for completion with a progress spinner,
     (c) opens eScriptorium's own editor in an embedded `<iframe>` sized to
         fill the tab. The annotator does the zone + baseline + transcription
         work in eScriptorium's native UI (purpose-built, no point
         reinventing); our tab just frames it and holds the document context.
   - **Provenance & Merge** — "Pull & Merge" button calls
     `escriptorium_client.export_pagexml()`, then runs `phase4_merger.py` +
     `cross_link.py`; shows the per-line win table + flagged rows for review.
2. Operator queue state stored in `data/ground_truth/operator_queue.json`; the
   tab renders pending items across sessions and tags each with its
   eScriptorium `document_id` so a session resume can pick up mid-transcription.
3. Graceful degradation: if `escriptorium_client.health_check()` fails, the tab
   shows a copy-pasteable `docker compose up` command and disables the Scalpel
   + Pull & Merge buttons; Triage still works locally because it only needs
   Python + our helpers.

**Acceptance:** an archivist walkthrough: upload `manuscript07.pdf` → triage
flags known DEGRADED pages → zone-annotate one page → run merge → new rows
appear in `anchor_registry_phase4.json` with a `review` flag on low-confidence
lines. All without leaving Streamlit.

### M10 · Evaluation harness + doctor's report (1 day)

*Why last: the §7 rubric demands a single consolidated report, not 9 scattered
test runs. This is the artefact the doctor actually grades against.*

1. `tests/fixtures/scholar_queries.jsonl` — 20 in-corpus queries with gold
   citations.
2. `tests/fixtures/out_of_corpus_queries.jsonl` — 20 queries whose correct
   answer is the "not in corpus" template; measures refusal precision.
3. `tests/fixtures/bilingual_queries.jsonl` — 10 EN + 10 AR paired queries;
   measures bilingual retrieval parity.
4. `tests/fixtures/persona_scenarios.jsonl` — one scripted scenario per persona
   user story (4 scenarios total).
5. `scripts/evaluate.py` — runs every fixture set through `orchestrator.py` and
   produces `data/evaluation_report.md` covering all four §7 axes:
   - **Correctness:** CER bucket distribution from `phase4_merge_audit.json`,
     Recall@5 on scholar set, Cohen's κ on CRAG grader, citation-resolvability
     rate (target 100%), refusal precision on OOC (target ≥ 0.90).
   - **Robustness:** loop-activation rate per conditional (HyDE, CRAG-requery,
     Self-RAG retry), retry-budget exhaustion rate, fallback-LLM invocation rate.
   - **Efficiency:** p50/p95 end-to-end latency (target p50 < 4 s, p95 < 8 s),
     per-stage breakdown, cost per 100 queries (read from Together/Groq usage
     tokens).
   - **Human judgment:** 5-point Likert slot reserved for a Nabati scholar on
     a fixed 15-response sample (faithfulness / dialect fidelity / usefulness).
6. Commit the populated report so the doctor can read the numbers without
   running anything.

**Acceptance:** `python scripts/evaluate.py` produces
`data/evaluation_report.md` with all four axes filled in; any metric that falls
below the architecture-doc claim is flagged honestly rather than hidden.

### M11 · Notebook walkthrough + deployment notes (2 h)

*Why very last: by now everything works, so the notebook becomes a clean
narrative of the persona scenarios, not a debugging log.*

1. `notebooks/nabat_ai_demo.ipynb` — four notebook sections, one per persona:
   load ground-truth JSON → build the index → run the persona's scripted query
   → show the LangGraph state after each stage → display the manuscript-image
   cutout with bounding box.
2. 20-line "How to run this on your laptop" section added to the top of
   `CLAUDE.md`.

**Acceptance:** `jupyter nbconvert --execute nabat_ai_demo.ipynb` finishes
without errors on a fresh clone with `requirements.txt` installed and `.env`
populated.

---

## 5 · Failure-handling budget (enforced in code)

*Why in the plan, not just the doc: these caps are what prevent the CRAG /
Self-RAG / fallback-LLM loops from running forever when the API is flaky. Every
number below is a constant in code, not a magic literal buried in a prompt.*

| Stage | Retry budget | Timeout | Fallback |
|---|---|---|---|
| Agent 1 translation (EN→AR) | 0 | 3 s | Raw query passed to Agent 2 |
| Agent 1 HyDE | 1 | 3 s | Proceed without `hyde_embedding` |
| Agent 1 Self-Query | 1 | 3 s | Unfiltered retrieval |
| Agent 2 each retriever | 0 | 800 ms | Dropped; carry on with remaining 2 |
| Agent 2 CRAG re-query | 1 | — | Scoped "not in corpus" refusal |
| Agent 2 Self-RAG reflection | 2 | — | Flag for expert review, emit with caveat |
| Any LLM call | exp-backoff×3 | 5 s | Switch to Mistral-7B after 2 consecutive Qwen failures |
| eScriptorium health check | 1 | 2 s | Disable Scalpel + Pull & Merge; show "start Docker" instruction |
| eScriptorium Kraken segmentation | 0 | 300 s | Surface error; let annotator fall back to upload-pre-segmented-XML |

---

## 6 · Risks worth flagging before we start

*Why here: §8 of the architecture covers corpus + model risks; this list covers
the execution risks that only become visible during implementation.*

- **Hosted-API rate limits.** Together.ai's free tier caps Qwen2.5 at ~60 req/min.
  A single user query can fire 5-8 LLM calls (HyDE + Self-Query + CRAG + Synth +
  Self-RAG × retries). Mitigation: exponential backoff in `llm.py`, request
  deduplication across the 3-5 bilingual variants, and a clear budget line in the
  evaluation report.
- **Arabic OCR on uploaded images.** `pytesseract` + Arabic pack handles printed
  Arabic reasonably; handwritten questions will fail. Mitigation: Agent 1 surfaces
  an honest "couldn't read this image clearly — please type your question"
  refusal rather than silently embedding garbage.
- **Poet-name ambiguity.** Some Nabati poets share names across manuscripts.
  Crosslink confidence threshold of 0.85 may need tuning on the first Phase-1 run;
  anything below 70% linked rate is a signal the match key is wrong, not a
  signal to lower the threshold.
- **Manuscript registry open questions.** Two items in
  `manuscript_registry.json` are flagged for Asma's confirmation before M1
  runs end-to-end: (a) "ألبرت سيوسيين" is transliterated as "Albert Sowssan"
  — the correct Latin spelling may differ; any fix should be made in the
  registry file first so the short_keys regenerate cleanly. (b) Ibn Yahya is
  currently split into 4 entries (#11-14) to keep volume_range filterable;
  if Asma prefers one entry with sub-volumes, the registry schema needs a
  `parent_short_key` field and the Qdrant filter logic in M4 updates
  accordingly. Both are one-file fixes; flag before M3 (index build).
- **eScriptorium Docker footprint.** eScriptorium's docker-compose stack is
  ~1.5 GB of images (Postgres + Redis + Celery + Django + nginx) and needs
  ~4 GB RAM to run comfortably. Mitigation: Tab A degrades gracefully when
  eScriptorium is not running (§M9); the retrieval demo (Tab B, the doctor's
  actual grading surface) runs without it. The `infra/escriptorium/README.md`
  is written so a reviewer without Docker can still understand the Archive
  Manager flow via screenshots + a recorded walkthrough.
- **iframe embedding of eScriptorium.** Some browsers block iframed Django apps
  that set `X-Frame-Options: DENY`. UB-Mannheim's build defaults to `SAMEORIGIN`
  which works when the Streamlit app and eScriptorium share `localhost`, but
  Safari is stricter. Mitigation: if iframe fails, Tab A falls back to a
  "Open eScriptorium in new tab" link with the document pre-selected; the
  Pull & Merge step still pulls the PAGE-XML back without user action.
- **Persona drift in evaluation.** Each persona's success criterion is qualitative
  ("feels like a universal translator"). Mitigation: the scripted scenarios in
  `persona_scenarios.jsonl` are the binary pass/fail proxy — either the scenario
  runs end-to-end and produces the documented output, or the persona fails.
- **Mistral-7B Arabic quality.** The fallback LLM is weaker at dialectal Arabic.
  If it fires, the response is tagged in the citation block so the reviewer can
  see the degradation; it is not hidden.

---

## 7 · Handing this to Claude Code

When kicking off Claude Code, point it at this file and use this prompt:

> Read `doc/IMPLEMENTATION_PLAN.md`, `doc/DATA_PIPELINE.md`, and
> `doc/Deliverable_2_Architecture_Updated_v4.docx`. Work through the milestones
> M0 → M11 **in order**. At each milestone, stop after the acceptance check
> passes and show me the diff before moving on. Respect the constraints in §0
> (hosted API only, Streamlit only, three-worker split, persona-first UI) and
> the failure-handling budgets in §5. Comments and docstrings lead with a short
> "why" line before the "what" (stable preference — see project memory).

This keeps scope visible, forces verification between milestones, and matches
the explanatory-line style I want used throughout the codebase.
