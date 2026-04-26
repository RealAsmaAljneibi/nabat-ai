# NABAT-AI — Traceability Matrix

**Use-case → Architecture → Code**  
**Course:** MAAI1704 · **Student:** Asma Salem Mubarak Najem Aljneibi

This document maps every use case and user persona to the architectural decision that serves it, and then to the specific code that implements it. A reviewer can open `Deliverable_2_Architecture_Updated_v4.docx` and find each reference (§X.Y) in the table below.

---

## Problem Statement

Khaleeji Nabati poetry is preserved in 25 handwritten manuscripts (76 pages, 2,222 verse anchors) that are inaccessible to non-specialist users because:
1. They are written in Khaleeji Arabic dialect (not MSA)
2. They exist as images, not searchable text
3. There is no bilingual (AR/EN) query interface

**NABAT-AI** solves this by digitising the manuscripts into a structured vector index and exposing them through a bilingual RAG agent.

---

## User Personas → Architecture → Code

### Persona 1: Cultural Institution (Archive Manager)

**Goal:** Ingest new manuscripts; monitor corpus quality; register new poets.

| Architecture Component | Code Location |
|------------------------|---------------|
| Worker 1 (Al-Nassikh) — ETL pipeline | `src/al_nassikh/` |
| PDF → per-page PNGs | `src/al_nassikh/ingest/pdf_to_pages.py` |
| Kraken HTR draft transcription | `src/al_nassikh/ingest/kraken_draft.py` |
| HITL review queue | `src/al_nassikh/ingest/queue_for_review.py` |
| Page quality operators (8 challenges) | `src/al_nassikh/operator/` |
| eScriptorium REST integration (§2.2) | `src/al_nassikh/escriptorium_client.py` |
| Tab B — Archive Manager UI | `app/streamlit_app.py` (Tab B) |
| Manuscript registry (25 MSS) | `data/ground_truth/manuscript_registry.json` |

**Traceability:** Architecture §2.2 (Worker 1) → `src/al_nassikh/` directory  
**UI surface:** Tab B — Archive Manager & Contributor Guide

---

### Persona 2: Academic Researcher (Scholar Workbench)

**Goal:** Find thematic verses ("poems about exile and longing"), filter by genre/poet, read cited sources.

| Architecture Component | Code Location |
|------------------------|---------------|
| Stage 1 — Bilingual Query Analysis (§2.4) | `agent1_query_understanding/nodes/bilingual_analyzer.py` |
| Stage 2a — HyDE dense augmentation (§2.4) | `agent1_query_understanding/nodes/hyde.py` |
| Stage 2b — Bilingual paraphrase expansion (§2.4) | `agent1_query_understanding/nodes/bilingual_expand.py` |
| Stage 3 — Self-Query typed filter extraction (§2.4) | `agent1_query_understanding/nodes/self_query.py` |
| Stage 4 — Triple hybrid retrieval (§2.5) | `agent2_retrieval_synthesis/nodes/retrieve.py` |
| Stage 5 — RRF fusion (§2.5) | `agent2_retrieval_synthesis/nodes/rrf_fuse.py` |
| Stage 7 — CRAG grading (§2.5) | `agent2_retrieval_synthesis/nodes/crag_grader.py` |
| Stage 8 — Grounded synthesis with citations (§2.5) | `agent2_retrieval_synthesis/nodes/synthesise.py` |
| Stage 10 — Multi-variant format (§2.5) | `agent2_retrieval_synthesis/nodes/format_variants.py` |
| Conversation history (multi-turn) | `bilingual_analyzer.py:_format_history_for_prompt()` |
| Genre + emotion sidebar filters | `app/streamlit_app.py` (Tab A, sidebar) |

**Traceability:** Architecture §3.2 (Researcher persona) → Stages 1–10 → `agent1/` + `agent2/` nodes  
**UI surface:** Tab A — Scholar Workbench, genre/emotion filter, al-Maktub / al-Mantuq variants

---

### Persona 3: Student

**Goal:** Ask questions in English ("What is Nabati poetry?"), learn about the corpus, discover poems by era or genre.

| Architecture Component | Code Location |
|------------------------|---------------|
| Stage 0.5a — Intent router (fast-path) | `agent1_query_understanding/nodes/intent_router.py` |
| Stage 0.5b — Semantic router (4-track LLM) | `agent1_query_understanding/nodes/semantic_router.py` |
| Deterministic answer (counting, provenance, age) | `agent2_retrieval_synthesis/nodes/deterministic_answer.py` |
| Capabilities fast-path | `intent_router.py:_CAPABILITIES_CUES` |
| Bilingual response (AR + EN) | `synthesise.py` + `format_variants.py` |
| Bilingual guardrails (refusal templates) | `src/fatat_al_arab/guardrails.py` |

**Traceability:** Architecture §3.3 (Student persona) → M2c intent router + deterministic answer path  
**UI surface:** Tab A — single query box, English accepted, bilingual responses

---

### Persona 4: Heritage Enthusiast

**Goal:** Upload a photo of a verse and ask "Who wrote this?"; search by poet name; hear the poem read aloud.

| Architecture Component | Code Location |
|------------------------|---------------|
| Image OCR → query (§2.4) | `src/fatat_al_arab/image_ocr.py` |
| Stage 0.5c — Image-grounded provenance routing | `intent_router.py:_IMAGE_PROVENANCE_CUES` |
| Image provenance node | `agent2_retrieval_synthesis/nodes/image_provenance.py` |
| Voice input (Whisper) | `src/fatat_al_arab/audio_input.py` |
| Poet biographical lookup | `data/ground_truth/poets_bio.json` |
| Al-Mantuq (oral reading) format variant | `agent2_retrieval_synthesis/nodes/format_variants.py` |

**Traceability:** Architecture §3.4 (Enthusiast persona) → image_ocr + audio_input + image_provenance  
**UI surface:** Tab A — image upload button, mic icon, al-Mantuq format selector

---

## Original Problem Statement → Architecture → Implementation

| Problem | Architecture Decision | Implementation |
|---------|-----------------------|----------------|
| Manuscripts are images, not text | Kraken HTR + eScriptorium OCR pipeline | `al_nassikh/ingest/`, `escriptorium_client.py` |
| 2,222 verses span 4 phases of digitisation | Phase 1–4 cross-link (M1) + unified Qdrant index | `cross_link.py`, `index.py`, `anchor_registry_full_enriched.json` |
| Khaleeji dialect ≠ MSA (retrieval gap) | Heritage resolution (Stage 6) + dialect-aware patterns | `resolve_heritage.py`, bilingual regex in `intent_router.py` |
| Factual queries fail in RAG (counting, age, provenance) | Stage 0.5 fast-path router before any LLM call | `intent_router.py`, `corpus_stats.py`, `deterministic_answer.py` |
| Short queries miss dense embedding targets | HyDE hypothetical verse + bilingual expansion | `hyde.py`, `bilingual_expand.py` |
| Retrieved passages may be irrelevant | CRAG grading with 1 bounded re-query | `crag_grader.py`, `retrieve.py` (re-query path) |
| Synthesised answers may hallucinate | Self-RAG faithfulness + relevance + completeness check | `reflect.py`, `synthesise.py` (retry with fix_instructions) |
| Users speak both Arabic and English | Bilingual analysis + bilingual refusal templates | `bilingual_analyzer.py`, `guardrails.py` |
| Demo must run without Docker/GPU | Hosted API (Together.ai/Groq) + stub fallback | `llm.py`, `.env.example` |
| Genre metadata not in raw manuscripts | Silver-baseline heuristic classifier | `genre_heuristic.py`, `nabati_taxonomy.py` |

---

## Three-Worker Architecture → Code Modules

```
Architecture Doc           → Code Module
─────────────────────────────────────────────────────────────────
§2.2 Worker 1 (Al-Nassikh) → src/al_nassikh/
§2.4 Worker 2 (Agent 1)    → src/fatat_al_arab/agent1_query_understanding/
§2.5 Worker 3 (Agent 2)    → src/fatat_al_arab/agent2_retrieval_synthesis/
§2.6 State contract        → src/fatat_al_arab/state.py
§2.7 LLM adapter           → src/fatat_al_arab/llm.py
§2.8 Failure budgets       → CRAG_REQUERY_MAX, SELF_RAG_MAX_RETRIES (constants in nodes)
§2.9 Tool registry         → agent1_query_understanding/tools.py, agent2.../tools.py
§3   Personas              → This document
§4   Corpus strategy       → data/ground_truth/anchor_registry_full_enriched.json
§5   Reliability / failure → §5 comments throughout nodes + orchestrator.py
```

---

## Milestone → Code Traceability

| Milestone | Description | Key Files |
|-----------|-------------|-----------|
| M0 | Repo scaffolding, env config | `requirements.txt`, `.env.example`, `src/` layout |
| M1 | Phase 1-3 ↔ Phase 4 cross-link | `cross_link.py`, `registry.py`, `verse_anchor_crosslink.json` |
| M2a | ETL operators (8 challenges) | `src/al_nassikh/operator/` (7 operator modules) |
| M2b | eScriptorium REST client + Tab B | `escriptorium_client.py`, `streamlit_app.py` Tab B |
| M2c | Intent router + deterministic answer | `intent_router.py`, `corpus_stats.py`, `deterministic_answer.py` |
| M2d | Genre/emotion silver classifier | `nabati_taxonomy.py`, `genre_heuristic.py`, `enrich_genre_heuristic.py` |
| M3 | Genre/emotions in Qdrant + Self-Query filters | `index.py`, `self_query.py`, `retrieve.py` |
| M4 | Agent 1 full graph (Stages 0.5–3) | `agent1_query_understanding/graph.py` + all nodes |
| M5 | Agent 2 full graph (Stages 4–10) | `agent2_retrieval_synthesis/graph.py` + all nodes |
| M6 | RRF fusion | `rrf.py`, `rrf_fuse.py` |
| M7 | Orchestrator (single entry point) | `orchestrator.py` |
| M8 | Streamlit UI (2 tabs, bilingual) | `app/streamlit_app.py` |
| M9 | Agentic loop upgrade (LLM-directed re-query + surgical retry) | `crag_grader.py`, `reflect.py`, `synthesise.py`, `state.py` |
| M10 | Evaluation harness (4 axes) | `scripts/evaluate.py` |
| M11 | Phase 1-3 data ingest + full index | `scripts/ingest_phases_123.py`, `scripts/rebuild_index.py` |
