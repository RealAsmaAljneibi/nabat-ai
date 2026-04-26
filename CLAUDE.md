# NABAT-AI — Khaleeji Nabati Poetry Digitization & RAG System

## Project Overview

AI system to digitise handwritten Khaleeji Nabati poetry manuscripts and expose them via a bilingual RAG agent. Graded MAAI1704 deliverable; audience is Asma's professor and a demo panel — portability ("runs everywhere") matters more than raw throughput.

**Course:** MAAI1704 – Generative AI | **Student:** Asma Salem Mubarak Najem Aljneibi  
**Architecture doc (authoritative):** `doc/Deliverable_2_Architecture_Updated_v4.docx`  
**Build plan:** `doc/IMPLEMENTATION_PLAN.md` (v2, updated 2026-04-23)

---

## How to run

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Copy the env template and fill in your API key
cp .env.example .env
# Set LLM_PROVIDER=together (or groq) and LLM_API_KEY=<your key>

# 3. Build the vector index (once, before first query)
PYTHONPATH=src python scripts/rebuild_index.py --registry data/ground_truth/anchor_registry_full_enriched.json --force

# 4. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

**Tab A — Scholar Workbench** runs entirely without Docker.  
**Tab B — Archive Manager** degrades gracefully if eScriptorium is not running — the contributor guide and rebuild button still work.

---

## API Keys — pick one


| Provider        | Where to get it                                                           | Notes                                         |
| --------------- | ------------------------------------------------------------------------- | --------------------------------------------- |
| **Together.ai** | [https://api.together.xyz](https://api.together.xyz) → Sign up → API Keys | Free tier: ~60 req/min. Primary.              |
| **Groq**        | [https://console.groq.com](https://console.groq.com) → API Keys           | Free tier, fast. Good backup.                 |
| **stub**        | None needed                                                               | Offline testing — canned responses, not real. |


Set `LLM_PROVIDER=together` (or `groq` or `stub`) and `LLM_API_KEY=<key>` in `.env`.

---

## Three-Worker Architecture

```
User query
    │
    ▼
Worker 2 — Agent 1: Query Understanding (Stages 0.5–3)
  intent_router → bilingual_analyzer → hyde → bilingual_expand → self_query
    │   (QueryContext)
    ▼
Worker 3 — Agent 2: Retrieval & Synthesis (Stages 4–10)
  retrieve → rrf_fuse → resolve_heritage → crag_grader
         → synthesise → reflect → format_variants
    │   (AgentState["final_response"])
    ▼
Streamlit UI (app/streamlit_app.py)

Worker 1 — Al-Nassikh: Deterministic ETL (offline / on ingest)
  PAGE-XML → parser → cross_link → registry_join → embed → Qdrant
```

The orchestrator (`src/fatat_al_arab/orchestrator.py`) is the single public entry point.

---

## Directory Structure — current state

Legend: ✅ Complete · 🔧 Placeholder (not blocking) · 🗑️ Legacy (ignore)

```
handwritten-poems/
│
├── .env.example                           LLM_PROVIDER, LLM_API_KEY, EMBED_MODEL
├── requirements.txt                       All dependencies pinned
├── CLAUDE.md                              ← this file
│
├── doc/
│   ├── IMPLEMENTATION_PLAN.md             ✅ v2 build plan (M0–M11), all milestones
│   ├── DATA_PIPELINE.md                   ✅ ETL workflow docs
│   ├── ARCHITECTURE_DIAGRAMS.md           ✅ Mermaid source for all architecture diagrams
│   ├── Deliverable_2_Architecture_Updated_v4.docx   ← authoritative architecture
│   └── Deliverable_2_Architecture_Poetry.pdf        (PDF render of v3 — for reference only)
│
├── infra/
│   └── escriptorium/
│       ├── docker-compose.yml             Self-host eScriptorium (UB-Mannheim image)
│       ├── nginx.conf                     Reverse proxy config
│       ├── .env.example                   eScriptorium env vars
│       └── README.md                      One-command bring-up instructions
│
├── src/
│   │
│   ├── al_nassikh/                        Worker 1 — Deterministic ETL
│   │   │
│   │   ├── corpus_stats.py                ✅ Deterministic aggregations (M2c)
│   │   │                                     count_poems / count_poets / manuscripts_overview
│   │   │                                     Used by: intent_router, streamlit sidebar
│   │   │
│   │   ├── cross_link.py                  ✅ Phase 1-3 ↔ Phase 4 fuzzy join (M1)
│   │   │                                     rapidfuzz WRatio; acceptance gate ≥70% @ 0.85
│   │   │                                     Output: data/ground_truth/verse_anchor_crosslink.json
│   │   │
│   │   ├── registry.py                    ✅ Manuscript registry loader + lookup
│   │   │                                     by_short_key / by_filename / list_all
│   │   │                                     Used by: self_query, archive UI, evaluate.py
│   │   │
│   │   ├── registry_join.py               ✅ Enrich anchor_registry with human-readable MS names
│   │   │                                     Re-runnable when registry or filename_map changes
│   │   │
│   │   ├── nabati_taxonomy.py             ✅ Frozen label space (M2d)
│   │   │                                     GENRES (10 + غير_محدد), EMOTIONS (10)
│   │   │                                     validate_genre / validate_emotions
│   │   │                                     Used by: genre_heuristic, embed, self_query, streamlit
│   │   │
│   │   ├── genre_heuristic.py             ✅ Silver-baseline genre+emotion classifier (M2d)
│   │   │                                     Corpus-grounded keyword lexicon; abstains to غير_محدد
│   │   │                                     43% genre coverage on 1,502 matla verses
│   │   │                                     Used by: scripts/enrich_genre_heuristic.py
│   │   │
│   │   ├── escriptorium_client.py         ✅ REST wrapper over escriptorium-connector (Tab B)
│   │   │                                     7 operations: ensure_project → export_pagexml
│   │   │                                     Graceful fallback when Docker not running
│   │   │
│   │   ├── reference_ingest.py            ✅ Ingests scholarly reference PDFs as context
│   │   │                                     4 Arabic books (Ibn Khaldun, desert culture, etc.)
│   │   │                                     OCR with pytesseract → level="reference" chunks
│   │   │                                     Output: data/ground_truth/reference_corpus.json
│   │   │
│   │   ├── parser.py                      🔧 PLACEHOLDER — PAGE-XML → stanza JSON
│   │   │                                     Not blocking: Phase 1 corpus already in JSON
│   │   │                                     Needed for: future eScriptorium-based ingestion
│   │   │
│   │   ├── phase4_merger.py               🔧 PLACEHOLDER — TOC merge script
│   │   │                                     Not blocking: Phase 4 merge complete (1,502 entries)
│   │   │                                     Needed for: re-running ETL on new manuscripts
│   │   │
│   │   ├── ingest/                        ✅ New-manuscript ingestion pipeline
│   │   │   ├── pdf_to_pages.py               PDF → per-page PNGs
│   │   │   ├── kraken_draft.py               Kraken v5 HTR draft transcription
│   │   │   └── queue_for_review.py           HITL review queue (pending/in_review/complete)
│   │   │
│   │   └── operator/                      ✅ 8-challenge HITL quality operators
│   │       ├── triage.py                     Page quality gating (NORMAL / DEGRADED)
│   │       ├── bleed_suppress.py             Ghost ink removal (reverse-page bleed)
│   │       ├── standardise.py                Deskew + binarise + resize
│   │       ├── intrusion_mask.py             Stamp/label exclusion via poly-masking
│   │       ├── style_profile.py              NASKH/RUQAH style ID from 5 sample pages
│   │       ├── zone_classify.py              Layout region labelling (TWO_COL, PROSE, MARGIN…)
│   │       └── qa_jury.py                    Confidence routing → auto / jury / HITL
│   │
│   └── fatat_al_arab/                     Workers 2 + 3 — RAG Pipeline
│       │
│       ├── llm.py                         ✅ Hosted-API adapter (Together / Groq / stub)
│       │                                     chat(prompt, system, json_schema, max_tokens)
│       │                                     3 retries + exponential backoff + Qwen→Mistral fallback
│       │                                     Used by: every LLM-calling node in both agents
│       │
│       ├── state.py                       ✅ LangGraph state contract
│       │                                     QueryContext TypedDict — Agent 1 output
│       │                                     AgentState TypedDict — full graph state
│       │                                     make_query_context / make_agent_state factories
│       │                                     crag_requery_strategy field (2026-04-25)
│       │
│       ├── orchestrator.py                ✅ Single public entry point
│       │                                     run(query_text, input_image_path=None) → dict
│       │                                     Wires Agent 1 → QueryContext → Agent 2
│       │                                     Dual mode: LangGraph full / direct nodes (CI)
│       │                                     Never raises — always returns displayable dict
│       │
│       ├── embed.py                       ✅ Dense + BM25 embeddings
│       │                                     GATE-AraBERT 768-dim (sentence-transformers)
│       │                                     Fallback: deterministic hash-seeded vectors (CI/offline)
│       │                                     Arabic normaliser: alef unify, harakat strip, ة→ه
│       │
│       ├── index.py                       ✅ File-backed Qdrant index builder (M3)
│       │                                     9-level chunking: verse / group / poem /
│       │                                       manuscript / poet / era / genre / emotion / reference
│       │                                     Auto-selects enriched registry when available
│       │                                     Payload: chunk_id, text, level, anchor_id, poet_name,
│       │                                              source_volume, source_page, manuscript_short_key,
│       │                                              genre, genre_confidence, genre_source, emotions
│       │                                     build_index(registry, qdrant_path, force_rebuild)
│       │                                     load_index(qdrant_path) — fast path, no re-encoding
│       │
│       ├── rrf.py                         ✅ Reciprocal Rank Fusion (M3 — ScoredChunk extended)
│       │                                     ScoredChunk: chunk_id, rrf_score, text, level,
│       │                                                  anchor_id, poet_name, manuscript_short_key,
│       │                                                  genre, genre_confidence, emotions, extra
│       │                                     fuse(ranked_lists, k=60, top_n) → list[ScoredChunk]
│       │
│       ├── guardrails.py                  ✅ Refusal templates + out-of-corpus guards
│       │                                     should_refuse(query, genre, results) → (bool, reason)
│       │                                     AR + EN bilingual refusal messages
│       │
│       ├── translate.py                   ✅ AR ↔ EN translation via LLM adapter
│       │                                     translate_ar_to_en / translate_en_to_ar
│       │
│       ├── image_ocr.py                   ✅ Pytesseract Arabic OCR wrapper
│       │                                     ocr_image_to_query(image_path) → str
│       │                                     Graceful degradation if tesseract unavailable
│       │
│       ├── audio_input.py                 ✅ Whisper-based voice query transcription
│       │                                     transcribe_audio(audio_path) → str
│       │                                     Used by: streamlit_app (mic input), deterministic_answer
│       │
│       ├── rag.py                         🔧 RESERVED NAMESPACE — intentionally empty
│       │                                     RAG logic lives in orchestrator + agents (not here)
│       │                                     See module docstring before importing
│       │
│       ├── agent1_query_understanding/    ✅ Worker 2 — Stages 0.5–3
│       │   ├── graph.py                      LangGraph StateGraph with conditional edges
│       │   ├── tools.py                      Tool registry (translate_query, extract_filters, etc.)
│       │   └── nodes/
│       │       ├── intent_router.py          Stage 0.5 — regex fast-path (M2c)
│       │       │                               7 intents, EN + AR + Khaleeji patterns
│       │       │                               Counting/age/provenance → registry_lookup
│       │       │                               Everything else → rag_pipeline
│       │       ├── bilingual_analyzer.py     Stage 1 — lang detect + intent + dialect ID
│       │       ├── semantic_router.py        Stage 1b — embedding-based semantic intent routing
│       │       │                               Fallback when regex fast-path is inconclusive
│       │       ├── hyde.py                   Stage 2a — hypothetical Nabati verse (3 s timeout)
│       │       ├── bilingual_expand.py       Stage 2b — 3-5 AR + EN paraphrases
│       │       └── self_query.py             Stage 3 — typed filter extraction (M3 updated)
│       │                                       Extracts: poet, manuscript, genre, emotions_any,
│       │                                                 page/verse range, theme
│       │                                       Hard (≥0.7) vs soft (<0.7) filter split
│       │                                       Registry-aware manuscript resolution (rapidfuzz)
│       │
│       ├── agent2_retrieval_synthesis/    ✅ Worker 3 — Stages 4–10
│       │   ├── graph.py                      LangGraph with CRAG re-query + Self-RAG retry loops
│       │   └── nodes/
│       │       ├── deterministic_answer.py   Stage 0.5 — registry-lookup answer (M2c)
│       │       │                               Bilingual template renderer; 🗂️ badge in UI
│       │       ├── image_provenance.py        Stage 0.5b — image-grounded provenance
│       │       │                               OCRs attached image → verse-level dense retrieval
│       │       │                               In-corpus: cite poet+MS+page; Out: "not in corpus"
│       │       ├── retrieve.py               Stage 4 — triple hybrid (M3 updated)
│       │       │                               BM25 + Dense + ColBERT; hard filters applied first
│       │       │                               Genre exact-match + emotions_any + §5 safety net
│       │       │                               On CRAG re-query: augments BM25 with crag_requery_strategy
│       │       ├── rrf_fuse.py               Stage 5 — RRF k=60, top-5
│       │       ├── resolve_heritage.py       Stage 6 — MSA→Khaleeji text swap
│       │       ├── crag_grader.py            Stage 7 — passage-level relevance grading
│       │       │                               Correct / Ambiguous / Incorrect; bounded 1× re-query
│       │       │                               Outputs requery_strategy: LLM-generated search hint
│       │       │                               Written to state["crag_requery_strategy"] (2026-04-25)
│       │       ├── synthesise.py             Stage 8 — grounded generation (mandatory citations)
│       │       ├── reflect.py                Stage 9 — Self-RAG: faithfulness+relevance+completeness
│       │       │                               retry ≤2× per §5 failure budget
│       │       │                               Outputs fix_instructions + failed_claims (2026-04-25)
│       │       │                               synthesise.py prefers fix_instructions over issues on retry
│       │       └── format_variants.py        Stage 10 — al-Maktub + orthographic + al-Mantuq
│       │
│       └── retrievers/                    ✅ Retriever implementations
│           ├── bm25.py                       BM25Okapi over normalised Arabic tokens
│           ├── dense.py                      Cosine similarity over AraBERT embeddings
│           └── colbert.py                    ColBERT stub (token-level; late interaction)
│
├── app/
│   ├── streamlit_app.py                   ✅ PRIMARY UI — two tabs
│   │                                         Tab A: Scholar Workbench (unified, all users)
│   │                                           • Single default view (mode toggle removed — v2)
│   │                                           • Genre + emotion sidebar filter (M3)
│   │                                           • M2c deterministic answer badge (🗂️)
│   │                                           • Conversation history (last 5 turns)
│   │                                         Tab B: Archive Manager & Contributor Guide
│   │                                           • Metadata schema reference table
│   │                                           • Manuscript registration guide
│   │                                           • Step-by-step contribution workflow
│   │                                           • Metadata quality tips (occasion field, bbox, etc.)
│   │                                           • Operator pipeline tools (triage / bleed / standardise)
│   │                                           • Index rebuild button
│   ├── _calligraphy_b64.py                ✅ Base-64 encoded calligraphy header image (UI asset)
│   └── static/calligraphy-sample.png      ✅ Hero image for the Streamlit header
│
├── data/
│   └── ground_truth/
│       ├── anchor_registry_phase4.json    ✅ 1,502 RAG-ready anchors (Phase 4 TOC only)
│       ├── anchor_registry_phase4_enriched.json  ✅ + genre / emotions / MS names (M2d)
│       ├── anchor_registry_full.json      ✅ 2,222 anchors — Phase 1-4 combined
│       ├── anchor_registry_full_enriched.json ✅ 2,222 anchors + genre/emotions ← CANONICAL SOURCE
│       ├── anchor_registry_phase4_cleaned.json   ✅ Cleaned version (QA pass)
│       ├── manuscript_registry.json       ✅ 25 manuscripts: short_key, AR/EN names, provenance
│       ├── manuscript_filename_map.json   ✅ filename stem → short_key mapping
│       ├── verse_anchor_crosslink.json    ✅ Phase 1-3 ↔ Phase 4 cross-links (M1 output)
│       ├── poets_bio.json                 ✅ 509 poet biographical entries
│       ├── phase2_poems.json              ✅ Phase 2 ground truth (12 pages, 577 stanza entries)
│       │                                     ms04, ms05, ms19, ms21 — absorbed into phase4 registry
│       ├── phase3_poems.json              ✅ Phase 3 ground truth (8 pages)
│       │                                     ms01, ms03, ms06, ms08 — absorbed into phase4 registry
│       ├── phase1_poems.json              ✅ Phase 1 ground truth
│       ├── genre_enrichment_audit.json    ✅ Coverage + distribution report (M2d output)
│       ├── registry_join_audit.json       ✅ Unmatched sources report
│       ├── cleaning_audit.json            ✅ Registry cleaning QA report
│       └── phase4_merge_audit.json        ✅ Merge QA report
│
├── data/qdrant/                           ✅ File-backed Qdrant index (rebuild with rebuild_index.py)
│   ├── chunks_meta.json                      4,747 chunks across 25 manuscripts (all phases indexed)
│   └── embeddings.npy                        AraBERT 768-dim float32 matrix
│
├── scripts/
│   ├── ingest_phases_123.py               ✅ Parse Phase 1/2/3 PAGE-XML → anchor entries → merge
│   │                                         python scripts/ingest_phases_123.py [--dry-run]
│   ├── rebuild_index.py                   ✅ Build/rebuild Qdrant index from enriched registry
│   │                                         python scripts/rebuild_index.py --registry data/ground_truth/anchor_registry_full_enriched.json [--force]
│   ├── enrich_genre_heuristic.py          ✅ Run genre classifier over all 1,502 anchors
│   │                                         python scripts/enrich_genre_heuristic.py [--dry-run]
│   ├── clean_registry.py                  ✅ One-time QA pass: fix swapped fields, remove leaked
│   │                                         non-Arabic tokens → produced anchor_registry_phase4_cleaned.json
│   └── evaluate.py                        ✅ M10 evaluation harness (4 axes: correctness,
│                                             robustness, efficiency, human)
│                                             LLM_PROVIDER=stub PYTHONPATH=src python scripts/evaluate.py
│
├── tests/                                 ✅ 12 test files, ~3,800 lines
│   ├── test_agent1_end_to_end.py             Agent 1 pipeline: all stages + failure budgets
│   ├── test_agent2_retrieval.py              Agent 2 Stage 4-5: triple hybrid + RRF
│   ├── test_agent2_synthesis.py              Agent 2 Stage 8-10: synthesise + reflect + format
│   ├── test_orchestrator.py                  End-to-end + direct node mode + failure handling
│   ├── test_streamlit_app.py                 Module-level helpers (_index_exists etc.)
│   ├── test_operator_triage.py               Al-Nassikh operators: triage, bleed, standardise
│   ├── test_llm_adapter.py                   LLM adapter: stub, fallback, retries
│   ├── test_index.py                         Index building: 3-level chunking, genre payload
│   ├── test_crosslink.py                     Cross-link algorithm + acceptance gate
│   ├── test_rrf.py                           RRF fusion + score normalisation
│   ├── test_semantic_intent_router.py        Semantic router: embedding-based intent classification
│   └── test_archive_manager.py              Streamlit Archive Manager fixtures
│
└── notebooks/
    └── nabat_ai_demo.ipynb                ✅ Demo notebook for live presentations
```

---

## Guiding Decisions (non-negotiable — cite in code comments)

1. **LLM inference = hosted API only.** Qwen2.5-7B primary + Mistral-7B fallback via Together.ai or Groq. Every LLM call routes through `src/fatat_al_arab/llm.py`. No CUDA / Ollama / vLLM.
2. **UI = Streamlit, two tabs.** `app/streamlit_app.py` is the only demo entrypoint. Tab A = Scholar Workbench (one unified query interface, single default view — mode toggle removed in v2). Tab B = Archive Manager & Contributor Guide.
3. **Three workers.** `agent1_query_understanding/` (Worker 2) and `agent2_retrieval_synthesis/` (Worker 3) are sibling packages. A reviewer can open §2.4/§2.5 in the architecture doc and the matching folder side-by-side.
4. **eScriptorium via REST API.** Self-hosted (`infra/escriptorium/docker-compose.yml`); driven through `al_nassikh/escriptorium_client.py`. Tab B degrades gracefully when Docker is not running.
5. **Silver-baseline genre tagging (🔸 badge required).** Genre/emotion tags come from `genre_heuristic.py` (heuristic_v1). Without a human-labelled gold set, accuracy cannot be claimed — only coverage (82.9% on the full 2,222-entry corpus) and distribution can be reported. The 🔸 badge must appear on all heuristic-tagged results in the UI.

---

## Ground Truth Status


| Phase                    | Manuscripts            | Pages    | Anchors            | Status                     |
| ------------------------ | ---------------------- | -------- | ------------------ | -------------------------- |
| Phase 1 — Momentum       | ms07, ms14, ms15, ms22 | 20       | 344 verse entries  | ✅ Vectorized (2026-04-26)  |
| Phase 2 — Core Sadr/Ajuz | ms04, ms05, ms19, ms21 | 12 pages | 240 verse entries  | ✅ Vectorized (2026-04-26)  |
| Phase 3 — Edge Cases     | ms01, ms03, ms06, ms08 | 8 pages  | 136 verse entries  | ✅ Vectorized (2026-04-26)  |
| Phase 4 — TOC Metadata   | vols 001–782           | 36       | 1,502              | ✅ Complete                 |
| **Combined**             | **25 manuscripts**     | **76**   | **2,222**          | ✅ All phases in Qdrant     |


> **Note:** Phase 1/2/3 full-verse data was ingested from PAGE-XML via `scripts/ingest_phases_123.py` (2026-04-26).
> The canonical registry is now **`anchor_registry_full_enriched.json`** (2,222 entries, 82.9% genre coverage).
> The vector index has **4,747 chunks** across all 25 manuscripts.
> Rebuild command: `python scripts/rebuild_index.py --registry data/ground_truth/anchor_registry_full_enriched.json --force`

---

## Completed Milestones


| Milestone | What it did                                                                                                                                       | Key files                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **M1**    | Phase 1-3 ↔ Phase 4 cross-link; manuscript registry                                                                                               | `cross_link.py`, `registry.py`, `registry_join.py`                         |
| **M2c**   | Intent router + deterministic answer path (< 2 ms)                                                                                                | `intent_router.py`, `corpus_stats.py`, `deterministic_answer.py`           |
| **M2d**   | Genre+emotion silver-baseline classifier (43% coverage)                                                                                           | `nabati_taxonomy.py`, `genre_heuristic.py`, `enrich_genre_heuristic.py`    |
| **M3**    | Genre+emotions in Qdrant payload + Self-Query filters                                                                                             | `index.py`, `rrf.py`, `self_query.py`, `retrieve.py`                       |
| **M8**    | Streamlit UI: unified Scholar tab + Contributor Guide tab                                                                                         | `app/streamlit_app.py`                                                     |
| **M9**    | Agentic-loop upgrade: CRAG re-query now LLM-directed via `requery_strategy`; Self-RAG retry now surgical via `fix_instructions` + `failed_claims` | `crag_grader.py`, `reflect.py`, `synthesise.py`, `retrieve.py`, `state.py` |


---

## Code Style

- Every function and module docstring leads with a **"Why this exists"** line before the "what".
- Comments lead with `Why:` when the reasoning is not obvious from the code.
- No magic literals — failure-handling budgets (retry counts, timeouts) are named constants.
- Graceful degradation throughout — every heavy dependency is a lazy import with a fallback.

