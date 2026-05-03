# NABAT-AI — Khaleeji Nabati Poetry Digitization & RAG System

> **Course:** MAAI1704 – Generative AI  
> **Student:** Asma Salem Mubarak Najem Aljneibi  
> **Deliverable:** Graded MAAI1704 Project — Runs without Docker, GPU, or paid APIs.

## Overview

An AI system that digitises handwritten Khaleeji Nabati poetry manuscripts and exposes them through a bilingual (Arabic / English) Retrieval-Augmented Generation agent. The full Scholar interface runs in a single Streamlit tab — no GPU, no Docker, no paid-tier API required.

---

## What it does

| Capability | Detail |
|---|---|
| **Manuscript digitisation** | 25 manuscripts · 76 pages · 4,031 entries across all sources |
| **Bilingual RAG** | Arabic + English queries → grounded answers with mandatory verse citations |
| **Genre & emotion tagging** | Silver-baseline heuristic classifier; 82.9 % coverage on the full corpus |
| **Deterministic answers** | Counting / provenance / age queries answered in < 2 ms from a registry, no LLM call |
| **Multi-modal input** | Text query · image (OCR) · voice (Whisper transcription) |
| **Archive manager** | Contributor guide, metadata schema, manuscript registration workflow |

---

## Architecture — Three Workers

```
User query
    │
    ▼
Worker 2 — Agent 1: Query Understanding  (Stages 0.5 – 3)
  intent_router → bilingual_analyzer → hyde → bilingual_expand → self_query
    │  (QueryContext)
    ▼
Worker 3 — Agent 2: Retrieval & Synthesis  (Stages 4 – 10)
  retrieve → rrf_fuse → resolve_heritage → crag_grader
  → synthesise → reflect → format_variants
    │  (final_response)
    ▼
Streamlit UI  (app/streamlit_app.py)

Worker 1 — Al-Nassikh: Deterministic ETL  (offline / on ingest)
  PAGE-XML → parser → cross_link → registry_join → embed → Qdrant
```

The orchestrator (`src/fatat_al_arab/orchestrator.py`) is the single public entry point that wires the three workers together.

---

## Agent Personas

Each worker has a named persona that is injected as the system-prompt header for every LLM call it makes. Defined in [`src/fatat_al_arab/personas.py`](src/fatat_al_arab/personas.py).

### Al-Nassikh · الناسخ · The Scribe

> *"A meticulous historical scribe who catalogues, cross-references, and counts with archival precision. Never guesses; always cites the registry."*

**Worker:** 1 (deterministic ETL, offline ingestion, registry-lookup answers)  
**Activated by:** counting, provenance, and dating queries (< 2 ms, no LLM generation)  
**Prompt role:** concise and exact — cites the registry field name; never speculates  
**Used in:** `deterministic_answer.py` · `corpus_stats.py` · `intent_router.py`

```
You are Al-Nassikh (الناسخ — The Scribe), NABAT-AI's archival metadata agent.
Your role: answer counting, dating, and provenance questions directly from the
manuscript registry. Be concise and exact. Cite the registry field name when
reporting a number or date. Never speculate beyond what the registry contains.
```

### Fatat Al-Arab · فتاة العرب · The Arabian Scholar

> *"A bilingual Khaleeji Nabati poetry scholar who grew up reading manuscript folios and speaks with measured scholarly precision. She quotes verses faithfully, respects dialectal register, and cites her sources by manuscript name and folio number. She never fabricates a verse or a poet."*

**Workers:** 2 + 3 (Agent 1 query understanding, Agent 2 retrieval & synthesis)  
**Activated by:** all thematic, semantic, and exploratory poetry queries  
**Prompt role:** bilingual (AR + EN), grounded citations mandatory, dialectal precision  
**Used in:** `self_query.py` · `crag_grader.py` · `synthesise.py` · `reflect.py`

```
You are Fatat Al-Arab (فتاة العرب — The Arabian Scholar), NABAT-AI's bilingual
Khaleeji Nabati poetry expert. You were raised on Gulf manuscript folios and speak
with scholarly precision in both Arabic and English. You quote verses faithfully
from the source texts, respect Khaleeji dialectal register, and always cite by
manuscript name and folio. You never fabricate a verse, a poet, or a manuscript
reference.
```

---

## Quick Start

### Prerequisites

- Python 3.10 +
- One free API key — [Together.ai](https://api.together.xyz) **or** [Groq](https://console.groq.com) (sign up takes ~2 minutes, no credit card)
- `tesseract` with Arabic language pack (only needed for image-based queries)

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ara
```

### Install & run

```bash
# 1. Clone
git clone https://github.com/<your-username>/handwritten-poems.git
cd handwritten-poems

# 2. Install Python dependencies  (no GPU needed)
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Open .env and set:
#   LLM_PROVIDER=together   (or groq)
#   LLM_API_KEY=<your key>

# 4. Build the vector index  (once — ~30 s on first run)
PYTHONPATH=src python scripts/rebuild_index.py \
  --registry data/ground_truth/anchor_registry_full_enriched.json \
  --force

# 5. Launch the app
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Offline / no-key mode

Set `LLM_PROVIDER=stub` in `.env` — the app runs with canned responses so you can inspect the UI layout without an internet connection.

---

## API Key Options

| Provider | Where to get it | Notes |
|---|---|---|
| **Together.ai** | [api.together.xyz](https://api.together.xyz) → Sign up → API Keys | Free tier · ~60 req/min · **Primary** |
| **Groq** | [console.groq.com](https://console.groq.com) → API Keys | Free tier · fast · Good backup |
| **stub** | None needed | Offline demo — canned responses only |

LLM models used: **Qwen2.5-7B-Instruct-Turbo** (primary) · **Mistral-7B-Instruct** (fallback).  
Embeddings run **locally** via AraBERT — no additional key needed.

---

## Project Structure

```
handwritten-poems/
├── app/
│   └── streamlit_app.py          Main UI — Tab A: Scholar Workbench · Tab B: Archive Manager
├── src/
│   ├── al_nassikh/               Worker 1 — Deterministic ETL & operators
│   │   ├── corpus_stats.py       Aggregations (poem/poet counts)
│   │   ├── genre_heuristic.py    Silver-baseline genre + emotion classifier
│   │   ├── nabati_taxonomy.py    Frozen label space (10 genres · 10 emotions)
│   │   ├── registry.py           Manuscript registry loader
│   │   └── operator/             8-challenge HITL quality operators
│   └── fatat_al_arab/            Workers 2 + 3 — RAG Pipeline
│       ├── orchestrator.py       Single public entry point
│       ├── llm.py                Hosted-API adapter (Together / Groq / stub)
│       ├── embed.py              GATE-AraBERT 768-dim dense embeddings
│       ├── index.py              File-backed Qdrant index builder (9 chunk levels)
│       ├── agent1_query_understanding/   Stages 0.5 – 3
│       ├── agent2_retrieval_synthesis/   Stages 4 – 10
│       └── retrievers/           BM25 · Dense · ColBERT
├── data/ground_truth/
│   ├── anchor_registry_full_enriched.json   2,175 manuscript anchors (Phase 1–4)
│   ├── manuscript_registry.json             25 manuscripts with AR/EN names
│   └── poets_bio.json                       Poet biographical entries
├── data/unified_registry.json    4,031 entries (manuscripts + oral + online)
├── data/qdrant/                  File-backed vector index (8,415 chunks)
├── scripts/
│   ├── rebuild_index.py          Build / rebuild the Qdrant index
│   ├── enrich_genre_heuristic.py Run genre classifier over all anchors
│   └── evaluate.py               M10 evaluation harness (4 axes)
├── tests/                        14 test files · ~4,750 lines
├── infra/escriptorium/           Self-hosted eScriptorium (optional)
├── requirements.txt
└── .env.example
```

---

## Corpus

| Phase | Manuscripts | Pages | Anchors | Status |
|---|---|---|---|---|
| Phase 1 — Momentum | ms07, ms14, ms15, ms22 | 20 | 344 | Vectorized |
| Phase 2 — Core Sadr/Ajuz | ms04, ms05, ms19, ms21 | 12 | 240 | Vectorized |
| Phase 3 — Edge Cases | ms01, ms03, ms06, ms08 | 8 | 136 | Vectorized |
| Phase 4 — TOC Metadata | vols 001–782 | 36 | 1,502 | Complete |
| **Combined** | **25 manuscripts** | **76** | **2,175** | **All in Qdrant** |
| **+ Oral tradition** | MAAI7103 | — | **106** | **All in Qdrant** |
| **+ Online digitized** | — | — | **1,750** | **All in Qdrant** |
| **Total** | — | — | **4,031** | **All in Qdrant** |

The vector index holds **8,415 chunks** at 9 granularity levels (verse → group → poem → manuscript → poet → era → genre → emotion → reference).

---

## UI Overview

**Tab A — Scholar Workbench**

- Single bilingual query box (Arabic or English)
- Genre + emotion sidebar filters
- Deterministic answer badge (🗂️) for registry-backed facts
- Silver-baseline genre tag badge (🔸) on heuristic-tagged results
- Conversation history (last 5 turns)
- Image upload and microphone input

**Tab B — Archive Manager & Contributor Guide**

- Metadata schema reference
- Manuscript registration guide
- Step-by-step contribution workflow
- Operator pipeline tools (triage / bleed suppress / standardise)
- One-click index rebuild button
- Gracefully degrades when eScriptorium Docker container is not running

---

## Running the Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

Offline stub mode is the default for all tests — no API key required.

```bash
# Evaluation harness (4 axes: correctness · robustness · efficiency · human)
LLM_PROVIDER=stub PYTHONPATH=src python scripts/evaluate.py
```

---

## Optional: eScriptorium (Tab B Archive Manager)

eScriptorium is a self-hosted HTR (Handwritten Text Recognition) platform. It is **not required** to run the Scholar Workbench — Tab A works fully without it.

```bash
cd infra/escriptorium
docker compose up -d
```

Then set `ESCR_BASE_URL` and `ESCR_API_TOKEN` in your `.env`. See `infra/escriptorium/README.md` for full instructions.

---

## Documentation

| Document | Purpose |
|---|---|
| [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) | Iterative development process — prompt engineering, debugging, AI-assisted decisions |
| [`CLAUDE.md`](CLAUDE.md) | Full architecture reference, directory structure, and guiding design decisions |

---


## Key Design Decisions

1. **Hosted-API LLM only.** No CUDA / Ollama / vLLM. Every LLM call routes through `src/fatat_al_arab/llm.py` with 3 retries and exponential backoff.
2. **Portable vector store.** Qdrant runs in file-backed mode — no separate server process needed.
3. **Silver-baseline genre tagging.** Without a human-labelled gold set, accuracy cannot be claimed. All heuristic-tagged results show a 🔸 badge. Coverage is 82.9 % on the manuscript entries.
4. **Graceful degradation throughout.** Every heavy dependency is a lazy import with a fallback so the app remains runnable even when optional components are missing.
5. **Three-worker architecture.** Al-Nassikh (ETL), Agent 1 (query understanding), Agent 2 (retrieval + synthesis) are independent workers wired by the orchestrator — mirrors the architecture document section by section.

---

## Completed Milestones

| Milestone | Description |
|---|---|
| M1 | Phase 1-3 ↔ Phase 4 cross-link; manuscript registry |
| M2c | Intent router + deterministic answer path (< 2 ms) |
| M2d | Genre + emotion silver-baseline classifier |
| M3 | Genre + emotions in Qdrant payload + self-query filters |
| M8 | Streamlit UI: unified Scholar tab + Contributor Guide tab |
| M9 | CRAG re-query (LLM-directed) + Self-RAG retry (surgical fix instructions) |

---

## License

This project is submitted as a graded deliverable for course MAAI1704 at MBZUAI. All manuscript content belongs to its respective rights holders.
