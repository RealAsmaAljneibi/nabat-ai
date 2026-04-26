# NABAT-AI — Khaleeji Nabati Poetry Digitization & RAG System

> **Course:** MAAI1704 – Generative AI  
> **Student:** Asma Salem Mubarak Najem Aljneibi  
> **Deliverable:** GenAI MAAI1704 Project — Runs without Docker, GPU, or paid APIs.

## Grading Rubric

The assessment rubric is stored at **`doc/RUBRIC.md`** for easy reference during evaluation. Each criterion links directly to the evidence in this repo:

| Criterion | Max | Where to find the evidence |
|---|---|---|
| Alignment with Proposal & Architecture | 10 | [`doc/TRACEABILITY.md`](doc/TRACEABILITY.md) · [`doc/IMPLEMENTATION_PLAN.md`](doc/IMPLEMENTATION_PLAN.md) · architecture §2.4/§2.5 mirrors `agent1_query_understanding/` + `agent2_retrieval_synthesis/` |
| Agentic System Realization | 20 | [`src/fatat_al_arab/orchestrator.py`](src/fatat_al_arab/orchestrator.py) · [`agent1_query_understanding/graph.py`](src/fatat_al_arab/agent1_query_understanding/graph.py) · [`agent2_retrieval_synthesis/graph.py`](src/fatat_al_arab/agent2_retrieval_synthesis/graph.py) |
| AI-Assisted Development Process | 15 | [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) — prompt iterations, debugging stories, code modifications |
| Code Quality & Modularity | 10 | `src/` directory structure · every module docstring · `.env.example` |
| System Integration | 15 | `data/qdrant/` (pre-built real index) · `app/streamlit_app.py` · `scripts/rebuild_index.py` |
| Memory, Tools & RAG | 8 | `data/qdrant/` (long-term vector store) · `bilingual_analyzer.py` (conversation history) · `retrievers/` |
| Evaluation & Reliability | 7 | `tests/` (15 files · 480+ test cases including edge cases in [`tests/test_edge_cases.py`](tests/test_edge_cases.py)) · `scripts/evaluate.py` |
| Understanding & Ownership | 10 | [`doc/TRACEABILITY.md`](doc/TRACEABILITY.md) · [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) · inline `Why:` comments throughout |
| Reflection on AI Usage | 5 | [`doc/AI_USAGE_REFLECTION.md`](doc/AI_USAGE_REFLECTION.md) |

An AI system that digitises handwritten Khaleeji Nabati poetry manuscripts and exposes them through a bilingual (Arabic / English) Retrieval-Augmented Generation agent. The full Scholar interface runs in a single Streamlit tab — no GPU, no Docker, no paid-tier API required.

---

## What it does

| Capability | Detail |
|---|---|
| **Manuscript digitisation** | 25 manuscripts · 76 pages · 2,222 verse anchors across 4 phases |
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
│   ├── anchor_registry_full_enriched.json   2,222 verse anchors (canonical)
│   ├── manuscript_registry.json             25 manuscripts with AR/EN names
│   └── poets_bio.json                       509 poet biographical entries
├── data/qdrant/                  File-backed vector index (4,747 chunks)
├── scripts/
│   ├── rebuild_index.py          Build / rebuild the Qdrant index
│   ├── enrich_genre_heuristic.py Run genre classifier over all anchors
│   └── evaluate.py               M10 evaluation harness (4 axes)
├── tests/                        12 test files · ~3,800 lines
├── doc/
│   ├── IMPLEMENTATION_PLAN.md
│   └── ARCHITECTURE_DIAGRAMS.md
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
| **Combined** | **25 manuscripts** | **76** | **2,222** | **All in Qdrant** |

The vector index holds **4,747 chunks** at 9 granularity levels (verse → group → poem → manuscript → poet → era → genre → emotion → reference).

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
| [`doc/AI_USAGE_REFLECTION.md`](doc/AI_USAGE_REFLECTION.md) | What worked / failed with code generators, limitations, improvements |
| [`doc/TRACEABILITY.md`](doc/TRACEABILITY.md) | Use-case → Architecture → Code mapping for all four personas |
| [`doc/IMPLEMENTATION_PLAN.md`](doc/IMPLEMENTATION_PLAN.md) | Build plan v2 — all milestones M0–M11 |
| [`doc/ARCHITECTURE_DIAGRAMS.md`](doc/ARCHITECTURE_DIAGRAMS.md) | Mermaid source for all pipeline diagrams |
| [`doc/DATA_PIPELINE.md`](doc/DATA_PIPELINE.md) | ETL workflow — 4-phase corpus strategy |

---


## Key Design Decisions

1. **Hosted-API LLM only.** No CUDA / Ollama / vLLM. Every LLM call routes through `src/fatat_al_arab/llm.py` with 3 retries and exponential backoff.
2. **Portable vector store.** Qdrant runs in file-backed mode — no separate server process needed.
3. **Silver-baseline genre tagging.** Without a human-labelled gold set, accuracy cannot be claimed. All heuristic-tagged results show a 🔸 badge. Coverage is 82.9 % on 2,222 entries.
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
