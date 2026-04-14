# NABAT-AI — Khaleeji Nabati Poetry Digitization & RAG System

## Project Overview
AI system to digitize handwritten Khaleeji Nabati poetry manuscripts and expose them via a RAG agent.
**Course:** MAAI1704 – Generative AI | **Student:** Asma Salem Mubarak Najem Aljneibi

## Structure
```
nabat-ai/
├── src/
│   ├── al_nassikh/       # Worker 1: PAGE-XML parser → structured JSON
│   │   └── parser.py     # Run: python -m src.al_nassikh.parser
│   └── fatat_al_arab/    # Worker 2: RAG pipeline
│       └── rag.py        # Run: python -m src.fatat_al_arab.rag
├── app/
│   └── index.html        # Frontend demo app (open in browser or deploy to GitHub Pages)
├── data/
│   ├── ground_truth/     # Al-Nassikh pipeline outputs (JSON)
│   └── demo_query_results.json
├── notebooks/
│   └── nabat_ai_demo.ipynb  # End-to-end demo
├── manuscripts/
│   ├── Ground_Truth_Exports/  # eScriptorium PAGE-XML exports
│   │   ├── export_doc5_phase_1_pagexml_*/  # Phase 1: 20 body pages
│   │   └── export_doc9_phase_4_pagexml_*/  # Phase 4: 4 TOC pages
│   ├── MVP_Ground_Truth_Images/  # Extracted page images per phase
│   ├── escriptorium/     # eScriptorium Docker setup
│   ├── manuscript*.pdf   # 23 raw manuscript PDFs
│   └── *.pdf             # Sowayan archive volumes (001-200, 201-400, etc.)
└── doc/                  # Deliverables and architecture documents
```

## Two-Worker Pipeline
- **Al-Nassikh (الناسخ)** — Deterministic ETL: PAGE-XML → multi-variant JSON (manuscript/standard/dialectal)
- **Fatat Al Arab (فتاة العرب)** — LangGraph RAG agent: chunk → embed → Qdrant → hybrid retrieval → cite

## Ground Truth Status
| Phase | Manuscripts | Pages | Status |
|---|---|---|---|
| Phase 1 — Momentum | ms07, ms14, ms15, ms22 | 20 | ✅ Complete |
| Phase 4 — TOC Metadata | 601-782 | 4 | 🔄 In Progress |
| Phase 2 — Core Sadr/Ajuz | ms04, ms05, ms19, ms21 | 15 | 📋 Planned |
| Phase 3 — Edge Cases | ms01, ms03, ms06, ms08 | 10 | 📋 Planned |

## Key Architecture Decisions
- Stanza-aware chunking: 3 levels (verse بيت / group مجموعة / full poem قصيدة)
- Multi-variant transcription: manuscript + MSA standard + Khaleeji dialectal per verse
- Anchor-based CER verification: HIGH (<10%) / MEDIUM (10-25%) / LOW (≥25%)
- BM25 + GATE-AraBERT-v1 hybrid search with RRF fusion
- Mandatory citation injection on every result (poet|volume|page|image)

## Run the Pipeline
```bash
cd src
python -m al_nassikh.parser          # Parse exports → data/ground_truth/
python -m fatat_al_arab.rag          # Build RAG + run demo queries
```

## Deploy
```bash
# GitHub Pages: push app/ to gh-pages branch
# Local: open app/index.html in browser
```
