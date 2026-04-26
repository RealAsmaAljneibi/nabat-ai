# Reflection on AI Usage — NABAT-AI

**Student:** Asma Salem Mubarak Najem Aljneibi · **Course:** MAAI1704  
**Date:** 2026-04-26

---

## 1. How AI Tools Were Used

### Code generation (Claude, ChatGPT, GitHub Copilot)

AI tools were used in three distinct modes throughout the project:

**Mode A — Architecture consultation:** Early in the project I described the problem (handwritten Khaleeji poetry, 2,222 verse anchors, bilingual users) and asked AI tools to suggest architectural patterns. The AI's initial suggestion (basic FAISS + LLM pipeline) was a useful starting point but had to be significantly extended — it didn't account for the fact that many user queries are factual (counting, provenance) rather than thematic, and the single-pipeline approach returned confusing refusals for those cases.

**Mode B — Prompt drafting and iteration:** Almost every LLM prompt in the system went through multiple iterations. I would generate a first draft using AI assistance, test it against real queries, identify failure modes, and revise. The CRAG grader prompt went through three major iterations (see `DEVELOPMENT_LOG.md`); the HyDE prompt went through three. In most cases the AI-generated first draft required substantial manual revision.

**Mode C — Boilerplate and test fixtures:** TypedDict definitions (`state.py`), test fixture structures, and `requirements.txt` dependency lists were generated with AI assistance and verified manually. These required less iteration because they are structurally simple.

---

## 2. What Worked Well

### 2a. Structured JSON output for LLM nodes

Asking the LLM to return JSON rather than prose was consistently the right call. The `bilingual_analyzer`, `self_query`, `crag_grader`, and `reflect` nodes all use JSON-schema constrained output. AI assistance helped draft the initial schemas; the key insight (mine, not the AI's) was that every schema needed a `confidence` float rather than just a label, because downstream logic needs to threshold on confidence (e.g., `intent_confidence < 0.5` → clarification path).

### 2b. LangGraph for conditional control flow

The AI helped scaffold the LangGraph `StateGraph` objects for both agents. The conditional edge pattern (`add_conditional_edges`) was well-explained in AI consultation and maps cleanly to the §2.4/§2.5 pipeline described in the architecture document. The agentic loops (CRAG re-query, Self-RAG retry) were easy to express as LangGraph conditions once I understood the state-passing model.

### 2c. Regex pattern generation for Arabic

When building the intent_router's bilingual regex patterns, I asked AI tools to suggest Arabic equivalents of English cue phrases. This was genuinely useful for MSA variants. However, the Khaleeji-specific forms (e.g., "شو عدد" instead of "كم عدد") were either absent or incorrect in AI suggestions — I had to add these manually after testing with native-speaker review.

### 2d. Error message quality

AI assistance improved error messages significantly. Early versions of nodes had bare `logger.warning("failed")` calls; the AI suggested including the exception type, the input that caused it, and what fallback was taken. The "§5 fallback: proceed with Ambiguous" pattern in `crag_grader.py` was AI-suggested and is now consistent across all nodes.

---

## 3. What Failed / Required Human Correction

### 3a. The "one-shot complete system" failure

My first attempt was to describe the entire system to Claude and ask it to generate a complete implementation. The result was a single 800-line Python file with all nodes inlined, no separation of agent concerns, and a hardcoded FAISS index path. It was not modular, not testable in isolation, and didn't match the three-worker architecture required by the deliverable.

**Resolution:** I scrapped the one-shot output and designed the directory structure manually first (`agent1_query_understanding/`, `agent2_retrieval_synthesis/`, `al_nassikh/`), then used AI for individual components within that structure.

**Learning:** AI code generation works well at the function or module level. At the full-system level it collapses concerns and ignores architectural constraints unless those constraints are stated very precisely in each sub-prompt.

### 3b. Arabic dialect errors in generated patterns

AI-suggested Khaleeji Arabic regex patterns had several errors:
- Used MSA negation (لا, لم) where Khaleeji uses (ما, مو)  
- Missed common Khaleeji discourse markers (بعد، فدوة، يبه)
- Incorrectly suggested `وين` as MSA (it's Khaleeji/Iraqi dialect)

**Resolution:** All Arabic patterns in `intent_router.py` were reviewed and corrected with dialect reference materials. The genre heuristic patterns (`genre_heuristic.py`) were spot-checked against actual corpus verses.

### 3c. Missing failure-handling in generated agentic loops

The first AI-generated version of the CRAG re-query loop had no cap on the number of re-queries. Testing showed it would loop indefinitely when the re-query strategy produced queries that retrieved equally poor results. The AI did not spontaneously add a counter guard.

**Resolution:** I added `CRAG_REQUERY_MAX = 1` (named constant, not magic literal) and the `crag_requery_count` counter in state. The `reflect` node had a similar issue — I added `SELF_RAG_MAX_RETRIES = 2` after the same pattern.

### 3d. ColBERT was over-promised

An early architecture consultation suggested ColBERT late-interaction retrieval as a third retriever. The AI made it sound straightforward to add. In practice, ColBERT requires a dedicated server or a substantial local index; neither was compatible with the "runs everywhere with pip" portability requirement.

**Resolution:** `retrievers/colbert.py` is a working stub that returns empty results by design. The dense + BM25 combination provides adequate retrieval quality without the infrastructure requirement. The stub is clearly documented; the "triple hybrid" claim in the architecture doc is accurate at the interface level even though ColBERT contributes empty results in the current deployment.

---

## 4. Limitations Observed

### 4a. LLM latency vs. retrieval quality trade-off

Every additional LLM call in the pipeline adds 400–1200ms of latency depending on provider load. The decision to add HyDE (Stage 2), bilingual expansion (Stage 2b), CRAG grading (Stage 7), and Self-RAG reflection (Stage 9) was architecturally sound but means the full pipeline takes 5–8 seconds end-to-end. For a production system, at least HyDE and bilingual expansion should be parallelised (they are independent). This was not implemented due to time constraints.

### 4b. Genre heuristic accuracy is not measurable without gold labels

The genre classifier covers 82.9% of the corpus. But coverage ≠ accuracy: we have no human-labelled gold set to measure false positives. The 🔸 badge on all heuristic-tagged results is intentional to signal this uncertainty. AI tools cannot substitute for domain-expert annotation here.

### 4c. Multi-turn memory is session-local

Conversation history is stored in `st.session_state` and passed to the pipeline per-turn. When the browser session ends, all history is lost. A returning researcher cannot resume a previous exploration. This is a genuine limitation; a production system would need a persistent session store.

### 4d. eScriptorium ingestion requires Docker

The full ingest pipeline (PDF → Kraken HTR → PAGE-XML → anchors → Qdrant) requires eScriptorium running in Docker. The graded demo can run without it (the index is pre-built), but adding a new manuscript requires Docker. This limits portability for the demo panel.

---

## 5. Improvements Suggested

### 5a. Persistent cross-session memory

Add a lightweight SQLite or JSON store for conversation sessions. Returning users could resume previous exploration threads. The state contract (`state.py`) already has `conversation_id` — persisting this to disk is a 2-hour implementation.

### 5b. Parallelise Agent 1 LLM calls

HyDE (Stage 2a) and bilingual expansion (Stage 2b) are independent. Running them in parallel with `asyncio.gather` would cut ~600ms from the median query latency. The `ThreadPoolExecutor` approach already used for the HyDE timeout could be extended to both stages.

### 5c. Human-in-the-loop genre labelling

Even labelling 200 verses from the corpus (10% sample) would enable proper precision/recall metrics for the genre heuristic and allow a fine-tuned classifier. The `qa_jury.py` operator already implements a confidence routing scaffold — the missing piece is a simple annotation UI.

### 5d. Real ColBERT retrieval

Replace the ColBERT stub with `pylate` or a small ColBERT checkpoint. This would genuinely diversify the retrieval signal beyond dense cosine similarity, which is what the architecture intended.

### 5e. Streaming responses in Streamlit

For long synthesised answers, streaming the LLM output word-by-word (using Together.ai's streaming API) would reduce perceived latency significantly. The LLM adapter (`llm.py`) does not currently support streaming; adding it would require extending `chat()` with a `stream=True` flag.

---

## Summary

AI tools accelerated the development of NABAT-AI significantly, particularly for:
- Scaffolding LangGraph graph structures
- Drafting LLM system prompts (which then required manual iteration)
- Generating TypedDict state contracts
- Boilerplate test fixtures

The most important contributions were mine, not the AI's:
- The three-worker architecture split (matching the deliverable requirement)
- The `requery_strategy` field that makes CRAG re-queries LLM-directed
- The `fix_instructions` field that makes Self-RAG retries surgical
- The confidence-weighted intent router scoring (0.7 threshold)
- The Arabic dialect corrections in all bilingual regex patterns
- The refusal filter in conversation history formatting
- The CRAG and Self-RAG failure budget caps

The pattern throughout: AI provides a fast, competent first draft. Turning that draft into a production-quality, architecturally-aligned, domain-correct module required substantive manual work.
