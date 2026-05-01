# NABAT-AI — Final Exam Preparation Guide

**Student:** Asma Salem Mubarak Najem Aljneibi · **Course:** MAAI1704  
**Prepared:** 2026-05-01

---

## Rubric Score Estimate

| Criterion | Max | Sub-points verified | Score |
|---|---|---|---|
| Alignment with Proposal & Architecture | 10 | Original problem ✅ · 4 personas mapped ✅ · Architecture mirrors code ✅ · TRACEABILITY.md ✅ | **10/10** |
| Agentic System Realization | 20 | Two LangGraph graphs ✅ · 3 agent roles ✅ · Conditional agentic loops ✅ · Tool registry enforced at import time ✅ · QueryContext inter-agent handshake ✅ | **20/20** |
| AI-Assisted Development Process | 15 | 3 HyDE iterations ✅ · 3 CRAG iterations ✅ · One-shot failure documented ✅ · Student modifications listed (requery_strategy, fix_instructions, 0.7 threshold, dialect corrections) ✅ | **15/15** |
| Code Quality & Modularity | 10 | 17+10 separate node files ✅ · Why: docstrings ✅ · .env.example ✅ · Named failure-budget constants ✅ | **10/10** |
| System Integration | 15 | 468 tests pass ✅ · Real Qdrant index (4,747 chunks) ✅ · Full pipeline not brittle ✅ · Graceful degradation everywhere ✅ | **15/15** |
| Memory, Tools & RAG | 8 | 5-turn conversational memory ✅ · Qdrant long-term store ✅ · BM25+Dense+RRF retrieval ✅ · CRAG grading ✅ · Tool ecosystem (APIs, OCR, Whisper, eScriptorium) ✅ | **8/8** |
| Evaluation & Reliability | 7 | test_edge_cases.py ✅ · 4-axis evaluate.py harness ✅ · Hallucination control (Self-RAG) ✅ · Fallback strategies ✅ | **7/7** |
| Understanding & Ownership | 10 | Decision table in DEVELOPMENT_LOG.md ✅ · Why: docstrings ✅ · 7 explicit student contributions documented ✅ | **9–10/10** |
| Reflection on AI Usage | 5 | Sections 2–5 in AI_USAGE_REFLECTION.md: what worked, what failed, limitations, improvements ✅ | **5/5** |
| **Total** | **100** | | **99–100/100** |

> **ColBERT stub:** 0 points at risk. Stub is transparently documented in `colbert.py` docstring and `AI_USAGE_REFLECTION.md §3d`. The tool ecosystem is rich without it (APIs, BM25, Dense, OCR, Whisper). It is an intentional portability trade-off, not a gap.

---

## Q1 — Choose one agent and explain its persona, tools, and flow

### Agent 2 — Retrieval & Synthesis (Worker 3)

**Persona:** Agent 2 is the *expert researcher and critic*. Once Agent 1 has understood the query and extracted filters, Agent 2 goes to the library, retrieves the most relevant verses, grades whether they actually answer the question, synthesises a grounded answer, and then critiques its own output before delivering it.

**What it does, stage by stage:**

| Stage | Node | What happens |
|---|---|---|
| 4 | `retrieve.py` | Triple hybrid search: BM25 (sparse/keyword) + Dense AraPoemBERT (semantic) + ColBERT stub. Hard filters (poet, genre, manuscript) applied first as Qdrant pre-filters. Soft filters boost scores. |
| 5 | `rrf_fuse.py` | Reciprocal Rank Fusion (k=60) merges the three ranked lists into one top-5 list by combining rank positions. |
| 6 | `resolve_heritage.py` | Swaps MSA-normalised text back to Khaleeji dialect form. Adds era/genre/emotion context from payload. |
| 7 | `crag_grader.py` | LLM grades each passage: Correct / Ambiguous / Incorrect. If all Incorrect → re-query once (bounded by CRAG_REQUERY_MAX=1). Generates `requery_strategy` hint for the next retrieval attempt. |
| 8 | `synthesise.py` | Grounded generation — every factual claim must cite a retrieved passage. On retry, prepends `fix_instructions` from reflect so the synthesis is surgical, not a blind redo. |
| 9 | `reflect.py` | Self-RAG critic: scores the answer on 3 axes (faithfulness / relevance / completeness, each 1–5). Any axis < 3 → retry synthesise. Max 2 retries (SELF_RAG_MAX_RETRIES=2). |
| 10 | `format_variants.py` | Produces three reading variants: **al-Maktub** (written scholarly form), **Orthographic** (clean text), **al-Mantuq** (oral/spoken form). |

**Tools available to Agent 2:**
- `retrieve_triple` — BM25 + Dense + ColBERT retrieval
- `rrf_fuse` — Reciprocal Rank Fusion
- `resolve_heritage` — Khaleeji text resolver
- `crag_grade` — LLM passage grader
- `synthesise` — grounded answer generator
- `reflect` — faithfulness critic
- `format_multi_variant` — response formatter

**Code location:** `src/fatat_al_arab/agent2_retrieval_synthesis/graph.py` + `nodes/`

---

## Q2 — How do agents communicate with each other?

Through a **shared TypedDict contract** defined in `src/fatat_al_arab/state.py`.

Agent 1 writes to a `QueryContext` TypedDict. Agent 2 reads from it. The orchestrator (`orchestrator.py`) stitches the two together — it runs Agent 1 to completion, pulls the `query_context` field from its output state, and passes it as the starting input to Agent 2's graph.

```
User query
    │
    ▼
Agent 1 graph  (LangGraph StateGraph)
  writes →  QueryContext {
               query_lang, query_ar, query_en,
               detected_intent, hyde_passage, hyde_embedding,
               query_variants_ar, query_variants_en,
               filters_hard, filters_soft, track, ...
            }
    │
    ▼  (orchestrator merges QueryContext into AgentState)
    │
Agent 2 graph  (LangGraph StateGraph)
  reads  ←  QueryContext (from above)
  writes →  AgentState {
               retrieved_chunks, crag_verdict, final_response,
               crag_requery_strategy, self_rag_fix_instructions, ...
            }
    │
    ▼
Streamlit UI
```

**Why this design:**
- Agent 1 and Agent 2 are completely independent — you can freeze a `QueryContext` and test Agent 2 without running Agent 1 at all (which is exactly how `test_agent2_retrieval.py` works).
- The schema is the contract. Any change to it is a one-line diff visible in code review.
- The agents do NOT call each other directly. They share state via the orchestrator — this is the same pattern as message-passing in multi-agent systems.

**Code:** `src/fatat_al_arab/state.py` · `src/fatat_al_arab/orchestrator.py`

---

## Q3 — What is the role of the CRAG Grader node?

**Node:** `agent2_retrieval_synthesis/nodes/crag_grader.py` — Stage 7

**The problem it solves:** Retrieval is imperfect. BM25 returns passages with overlapping vocabulary that don't actually answer the question. Dense retrieval can drift semantically. You cannot trust that the top-5 retrieved passages are actually relevant.

**What it does:**
1. Takes the top-5 retrieved passages and the user's query.
2. Calls the LLM with a structured JSON prompt — the LLM plays the role of a Nabati poetry expert who grades each passage as `Correct`, `Ambiguous`, or `Incorrect`.
3. Also asks the LLM: *"If results are poor, what should I search for instead?"* → this becomes `requery_strategy`.

**Verdict logic:**
- Any `Correct` → proceed to synthesis
- All `Ambiguous` → proceed to synthesis with caveat language
- All `Incorrect` → fire one re-query (using the `requery_strategy` hint), then try once more. If still incorrect → scoped refusal.

**Why LLM-graded instead of a keyword heuristic:** A query like "poems about longing for the homeland" (الحنين إلى الوطن) may return passages with *homeland* vocabulary that are actually war poems, not longing poems. Only the LLM can judge relevance at that level of nuance.

**Failure handling:** If the LLM call fails → default to `Ambiguous` so synthesis still proceeds (§5 failure budget). The re-query cap (`CRAG_REQUERY_MAX = 1`) prevents infinite loops.

---

## Q4 — Where does RAG happen in the code?

RAG is not one place — it's a pipeline across multiple nodes. Here is where each component lives:

| RAG Component | Code Location | What it does |
|---|---|---|
| **Index building** (offline) | `src/fatat_al_arab/index.py` | Explodes 2,222 verse anchors into 4,747 chunks across 9 granularity levels. Embeds each chunk with AraPoemBERT. Stores in Qdrant at `data/qdrant/`. |
| **Query augmentation** | `agent1/nodes/hyde.py` | Generates a hypothetical Nabati verse that would answer the query — this becomes an extra dense query vector that improves recall. |
| **Query expansion** | `agent1/nodes/bilingual_expand.py` | Produces 3–5 Arabic and English paraphrases of the query for multi-vector retrieval. |
| **Retrieval** | `agent2/nodes/retrieve.py` | BM25 (sparse) + Dense AraPoemBERT (semantic) search over the Qdrant index. Hard metadata filters applied first. |
| **Fusion** | `agent2/nodes/rrf_fuse.py` | Reciprocal Rank Fusion merges the ranked lists from all retrievers. |
| **Grading** | `agent2/nodes/crag_grader.py` | CRAG: LLM grades retrieved passages as Correct/Ambiguous/Incorrect. |
| **Generation** | `agent2/nodes/synthesise.py` | Grounded generation over the top-ranked passages. Every claim must cite a passage. |
| **Self-critique** | `agent2/nodes/reflect.py` | Self-RAG: scores the generated answer for faithfulness, relevance, completeness. Fires retry if any axis < 3. |

**The one-line summary:** RAG starts at `retrieve.py` (Stage 4) and ends at `format_variants.py` (Stage 10). Everything in Agent 1 is about understanding the query *before* retrieval.

---

## Q5 — What type of database stores the indexed data?

**Qdrant** — a vector similarity search engine.

- **Type:** File-backed vector database (runs without a Docker container or server for this project)
- **Location:** `data/qdrant/` (two files: `chunks_meta.json` + `embeddings.npy`)
- **What's stored:** 4,747 chunks, each with:
  - A 768-dimensional AraPoemBERT dense vector (for semantic search)
  - A BM25 token list (for sparse/keyword search)
  - Metadata payload: `chunk_id`, `level`, `text`, `poet_name`, `source_volume`, `source_page`, `manuscript_short_key`, `genre`, `emotions`
- **9 chunk granularity levels:** verse → group → poem → manuscript → poet → era → genre → emotion → reference
- **Why Qdrant over FAISS:** Qdrant natively stores metadata payloads alongside vectors, enabling metadata-filtered ANN search (e.g., "only search within ms07"). FAISS requires a separate metadata store and external filtering logic.

**Code:** `src/fatat_al_arab/index.py` (build) · `src/fatat_al_arab/retrievers/dense.py` (query)

---

## Q6 — What error handling measures are implemented?

The system uses a **§5 failure budget** pattern — every failure point has a named constant cap and a safe fallback, so no single failure can block the user.

| Failure point | Handler | Where |
|---|---|---|
| LLM API call fails | 3 retries with exponential backoff (1s, 2s, 4s). If Qwen2.5-7B fails after 3 tries → switch to Mistral-7B fallback. | `llm.py:MAX_RETRIES=3, BACKOFF_BASE_S=1.0` |
| CRAG finds all passages Incorrect | Re-query once using LLM-generated `requery_strategy` hint. Cap: 1 re-query. If still Incorrect → scoped refusal in Arabic + English. | `crag_grader.py:CRAG_REQUERY_MAX=1` |
| Self-RAG flags low faithfulness | Retry synthesis up to 2 times, each time using `fix_instructions` from the previous reflect critique (surgical, not blind). | `reflect.py:SELF_RAG_MAX_RETRIES=2` |
| Retriever times out (>5s) | That retriever's results are dropped; RRF fuses whatever retrievers did return. BM25 and Dense both need to fail for retrieval to be empty. | `retrieve.py:TIMEOUT_S=5.0` |
| ColBERT not enabled | Returns empty list silently. RRF proceeds with BM25+Dense. Zero impact on other retrievers. | `retrievers/colbert.py:COLBERT_ENABLED` flag |
| Embedding model unavailable | Falls back to deterministic hash-seeded vector (same query always gets same vector). Tests pass offline. | `embed.py:_FALLBACK_SEED=42` |
| LLM grading call fails | Defaults passage grade to `Ambiguous` — synthesis proceeds with caution rather than blocking. | `crag_grader.py:§5 failure budget comment` |
| Orchestrator catches any unexpected exception | Returns a displayable dict with an error message. The UI always gets a response — it never crashes. | `orchestrator.py: try/except at top level` |
| Guardrail: hallucinated verse | `guardrails.py` checks that every verse in the response is verbatim in the approved passage set. Fires scoped refusal if not. | `guardrails.py:check_verbatim_verse()` |
| Guardrail: unresolvable citation | Checks every citation resolves to a real anchor registry entry. Blocks response if not. | `guardrails.py:check_citation_resolvable()` |

**The design principle:** *Never raise from the public entry point.* `orchestrator.run()` always returns a displayable dict. Failures are logged and surfaced as user-friendly messages, not tracebacks.

---

## Q7 — What type of memory does the agent have?

The system implements **two types of memory**, matching the course taxonomy:

### Short-term (Conversational / Working Memory)
- **What:** The last 5 conversation turns (query + response pairs) stored in `st.session_state`.
- **How used:** At the start of every new query, `bilingual_analyzer.py` reads this history and includes it in the LLM prompt. This lets the system resolve pronouns across turns — "What else did he write?" after "Who is Al-Hazani?" correctly resolves "he" to Al-Hazani.
- **Limitation:** Session-local. Lost when the browser tab closes.
- **Code:** `bilingual_analyzer.py:_format_history_for_prompt()` · `state.py:conversation_history` field

### Long-term (Semantic / Vector Memory)
- **What:** The Qdrant vector index — 4,747 chunks across 25 manuscripts, persistent on disk.
- **How used:** Every query searches this index. It does not change per-session; it accumulates as new manuscripts are added and the index is rebuilt.
- **Why this counts as memory:** The system "knows" 2,222 verse anchors across 4 digitisation phases. That knowledge persists across all sessions, all users, indefinitely — it is not re-loaded from the LLM's parametric weights.
- **Code:** `data/qdrant/` · `src/fatat_al_arab/index.py`

> **Course connection:** Short-term ↔ Agent working memory. Long-term ↔ external knowledge store. Together they implement the full memory stack from the Agents lecture.

---

## Q8 — What types of retrievers does the system have?

The system uses **triple hybrid retrieval** — three complementary approaches fused via Reciprocal Rank Fusion:

| Retriever | Type | Strengths | Code |
|---|---|---|---|
| **BM25** (BM25Okapi) | Sparse / keyword | Exact token matches — poet names, volume numbers, rare Arabic roots. Fails on synonyms. | `retrievers/bm25.py` |
| **Dense** (AraPoemBERT 768-dim) | Dense / semantic | Thematic similarity — "poems about longing" finds relevant verses even without keyword overlap. Fails on very rare names. | `retrievers/dense.py` |
| **ColBERT** | Late-interaction | Per-token MaxSim matching — more precise than bi-encoder dense. Currently stubbed (returns []) due to 2GB dependency + GPU requirement. | `retrievers/colbert.py` |

**Plus two retrieval augmentation techniques:**
- **HyDE** (Hypothetical Document Embedding): generate a fake Nabati verse that would answer the query, embed it, use that vector for retrieval. Bridges the query-document embedding gap.
- **Bilingual expansion**: generate 3–5 Arabic + English paraphrases of the query, run retrieval for each, fuse results.

**Fusion:** Reciprocal Rank Fusion (RRF, k=60) combines all ranked lists: `score = Σ 1/(k + rank)`. This rewards passages that rank highly in multiple lists.

---

## Course Journey — How NABAT-AI Reflects Every Module

The course path was: **Foundations → Prompting → RAG → Fine-tuning → Agents → Multi-Agent → Governance → Evaluation → RL**

This project is a working demonstration of that entire path applied to one real problem.

---

### Foundations (Transformers, autoregressive generation, diffusion)

**Applied in NABAT-AI:**
- **AraPoemBERT** is a Transformer encoder (BERT architecture) fine-tuned on 2M+ Arabic poetry verses. It produces the dense vectors that make semantic retrieval work.
- **Qwen2.5-7B / Mistral-7B** are autoregressive Transformers — they generate the HyDE verse, the synthesised answer, the CRAG grades, and the Self-RAG critique.
- Understanding scaling laws explains *why* a 7B hosted model gives usable output for poetry tasks without GPU.

**What to say to the jury:** "The foundation of the system is two Transformer models — one encoder (AraPoemBERT) for understanding verse meaning in the vector space, and one autoregressive decoder (Qwen2.5-7B) for generation and reasoning. Without understanding how attention works, I wouldn't have known why AraPoemBERT outperforms generic Arabic models for this domain."

---

### Prompting = Programming

**Applied in NABAT-AI:**
- Every LLM call in the system is a precisely engineered prompt. None are one-shot.
- **Role-based:** `"أنت شاعر متمرس في الشعر النبطي الخليجي"` (you are an expert Nabati poet) — tells the LLM its persona for HyDE generation.
- **Structured output:** all LLM calls return JSON with a schema — `bilingual_analyzer`, `self_query`, `crag_grader`, `reflect` all use `json_schema` constraints.
- **Iterative:** HyDE went through 3 iterations before the dialect was correct. CRAG grader went through 3 iterations before per-chunk granularity worked. Self-RAG reflect went through 2 iterations before `fix_instructions` made retries surgical.
- **Key insight applied:** "You don't retrain — you program behavior." The system has zero fine-tuning. All behavior is controlled by prompts.

**Decision table (from DEVELOPMENT_LOG.md):**

| Decision | What AI suggested | What I changed | Why |
|---|---|---|---|
| HyDE system prompt | English prompt | Arabic role prompt | Better dialect adherence |
| CRAG output format | Single aggregate grade | Per-chunk array + `requery_strategy` field | Re-query loop needs per-passage granularity |
| Self-RAG output | Binary pass/fail | 3-axis score + `fix_instructions` | Retries need actionable guidance |
| Intent router threshold | Binary match | Weighted scoring (0.7 gate) | Binary had too many false positives |

---

### RAG vs Fine-tuning vs Prompting — Choosing the Right Tool

**Why RAG and not fine-tuning:**
1. The corpus grows — new manuscripts can be added. Fine-tuning would need to be re-run every time.
2. Fine-tuning with 2,222 verses would almost certainly overfit — too small a dataset.
3. RAG gives **mandatory citations** (volume, page, manuscript). Fine-tuning memorises facts but cannot cite them.
4. RAG is transparent — you can see exactly which passages were retrieved. Fine-tuning is a black box.

**Why not prompting alone:**
- Stuffing all 2,222 verses into a context window would need ~500K tokens — impossible and expensive.
- The LLM has no knowledge of these specific manuscripts in its parametric memory.

**What to say:** "I applied the course's key skill: choosing the right tool. RAG is 'external knowledge' — exactly the use case the course identified. Fine-tuning would have changed model behavior but couldn't give citations. Prompting alone couldn't access the corpus."

---

### Systems Thinking (Intelligence = Model + Data + System)

**Applied in NABAT-AI:**
- **Model** = Qwen2.5-7B (reasoning, generation, grading) + AraPoemBERT (embedding)
- **Data** = 2,222 verse anchors in `anchor_registry_full_enriched.json` + Qdrant index of 4,747 chunks
- **System** = Three-worker orchestration (Al-Nassikh ETL + Agent 1 Query Understanding + Agent 2 Retrieval Synthesis) + LangGraph conditional edges + failure budget caps

None of these three alone would work. The LLM without the data answers "I don't know." The data without the system is a static file. The system without the model is a keyword search engine.

---

### Agents (Planner, Executor, Memory, Tools — Plan→Act→Observe→Reflect→Repeat)

**Applied in NABAT-AI — the ReAct loop is explicit:**

| ReAct step | Maps to |
|---|---|
| **Plan** | Agent 1: intent routing, HyDE hypothesis, bilingual expansion, filter extraction |
| **Act** | Agent 2 Stage 4–5: triple hybrid retrieval + RRF fusion |
| **Observe** | Agent 2 Stage 7: CRAG grader examines retrieved passages |
| **Reflect** | Agent 2 Stage 9: Self-RAG scores the synthesised answer |
| **Repeat** | Conditional edges fire re-query (CRAG) or re-synthesis (Self-RAG) loops |

The system is not a single-step prompt — it is a multi-step reasoning system that decides at each stage whether to continue, retry, or stop.

---

### Agent Frameworks (LangGraph, CrewAI, AutoGen, n8n)

**Applied in NABAT-AI:**
- **LangGraph** is the framework — two separate `StateGraph` objects, one per agent.
- Chose LangGraph over CrewAI because: LangGraph gives explicit control over conditional edges, which is exactly what the CRAG re-query loop and Self-RAG retry loop require. In CrewAI, agents call each other by role — you cannot set a hard cap of 1 re-query as a named constant.
- The `add_conditional_edges` pattern maps directly to the architecture document's stage diagrams.

---

### Multi-Agent Systems (Specialization, Communication protocols, Coordination challenges)

**Applied in NABAT-AI:**

**Specialization:**
- Agent 1 = Query Specialist. Cannot call any retrieval tool (enforced at import time by `tools.py:ToolNotPermittedError`).
- Agent 2 = Retrieval & Synthesis Specialist. Cannot call translation or expansion tools.
- Worker 1 (Al-Nassikh) = ETL Specialist. Runs offline, not during query time.

**Communication protocol:**
- The `QueryContext` TypedDict is the formal message-passing protocol between agents.
- It is explicitly versioned — if Agent 1 adds a new field, Agent 2 fails loudly at import time (not silently at runtime) if it reads a field that doesn't exist.

**Coordination challenges solved:**
- Infinite re-query loops → `CRAG_REQUERY_MAX = 1`
- Infinite retry loops → `SELF_RAG_MAX_RETRIES = 2`
- Agent 1 failure not breaking Agent 2 → orchestrator has a direct-node fallback path
- Race condition on embedding model load → singleton pattern in `embed.py`

---

### Governance (Safety, Accountability, Explainability, Guardrails)

**Applied in NABAT-AI — three hard guardrails in `guardrails.py`:**

| Governance principle | Implementation |
|---|---|
| **Safety** | Scoped refusal template (AR + EN) fires when CRAG finds no relevant passages. The LLM cannot make up an answer. |
| **Accountability** | Every response carries mandatory citations (poet name, manuscript short key, volume, page). You can trace every claim back to a specific page of a specific manuscript. |
| **Explainability** | 🔸 badge on all heuristic-tagged genre results — signals to the user that this is silver-baseline classification, not gold. 🗂️ badge on deterministic registry answers — signals no LLM was involved. |
| **Guardrails** | (a) Every verse in the response must be verbatim in the approved passage set — paraphrase detection blocks hallucinated verse text. (b) Every citation must resolve to a real anchor registry entry. (c) Out-of-corpus queries → fixed refusal template, no decoration. |

**What to say:** "Governance wasn't an afterthought — it's baked into the architecture. The guardrails run as the last step before anything reaches the UI. The 🔸 badge was a specific design decision: without a gold-labelled genre dataset, claiming accuracy would be dishonest. The badge makes the uncertainty visible."

---

### Evaluation (Test cases, metrics, qualitative)

**Applied in NABAT-AI:**
- **468 tests** across 15 test files, including `test_edge_cases.py` (empty strings, very long queries, mixed language, adversarial out-of-corpus inputs).
- **4-axis evaluation harness** in `scripts/evaluate.py`:
  - *Correctness:* CER buckets, Recall@5, CRAG κ agreement, citation resolvability, refusal precision
  - *Robustness:* loop activation rates, retry exhaustion, fallback LLM invocations
  - *Efficiency:* p50/p95 latency, per-stage breakdown, token cost estimate
  - *Human:* 5-point Likert scaffold (filled by a Nabati scholar)
- Evaluation runs in stub mode (no API key needed): `LLM_PROVIDER=stub PYTHONPATH=src python3 -m pytest tests/ -q`

---

### RL & Feedback (RLHF, Self-critique, Reward signals, Feedback loops)

**Applied in NABAT-AI — two internal feedback loops:**

**Loop 1 — CRAG re-query (retrieval feedback):**
- After grading passages, if verdict is `Incorrect`, the system generates a `requery_strategy` — an LLM-written description of what to search for instead.
- The next retrieval call reads this hint and augments its BM25 query.
- This is a **closed feedback loop**: observe result → generate improvement signal → act on it.

**Loop 2 — Self-RAG retry (generation feedback):**
- The `reflect` node scores the synthesised answer on faithfulness, relevance, and completeness (1–5 each).
- If any axis < 3, it generates `fix_instructions` — specific sentences to correct in the next synthesis attempt.
- The next `synthesise` call prepends these instructions: it knows exactly what failed and what to fix.
- This mirrors the **self-critique** concept from the RL lecture: the system critiques itself without external human feedback.

**What is NOT implemented (honest):** RLHF (no human preference labels). This would require a gold-labelled dataset of preferred/rejected responses, which requires domain expert annotation. That is flagged in `AI_USAGE_REFLECTION.md §5c` as the next step.

---

## Critical Questions to Prepare For

### "Are you training the model to read handwriting?"

**No — and this is important to explain clearly.**

The system does NOT train any model to read handwriting. The digitisation pipeline works like this:

1. **Kraken HTR** (a pre-trained Handwritten Text Recognition model specifically built for Arabic manuscripts) produces a draft transcription of each manuscript image.
2. **eScriptorium** (a web interface for manuscript annotation) lets a human expert review and correct Kraken's draft.
3. The corrected text is exported as **PAGE-XML** and parsed by `scripts/ingest_phases_123.py` into the anchor registry.
4. The RAG system is built on top of this already-transcribed text — it never sees the images during query time.

**The AI "learns" nothing from the images.** Kraken was already trained (by a separate research team) on thousands of Arabic manuscripts. We use it as a pre-trained tool, the same way you use pytesseract for printed OCR.

**What the current system CAN do with images:** If a user uploads a photo of a verse, `image_ocr.py` uses pytesseract to extract the text, which is then used as the query text for the RAG pipeline. This is OCR for query, not OCR for indexing.

**Honest limitation:** For new manuscripts not yet in the system, the full pipeline (scan → Kraken → eScriptorium review → export → rebuild index) is a manual process. Tab B of the UI (Archive Manager) documents this workflow for contributors.

---

### "What about Tashkeel (diacritical marks / short vowels)?"

**Tashkeel** (التشكيل) refers to the short-vowel diacritical marks in Arabic: fatḥa (َ), ḍamma (ُ), kasra (ِ), sukūn (ْ), shadda (ّ), tanwīn, etc.

**What the system does with Tashkeel:**

During embedding (`embed.py`), all harakat (diacritics) are stripped before encoding:
```python
_HARAKAT = re.compile(r"[ً-ٰٟ]")
```

**Why strip Tashkeel for retrieval:**
- Retrieval recall: a user who types `الوطن` (no diacritics) must match a verse that reads `الوَطَن` (with diacritics). Stripping both ensures the match.
- Practical reality: most users type without diacritics. Keeping them in the index would mean queries without diacritics miss verses.

**What is lost:**
- Tashkeel carries phonological and metrical information. In Nabati poetry, *wazn* (وزن, meter) and *qāfiya* (قافية, rhyme) depend on diacritisation. Stripping it means the index is metre-agnostic — you cannot search by specific metre pattern.
- Tashkeel also disambiguates homographs: `عِلم` (knowledge) vs `عَلَم` (flag/name). After stripping, these are the same token — a retrieval error that is rare but possible.

**How to answer if a juror presses:**
> "We made a deliberate trade-off: retrieval recall over metrical precision. For a scholarly system that needs metre-aware search, you would need a Tashkeel-preserving index with a tashkeel-normalisation that handles equivalences rather than stripping. That is documented as a future improvement — it requires a human expert to annotate which diacritical variants are metrical vs. dialectal."

**The original manuscripts:** Many Nabati manuscripts are *not* fully diacritised — the handwritten text itself often lacks Tashkeel. So even if we preserved it in the index, the coverage would be uneven across the 25 manuscripts.

---

## Reviewer Questions from the Course Slides — Prepared Answers

### "Is this over-engineering?"

No — every component solves a real problem that simpler approaches couldn't:
- **Intent router before LLM:** without it, "how many poems do I have?" goes into retrieval and returns a scoped refusal. Demonstrated in DEVELOPMENT_LOG.md Day 8.
- **HyDE:** without it, a 3-word query like "exile and longing" retrieves unrelated verses. Dense retrieval works much better with a full verse as the query.
- **CRAG grader:** without it, BM25 returns topically adjacent but thematically wrong passages and the LLM confidently synthesises a wrong answer.
- **Self-RAG:** without it, the synthesised answer sometimes cites a verse for a claim the verse doesn't actually support (hallucination).

Each node exists because the system *broke* without it during development. See DEVELOPMENT_LOG.md for the specific failure stories.

### "Do you need agents for this? Can't prompting alone do it?"

No — prompting alone cannot:
1. Access the 2,222 verse corpus (would need ~500K tokens of context).
2. Know at query time whether the retrieved passages are relevant (you need to see them before grading).
3. Enforce citation traceability (a single prompt generates uncited prose).
4. Route factual queries away from RAG without adding 400ms LLM latency to every request.

The agent structure is not cosmetic — the conditional edges encode real decision logic that a single prompt cannot make.

### "What is the biggest flaw?"

Honest answer: **the ColBERT retriever is a stub.** The "triple hybrid" only uses two retrievers. This is documented, and the reason is portability (2GB dependencies, GPU memory requirement), but it means token-level retrieval precision is missing.

Second honest answer: **genre classification is silver-baseline only.** 82.9% of the corpus has a genre tag, but without a gold-labelled dataset we cannot measure precision. The 🔸 badge makes this visible.

### "What would you remove?"

The prototype semantic router (`semantic_router.py` + `test_prototype_router.py`). It adds 80–120ms latency on first call (model load) for a marginal 3-of-20 hit rate improvement over the regex intent router. It's already kept as a lazy Tier 2 layer, but if asked to simplify, this would go first.

### "What would you improve?"

From `AI_USAGE_REFLECTION.md §5`:
1. **Persistent session memory** — currently lost on browser refresh. A SQLite store for `conversation_id` would fix this in ~2 hours.
2. **Parallel Agent 1 LLM calls** — HyDE and bilingual expansion are independent but run sequentially. `asyncio.gather` would cut ~600ms from median latency.
3. **Real ColBERT** — replace the stub with `pylate`. Adds genuine token-level retrieval diversity.
4. **Gold genre labels** — annotate 200 verses (10% sample) to get real precision/recall metrics for the genre classifier.

### "Real-world viability?"

The system runs today without Docker, GPU, or paid-tier API. Deployment concerns:
- **Latency:** 5–8s end-to-end is acceptable for a scholar workbench but not for a mobile app.
- **Scale:** Qdrant file-backed mode works at 4,747 chunks; for 50K+ chunks, switch to Qdrant server mode.
- **Persistence:** session memory needs a store; currently session-local.
- **Corpus growth:** adding a new manuscript is a documented 5-step process in Tab B, ending with `python scripts/rebuild_index.py`.

The system is not a demo — it has 468 tests and handles failure at every node. It is production-viable at this scale.

---

## Quick Reference — Key Files for the Final

| Question topic | Go to |
|---|---|
| Full score evidence | `README.md` rubric table |
| Use-case → code tracing | `doc/TRACEABILITY.md` |
| AI iteration history | `DEVELOPMENT_LOG.md` |
| AI limitations + reflection | `doc/AI_USAGE_REFLECTION.md` |
| Inter-agent state contract | `src/fatat_al_arab/state.py` |
| Agent 1 graph (LangGraph) | `src/fatat_al_arab/agent1_query_understanding/graph.py` |
| Agent 2 graph (LangGraph) | `src/fatat_al_arab/agent2_retrieval_synthesis/graph.py` |
| Orchestrator (single entry point) | `src/fatat_al_arab/orchestrator.py` |
| CRAG grader prompt (3 iterations) | `agent2/.../crag_grader.py` |
| Self-RAG reflect prompt | `agent2/.../reflect.py` |
| Guardrails | `src/fatat_al_arab/guardrails.py` |
| Tashkeel stripping | `src/fatat_al_arab/embed.py:_HARAKAT` |
| ColBERT stub (why + what's needed) | `src/fatat_al_arab/retrievers/colbert.py` |
| Evaluation harness | `scripts/evaluate.py` |
| Run all tests | `LLM_PROVIDER=stub PYTHONPATH=src python3 -m pytest tests/ -q` |
