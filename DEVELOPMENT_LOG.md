# NABAT-AI — Development Log

**Student:** Asma Salem Mubarak Najem Aljneibi · **Course:** MAAI1704  
**Period:** 2026-04-10 to 2026-04-26

This log records the iterative development process: which AI tools were used (Claude, ChatGPT), which prompts were engineered, what failed, and how decisions were refined. It demonstrates that the system was built through active, driver-led iteration — not one-shot generation.

---

## Week 1 (2026-04-10 – 2026-04-14): Architecture & ETL Scaffold

### Day 1 — Problem Framing with AI Assistance

**Goal:** Decide how to structure the RAG pipeline for poetry.

I started by describing the problem to Claude and asking for architecture options:

> **Prompt v1:**  
> *"I have 25 handwritten manuscript images of Khaleeji Nabati poetry, about 2,200 verses. I want to build a RAG system so a researcher can ask 'show me love poems by Al-Hazani'. What architecture would work?"*

**AI response (summary):** Suggested a basic RAG: embed docs → FAISS index → query → retrieve → LLM answer.

**Problem with v1:** This treats every query identically. But a researcher asking "how many poems do I have?" should NOT go through retrieval at all — they need a deterministic registry lookup. I pushed back.

> **Prompt v2:**  
> *"The problem is that some queries are factual ('how many poems?') while others are thematic ('find love poems'). How do I separate these without adding 400ms of LLM latency to every query?"*

**AI response:** Suggested a regex fast-path router as Stage 0 before any LLM call.  
**Decision:** This became the `intent_router.py` design (Stage 0.5). The regex confidence scoring (≥0.7 threshold to fire) was my addition after testing — the AI suggested a simpler binary match, but binary matching had too many false positives on partial matches like "how" without a counting noun.

---

### Day 2 — Three-Worker Split

**Goal:** Decide whether agents should be one graph or two.

> **Prompt:**  
> *"Should I use one LangGraph graph or two? Agent 1 does query understanding (language detection, HyDE, filter extraction). Agent 2 does retrieval and synthesis. They pass state between them."*

**AI suggested:** One graph with conditional edges is simpler to debug.

**My pushback:** The architecture document (v4) requires two separate workers that can be tested independently. I need to match the deliverable architecture.

**Decision:** Two separate `StateGraph` objects wired by the orchestrator. Agent 1 emits a `QueryContext` TypedDict; Agent 2 consumes it. This meant writing `state.py` with two distinct type contracts — the AI generated a good starting draft that I then extended with `crag_requery_strategy` and `self_rag_fix_instructions` fields (not in the AI's draft, added during M9 debugging).

---

### Day 3 — HyDE Prompt Engineering

**Goal:** Get the LLM to generate a plausible Nabati verse that improves dense retrieval.

**Iteration 1** — English system prompt:
```
You are a poetry assistant. Given a query, write a short Arabic poem that would answer it.
```
**Result:** The LLM generated modern Arabic, not Khaleeji Nabati. Retrieval quality was poor — the dense embeddings drifted away from the corpus.

**Iteration 2** — Added Khaleeji constraint:
```
Write one verse in Gulf Arabic dialect (Nabati poetry style) that would answer this query.
```
**Result:** Better dialect, but the LLM sometimes added explanations after the verse ("Here is a verse: ..."). The explanation text inflated the embedding and hurt retrieval.

**Iteration 3** — Added strict output format + length cap:
```
أنت شاعر متمرس في الشعر النبطي الخليجي. مهمتك أن تكتب بيتاً شعرياً خليجياً واحداً
يمكن أن يكون إجابةً على السؤال التالي. يجب أن يكون البيت:
- مكتوباً بالعربية الخليجية (النبطية) أو الفصحى الشعرية.
- مكوناً من شطرين (صدر وعجز) يفصل بينهما مسافة أو شرطة.
- مصدراً خاماً بدون شرح أو تعليق.
أعد البيت الشعري فقط.
```
**Result:** Clean verse output, no explanations. I added a guard: `len(verse) > 10` to catch the rare empty-string response. This prompt is in `hyde.py` as `_SYSTEM_HYDE`.

**Key learning:** Prompting in Arabic for an Arabic task gives dramatically better dialect adherence than prompting in English for Arabic output.

---

## Week 2 (2026-04-15 – 2026-04-20): RAG Core & Retrieval

### Day 5 — CRAG Grader Prompt Engineering

**Goal:** Grade retrieved passages as Correct/Ambiguous/Incorrect to decide whether to re-query.

**Iteration 1** — Simple grading prompt:
```
Does this passage answer the query? Grade: Correct, Ambiguous, or Incorrect.
```
**Problem:** The LLM returned prose ("This passage seems relevant because...") instead of structured output. Could not parse reliably.

**Iteration 2** — Added JSON schema requirement:
```
Return a JSON object: {"grade": "Correct"|"Ambiguous"|"Incorrect", "rationale": "..."}
```
**Problem:** When given multiple passages, the LLM returned a single aggregate grade instead of per-passage grades. Re-querying logic needed per-chunk granularity.

**Iteration 3** — Switched to array output + added `requery_strategy` field:
```json
{
  "grades": [{"chunk_id": "...", "label": "Correct|Ambiguous|Incorrect", "confidence": 0.0-1.0, "rationale": "..."}],
  "requery_strategy": "<what to search for next if grades are poor>"
}
```
**Result:** Per-chunk grades enabled per-passage CRAG logic. The `requery_strategy` field (my addition — the AI draft didn't include it) became the LLM-directed re-query hint used in M9 (`crag_requery_strategy` in state). Without it, re-queries repeated the same query — the re-query loop never improved results.

**Debugging story:** The CRAG loop was firing infinitely in early testing. Root cause: the re-query counter was not being checked before firing. I added `crag_requery_count < CRAG_REQUERY_MAX` guard. The AI had generated the loop logic but omitted the cap — I caught this by reading the generated code carefully.

---

### Day 7 — Self-RAG Reflect Prompt Engineering

**Goal:** After synthesis, check whether the generated answer faithfully cites the retrieved evidence.

**Iteration 1:**
```
Review this answer. Is it faithful to the retrieved passages? Return: pass or fail.
```
**Problem:** Binary pass/fail gave no actionable information on retry. The second synthesis call had no idea *what* to fix.

**Iteration 2:** Added three-axis scoring + surgical fix instructions:
```
Score on three axes (1-5):
1. Faithfulness: are all claims supported by the retrieved passages?
2. Relevance: does the answer address what the user asked?
3. Completeness: are all key retrieved facts included?

If any axis < 3, return:
- failed_claims: list of specific claims not supported by retrieved text
- fix_instructions: exact instructions for what to correct in the next synthesis attempt
```
**Result:** The `fix_instructions` field (my design addition) allowed `synthesise.py` to prepend the critique to the next attempt, making retry surgical rather than blind. This was a significant improvement over the AI's initial suggestion.

**Performance finding:** Retry fired ~28% of the time in stub-mode testing. Setting the failure threshold at axis score < 3 (not < 4) reduced false-positive retries while still catching real hallucinations.

---

### Day 8 — Intent Router: The "Count" Bug

**Context:** While testing on 2026-04-23, I typed "how many poems do I have?" into the Scholar tab. Expected: a count from the registry. Actual: the query went into retrieval, CRAG marked retrieved verses as "Incorrect" (none contained a count), and the system returned the refusal template.

**Debugging with AI:**
> *"Here is my CRAG grader output for the query 'how many poems'. It's marking all passages as Incorrect and firing the refusal. How do I fix this without running all queries through an LLM router?"*

**AI suggestion:** Add a regex pre-check before the LLM pipeline.  
**My implementation:** Designed the two-pass confidence scoring system (cue score 0.6 + slot score 0.3 + question mark bonus 0.1, threshold 0.7). The AI suggested a simpler threshold of 0.5 — I raised it to 0.7 after testing showed that "number of stars in poems" was incorrectly triggering the counting path at 0.5.

**Further iteration:** The initial intent_router only handled English. I extended the patterns manually for Arabic ("كم", "شو عدد") and Khaleeji variants after testing with a native-speaker colleague.

---

## Week 3 (2026-04-21 – 2026-04-26): Integration, Polish, Evaluation

### Day 11 — bilingual_analyzer: Conversation History

**Problem found:** Multi-turn queries like "what else did he write?" returned nonsense because each turn was treated independently.

> **Prompt to AI:**  
> *"My bilingual analyzer doesn't know about previous conversation turns. The user asks 'what else did he write?' and the analyzer doesn't know who 'he' is. How do I pass conversation context without blowing the token budget?"*

**AI suggestion:** Pass last N turns in the prompt, truncate long responses.  
**My implementation:** `_format_history_for_prompt()` — filters out refusals (they carry no useful context), caps at 5 turns, truncates responses to 300 chars. I added the refusal filter myself; the AI draft included all turns.

**Result:** Multi-turn pronoun resolution now works. Tested with: "من هو ناصر الهزاني؟" → "ماذا كتب؟" — second turn correctly resolves "كتب" to Nasser Al-Hazani.

---

### Day 12 — Prototype Router: Tier 2 Embedding Classifier

**Attempt:** Tried to add an embedding-based similarity classifier (AraBERT cosine vs. prototype queries) to catch paraphrases of meta-questions that the regex misses.

**Implementation:** `prototype_router.py` — uses sentence-transformers when available, TF-IDF fallback for CI.

**Problem encountered:** The prototype router had a marginal hit rate on real queries (3 of 20 test queries were new hits over the regex router). Integrating it added 80-120ms latency on cache-cold first calls (model load). Given that the regex fast-path already covers 94% of meta-questions, the latency cost outweighed the gain for the graded demo.

**Decision:** Kept as an optional Tier 2 layer called from within `intent_router_node` (Check 5, lazy import) but not added as a separate node in `agent1/graph.py`. It runs when available and fails silently if not. This way it adds coverage without adding fragility to the demo path.

---

### Day 13 — Genre Heuristic Classifier

**Goal:** Tag each of the 2,222 verse anchors with a genre label (غزل, رثاء, مديح, etc.) without a human-labelled gold set.

**AI-assisted approach:** I described the 10-genre taxonomy and asked the AI to suggest keyword patterns for each genre in both MSA and Khaleeji Arabic.

**Problem:** The AI's suggested patterns for `غزل` (love poetry) included words like `قلب` (heart) and `عين` (eye) — but these appear frequently in other genres too (e.g., `حكمة`). Coverage was high but precision was low.

**My fix:** Added context-aware pattern weighting: words that appear in ≥3 genres are downweighted; words that uniquely predict a genre (e.g., `رثاء` for elegy, specific verb forms like `أندب`) are upweighted. This improved genre precision on spot-check from ~60% to ~82%.

**Final coverage:** 82.9% of 2,222 anchors tagged. The 17.1% that abstain go to `غير_محدد` — this is intentional; forcing a label on every verse would be misleading.

---

### Day 14 — Evaluation Harness

**Goal:** Measure system quality on 4 axes: correctness, robustness, efficiency, human.

**AI-generated:** Initial `evaluate.py` structure with fixture queries and correctness scoring.  
**My additions:**  
- Robustness axis: adversarial queries (out-of-corpus, very long, mixed language)
- Efficiency axis: wall-clock timing per query (deterministic < 2ms gate)
- The fixture query set was expanded from AI's 5 examples to 28 test cases

---

## Code Quality Decisions Made During Development

| Decision | Why | What the AI suggested | What I changed |
|----------|-----|----------------------|----------------|
| Regex confidence ≥ 0.7 threshold | Empirical: 0.5 had false positives | Binary match | Weighted scoring with 0.7 gate |
| HyDE prompt in Arabic | Better dialect adherence | English prompt | Switched to Arabic system prompt |
| `requery_strategy` field in CRAG | Makes re-queries LLM-directed | Not suggested | Added to prompt schema and state |
| `fix_instructions` in Self-RAG | Makes retries surgical | Not suggested | Added to reflect prompt + state |
| Refusal filter in history formatter | Refusals carry no context | Include all turns | Filter `is_refusal=True` turns |
| Prototype router as lazy Tier 2 | 80-120ms load cost too high | Separate graph node | Lazy import inside intent_router |
| Genre precision via downweighting | Ambiguous words inflate false positives | Keyword match | Frequency-based weight decay |

---

## Tools Used

| Tool | Usage |
|------|-------|
| Claude (claude.ai) | Architecture design, prompt drafting, debugging assistance |
| ChatGPT | Cross-checking Arabic pattern suggestions for intent router |
| GitHub Copilot | Autocomplete for boilerplate (TypedDicts, test fixtures) |
| Manual testing | Running `streamlit run` and typing real queries to catch edge cases |
