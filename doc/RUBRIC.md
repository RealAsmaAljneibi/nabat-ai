# MAAI1704 — Assessment Rubric

| Criteria | Description | Detailed Description | Max Points |
|---|---|---|---|
| **Alignment with Proposal & Architecture deliverables** | Traceability from use-case to architecture to implementation | Implementation reflects: Original problem statement · Target users & value · Designed architecture (agents, tools, workflows). **Positive:** Clear traceability: Use-case → Architecture → Code. **Negative:** Implementation deviates or is ad-hoc. | 10 |
| **Agentic System Realization** | Multi-agent design, orchestration, tool use, communication | Evaluate whether you actually built an agentic system, not just an LLM wrapper. Multiple agents with defined roles · Clear orchestration: Planner–Executor / Supervisor / Graph-based flow (e.g., LangGraph) · Tool usage (APIs, retrieval, external functions) · Inter-agent communication. **Positive:** True multi-agent coordination with reasoning flow or true implementation of agentic solution. **Negative:** Single prompt pipeline disguised as "agents". | 20 |
| **AI-Assisted Development Process** | Prompting, iteration, debugging, code modification | How effectively you used code generators (LLMs, Copilot, etc.): Prompt engineering (structured, role-based, iterative) · Debugging using AI (error tracing, refinement) · Evidence of iteration (not one-shot generation) · Modification/customization of generated code. **Positive:** Student drives the system; AI is a tool [a few examples of development notes/logs]. **Negative:** Blind copy-paste from LLM. | 15 |
| **Code Quality & Modularity** | Readable, modular, documented code | Clean structure (agents, tools, pipelines separated) · Reusability and modularity · Documentation/comments · Proper configuration (env variables, configs). | 10 |
| **System Integration** | End-to-end working system with real inputs | Full pipeline works: Input → agent reasoning → tool use → output · Real execution (not mocked) · Handles realistic scenarios. **Positive:** Reliable execution across cases / deployed. **Negative:** Demo-only or brittle system. | 15 |
| **Memory, Tools & RAG** | Use of memory, retrieval, APIs/tools | Evaluate depth of agent capabilities: Use of memory: Short-term / conversational · Long-term (vector DB) · RAG pipeline: Retrieval quality, Context injection · Tool ecosystem: APIs, search, structured tools. | 8 |
| **Evaluation & Reliability** | Test cases, edge handling, robustness | Test cases (normal + edge cases) · Handling failures: hallucination control, fallback strategies · Metrics or qualitative evaluation. | 7 |
| **Understanding & Ownership** | Ability to explain and modify system | Can explain: Agent flow · Tool usage · Key implementation decisions. | 10 |
| **Reflection on AI Usage** | Insights on AI usage, limitations, improvements | What worked / failed with code generators · Limitations observed · Improvements suggested. | 5 |

**Total: 100 points**

---

## Evidence Map

| Criterion | Evidence in this repo |
|---|---|
| Alignment (10 pts) | [`doc/TRACEABILITY.md`](TRACEABILITY.md) maps every use-case → §ref → file path. [`doc/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) shows milestone-to-code traceability M0–M11. The three-worker code layout (`agent1_query_understanding/`, `agent2_retrieval_synthesis/`, `al_nassikh/`) mirrors §2.4, §2.5, §2.2 of the architecture document section-by-section. |
| Agentic System (20 pts) | Two LangGraph `StateGraph` objects: `agent1/graph.py` (6 nodes, conditional edges) and `agent2/graph.py` (10 nodes, 3 agentic loops — CRAG re-query × 1, CRAG verdict re-query × 1, Self-RAG retry × 2). Orchestrator (`orchestrator.py`) wires them via a `QueryContext` TypedDict state contract. Tool registries: `agent1/tools.py` (4 tools), `agent2/tools.py` (7 tools). Cross-agent tool use is enforced by ToolNotPermittedError at import time. |
| AI-Assisted Dev (15 pts) | [`DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md) — dated iterative log with prompt v1/v2/v3 for HyDE, CRAG grader, Self-RAG reflect; the "count bug" debugging story; and a table of AI suggestions vs. manual corrections. |
| Code Quality (10 pts) | Every module docstring leads with "Why this file exists". Named constants for all failure budgets (`CRAG_REQUERY_MAX`, `SELF_RAG_MAX_RETRIES`, `HYDE_TIMEOUT_S`). `.env.example` with all config keys. Three-layer directory structure separates ETL, Agent 1, Agent 2. |
| System Integration (15 pts) | Pre-built real Qdrant index at `data/qdrant/` (4,747 chunks, AraBERT 768-dim embeddings). Triple hybrid retrieval (BM25 + Dense + ColBERT). Real LLM calls via Together.ai / Groq through `llm.py`. Streamlit app handles text, image (OCR), and voice (Whisper) inputs end-to-end. |
| Memory, Tools & RAG (8 pts) | Long-term: Qdrant file-backed vector store with 9-level chunking. Short-term: conversation history in `st.session_state`, passed to `bilingual_analyzer.py` (`_format_history_for_prompt()`). RAG: triple hybrid retrieval → RRF fusion → CRAG grading → grounded synthesis with mandatory citations → Self-RAG reflection. |
| Evaluation (7 pts) | 15 test files, 480+ test cases. `tests/test_edge_cases.py` (53 tests) covers empty inputs, long inputs, injection attempts, malformed history, non-string inputs, failure budget caps, return-shape invariants. `scripts/evaluate.py` — 4-axis harness (correctness, robustness, efficiency, human). |
| Understanding (10 pts) | [`doc/TRACEABILITY.md`](TRACEABILITY.md) explains agent flow and tool usage end-to-end for all 4 personas. Inline `Why:` comments throughout nodes explain non-obvious design decisions. [`DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md) documents key implementation decisions and trade-offs. |
| Reflection (5 pts) | [`doc/AI_USAGE_REFLECTION.md`](AI_USAGE_REFLECTION.md) — what worked (JSON schema prompting, LangGraph scaffolding), what failed (one-shot generation, Arabic dialect errors, missing failure caps), limitations (latency, session-local memory, ColBERT stub), 5 concrete improvements. |
