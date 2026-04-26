"""
src/fatat_al_arab/rag.py
========================
Why this file exists: reserved namespace — this module name appears in
architecture diagrams and external references. It is intentionally empty.

Why the RAG logic is NOT here: the v2 architecture (2026-04-22) replaced a
monolithic RAG class with a two-agent LangGraph pipeline:
  - Agent 1 (agent1_query_understanding/) — query understanding
  - Agent 2 (agent2_retrieval_synthesis/) — retrieval + synthesis

The entry point for the pipeline is:
    from fatat_al_arab.orchestrator import run
    result = run("your question here")

Nothing should import from this module. If you see an import of
`fatat_al_arab.rag` in any file, it is stale and should be updated to
import from the appropriate agent or orchestrator module instead.

Architecture ref: §0 guiding decision 3 (three-worker pipeline).
"""
# Intentionally empty — see module docstring above.
