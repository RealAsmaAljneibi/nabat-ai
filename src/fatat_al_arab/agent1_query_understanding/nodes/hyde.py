"""
agent1_query_understanding/nodes/hyde.py
=========================================
Why this node exists: §2.4 Stage 2 (HyDE — Hypothetical Document Embeddings)
addresses a fundamental mismatch between user queries and poetry: a query like
"poems about camels at dusk" is short and sparse, while the verses it should match
are long and richly idiomatic. HyDE bridges this by asking the LLM to generate a
plausible Nabati verse that *would* answer the query, then embedding that verse
rather than the raw query. The verse's dense vector lands much closer to the actual
corpus vectors than the query itself would.

Failure handling (§5 budget): HyDE has 1 retry and a 3-second cap per the §5
table ("Agent 1 HyDE: 1 retry, 3 s timeout — fallback: proceed without
hyde_embedding"). If the LLM call fails or times out, we log and continue with
hyde_passage=None, hyde_embedding=None. The retriever (Agent 2) checks for None
and falls back to embedding the query directly.

Architecture refs: §2.4 Stage 2 (HyDE), §2.9 tool registry ('hyde_passage'),
§5 (failure budget), §3.2 Philology view (HyDE passage displayed to researcher).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ...llm import chat
from ...state import AgentState

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
HYDE_TIMEOUT_S   = 3.0     # §5: 3 s hard timeout for the HyDE LLM call
HYDE_MAX_TOKENS  = 120     # one verse should not exceed ~120 tokens

_SYSTEM_HYDE = """\
أنتِ فتاة العرب (الباحثة العربية)، خبيرة الشعر النبطي الخليجي في نظام NABAT-AI.
المهمة — المرحلة 2أ (HyDE): اكتبي بيتاً شعرياً خليجياً واحداً يمكن أن يكون \
إجابةً على السؤال التالي. يجب أن يكون البيت:
- مكتوباً بالعربية الخليجية (النبطية) أو الفصحى الشعرية.
- مكوناً من شطرين (صدر وعجز) يفصل بينهما مسافة أو شرطة.
- مصدراً خاماً بدون شرح أو تعليق.
أعد البيت الشعري فقط.
"""


def _hyde_with_timeout(query_ar: str) -> Optional[str]:
    """
    Why threading: we need a hard timeout on the LLM call that works regardless
    of the provider's own timeout implementation. The §5 budget says 3 s max for
    HyDE — we honour that by running the call in a daemon thread and abandoning
    it if it exceeds HYDE_TIMEOUT_S.

    Returns the hypothetical verse text, or None on timeout/failure.
    """
    result: list[Optional[str]] = [None]
    error:  list[Optional[Exception]] = [None]

    def _target() -> None:
        try:
            result[0] = chat(
                prompt=f"السؤال: {query_ar}",
                system=_SYSTEM_HYDE,
                max_tokens=HYDE_MAX_TOKENS,
            )
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=HYDE_TIMEOUT_S)

    if t.is_alive():
        logger.warning(
            "hyde: LLM call exceeded %.1f s timeout — proceeding without hyde_embedding (§5).",
            HYDE_TIMEOUT_S,
        )
        return None

    if error[0]:
        logger.warning("hyde: LLM call failed — %s — proceeding without hyde_embedding.", error[0])
        return None

    verse = (result[0] or "").strip()
    # Guard against empty or very short output that's clearly not a verse
    return verse if len(verse) > 10 else None


def _embed_text(text: str) -> Optional[list[float]]:
    """
    Why lazy import: the sentence-transformers model is large and slow to load.
    Importing here (not at module level) means that tests and agent-graph imports
    don't pay the load cost unless HyDE actually runs.

    Returns a list[float] embedding or None if the model is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        import os

        model_name = os.getenv("EMBED_MODEL", "aubmindlab/bert-base-arabertv02")
        # Module-level cache to avoid reloading the model on every call
        if not hasattr(_embed_text, "_model") or _embed_text._model_name != model_name:
            logger.info("hyde: loading embedding model %s …", model_name)
            _embed_text._model      = SentenceTransformer(model_name)
            _embed_text._model_name = model_name

        embedding = _embed_text._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as exc:
        logger.warning("hyde: embedding failed — %s — returning None.", exc)
        return None


def hyde_node(state: AgentState) -> AgentState:
    """
    LangGraph node — §2.4 Stage 2 (HyDE half).

    Reads:   state["query_context"]["query_ar"]
    Writes:  state["query_context"]["hyde_passage"], ["hyde_embedding"]

    Why we skip when clarification is flagged: we don't invest LLM calls on a
    query the user is about to rephrase.
    """
    qc = state.get("query_context", {})

    if qc.get("needs_clarification"):
        logger.debug("hyde: skipping — clarification path active.")
        return state

    query_ar: str = qc.get("query_ar", state.get("raw_query", ""))

    # §5 budget: 1 retry → we call _hyde_with_timeout which already handles the
    # internal retry via llm.py's MAX_RETRIES. The timeout wrapper adds the wall-
    # clock cap that llm.py alone cannot guarantee.
    hyde_passage = _hyde_with_timeout(query_ar)
    hyde_embedding: Optional[list[float]] = None

    if hyde_passage:
        hyde_embedding = _embed_text(hyde_passage)
        logger.debug("hyde: passage=%r embedding_dim=%s",
                     hyde_passage[:60], len(hyde_embedding) if hyde_embedding else "None")

    updated_qc = {
        **qc,
        "hyde_passage":   hyde_passage,
        "hyde_embedding": hyde_embedding,
    }
    from ...state import trace_append
    if hyde_passage:
        preview = hyde_passage[:70] + "…" if len(hyde_passage) > 70 else hyde_passage
        trace_summary = f'"{preview}"'
    else:
        trace_summary = "Skipped (timeout / no passage generated)"
    return {
        **state,
        "query_context": updated_qc,
        "agent_trace": trace_append(state, stage="2a", icon="🔬", label="HyDE — Hypothetical Verse", summary=trace_summary),
    }


def hyde_passage(query_ar: str) -> dict:
    """
    Tool-registry entry point (§2.9 'hyde_passage').
    Returns {"hyde_passage": <str or None>, "hyde_embedding": <list or None>}
    """
    passage = _hyde_with_timeout(query_ar)
    embedding = _embed_text(passage) if passage else None
    return {"hyde_passage": passage, "hyde_embedding": embedding}
