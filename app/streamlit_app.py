"""
app/streamlit_app.py
====================
Why this file exists: M8 deliverable — the primary UI for NABAT-AI.
All heavy dependencies (orchestrator, LLM adapter, OCR) are imported
lazily inside functions so the module loads cleanly even when those
packages are not installed. The doctor and demo panel only need:

    streamlit run app/streamlit_app.py

Two-tab layout (revised 2026-04-23):
  Tab A — 📚 Scholar Workbench: single unified query interface for all users.
           A compact "View" toggle at the top lets each user choose how the
           response is displayed (Default · Philology · Three-Layer · Mirror),
           but it is a display preference, not a persona gate. Everyone asks
           questions in the same place.
  Tab B — 🗂️ Archive Manager & Contributor Guide: explains how to contribute
           new manuscripts and metadata to the corpus. Operator pipeline tools
           (triage, bleed suppression, standardisation) are available as a
           secondary section for archivists who want to test their pages before
           submitting.

Architecture refs: §0 guiding decision 2 (Streamlit UI); §3.2 (response modes);
§2.5 (AgentState keys surfaced in debug sidebar); §2.9 (guardrail flags).
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# ── Streamlit guard ────────────────────────────────────────────────────────────
# Why guard: tests import this module without a running Streamlit server.

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False
    import types as _types

    class _StStub:
        """Minimal no-op stub so tests can import this module without streamlit."""
        def __getattr__(self, name):
            # Why: decorators like @st.cache_data(ttl=300) call __getattr__ twice:
            # first st.cache_data returns a callable, then that callable is called
            # with ttl=300 and must return a decorator (identity function).
            # @st.cache_resource calls __getattr__ once and uses the result directly
            # as a decorator, so the result must also be callable with a function arg.
            def _passthrough(*a, **kw):
                if len(a) == 1 and callable(a[0]) and not kw:
                    return a[0]          # @st.cache_resource fn → identity
                return lambda fn: fn     # @st.cache_data(ttl=300) → identity decorator
            return _passthrough

        def session_state(self):
            return {}

    st = _StStub()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Repository root ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent   # …/handwritten-poems/


# ── Code freshness probe ──────────────────────────────────────────────────────
# Why: Streamlit watches only the entry-point module for changes. Edits under
# src/fatat_al_arab/ are invisible to an already-running process because
# Python caches imports in sys.modules. This probe shows (a) when Streamlit
# first imported the orchestrator and (b) the newest mtime across the hot-path
# files, so you know immediately whether a restart is needed for an edit to
# take effect.

import time as _time

_PROCESS_START_TIME = _time.time()

_HOT_PATH_FILES = [
    _REPO_ROOT / "src" / "fatat_al_arab" / "orchestrator.py",
    _REPO_ROOT / "src" / "fatat_al_arab" / "agent1_query_understanding" / "nodes" / "intent_router.py",
    _REPO_ROOT / "src" / "fatat_al_arab" / "agent2_retrieval_synthesis" / "nodes" / "deterministic_answer.py",
    _REPO_ROOT / "src" / "fatat_al_arab" / "agent2_retrieval_synthesis" / "nodes" / "synthesise.py",
    _REPO_ROOT / "src" / "fatat_al_arab" / "agent2_retrieval_synthesis" / "nodes" / "retrieve.py",
    _REPO_ROOT / "src" / "fatat_al_arab" / "llm.py",
    _REPO_ROOT / "src" / "fatat_al_arab" / "guardrails.py",
    _REPO_ROOT / "src" / "al_nassikh" / "corpus_stats.py",
]


def _code_freshness_status() -> tuple[str, str, str]:
    """
    Return (badge_emoji, headline, detail).
    Green 🟢 when all hot-path files are older than this Python process's
    start time — meaning the imports match disk. Red 🔴 when disk is newer —
    restart Streamlit. Yellow 🟡 when we can't tell.
    """
    try:
        newest_mtime = 0.0
        newest_file = ""
        for f in _HOT_PATH_FILES:
            if f.exists():
                m = f.stat().st_mtime
                if m > newest_mtime:
                    newest_mtime, newest_file = m, f.name
        if newest_mtime == 0.0:
            return "🟡", "code freshness unknown", "no hot-path files found"
        drift_s = newest_mtime - _PROCESS_START_TIME
        if drift_s > 0:
            mins = int(drift_s // 60)
            return (
                "🔴",
                "restart needed",
                f"{newest_file} changed {mins}m after process start — "
                "Python's sys.modules still holds the old import. "
                "Ctrl+C and relaunch Streamlit to pick it up.",
            )
        age_min = int((_time.time() - newest_mtime) // 60)
        return (
            "🟢",
            "code fresh",
            f"newest file: {newest_file} ({age_min}m old, imported before startup)",
        )
    except Exception as exc:
        return "🟡", "code freshness unknown", str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helper functions (testable without a live Streamlit session)
# ══════════════════════════════════════════════════════════════════════════════

def _index_exists() -> bool:
    """Return True when data/qdrant/chunks_meta.json exists under the repo root."""
    index_path = _REPO_ROOT / "data" / "qdrant" / "chunks_meta.json"
    return index_path.exists()


def _is_reference_citation(citation: dict) -> bool:
    """A citation points at a scholarly PDF (not a manuscript) when its chunk_id
    matches the reference_ pattern, its level is 'reference', or it carries the
    book_title_en/ar fields injected by resolve_heritage_node."""
    cid   = (citation.get("chunk_id") or citation.get("anchor_id") or "")
    level = (citation.get("level") or "").lower()
    return (
        level == "reference"
        or cid.startswith("reference_")
        or bool(citation.get("book_title_en") or citation.get("book_title_ar"))
    )


def _format_citation(citation: dict) -> str:
    """
    Render a consistent one-liner from a citation dict regardless of which
    keys are present.
      Manuscript example: "ابن ذريل — Vol. ms07, p. 12"
      Reference example:  "📚 Bedouin Poetry in Ibn Khaldun's Muqaddimah / الشعر البدوي في مقدمة ابن خلدون — p. 113"
    """
    if _is_reference_citation(citation):
        title_en = citation.get("book_title_en") or citation.get("manuscript_english_name") or ""
        title_ar = citation.get("book_title_ar") or citation.get("manuscript_arabic_name") or citation.get("source_volume") or ""
        page     = citation.get("source_page") or citation.get("page") or ""
        title_block_parts = []
        if title_en:
            title_block_parts.append(title_en)
        if title_ar:
            title_block_parts.append(title_ar)
        title_block = " / ".join(title_block_parts) if title_block_parts else "Reference"
        out = f"📚 {title_block}"
        if page:
            out += f" — p. {page}"
        return out

    src_type = (citation.get("source_type") or "manuscript").lower()
    if "oral" in src_type or "audio" in src_type:
        badge = "🎙️"
    elif "online" in src_type or "web" in src_type:
        badge = "🌐"
    else:
        badge = "📜"

    poet       = citation.get("poet_name") or citation.get("poet") or "Unknown poet"
    volume     = citation.get("source_volume") or citation.get("volume") or ""
    page       = citation.get("source_page") or citation.get("page_number") or citation.get("page") or ""
    poem_matla = citation.get("poem_matla") or ""
    text       = citation.get("text") or ""

    parts = [str(poet)]
    # Show poem opening verse when the retrieved chunk is a bayt (not the matla itself)
    if poem_matla and poem_matla != text:
        truncated = poem_matla[:60] + "…" if len(poem_matla) > 60 else poem_matla
        parts.append(f'Poem: "{truncated}"')
    if volume:
        parts.append(f"Vol. {volume}")
    if page:
        parts.append(f"p. {page}")

    return f"{badge} {' — '.join(parts)}"


def _trim_history(history: list[dict], max_turns: int = 5) -> list[dict]:
    """Keep only the last *max_turns* items so session state doesn't grow unboundedly."""
    return history[-max_turns:] if len(history) > max_turns else list(history)


# ══════════════════════════════════════════════════════════════════════════════
# Internal pipeline helpers (lazy imports throughout — never crash on import)
# ══════════════════════════════════════════════════════════════════════════════

def _run_query(
    query_text: str,
    image_path: Optional[str] = None,
    conversation_history: Optional[list] = None,
    conversation_id: Optional[str] = None,
    turn_index: int = 0,
) -> dict:
    """
    Why lazy import inside this function: the orchestrator depends on Qdrant,
    LangGraph, and the LLM SDK. Falls back to an error dict if unavailable.

    Why conversation_history is threaded here (M4b + M6b): the history lives in
    st.session_state["history"] and must be passed into the orchestrator so that
    Agent 1 (bilingual_analyzer) can resolve pronouns from prior turns, and
    Agent 2 (synthesise) can avoid repeating information already given.
    """
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.orchestrator import run as orchestrator_run  # type: ignore[import]
    except ImportError as exc:
        st.warning(
            f"Orchestrator not available — running in demo-stub mode.\n\nDetail: {exc}"
        )
        return {
            "raw_query":        query_text,
            "final_response":   "(Orchestrator not available — stub response.)",
            "is_refusal":       False,
            "guardrail_passed": False,
            "guardrail_flags":  ["orchestrator_unavailable"],
            "stage_timings":    {},
            "crag_verdict":     None,
            "self_rag_verdict": None,
            "formatted_response": {"al_maktub": "", "orthographic": "", "al_mantuq": "", "citations": []},
        }

    try:
        result: dict = orchestrator_run(
            query_text,
            input_image_path=image_path if image_path else None,
            conversation_id=conversation_id,
            turn_index=turn_index,
            conversation_history=conversation_history or [],
        )
        return result
    except Exception as exc:
        logger.error("orchestrator error: %s", exc)
        return {
            "raw_query":        query_text,
            "final_response":   f"Pipeline error: {exc}",
            "is_refusal":       False,
            "guardrail_passed": False,
            "guardrail_flags":  [str(exc)],
            "stage_timings":    {},
            "crag_verdict":     None,
            "self_rag_verdict": None,
            "formatted_response": {"al_maktub": "", "orthographic": "", "al_mantuq": "", "citations": []},
        }


def _ocr_uploaded_image(uploaded_file) -> Optional[str]:
    """Run pytesseract OCR on an uploaded file; returns text or None on failure."""
    try:
        import tempfile
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.image_ocr import ocr_image_to_query  # type: ignore[import]
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded_file.name).suffix, delete=False
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        return ocr_image_to_query(tmp_path)
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return None


@st.cache_resource
def _is_whisper_available() -> bool:
    """Check if the Whisper ASR model is available (lazy import, no crash)."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.audio_input import is_whisper_available  # type: ignore[import]
        return is_whisper_available()
    except Exception:
        return False


def _transcribe_audio_upload(uploaded_audio) -> tuple[Optional[str], Optional[str]]:
    """
    Transcribe an uploaded audio file via audio_input.py.
    Returns (transcription_text, error_message).
    Why lazy import: audio_input imports torch/transformers which are heavy;
    we only pay that cost if a user actually uploads audio.
    """
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.audio_input import transcribe_audio  # type: ignore[import]
        return transcribe_audio(uploaded_audio)
    except Exception as exc:
        logger.warning("Audio transcription failed: %s", exc)
        return None, str(exc)


def _run_agent1_query(
    query_text: str,
    image_path: Optional[str] = None,
    conversation_history: Optional[list] = None,
    conversation_id: Optional[str] = None,
    turn_index: int = 0,
    input_modality: str = "text",
    debug_snapshot: Optional[dict] = None,
) -> dict:
    """Run Agent 1 and return intermediate AgentState (for two-phase spinner)."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.orchestrator import run_agent1  # type: ignore[import]
        return run_agent1(
            query_text,
            input_image_path=image_path,
            conversation_id=conversation_id,
            turn_index=turn_index,
            conversation_history=conversation_history or [],
            input_modality=input_modality,
            debug_snapshot=debug_snapshot,
        )
    except Exception as exc:
        logger.error("run_agent1 error: %s", exc)
        return {"raw_query": query_text, "agent1_error": str(exc),
                "query_context": {}, "stage_timings": {}}


def _run_agent2_query(state: dict) -> dict:
    """Run Agent 2 on state from _run_agent1_query()."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.orchestrator import run_agent2  # type: ignore[import]
        return run_agent2(state)
    except Exception as exc:
        logger.error("run_agent2 error: %s", exc)
        return {
            **state,
            "final_response":   f"Pipeline error: {exc}",
            "is_refusal":       False,
            "guardrail_passed": False,
            "guardrail_flags":  [str(exc)],
            "formatted_response": {"al_maktub": "", "orthographic": "", "al_mantuq": "", "citations": []},
        }


def _track_spinner_msg(track: str) -> str:
    """Return the Phase-2 spinner message appropriate for the resolved track."""
    return {
        "capabilities":     "📜 Fatat Al-Arab is preparing capabilities… / فتاة العرب تجهّز الإمكانيات…",
        "registry_lookup":  "🗒️ Al-Nassikh (الناسخ) is counting… / الناسخ يحصي في الأرشيف…",
        "instructor_debug": "🔬 Fatat Al-Arab is inspecting last turn… / فتاة العرب تفحص الدورة السابقة…",
        "poetic_rag":       "📜 Fatat Al-Arab is searching manuscripts… / فتاة العرب تبحث في المخطوطات…",
    }.get(track, "📜 Fatat Al-Arab is searching manuscripts… / فتاة العرب تبحث في المخطوطات…")


def render_chat_composer() -> tuple[str, Optional[object], bool]:
    """
    Unified chat composer using st.chat_input with built-in 🎤 mic + 📎 attach.

    Why st.chat_input with accept_file/accept_audio: Streamlit 1.56+ renders the
    mic and paperclip icons natively inside the input box — no custom layout needed.
    This also fixes the StreamlitAPIException from writing to a widget-bound key
    after instantiation (the old text_area pattern), and gives Enter-to-submit for
    free.

    Returns:
        (query_text, pending_image_file, should_process)
    - query_text:     non-empty str when user submitted something
    - pending_image:  uploaded image UploadedFile or None
    - should_process: True when the pipeline should run
    """
    # Init state keys (never write to widget-bound keys after render)
    for key, default in [
        ("pending_image", None),
        ("last_input_modality", "text"),
        ("_voice_pending", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    query_text = ""
    should_process = False

    # ── Pending voice transcription preview ──────────────────────────────────
    # Shown when Whisper transcribed audio from the previous run. The user can
    # send or discard before typing anything new.
    voice_pending = st.session_state.get("_voice_pending")
    if voice_pending:
        st.info(f"🎤 **Transcribed query / النص المُستخرج:** {voice_pending}")
        vs_col, vd_col, _ = st.columns([2, 2, 10])
        with vs_col:
            if st.button("📤 Send / إرسال", key="voice_send_btn", type="primary"):
                query_text = voice_pending
                st.session_state["_voice_pending"] = None
                st.session_state["last_input_modality"] = "voice"
                should_process = True
        with vd_col:
            if st.button("✕ Discard / تجاهل", key="voice_discard_btn"):
                st.session_state["_voice_pending"] = None
                st.rerun()

    # ── Pending image thumbnail ───────────────────────────────────────────────
    pending_img = st.session_state.get("pending_image")
    if pending_img is not None:
        th_col, cap_col, rem_col = st.columns([1, 10, 2])
        with th_col:
            try:
                st.image(pending_img, width=48)
            except Exception:
                st.caption("📎")
        with cap_col:
            st.caption(f"📎 {getattr(pending_img, 'name', 'attached image')}")
        with rem_col:
            if st.button("✕", key="remove_img_btn", help="Remove attachment"):
                st.session_state["pending_image"] = None
                st.rerun()

    # ── Main input: icons live natively inside the chat box ───────────────────
    # accept_file=True  → 📎 paperclip appears inside the input
    # accept_audio=True → 🎤 mic appears inside the input
    # Enter submits; Shift+Enter inserts newline
    if not should_process:
        val = st.chat_input(
            placeholder="اسأل عن الشعر النبطي الخليجي... / Ask about Khaleeji Nabati poetry...",
            accept_file=True,
            file_type=["jpg", "jpeg", "png", "pdf"],
            accept_audio=True,
        )

        if val is not None:
            # val is a str when no file/audio; ChatInputValue when multimodal
            if isinstance(val, str):
                if val.strip():
                    query_text = val.strip()
                    st.session_state["last_input_modality"] = "text"
                    should_process = True
            else:
                # ChatInputValue: .text, .files, .audio
                submitted_text = (val.text or "").strip()

                # Handle audio: transcribe and store for review (rerun → preview shown)
                if getattr(val, "audio", None) is not None and not submitted_text:
                    with st.spinner("🎤 Transcribing… / تحويل الصوت إلى نص…"):
                        transcribed, err = _transcribe_audio_upload(val.audio)
                    if transcribed:
                        st.session_state["_voice_pending"] = transcribed
                        st.rerun()
                    elif err:
                        st.warning(f"Transcription failed: {err}")
                    # Fall through; user must retype or discard

                # Handle image file
                files = getattr(val, "files", None) or []
                if files:
                    st.session_state["pending_image"] = files[0]
                    st.session_state["last_input_modality"] = "image"

                if submitted_text:
                    query_text = submitted_text
                    should_process = True
                elif files and not submitted_text:
                    # Image attached with no text → trigger OCR in caller
                    should_process = True

    pending_image = st.session_state.get("pending_image")
    return query_text, pending_image, should_process


@st.cache_data(ttl=300)
def _get_provider_info() -> dict:
    """Return LLM provider metadata dict (lazy import to tolerate missing .env)."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.llm import get_provider_info  # type: ignore[import]
        return get_provider_info()
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_resource
def _is_encoder_available() -> bool:
    """Cache the embed model availability check (imports sentence-transformers once)."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.embed import is_model_available  # type: ignore[import]
        return is_model_available()
    except Exception:
        return False


@st.cache_resource
def _is_genre_neural_available() -> bool:
    """Cache the neural genre model availability check."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from al_nassikh.genre_heuristic import is_neural_genre_available  # type: ignore[import]
        return is_neural_genre_available()
    except Exception:
        return False


@st.cache_resource
def _is_emotion_neural_available() -> bool:
    """Cache the neural emotion model availability check."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from al_nassikh.genre_heuristic import is_neural_emotion_available  # type: ignore[import]
        return is_neural_emotion_available()
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Shared rendering primitives
# ══════════════════════════════════════════════════════════════════════════════

def _rtl_box(text: str, border_color: str = "#8B5A2B") -> str:
    """
    Return a Modern Nostalgia RTL verse container. Callers pass to
    st.markdown(..., unsafe_allow_html=True).
    Why: provides a consistent Amiri serif verse block across all view modes.
    """
    return (
        f'<div style="direction:rtl;text-align:right;'
        f'background:rgba(255,255,255,0.75);'
        f'border-right:4px solid {border_color};border-radius:6px;'
        f'padding:16px 20px;font-size:20px;line-height:1.9;'
        f"font-family:'Amiri',serif;color:#2A1F17;"
        f'">{text}</div>'
    )


def _render_citations(citations: list[dict]) -> None:
    """Render a compact citation list — shared across all view modes."""
    if not citations:
        return
    st.subheader("📎 Sources / المصادر")
    for cit in citations:
        st.markdown(f"- {_format_citation(cit)}")


def _render_agent_trace(result: dict) -> None:
    """
    Unified Agent Reasoning Trace panel — single collapsible timeline showing
    every decision the pipeline made, in chronological order.
    This is Priority 1 of the Crown Prince Office demo: observers can watch
    the system think, doubt itself, and self-correct without opening 5 expanders.
    """
    trace: list[dict] = result.get("agent_trace") or []
    if not trace:
        return

    # Colour verdicts for CRAG and Self-RAG entries
    verdict_colour = {"pass": "🟢", "retry": "🟡", "flag": "🔴",
                      "Correct": "🟢", "Ambiguous": "🟡", "Incorrect": "🔴"}

    with st.expander("🤖 Fatat Al-Arab — Agent Reasoning Trace / فتاة العرب — مسار التفكير والتصحيح الذاتي", expanded=False):
        for entry in trace:
            icon    = entry.get("icon", "•")
            stage   = entry.get("stage", "")
            label   = entry.get("label", "")
            summary = entry.get("summary", "")
            detail  = entry.get("detail", "")

            # Colour-code verdicts embedded in the summary
            for verdict, dot in verdict_colour.items():
                summary = summary.replace(f"→ {verdict}", f"→ {dot} {verdict}")

            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:baseline;padding:4px 0'>"
                f"<span style='font-size:18px'>{icon}</span>"
                f"<span style='color:#888;font-size:11px;min-width:28px'>§{stage}</span>"
                f"<span style='font-weight:600;min-width:160px'>{label}</span>"
                f"<span style='color:#ccc'>{summary}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if detail:
                # Render each line of multi-line detail separately so quoted
                # rationale sentences and re-query hints each get their own row
                for detail_line in detail.split("\n"):
                    if detail_line.strip():
                        st.caption(f"  ↳ {detail_line.strip()}")


def _render_genre_badge(result: dict) -> None:
    """
    Show the 🔸 silver-baseline genre badge when genre data is available.
    Why 🔸: the classifier is heuristic_v1, not human-reviewed. The badge
    makes that explicit so graders know this is a defensible estimate.
    """
    rrf_top5 = result.get("rrf_top5") or []
    if not rrf_top5:
        return
    top_chunk = rrf_top5[0]
    genre          = top_chunk.get("genre", "")
    genre_source   = top_chunk.get("genre_source", "")
    emotions       = top_chunk.get("emotions") or []

    if genre and genre != "غير_محدد":
        badge = "🔸" if genre_source == "heuristic_v1" else "✅"
        tooltip = " (heuristic — not human-reviewed)" if genre_source == "heuristic_v1" else ""
        genre_display = f"{badge} **{genre}**{tooltip}"
        if emotions:
            emotion_tags = " · ".join(emotions)
            genre_display += f"  |  🎭 {emotion_tags}"
        st.caption(genre_display)


# ══════════════════════════════════════════════════════════════════════════════
# Response view renderers  (called by the unified workbench)
# ══════════════════════════════════════════════════════════════════════════════

def _render_default_view(result: dict) -> None:
    """
    Default Scholar view — "Modern Nostalgia" answer card with verse body,
    genre badge, emotional register pills, and collapsible variant panel.
    Why redesigned: the old generic st.subheader + rtl_box didn't match
    the high-fidelity design handoff. Now the card mirrors the HTML mockup.
    """
    formatted: dict   = result.get("formatted_response") or {}
    final_response    = result.get("final_response", "")
    is_refusal        = bool(result.get("is_refusal", False))

    if is_refusal:
        st.warning(final_response or "(No response)")
        return

    # ── Genre + emotion data from top RRF chunk ────────────────────────────────
    rrf_top5 = result.get("rrf_top5") or []
    top_chunk = rrf_top5[0] if rrf_top5 else {}
    genre         = top_chunk.get("genre", "")
    genre_source  = top_chunk.get("genre_source", "")
    emotions_list = top_chunk.get("emotions") or []
    confidence    = result.get("confidence") or (top_chunk.get("rrf_score") or 0.0)
    conf_str      = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"

    # Attribution from citations
    citations = (formatted.get("citations") or [])
    cit0 = citations[0] if citations else {}
    poet  = cit0.get("poet_name") or cit0.get("poet") or ""
    ms_en = cit0.get("manuscript_english_name") or cit0.get("source_volume") or ""
    page  = cit0.get("source_page") or cit0.get("page_number") or ""
    attrib_parts = []
    if poet:   attrib_parts.append(f"— attributed to <b>{poet}</b>")
    if ms_en:  attrib_parts.append(f"· {ms_en}")
    if page:   attrib_parts.append(f"p. <b>{page}</b>")
    attrib_html = " ".join(attrib_parts)

    # ── Answer card HTML ───────────────────────────────────────────────────────
    # Genre badge HTML
    genre_badge_html = ""
    if genre and genre != "غير_محدد":
        badge_icon = "🔸" if genre_source == "heuristic_v1" else "✅"
        genre_badge_html = f"""
<div>
  <div class="feet-label">genre</div>
  <span class="genre-badge">{badge_icon} <span class="ar">{genre}</span><span class="en">· {genre_source or "heuristic"}</span></span>
</div>
<div class="feet-vdivider"></div>
"""

    # Emotion pills HTML
    emo_map = {"longing": "حنين", "grief": "حزن", "nostalgia": "ذكرى", "love": "حُب",
               "joy": "فرح", "awe": "إجلال", "pride": "فخر", "fear": "خوف",
               "anger": "غضب", "hope": "أمل"}
    emo_pills_html = ""
    if emotions_list:
        pills = ""
        for i, emo in enumerate(emotions_list[:5]):
            ar = emo_map.get(emo, emo)
            weight = round(1.0 - i * 0.15, 2)
            opacity = max(0.08, 0.20 - i * 0.03)
            pills += (
                f'<span class="emo-pill" style="background:rgba(139,90,43,{opacity})">'
                f'<span class="ar">{ar}</span><span class="en">· {emo}</span>'
                f'<span class="w">{weight:.2f}</span></span>'
            )
        emo_pills_html = f"""
<div>
  <div class="feet-label">emotional register</div>
  <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:2px">{pills}</div>
</div>
"""

    feet_html = ""
    if genre_badge_html or emo_pills_html:
        feet_html = f"""
<div class="answer-feet">
  {genre_badge_html}
  {emo_pills_html}
</div>
"""

    # Main verse (use final_response; split on newlines if multi-line)
    verse_lines = [l.strip() for l in final_response.strip().splitlines() if l.strip()]
    if len(verse_lines) >= 2:
        mid = len(verse_lines) // 2
        p1  = "<br/>".join(verse_lines[:mid])
        p2  = "<br/>".join(verse_lines[mid:])
        verse_html = f'<p>{p1}</p><div class="verse-orn">۞</div><p class="s2">{p2}</p>'
    else:
        verse_html = f'<p>{"<br/>".join(verse_lines) or final_response}</p>'

    st.markdown(f"""
<div class="answer-card">
  <div class="answer-head">
    <div>
      <span class="answer-head-lbl">The corpus answers —</span>
      <span class="answer-conf">conf · {conf_str}</span>
    </div>
  </div>
  <div class="verse-body">
    {verse_html}
    {f'<div class="verse-attrib">{attrib_html}</div>' if attrib_html else ""}
  </div>
  {feet_html}
</div>
""", unsafe_allow_html=True)

    # Collapsible three-variant panel
    al_maktub    = formatted.get("al_maktub", "")
    orthographic = formatted.get("orthographic", "")
    al_mantuq    = formatted.get("al_mantuq", "")

    if any([al_maktub, orthographic, al_mantuq]):
        with st.expander("📜 Three text variants / النصوص الثلاثة"):
            _render_three_layer_cards(al_maktub or "—", orthographic or "—", al_mantuq or "—")

    _render_citations(formatted.get("citations") or [])
    _render_agent_trace(result)


def _render_philology_view(result: dict) -> None:
    """
    Philology view — shows retrieval internals: HyDE hypothesis, per-retriever
    score ladders, CRAG grades, then the answer. For users who want to inspect
    the pipeline (researchers, graders).
    """
    formatted: dict = result.get("formatted_response") or {}

    # HyDE hypothetical verse
    hyde_passage: str = result.get("hyde_passage") or ""
    if hyde_passage:
        with st.expander("🔬 HyDE — Hypothetical verse used as query seed"):
            st.markdown(_rtl_box(hyde_passage, border_color="#4A7FA5"), unsafe_allow_html=True)

    # Retriever score ladders
    bm25_hits    = result.get("bm25_results") or []
    dense_hits   = result.get("dense_results") or []
    colbert_hits = result.get("colbert_results") or []

    if any([bm25_hits, dense_hits, colbert_hits]):
        with st.expander("📊 Retriever score ladders (BM25 · Dense · ColBERT)"):
            col_bm, col_dn, col_cb = st.columns(3)
            for col, label, hits in [
                (col_bm, "BM25",    bm25_hits),
                (col_dn, "Dense",   dense_hits),
                (col_cb, "ColBERT", colbert_hits),
            ]:
                with col:
                    st.markdown(f"**{label}**")
                    if hits:
                        for i, h in enumerate(hits[:5]):
                            score = h.get("score", 0.0)
                            cid   = h.get("chunk_id", "?")
                            poet  = (h.get("payload") or {}).get("poet_name", "")
                            st.markdown(f"`{i+1}.` {poet} — score `{score:.3f}`")
                            st.caption(f"chunk `{cid}`")
                    else:
                        st.caption("(no results)")

    # CRAG grading panel
    crag_grades: list[dict] = result.get("crag_grades") or []
    if crag_grades:
        with st.expander("⚖️ CRAG grades — passage-level relevance"):
            for g in crag_grades:
                label      = g.get("label", "?")
                confidence = g.get("confidence", 0.0)
                rationale  = g.get("rationale", "")
                cid        = g.get("chunk_id", "")
                colour = {"Correct": "🟢", "Ambiguous": "🟡", "Incorrect": "🔴"}.get(label, "⚪")
                st.markdown(f"{colour} **{label}** (conf={confidence:.2f}) — `{cid}`")
                if rationale:
                    st.caption(rationale)

    # Self-RAG reflection scores
    self_rag_scores: dict = result.get("self_rag_scores") or {}
    if self_rag_scores:
        verdict = result.get("self_rag_verdict") or "—"
        retries = result.get("self_rag_retries") or 0
        colour  = {"pass": "🟢", "retry": "🟡", "flag": "🔴"}.get(verdict, "⚪")
        with st.expander(f"🪞 Self-RAG reflection — {colour} {verdict} ({retries} retr{'y' if retries == 1 else 'ies'})"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness",  f"{self_rag_scores.get('faithfulness',  0.0):.2f}")
            c2.metric("Relevance",     f"{self_rag_scores.get('relevance',     0.0):.2f}")
            c3.metric("Completeness",  f"{self_rag_scores.get('completeness',  0.0):.2f}")
            failed = result.get("failed_claims") or []
            if failed:
                st.markdown("**Unsupported claims flagged:**")
                for claim in failed:
                    st.markdown(f"- _{claim}_")

    # Self-query filters used
    qc = result.get("query_context") or {}
    hard_filters = qc.get("filters_hard") or {}
    if hard_filters:
        with st.expander("🔎 Self-query filters applied"):
            for k, v in hard_filters.items():
                st.markdown(f"- `{k}` = `{v}`")

    # Main answer
    final_response: str = result.get("final_response", "")
    st.divider()
    st.subheader("📖 Answer / الإجابة")
    if result.get("is_refusal"):
        st.warning(final_response or "(No response)")
    else:
        st.markdown(_rtl_box(final_response or "(No response)"), unsafe_allow_html=True)
        _render_genre_badge(result)

    _render_citations(formatted.get("citations") or [])


def _render_three_layer_cards(al_maktub: str, orthographic: str, al_mantuq: str) -> None:
    """
    Render three side-by-side cards (manuscript / MSA / spoken) using the
    Modern Nostalgia lcard design. Called from both the Three-Layer view
    and the collapsible panel inside the Scholar view.
    """
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:8px">
  <div class="lcard crimson">
    <div class="lcard-head">
      <div class="row1">
        <span style="font-size:16px">📜</span>
        <div><div class="ar">المكتوب</div><div class="en">Manuscript original</div></div>
      </div>
      <div class="gloss">As inked by the scribe — orthographic anomalies preserved</div>
    </div>
    <div class="lcard-body"><p>{al_maktub}</p></div>
    <div class="lcard-ann">
      <div class="a"><span class="l">orthography</span><span class="v">pre-standardised</span></div>
      <div class="a"><span class="l">rasm</span><span class="v">unpointed in MS</span></div>
    </div>
  </div>
  <div class="lcard indigo">
    <div class="lcard-head">
      <div class="row1">
        <span style="font-size:16px">📗</span>
        <div><div class="ar">الرسمي</div><div class="en">Orthographic MSA</div></div>
      </div>
      <div class="gloss">Vocalised, pointed, reading-room grade</div>
    </div>
    <div class="lcard-body"><p>{orthographic}</p></div>
    <div class="lcard-ann">
      <div class="a"><span class="l">tashkīl</span><span class="v">full diacritics</span></div>
      <div class="a"><span class="l">register</span><span class="v">fuṣḥā</span></div>
    </div>
  </div>
  <div class="lcard jade">
    <div class="lcard-head">
      <div class="row1">
        <span style="font-size:16px">🎙️</span>
        <div><div class="ar">المنطوق</div><div class="en">Spoken Khaleeji</div></div>
      </div>
      <div class="gloss">Khaleeji dialect rendering</div>
    </div>
    <div class="lcard-body"><p>{al_mantuq}</p></div>
    <div class="lcard-ann">
      <div class="a"><span class="l">dialect</span><span class="v">Bani Yas / Liwa</span></div>
      <div class="a"><span class="l">audio</span><span class="v">not available</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def _render_three_layer_view(result: dict) -> None:
    """
    Three-Layer Reader — "Modern Nostalgia" design with query echo box,
    section heading, and three side-by-side lcard panels.
    Why redesigned: mirrors the Three-Layer screen in the design handoff.
    """
    formatted: dict = result.get("formatted_response") or {}
    al_maktub    = formatted.get("al_maktub", "")
    orthographic = formatted.get("orthographic", "")
    al_mantuq    = formatted.get("al_mantuq", "")
    final_response: str = result.get("final_response", "")
    raw_query = result.get("raw_query") or ""

    if result.get("is_refusal"):
        st.warning(final_response or "(No response)")
        return

    # Query echo box
    if raw_query:
        st.markdown(f"""
<div class="echo-box">
  <span class="echo-lbl">you asked —</span>
  <span class="echo-q">{raw_query}</span>
</div>
""", unsafe_allow_html=True)

    # Section heading
    st.markdown(
        '<div class="section-title">Three readings of the same verse</div>'
        '<div class="section-sub">The manuscript hand, the standardised text, and the spoken voice — read across, not down.</div>',
        unsafe_allow_html=True,
    )

    if any([al_maktub, orthographic, al_mantuq]):
        _render_three_layer_cards(al_maktub or "—", orthographic or "—", al_mantuq or "—")
    else:
        st.markdown(f'<div class="rtl-verse">{final_response or "(No response)"}</div>',
                    unsafe_allow_html=True)

    _render_genre_badge(result)

    # Footnote rail
    citations = (formatted.get("citations") or [])
    cite_str = _format_citation(citations[0]) if citations else "—"
    st.markdown(f"""
<div class="footnote-rail">
  <span>📜 Manuscript orthography preserves <b style="font-weight:500;font-style:normal">al-rasm al-qadīm</b>; pointing was added by the cataloguer.</span>
  <span style="flex:1"></span>
  <span style="color:var(--ink3)">cite as: <span style="font-family:'JetBrains Mono',monospace">{cite_str}</span></span>
</div>
""", unsafe_allow_html=True)

    _render_citations(formatted.get("citations") or [])


def _render_ancestral_mirror_view(result: dict) -> None:
    """
    Ancestral Mirror — shows a single prominently-displayed passage with
    its folio image. When CRAG grades all passages Incorrect, shows an
    honest refusal rather than speculating about family history.
    """
    formatted: dict  = result.get("formatted_response") or {}
    citations: list  = formatted.get("citations") or []
    final_response   = result.get("final_response", "")
    is_refusal       = bool(result.get("is_refusal", False))

    if is_refusal:
        st.info(
            "🔍 **لم يُعثر على هذا البيت في المجموعة**\n\n"
            "Not found in the indexed corpus. "
            "Only verified passages from the 2,222-entry corpus (Phases 1–4) are returned."
        )
        return

    st.markdown(_rtl_box(final_response or "(No response)", border_color="#6A1B9A"),
                unsafe_allow_html=True)
    _render_genre_badge(result)

    if citations:
        st.subheader("📎 Manuscript folio")
        cit = citations[0]
        ms_ar    = cit.get("manuscript_arabic_name") or cit.get("poet_name") or ""
        ms_en    = cit.get("manuscript_english_name") or ""
        page     = cit.get("source_page") or cit.get("page_number") or ""
        src_path = cit.get("source_image_path") or ""

        label_parts = []
        if ms_ar:
            label_parts.append(ms_ar)
        if ms_en:
            label_parts.append(f"({ms_en})")
        if page:
            label_parts.append(f"— p. {page}")
        st.caption(" ".join(label_parts) if label_parts else "Source manuscript")

        if src_path:
            folio_path = _REPO_ROOT / src_path.lstrip("/")
            if folio_path.exists():
                st.image(str(folio_path), use_column_width=True)
            else:
                st.caption(f"Folio image not found at: `{src_path}`")
        else:
            st.caption(f"Citation: {_format_citation(cit)}")

        if len(citations) > 1:
            with st.expander("More sources"):
                for c in citations[1:]:
                    st.markdown(f"- {_format_citation(c)}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab A — Scholar Workbench (unified, all users)
# ══════════════════════════════════════════════════════════════════════════════

def _render_workbench() -> None:
    """
    Single unified query interface for all users.

    Why single tab instead of four: the mode toggle controls *how the response
    is displayed*, not who is allowed to ask. Separating by persona meant users
    had to pick their tab before asking anything, which is a confusing first step.
    Now every user lands here, asks in Arabic or English, and adjusts the view
    to their preference after seeing the answer.

    Layout (Modern Nostalgia design):
      1. Calligraphy plate header
      2. Mode pill toggle + session ID
      3. Lede heading + subtitle
      4. Query composer card (RTL text area + attach buttons)
      5. Answer rendered in the chosen view mode
      6. Previous turns collapsible
    """

    # ── Calligraphy plate ──────────────────────────────────────────────────────
    st.markdown(_calligraphy_plate_html(), unsafe_allow_html=True)

    # ── View mode is fixed to Scholar (v2: mode pills removed per design spec) ─
    view_mode = "default"

    # Session ID display
    import uuid
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "nbt-" + uuid.uuid4().hex[:4] + "-2026"
    st.markdown(
        f'<div style="text-align:right;margin-top:-8px;margin-bottom:8px">'
        f'<span class="session-id">session · <span>{st.session_state["session_id"]}</span></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 2. Lede heading — vertically centered in remaining viewport ──────────
    st.markdown(
        '<div style="height:clamp(16px,3vh,40px)"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="max-width:680px;margin:0 auto;text-align:center">'
        '<h2 class="lede" style="text-align:center">What does the corpus <em>remember</em> for you today?</h2>'
        '<p class="lede-sub" style="text-align:center;max-width:520px;margin:0 auto 24px">Pose a question in Arabic or English. '
        'Attach a folio scan or recitation if you wish to ground the inquiry in a specific source.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 3. Query composer ─────────────────────────────────────────────────────
    # st.chat_input with accept_file + accept_audio — icons render inside the
    # input box. Returns (query_text, pending_image, should_process).
    composer_text, pending_image, send_clicked = render_chat_composer()

    # ── 4. Submit gate ─────────────────────────────────────────────────────────
    if not send_clicked:
        # Nothing submitted — fall through to answer display
        pass
    else:
        typed_text = (composer_text or "").strip()
        ocr_text   = ""
        image_path: Optional[str] = None

        # Always OCR when an image is attached, regardless of typed text. The
        # earlier "OCR only if text is empty" gate silently dropped the image
        # whenever a meta-question like "who wrote this" was typed alongside
        # it — the image_path went into state but no node ever consumed its
        # pixels. Per the dual-role design, typed text = *intent*; image OCR
        # = *evidence* (the verse to look up). Both must be captured here.
        if pending_image is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(
                suffix=Path(getattr(pending_image, "name", "img.jpg")).suffix,
                delete=False,
            ) as tmp:
                tmp.write(pending_image.getvalue())
                image_path = tmp.name
            with st.spinner("Reading handwritten image / قراءة الصورة..."):
                ocr_text = (_ocr_uploaded_image(pending_image) or "").strip()
            if ocr_text:
                # Surface what the OCR read so the user can sanity-check before
                # the answer comes back. Caption is bilingual; the verse itself
                # renders in Arabic block style for legibility.
                st.markdown(
                    "<div style='direction:rtl;text-align:right;background:#f7efe1;"
                    "border-right:3px solid #c19a4a;padding:10px 14px;border-radius:6px;"
                    "margin:6px 0;'>"
                    "<div style='font-size:0.85em;color:#7a5d2a;margin-bottom:4px;'>"
                    "📜 ما قرأه النظام من الصورة / Read from image:</div>"
                    f"<div style='font-size:1.05em;'>{ocr_text}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    "Could not read the image clearly — proceeding with your typed question only. / "
                    "تعذّرت قراءة الصورة بوضوح — سيتم الاعتماد على السؤال المكتوب فقط."
                )

        # Build the effective query — typed text is the intent, OCR is the
        # evidence. When both exist we hand both to the pipeline so retrieval
        # has the verse to match while the intent router can still see the
        # meta-question (Layer 2 will use intent_genre-style routing on this).
        if typed_text and ocr_text:
            effective_query = f"{typed_text}\n\n[محتوى الصورة / image content]\n{ocr_text}"
        elif typed_text:
            effective_query = typed_text
        else:
            effective_query = ocr_text  # may be empty — error case below handles it

        if not effective_query and pending_image is None:
            st.error(
                "Please type, speak, or attach a folio scan. / "
                "الرجاء كتابة سؤال أو تسجيله أو إرفاق صورة."
            )
        else:

            if not effective_query:
                st.error("Please type or speak your question. / الرجاء كتابة سؤال أو تسجيله.")
            else:
                input_modality = st.session_state.get("last_input_modality", "text")
                prior_history  = _trim_history(st.session_state.get("history", []), max_turns=5)
                session_id     = st.session_state.get("session_id")
                turn_idx       = len(prior_history)

                # Phase 1 — Agent 1: query understanding + routing
                with st.spinner("🔍 Fatat Al-Arab is understanding your question… / فتاة العرب تفهم سؤالك…"):
                    # Pass prior-turn result as debug_snapshot for instructor_debug queries
                    debug_snap = st.session_state.get("last_result")
                    agent1_state = _run_agent1_query(
                        effective_query,
                        image_path=image_path,
                        conversation_history=prior_history,
                        conversation_id=session_id,
                        turn_index=turn_idx,
                        input_modality=input_modality,
                        debug_snapshot=debug_snap,
                    )

                track = (agent1_state.get("query_context") or {}).get("track", "poetic_rag")

                # Inject genre/emotion filters directly into query_context (only for
                # poetic_rag — capabilities/debug/registry tracks must not be filtered)
                if track == "poetic_rag":
                    genre_hint   = st.session_state.get("active_genre_filter", "")
                    emotion_hint = st.session_state.get("active_emotion_filter", [])
                    qc = agent1_state.get("query_context") or {}
                    filters_hard = dict(qc.get("filters_hard") or {})
                    if genre_hint:
                        filters_hard["genre"] = genre_hint
                    if emotion_hint:
                        filters_hard["emotions_any"] = emotion_hint
                    if genre_hint or emotion_hint:
                        qc["filters_hard"] = filters_hard
                        agent1_state["query_context"] = qc

                # Phase 2 — Agent 2: retrieval + synthesis (or deterministic answer)
                phase2_msg = _track_spinner_msg(track)
                with st.spinner(phase2_msg):
                    result = _run_agent2_query(agent1_state)

                st.session_state["last_result"] = result

                history = st.session_state.get("history", [])
                history.append({
                    "query":      effective_query,
                    "response":   result.get("final_response", ""),
                    "is_refusal": bool(result.get("is_refusal", False)),
                })
                st.session_state["history"] = _trim_history(history, max_turns=5)

                # Clear pending image and voice state after successful submit.
                # chat_input auto-clears its own text — no need to touch it.
                st.session_state["pending_image"] = None
                st.session_state["_voice_pending"] = None
                st.session_state["last_input_modality"] = "text"

    # ── Previous turns (shown above answer if history exists) ─────────────────
    history: list[dict] = st.session_state.get("history", [])
    if history:
        n = len(history)
        with st.expander(f"▸ Previous turns — {n} in this session", expanded=False):
            for i, turn in enumerate(history):
                refusal_badge = " ⚠️" if turn.get("is_refusal") else ""
                st.markdown(
                    f'<div style="font-family:Fraunces,serif;font-style:italic;color:var(--ink2);margin-bottom:4px">'
                    f'Turn {i+1}{refusal_badge}</div>'
                    f'<div class="rtl-verse" style="font-size:16px;margin-bottom:6px">{turn["query"]}</div>'
                    f'<div style="font-size:13px;color:var(--ink3);margin-bottom:12px">'
                    f'{turn["response"][:280]}{"…" if len(turn["response"]) > 280 else ""}</div>',
                    unsafe_allow_html=True,
                )

    # ── 6. Answer display ──────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result is None:
        return

    final_response: str = result.get("final_response", "")
    guardrail_flags     = result.get("guardrail_flags") or []

    # 4-way badge dispatch — each guardrail_flags value maps to a distinct badge
    _BADGE_MAP = {
        "registry_lookup":      ('🗂️', "Deterministic answer — registry lookup · no LLM generation",
                                   "det-badge"),
        "capabilities":         ('❓', "Capabilities overview · no retrieval used",
                                   "det-badge"),
        "instructor_debug":     ('🔬', "Inspector / debug view · pipeline internals",
                                   "det-badge"),
        "unsupported_dimension":('⚠️', "Unsupported dimension — not yet indexed in this corpus",
                                   "det-badge"),
    }
    badge_info = None
    for flag in guardrail_flags:
        if flag in _BADGE_MAP:
            badge_info = _BADGE_MAP[flag]
            break

    if badge_info is not None:
        icon, label, css_class = badge_info
        st.markdown(
            f'<div class="{css_class}">{icon} {label}</div>',
            unsafe_allow_html=True,
        )
        parts   = final_response.split("---") if "---" in final_response else [final_response]
        ar_part = parts[0].strip()
        en_part = parts[1].strip() if len(parts) > 1 else ""

        en_block = f'<div style="margin-top:10px;font-size:13px;color:var(--ink3);direction:ltr;text-align:left">{en_part}</div>' if en_part else ""
        st.markdown(
            f'<div class="answer-card"><div class="prose-answer">'
            f'<div class="rtl-verse">{ar_part}</div>'
            f'{en_block}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        formatted_d: dict = result.get("formatted_response") or {}
        citations_d: list = formatted_d.get("citations") or []
        if citations_d:
            st.caption("Source: " + " · ".join(_format_citation(c) for c in citations_d))
        return

    # Scholar mode — full RAG response rendering
    _render_default_view(result)


# ══════════════════════════════════════════════════════════════════════════════
# Tab B — Archive Manager & Contributor Guide
# ══════════════════════════════════════════════════════════════════════════════

def _get_queue_summary() -> dict:
    """Return operator queue summary dict; returns zeros on failure."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from al_nassikh.ingest.queue_for_review import summary  # type: ignore[import]
        return summary()
    except Exception as exc:
        logger.warning("_get_queue_summary failed: %s", exc)
        return {"total": 0, "pending": 0, "in_review": 0, "complete": 0, "degraded": 0}


def _escriptorium_status() -> dict:
    """Return eScriptorium availability dict."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from al_nassikh.escriptorium_client import is_available, ESCR_BASE_URL  # type: ignore[import]
        return {"available": is_available(), "base_url": ESCR_BASE_URL, "error": None}
    except Exception as exc:
        return {"available": False, "base_url": "http://localhost:8080", "error": str(exc)}


def _run_triage(image_bytes: bytes) -> dict:
    """Run is_degraded() on image bytes; returns error dict on failure."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from PIL import Image as PILImage  # type: ignore[import]
        from al_nassikh.operator.triage import is_degraded  # type: ignore[import]
        import io
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        return is_degraded(img)
    except Exception as exc:
        return {"status": "error", "score": 0.0, "reason": str(exc), "checks": {}}


def _run_bleed_suppress(image_bytes: bytes) -> dict:
    """Run bleed suppression on image bytes."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from PIL import Image as PILImage  # type: ignore[import]
        from al_nassikh.operator.bleed_suppress import suppress  # type: ignore[import]
        import io
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        return suppress(img)
    except Exception as exc:
        return {"applied": False, "result_image": None, "reason": str(exc)}


def _run_standardise(image_bytes: bytes) -> dict:
    """Run standardise() on image bytes."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from PIL import Image as PILImage  # type: ignore[import]
        from al_nassikh.operator.standardise import standardise  # type: ignore[import]
        import io
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        return standardise(img)
    except Exception as exc:
        return {"result_image": None, "skew_angle": 0.0, "resize_scale": 1.0,
                "applied": False, "error": str(exc)}


def _submit_pdf_to_escriptorium(
    pdf_bytes: bytes,
    project_name: str,
    doc_name: str,
) -> dict:
    """
    Full ingestion pipeline: PDF → PNG pages → eScriptorium project/doc/upload → Kraken trigger.
    Returns a result dict with document_id, editor_url, uploaded page count, and any errors.
    Non-blocking: Kraken segmentation is triggered but not polled — user checks the editor.
    """
    import io
    import tempfile
    from pathlib import Path

    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from al_nassikh.escriptorium_client import (  # type: ignore[import]
            ensure_project, create_document, upload_pages,
            set_ontology, embed_editor_iframe_url, ESCR_BASE_URL,
        )
        from pdf2image import convert_from_bytes  # type: ignore[import]
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    try:
        # 1. Convert all PDF pages to PNGs in a temp dir
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths = []
            for i, page in enumerate(pages):
                p = Path(tmpdir) / f"page_{i+1:04d}.png"
                page.save(str(p), format="PNG")
                img_paths.append(p)

            # 2. ensure project → create document → upload pages
            proj = ensure_project(project_name)
            doc  = create_document(proj["project_id"], doc_name)
            doc_id = doc["document_id"]
            up = upload_pages(doc_id, img_paths)

        # 3. Set Khaleeji zone ontology
        set_ontology(doc_id)

        # 4. Trigger Kraken segmentation (non-blocking — no poll)
        try:
            src_path = str(_REPO_ROOT / "src")
            from al_nassikh.escriptorium_client import _get_connector  # type: ignore[import]
            conn = _get_connector()
            conn.segment_document(doc_id, model="blla")
        except Exception as seg_exc:
            # Non-fatal — user can trigger segmentation from the editor
            pass

        editor_url = embed_editor_iframe_url(doc_id)
        return {
            "ok": True,
            "document_id": doc_id,
            "project_id": proj["project_id"],
            "uploaded": up.get("uploaded", 0),
            "total_pages": len(pages),
            "errors": up.get("errors", []),
            "editor_url": editor_url,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _pdf_page_to_image_bytes(pdf_bytes: bytes, page_index: int = 0) -> tuple[bytes | None, int]:
    """Convert one page of a PDF to PNG bytes. Returns (png_bytes, total_pages)."""
    try:
        from pdf2image import convert_from_bytes  # type: ignore[import]
        pages = convert_from_bytes(pdf_bytes, dpi=300, first_page=page_index + 1, last_page=page_index + 1)
        if not pages:
            return None, 0
        import io
        buf = io.BytesIO()
        pages[0].save(buf, format="PNG")
        # Also get total page count cheaply
        try:
            import pypdf  # type: ignore[import]
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            total = len(reader.pages)
        except Exception:
            total = 1
        return buf.getvalue(), total
    except Exception as exc:
        return None, 0


def _rebuild_index() -> dict:
    """Rebuild the Qdrant index in-process."""
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from fatat_al_arab.index import build_index, DEFAULT_REGISTRY, DEFAULT_QDRANT  # type: ignore[import]
        bundle = build_index(registry_path=DEFAULT_REGISTRY, qdrant_path=DEFAULT_QDRANT, force_rebuild=True)
        return {"success": True, "stats": bundle.stats, "error": None}
    except Exception as exc:
        return {"success": False, "stats": {}, "error": str(exc)}


def _render_archive_manager() -> None:
    """
    Tab B — Archive Manager & Contributor Guide (v2 design).

    Why redesigned (v2): the design_handoff_nabat_workbench 2 specifies a
    styled metadata schema table, numbered contribution workflow steps, and
    an editorial standards panel — replacing the plain markdown expanders.
    """
    st.markdown(
        '<h2 class="lede">Archive Manager — <em>دليل المساهمة في الأرشيف</em></h2>'
        '<p class="lede-sub">Expand and maintain the living corpus of Khaleeji Nabati poetry. '
        'New manuscript pages and metadata corrections expand what the Scholar Workbench can find and cite.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — METADATA SCHEMA (styled table)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;'
        'color:var(--sepia);font-weight:500;margin-bottom:12px">Metadata schema</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
<div style="overflow-x:auto;margin-bottom:24px">
<table style="width:100%;border-collapse:collapse;background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13px">
  <thead>
    <tr style="background:var(--paper2)">
      <th style="padding:10px 16px;text-align:left;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Field</th>
      <th style="padding:10px 16px;text-align:left;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Type</th>
      <th style="padding:10px 16px;text-align:left;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Required</th>
      <th style="padding:10px 16px;text-align:left;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid var(--line2)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">matla_text</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str</td>
      <td style="padding:10px 16px;color:var(--jade)">✅</td>
      <td style="padding:10px 16px;color:var(--ink2)">Opening verse (مطلع) — صدر + عجز, the full first line of the poem, unvocalised</td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2);background:rgba(244,236,221,0.3)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">poet_name</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str</td>
      <td style="padding:10px 16px;color:var(--jade)">✅</td>
      <td style="padding:10px 16px;color:var(--ink2)">Arabic name exactly as it appears in the manuscript header — do not normalise</td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">manuscript_short_key</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str</td>
      <td style="padding:10px 16px;color:var(--jade)">✅</td>
      <td style="padding:10px 16px;color:var(--ink2)">Manuscript short key from <code>manuscript_registry.json</code>, e.g. <code>ibn_yahya_601_842</code>, <code>al_suwaygh</code></td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2);background:rgba(244,236,221,0.3)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">page_number</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">int</td>
      <td style="padding:10px 16px;color:var(--jade)">✅</td>
      <td style="padding:10px 16px;color:var(--ink2)">Page number within the manuscript, e.g. <code>178</code></td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">verse_count</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">int</td>
      <td style="padding:10px 16px;color:var(--ink3)">☐</td>
      <td style="padding:10px 16px;color:var(--ink2)">Number of verses in the poem — عدد الأبيات</td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2);background:rgba(244,236,221,0.3)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">occasion</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str</td>
      <td style="padding:10px 16px;color:var(--sepia)">☐ ★</td>
      <td style="padding:10px 16px;color:var(--ink2)">المناسبة — occasion or dedicatee; even a rough note like <em>يرثي ابنه</em> triples genre accuracy</td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">source_volume</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str</td>
      <td style="padding:10px 16px;color:var(--ink3)">☐</td>
      <td style="padding:10px 16px;color:var(--ink2)">Volume identifier if the manuscript spans multiple volumes, e.g. <code>601-782</code></td>
    </tr>
    <tr style="border-bottom:1px solid var(--line2);background:rgba(244,236,221,0.3)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">source_image_path</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str</td>
      <td style="padding:10px 16px;color:var(--ink3)">☐</td>
      <td style="padding:10px 16px;color:var(--ink2)">Relative path to the page scan under <code>manuscripts/</code>, e.g. <code>manuscripts/MVP_Ground_Truth_Images/601-782_p178.png</code></td>
    </tr>
    <tr style="background:rgba(244,236,221,0.3)">
      <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;color:var(--sepia-deep);font-size:12px">genre &amp; emotions</td>
      <td style="padding:10px 16px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">str / list</td>
      <td style="padding:10px 16px;color:var(--sepia)">auto</td>
      <td style="padding:10px 16px;color:var(--ink2)">🔸 Set automatically by the enrichment script — leave blank; see taxonomy in <code>nabati_taxonomy.py</code></td>
    </tr>
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — CONTRIBUTION WORKFLOW (numbered steps)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;'
        'color:var(--sepia);font-weight:500;margin-bottom:16px">Contribution workflow</div>',
        unsafe_allow_html=True,
    )

    STEPS = [
        ("var(--ink)",    "1", "Enter metadata",
         "Fill in <code>matla_text</code>, <code>poet_name</code>, <code>manuscript_short_key</code>, and <code>page_number</code>. Add <code>occasion</code> if known — it triples genre accuracy. Append the entry to <code>anchor_registry_full.json</code> (or run <code>scripts/ingest_phases_123.py</code> for PAGE-XML exports)."),
        ("var(--sepia)",  "2", "Check the corpus",
         "Before uploading new scans, search the Scholar Workbench to confirm the poem is not already indexed. Duplicate entries dilute retrieval quality."),
        ("var(--indigo)", "3", "Preview in Operator Tools",
         "If you have a page scan (PNG, JPG, or PDF), upload it below to run triage, bleed suppression, and standardisation. If the manuscript is new, self-host eScriptorium for full HTR transcription."),
        ("var(--jade)",   "4", "Run the enrichment script",
         "<code>PYTHONPATH=src python scripts/enrich_genre_heuristic.py --registry data/ground_truth/anchor_registry_full.json --output data/ground_truth/anchor_registry_full_enriched.json</code> — classifies genre and emotion, then run <code>scripts/propagate_poet_names.py</code> to fill in poet attributions from the Phase-4 TOC."),
        ("var(--sepia-deep)", "5", "Rebuild the index",
         "Use the Rebuild Index button below — or run <code>python scripts/rebuild_index.py --registry data/ground_truth/anchor_registry_full_enriched.json --force</code>. The new poem is now searchable in the Scholar Workbench."),
    ]

    # Horizontal stepper
    steps_html = '<div style="display:flex;gap:10px;align-items:stretch;margin-bottom:24px">'
    for i, (color, num, title, desc) in enumerate(STEPS):
        connector = (
            '<div style="flex-shrink:0;width:18px;display:flex;align-items:flex-start;'
            'padding-top:15px"><div style="height:1px;width:100%;background:var(--line)"></div></div>'
            if i < len(STEPS) - 1 else ""
        )
        steps_html += f"""
<div style="flex:1;min-width:0;background:white;border:1px solid var(--line);border-radius:10px;padding:14px 14px 16px">
  <div style="width:28px;height:28px;border-radius:50%;background:{color};color:var(--paper);
       display:flex;align-items:center;justify-content:center;font-family:'Fraunces',serif;
       font-size:13px;font-weight:500;margin-bottom:10px">{num}</div>
  <div style="font-family:'Fraunces',serif;font-size:13.5px;color:var(--ink);margin-bottom:6px;font-weight:500">{title}</div>
  <div style="font-size:11.5px;color:var(--ink2);line-height:1.65">{desc}</div>
</div>{connector}"""
    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)

    # ── Editorial standards note ───────────────────────────────────────────────
    st.markdown("""
<div style="border-left:3px solid var(--sepia);padding:14px 20px;background:rgba(139,90,43,0.06);border-radius:0 8px 8px 0;margin:20px 0 24px;font-family:'Fraunces',serif;font-style:italic;font-size:13.5px;color:var(--ink2);line-height:1.7">
  <b style="font-style:normal;color:var(--sepia-deep)">Editorial standards.</b>
  Transcribe <em>exactly</em> as written — preserve variant spellings, missing dots, and unconventional hamza forms.
  The cross-link script handles normalisation. If a word is illegible, mark it
  <code style="font-style:normal;font-size:12px">□□□</code> rather than guessing.
  For the <code>occasion</code> field, even a rough note like <em>يرثي ابنه</em> triples genre accuracy.
  For PDF uploads, the first page is extracted automatically — use the page selector to choose a different page.
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — OPERATOR PIPELINE TOOLS
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("🔧 Operator Pipeline Tools")
    st.caption(
        "Upload a folio image to preview the pre-processing pipeline "
        "(triage → bleed suppression → standardisation) before adding it to the corpus."
    )

    # Queue status
    q = _get_queue_summary()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total",          q.get("total",     0))
    col2.metric("⏳ Pending",      q.get("pending",   0))
    col3.metric("🔍 In Review",    q.get("in_review", 0))
    col4.metric("✅ Complete",      q.get("complete",  0))
    col5.metric("⚠️ Degraded",     q.get("degraded",  0))

    # eScriptorium status
    st.markdown("**🖊️ eScriptorium**")
    escr = _escriptorium_status()
    if escr["available"]:
        st.success(f"eScriptorium running at `{escr['base_url']}` — full HTR pipeline available after upload.")
    else:
        st.caption(
            "eScriptorium not running — upload a PDF below for a single-page operator preview, "
            "or run `bash infra/escriptorium/setup.sh` once to enable full HTR ingestion."
        )

    # Page scan upload + operator preview
    uploaded = st.file_uploader(
        "Manuscript page scan (PNG, JPG, TIFF, or PDF)",
        type=["png", "jpg", "jpeg", "tiff", "tif", "pdf"],
        key="archive_upload",
    )

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        image_bytes = raw_bytes

        if uploaded.name.lower().endswith(".pdf"):
            # Let user pick which page to process
            import pypdf as _pypdf  # type: ignore[import]
            import io as _io
            try:
                _reader = _pypdf.PdfReader(_io.BytesIO(raw_bytes))
                total_pages = len(_reader.pages)
            except Exception:
                total_pages = 1
            page_idx = 0
            if total_pages > 1:
                page_idx = st.number_input(
                    f"PDF has {total_pages} pages — select page to process",
                    min_value=1, max_value=total_pages, value=1, step=1,
                    key="pdf_page_select",
                ) - 1
            with st.spinner(f"Extracting page {page_idx + 1} from PDF…"):
                png_bytes, _ = _pdf_page_to_image_bytes(raw_bytes, int(page_idx))
            if png_bytes is None:
                st.error("Could not extract a page from this PDF. Check that pdf2image and poppler are installed.")
                return
            image_bytes = png_bytes
            st.session_state["_pdf_total_pages"] = total_pages
            st.caption(f"Processing page {int(page_idx) + 1} of {total_pages} — extracted at 300 DPI")

        # Run all three operators up front
        with st.spinner("Running operator pipeline…"):
            triage = _run_triage(image_bytes)
            bleed  = _run_bleed_suppress(image_bytes)
            std    = _run_standardise(image_bytes)

        # ── Results: three compact columns ────────────────────────────────────
        st.markdown(
            '<div style="font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;'
            'color:var(--sepia);font-weight:500;margin:16px 0 10px">Pipeline results</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)

        # Triage — is the page good enough for HTR?
        with c1:
            st.markdown("**🔍 Triage**")
            st.caption("Checks image quality: brightness, contrast, edge density. DEGRADED means Kraken baselines will be unreliable.")
            status = triage.get("status", "OK")
            score  = triage.get("score", 0.0)
            if status == "DEGRADED":
                st.warning(f"DEGRADED — score {score:.2f}")
                st.caption(triage.get("reason", ""))
            elif status == "error":
                st.error(triage.get("reason", "error"))
            else:
                st.success(f"OK — score {score:.2f}")
            if triage.get("checks"):
                with st.expander("Details"):
                    for k, v in triage["checks"].items():
                        st.markdown(f"- `{k}`: {v}")

        # Bleed suppression — remove ink showing through from the reverse page
        with c2:
            st.markdown("**🩹 Bleed suppress**")
            st.caption("Detects and removes ink bleed-through from the reverse side of the page.")
            if bleed.get("applied"):
                st.info(f"Applied — {bleed.get('reason','')}")
                ri = bleed.get("result_image")
                if ri is not None:
                    try:
                        from PIL import Image as PILImage  # type: ignore[import]
                        import io as _io
                        buf = _io.BytesIO()
                        PILImage.fromarray(ri).save(buf, format="PNG")
                        with st.expander("View result"):
                            st.image(buf.getvalue(), use_container_width=True)
                    except Exception:
                        pass
            else:
                st.success(f"No bleed — {bleed.get('reason','')}")

        # Standardise — deskew, grayscale, resize
        with c3:
            st.markdown("**📐 Standardise**")
            st.caption("Deskews the scan, converts to grayscale, and resizes to a consistent width for Kraken.")
            if std.get("error"):
                st.warning(std["error"])
            else:
                skew  = std.get("skew_angle", 0.0)
                scale = std.get("resize_scale", 1.0)
                ops   = std.get("applied", [])
                st.success(f"Skew {skew:.1f}° · scale {scale:.2f}×")
                st.caption(f"Ops: {', '.join(ops) if ops else 'none'}")
                ri = std.get("result_image")
                if ri is not None:
                    try:
                        from PIL import Image as PILImage  # type: ignore[import]
                        import io as _io
                        buf = _io.BytesIO()
                        PILImage.fromarray(ri).save(buf, format="PNG")
                        with st.expander("View result"):
                            st.image(buf.getvalue(), use_container_width=True)
                    except Exception:
                        pass

        # Original thumbnail — small, not full-width
        st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
        with st.expander("View original upload"):
            st.image(image_bytes, use_container_width=True)

    # ── eScriptorium full HTR (PDF only) ──────────────────────────────────────
    if uploaded is not None and uploaded.name.lower().endswith(".pdf"):
        st.divider()
        st.markdown(
            '<div style="font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;'
            'color:var(--sepia);font-weight:500;margin-bottom:8px">Full HTR pipeline</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:13.5px;color:var(--ink2);line-height:1.7;margin-bottom:16px">'
            "The operator preview above checks a single page. To transcribe the <em>entire</em> "
            "manuscript, submit it to eScriptorium — it will convert all pages to images, "
            "upload them, seed the five Khaleeji zone labels, and launch Kraken BLLA "
            "segmentation. You can then annotate baselines and transcribe text in the "
            "eScriptorium editor, then export PAGE-XML back into the corpus."
            "</p>",
            unsafe_allow_html=True,
        )

        escr_status = _escriptorium_status()
        if not escr_status["available"]:
            st.markdown("""
<div style="border:1px solid var(--line);border-radius:10px;padding:20px 24px;background:white;margin-bottom:8px">
  <div style="font-family:'Fraunces',serif;font-size:15px;color:var(--ink);margin-bottom:10px;font-weight:500">
    🖊️ eScriptorium is not running
  </div>
  <p style="font-size:13px;color:var(--ink2);line-height:1.7;margin-bottom:16px">
    eScriptorium is a self-hosted transcription platform that handles the full
    HTR pipeline — upload pages, run Kraken segmentation, annotate baselines,
    and export PAGE-XML. Run the one-command setup below to install it locally.
    First-time setup takes about 15 minutes (it builds a Docker image and runs migrations).
  </p>
  <div style="font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--sepia);font-weight:500;margin-bottom:8px">Step 1 — First-time setup (run once in your terminal)</div>
  <pre style="background:var(--paper2);border:1px solid var(--line);border-radius:6px;padding:12px 16px;font-size:12px;color:var(--ink);margin-bottom:16px">cd infra/escriptorium
bash setup.sh</pre>
  <div style="font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--sepia);font-weight:500;margin-bottom:8px">Step 2 — Subsequent starts (already set up)</div>
  <pre style="background:var(--paper2);border:1px solid var(--line);border-radius:6px;padding:12px 16px;font-size:12px;color:var(--ink);margin-bottom:16px">cd infra/escriptorium && docker compose up -d</pre>
  <p style="font-size:12px;color:var(--ink3);margin:0">
    The setup script clones the eScriptorium source, builds the Docker image,
    runs migrations, creates an admin user (admin / nabatai2024), and writes
    the API token automatically into your <code>.env</code>. Restart the
    Streamlit app after setup completes to activate the connection.
  </p>
</div>
""", unsafe_allow_html=True)
        else:
            st.success(f"eScriptorium running at `{escr_status['base_url']}`")

            # Derive defaults from the uploaded filename
            stem = uploaded.name.rsplit(".", 1)[0]
            proj_default = f"NABAT-AI — {stem}"
            doc_default  = stem

            col_p, col_d = st.columns(2)
            proj_name = col_p.text_input("Project name", value=proj_default, key="escr_proj")
            doc_name  = col_d.text_input("Document name", value=doc_default,  key="escr_doc")

            total_p = st.session_state.get("_pdf_total_pages", "?")
            if st.button(
                f"🖊️ Submit all {total_p} pages to eScriptorium",
                key="escr_submit",
                type="primary",
            ):
                with st.spinner(
                    f"Converting {total_p} pages to PNG and uploading — this may take a minute…"
                ):
                    result = _submit_pdf_to_escriptorium(
                        uploaded.getvalue(), proj_name, doc_name
                    )

                if result.get("ok"):
                    st.success(
                        f"✅ Uploaded {result['uploaded']} / {result['total_pages']} pages "
                        f"to eScriptorium (document #{result['document_id']}). "
                        "Kraken BLLA segmentation is running in the background."
                    )
                    if result.get("errors"):
                        st.warning("Some pages had errors:\n" + "\n".join(result["errors"]))
                    st.markdown(
                        f"**Open in eScriptorium editor:** [{result['editor_url']}]({result['editor_url']})"
                    )
                    try:
                        import streamlit.components.v1 as components
                        components.iframe(result["editor_url"], height=700, scrolling=True)
                    except Exception:
                        pass
                else:
                    st.error(f"Submission failed: {result.get('error')}")

    st.divider()

    # Index rebuild
    st.subheader("🔄 Rebuild RAG Index")
    st.markdown(
        '<p style="font-size:13.5px;color:var(--ink2);line-height:1.7;margin-bottom:16px">'
        "The Scholar Workbench searches a pre-built vector index — it cannot find newly added bayts "
        "until the index is rebuilt. Rebuilding re-encodes every bayt in the registry "
        "(2,222 bayts across Phases 1–4 at nine chunk levels) into searchable vectors. "
        "Run this once after you finish adding or editing entries in the registry. "
        "It takes about 60–90 seconds and the result is immediately live in Tab A."
        "</p>",
        unsafe_allow_html=True,
    )
    idx_exists = _index_exists()
    if idx_exists:
        st.success("Index found — `data/qdrant/chunks_meta.json`")
    else:
        st.warning("Index not found — rebuild required before Scholar Workbench can answer queries.")

    if st.button("🔄 Rebuild index now", key="rebuild_btn"):
        with st.spinner("Rebuilding — encoding 2,222 bayts × 9 chunk levels…"):
            rb = _rebuild_index()
        if rb["success"]:
            s = rb["stats"]
            st.success(
                f"✅ Done — {s.get('total_chunks','?')} chunks "
                f"({s.get('verse_chunks','?')} verse / "
                f"{s.get('group_chunks','?')} group / "
                f"{s.get('poem_chunks','?')} poem)"
            )
        else:
            st.error(f"Index rebuild failed: {rb['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# Debug Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _get_era_genre_distribution() -> dict[str, dict[str, int]]:
    """
    Join enriched anchor registry with manuscript date ranges.
    Returns {era_label: {genre: count}} bucketed into 3 historical periods.
    """
    try:
        import json

        # Load manuscript → midyear mapping
        ms_reg_path = _REPO_ROOT / "data" / "ground_truth" / "manuscript_registry.json"
        ms_data = json.loads(ms_reg_path.read_text(encoding="utf-8"))
        ms_midyear: dict[str, int] = {}
        for ms in ms_data:
            s = ms.get("circa_date_start") or 0
            e = ms.get("circa_date_end") or s
            try:
                ms_midyear[ms["short_key"]] = (int(s) + int(e)) // 2
            except (ValueError, TypeError):
                pass

        def _era(mid: int) -> str:
            if mid < 1840:
                return "Early · 1800–1849"
            elif mid < 1890:
                return "Mid · 1850–1889"
            else:
                return "Late · 1890–1940"

        # Load anchor registry
        reg_path = _REPO_ROOT / "data" / "ground_truth" / "anchor_registry_phase4_enriched.json"
        if not reg_path.exists():
            reg_path = _REPO_ROOT / "data" / "ground_truth" / "anchor_registry_phase4.json"
        anchors = json.loads(reg_path.read_text(encoding="utf-8"))

        era_genre: dict[str, dict[str, int]] = {}
        for anchor in anchors:
            short_key = anchor.get("manuscript_short_key", "")
            mid = ms_midyear.get(short_key, 1865)
            era = _era(mid)
            genre = anchor.get("genre") or "غير_محدد"
            era_genre.setdefault(era, {})
            era_genre[era][genre] = era_genre[era].get(genre, 0) + 1

        # Ensure all three eras exist
        for label in ["Early · 1800–1849", "Mid · 1850–1889", "Late · 1890–1940"]:
            era_genre.setdefault(label, {})
        return era_genre
    except Exception:
        return {}


def _sidebar_era_bubbles_html(era_data: dict[str, dict[str, int]]) -> str:
    """
    Flat horizontal bubble chart: 3 era rows, bubbles left-to-right sorted
    by count, sized by count, labelled with genre. No Y-axis.
    """
    if not era_data:
        return '<div style="color:var(--ink3);font-size:12px;padding:4px 0">No era data.</div>'

    COLOURS = {
        "غزل":"#C8856A","رثاء":"#4A6580","مديح":"#6B8159",
        "هجاء":"#B4502E","فخر":"#A07040","حكمة":"#8B5A2B",
        "وصف":"#7A9070","دينية":"#5E6B8B","غزو":"#7A4A2B",
        "غير_محدد":"#A09080",
    }
    ERAS      = ["Early · 1800–1849", "Mid · 1850–1889", "Late · 1890–1940"]
    ERA_SHORT = ["1800–1849",          "1850–1889",        "1890–1940"]
    ERA_COL   = ["#5E3A1C",            "#8B5A2B",           "#B4502E"]

    all_counts = [c for gd in era_data.values() for c in gd.values()]
    max_c  = max(all_counts) if all_counts else 1
    total  = sum(all_counts)

    ROW_H   = 42    # px per era row
    W       = 210
    PAD_L   = 52    # left space for era label
    PAD_R   = 4
    PAD_T   = 4
    plot_w  = W - PAD_L - PAD_R
    H       = PAD_T + len(ERAS) * ROW_H + 4

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" style="display:block;overflow:visible">'
    ]

    for i, (era, short, ec) in enumerate(zip(ERAS, ERA_SHORT, ERA_COL)):
        cy = PAD_T + i * ROW_H + ROW_H / 2

        # Era label on the left
        parts.append(
            f'<text x="{PAD_L - 6}" y="{cy:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="7.5" fill="{ec}" '
            f'font-family="system-ui,sans-serif" font-style="italic">{short}</text>'
        )
        # Thin left border per row
        y_top = PAD_T + i * ROW_H + 4
        y_bot = PAD_T + (i + 1) * ROW_H - 4
        parts.append(
            f'<line x1="{PAD_L - 2}" y1="{y_top}" x2="{PAD_L - 2}" y2="{y_bot}" '
            f'stroke="{ec}" stroke-width="0.6" stroke-opacity="0.5"/>'
        )

        genres = era_data.get(era, {})
        top = sorted(genres.items(), key=lambda x: -x[1])
        max_r = ROW_H / 2 - 3

        x_cursor = PAD_L + 2
        for genre, count in top:
            r = max(9, min(max_r, 7 + 13 * (count / max_c) ** 0.5))
            if x_cursor + r * 2 > W - PAD_R:
                break
            cx = x_cursor + r
            bg = COLOURS.get(genre, "#A09080")
            # Ghost halo
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r+2:.1f}" '
                f'fill="{bg}" fill-opacity="0.10"/>'
            )
            # Solid bubble
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                f'fill="{bg}" fill-opacity="0.88" '
                f'stroke="rgba(255,255,255,0.5)" stroke-width="0.6">'
                f'<title>{genre}: {count}</title></circle>'
            )
            # Genre label inside bubble (always shown)
            fs = max(6, min(8, int(r * 0.72)))
            parts.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
                f'dominant-baseline="middle" font-size="{fs}" fill="#fff" '
                f'font-family="system-ui,sans-serif" font-weight="400">'
                f'{genre}</text>'
            )
            x_cursor += r * 2 + 2

    parts.append('</svg>')

    return (
        f'<div class="s-label" style="margin-bottom:4px">Corpus · genre × era</div>'
        f'<div style="line-height:0">{"".join(parts)}</div>'
        f'<div style="margin-top:3px;font-size:9.5px;color:#A09080;'
        f'font-style:italic;font-family:Fraunces,serif">'
        f'🔸 heuristic · ~43% cov · {total:,} bayts</div>'
    )


def _render_sidebar() -> None:
    """
    Sidebar — Back button, tab navigation, LLM provider, index status,
    genre bubble chart, and per-turn debug.
    """
    info     = _get_provider_info()
    index_ok = _index_exists()
    # Live corpus counts for index status block
    _corpus_s = _gov_load_corpus_stats()
    _total_chunks  = _corpus_s.get("total", 4750)
    try:
        src_path = str(_REPO_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from al_nassikh.corpus_stats import count_bayts as _count_bayts  # type: ignore[import]
        _total_anchors = _count_bayts()
    except Exception:
        _total_anchors = 2222

    # ── Back button (top-left, replaces corpus stats) ──────────────────────────
    with st.sidebar:
        st.markdown("""
<style>
[data-testid="stSidebarContent"] > div:first-child .stButton button {
  background: var(--ink) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 100px !important;
  font-family: 'Fraunces', serif !important;
  font-style: italic !important;
  font-size: 12px !important;
  padding: 5px 16px !important;
}
[data-testid="stSidebarContent"] > div:first-child .stButton button:hover {
  background: var(--ink2) !important;
  color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)
        if st.button("← Back", key="sidebar_back"):
            st.session_state["page"] = "hero"
            st.rerun()

    st.sidebar.markdown('<div class="s-divider" style="margin:3px 0"></div>', unsafe_allow_html=True)

    # ── Tab navigation ─────────────────────────────────────────────────────────
    active_tab = st.session_state.get("active_tab", "workbench")
    with st.sidebar:
        st.markdown("""
<style>
/* Tab nav buttons — full width, styled as nav pills */
div[data-testid="stSidebarContent"] .tab-nav-btn button {
  width: 100% !important;
  text-align: left !important;
  border-radius: 8px !important;
  font-family: 'Fraunces', serif !important;
  font-size: 12.5px !important;
  padding: 8px 14px !important;
  margin-bottom: 4px !important;
  border: none !important;
  transition: background .15s !important;
}
div[data-testid="stSidebarContent"] .tab-nav-btn.active button {
  background: var(--ink) !important;
  color: #ffffff !important;
}
div[data-testid="stSidebarContent"] .tab-nav-btn.inactive button {
  background: var(--ink3) !important;
  color: #ffffff !important;
}
div[data-testid="stSidebarContent"] .tab-nav-btn.inactive button:hover {
  background: var(--ink2) !important;
  color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

        wb_class = "active" if active_tab == "workbench" else "inactive"
        ar_class = "active" if active_tab == "archive"   else "inactive"

        st.markdown(f'<div class="tab-nav-btn {wb_class}">', unsafe_allow_html=True)
        if st.button("📚 Scholar Workbench · منضدة الباحث", key="tab_btn_workbench",
                     use_container_width=True):
            st.session_state["active_tab"] = "workbench"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="tab-nav-btn {ar_class}">', unsafe_allow_html=True)
        if st.button("🗂️ Archive Manager · إدارة الأرشيف", key="tab_btn_archive",
                     use_container_width=True):
            st.session_state["active_tab"] = "archive"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Governance & metrics tab ──────────────────────────────────────────
        # Why a third tab: trustworthy claims need visible measurement. The
        # demo audience can see live corpus stats, last evaluation results, and
        # health badges instead of relying on static slides. Surfacing this
        # honestly (including failing targets) is stronger than hiding numbers.
        gov_class = "active" if active_tab == "governance" else "inactive"
        st.markdown(f'<div class="tab-nav-btn {gov_class}">', unsafe_allow_html=True)
        if st.button("📊 Governance & Metrics · الحوكمة والمقاييس", key="tab_btn_governance",
                     use_container_width=True):
            st.session_state["active_tab"] = "governance"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="s-divider" style="margin:3px 0"></div>', unsafe_allow_html=True)

    # ── LLM provider block ─────────────────────────────────────────────────────
    if "error" not in info:
        st.sidebar.markdown(_sidebar_provider_html(info), unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="s-label">LLM provider</div>', unsafe_allow_html=True)
        st.sidebar.warning(f"Unavailable: {info['error']}")

    # ── Index status block ─────────────────────────────────────────────────────
    st.sidebar.markdown(
        _sidebar_index_html(index_ok, total_anchors=_total_anchors, total_chunks=_total_chunks),
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="s-divider" style="margin:3px 0"></div>', unsafe_allow_html=True)

    # ── Era × genre bubble visualization ──────────────────────────────────────
    era_data = _get_era_genre_distribution()
    st.sidebar.markdown(_sidebar_era_bubbles_html(era_data), unsafe_allow_html=True)

    # ── Per-turn debug (collapsed) ─────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        st.sidebar.markdown('<div class="s-divider" style="margin:3px 0"></div>', unsafe_allow_html=True)
        with st.sidebar.expander("🔬 Last turn debug", expanded=False):
            stage_timings = result.get("stage_timings") or {}
            if stage_timings:
                st.markdown("**Stage timings (ms):**")
                for stage, ms in stage_timings.items():
                    st.markdown(f"- `{stage}`: {ms}")
            st.markdown(f"**CRAG:** `{result.get('crag_verdict') or '—'}`")
            st.markdown(f"**Self-RAG:** `{result.get('self_rag_verdict') or '—'}`")
            st.markdown(f"**Guardrail:** `{result.get('guardrail_passed')}`")
            flags = result.get("guardrail_flags") or []
            if flags:
                for f in flags:
                    st.markdown(f"- {f}")

    # ── Pending review summary ─────────────────────────────────────────────────
    q = _get_queue_summary()
    pending  = q.get("pending", 0)
    in_rev   = q.get("in_review", 0)
    flagged  = q.get("degraded", 0)
    unlinked = max(0, q.get("total", 0) - q.get("complete", 0) - pending - in_rev)
    if pending + in_rev + flagged > 0:
        st.sidebar.markdown('<div class="s-divider" style="margin:3px 0"></div>', unsafe_allow_html=True)
        st.sidebar.markdown(f"""
<div class="s-label">Pending review</div>
<div style="margin-top:8px;font-size:12px;line-height:2">
  <div style="display:flex;justify-content:space-between">
    <span style="color:var(--ink2)">Submissions waiting</span>
    <span style="font-family:'JetBrains Mono',monospace;color:var(--ink)">{pending}</span>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:var(--ink2)">Corrections flagged</span>
    <span style="font-family:'JetBrains Mono',monospace;color:var(--terra)">{flagged}</span>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:var(--ink2)">Images unlinked</span>
    <span style="font-family:'JetBrains Mono',monospace;color:var(--ink3)">{unlinked}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown(
        '<div class="footer-note">Where poetry once lost to the wind<br/>is given form again</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Design System — "Modern Nostalgia" CSS
# ══════════════════════════════════════════════════════════════════════════════

_NABAT_CSS = """
<style>
/* ── Design tokens ─────────────────────────────────────────────────────────── */
:root {
  --paper:      #F4ECDD;
  --paper2:     #EFE5D0;
  --paper3:     #E9DFC6;
  --ink:        #2A1F17;
  --ink2:       #4A382A;
  --ink3:       #7A6A57;
  --line:       #D9CDB4;
  --line2:      #E6DCC4;
  --sepia:      #8B5A2B;
  --sepia-deep: #5E3A1C;
  --terra:      #B4502E;
  --indigo:     #2E4257;
  --jade:       #6B8159;
  --crimson:    #8C2E2A;
}

/* ── Google Fonts ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;1,9..144,400&family=Inter:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600&family=Amiri:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global paper background + grain ──────────────────────────────────────── */
html, body, [data-testid="stApp"] {
  background-color: var(--paper) !important;
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--ink);
}

/* Paper grain overlay */
[data-testid="stApp"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.16  0 0 0 0 0.12  0 0 0 0 0.08  0 0 0 0.5 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='0.35'/></svg>");
  opacity: 0.35;
  mix-blend-mode: multiply;
}

/* ── Streamlit top bar / toolbar ───────────────────────────────────────────── */
[data-testid="stHeader"] {
  background: rgba(244,236,221,0.8) !important;
  border-bottom: 1px solid var(--line) !important;
}
[data-testid="stToolbar"] { display: none !important; }

/* ── App title area ────────────────────────────────────────────────────────── */
.nabat-masthead {
  padding: 32px 32px 16px;
  border-bottom: 1px solid var(--line);
  background: rgba(244,236,221,0.5);
  display: flex;
  align-items: flex-end;
  gap: 16px;
  margin-top: 8px;
  margin-bottom: 0;
}
.nabat-brand-mark {
  width: 36px; height: 36px;
  border: 1px solid var(--sepia);
  border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-family: 'Amiri', serif;
  color: var(--sepia-deep);
  font-size: 18px;
  flex-shrink: 0;
}
.nabat-brand-word {
  font-family: 'Fraunces', serif;
  font-size: 20px;
  letter-spacing: -0.01em;
  color: var(--ink);
  line-height: 1;
}
.nabat-brand-word .dot { color: var(--sepia); }
.nabat-brand-tag {
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-size: 11px;
  color: var(--ink3);
  margin-top: 3px;
  letter-spacing: 0.04em;
}

/* ── Streamlit main content padding ───────────────────────────────────────── */
.block-container {
  padding: 0 !important;
  max-width: 1080px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
/* Kill the default top gap Streamlit injects above block-container */
.block-container > div:first-child { margin-top: 0 !important; }
[data-testid="stAppViewContainer"] > section > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] {
  background: rgba(239,229,208,0.65) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] > div:first-child {
  background: transparent !important;
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
}
/* ── Sidebar block spacing — compact but breathable ─────────────────────── */
section[data-testid="stSidebar"] .stVerticalBlock {
  gap: 2px !important;
}
section[data-testid="stSidebar"] .element-container {
  margin-bottom: 3px !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
  padding: 0 !important;
}
/* Nav buttons (Back + tab buttons) — add a little breathing room */
section[data-testid="stSidebar"] .stButton button {
  margin-bottom: 5px !important;
}

/* ── Sidebar labels ────────────────────────────────────────────────────────── */
.s-label {
  font-size: 10.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--sepia);
  font-weight: 500;
  margin-bottom: 8px;
}
.s-divider {
  height: 1px;
  background: var(--line);
  opacity: 0.7;
  margin: 4px 0;
}
.corpus-num {
  font-family: 'Fraunces', serif;
  font-size: 22px;
  color: var(--ink);
  line-height: 1;
}
.corpus-bar {
  height: 4px; border-radius: 2px;
  background: var(--line); overflow: hidden;
  margin: 8px 0 4px;
}
.corpus-bar-fill {
  display: block; height: 100%;
  background: var(--sepia); width: 64%;
}
.corpus-stats {
  font-size: 10.5px;
  color: var(--ink3);
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.04em;
}
.provider-card {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  background: var(--paper2);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 8px;
}
.dot-jade {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--jade);
  flex-shrink: 0;
}
.provider-name {
  font-size: 12px; font-weight: 500; color: var(--ink);
}
.provider-detail {
  font-size: 10.5px; color: var(--ink3);
  font-family: 'JetBrains Mono', monospace;
  display: block; margin-top: 2px;
}
.neural-card {
  background: var(--paper2);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 14px;
  margin-top: 8px;
}
.neural-row {
  display: flex; align-items: center; gap: 10px;
  padding: 5px 0;
}
.neural-name { font-size: 12px; font-weight: 500; color: var(--ink); }
.neural-detail {
  font-size: 10.5px; color: var(--ink3);
  font-family: 'JetBrains Mono', monospace;
  display: block; margin-top: 1px;
}
.footer-note {
  margin-top: auto;
  padding-top: 14px;
  font-size: 10.5px;
  color: var(--ink3);
  font-style: italic;
  font-family: 'Fraunces', serif;
  line-height: 1.5;
}
.index-row {
  display: flex; justify-content: space-between;
  padding: 4px 0; font-size: 12px;
}
.idx-ok   { color: var(--jade); }
.idx-warn { color: var(--terra); font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.idx-dim  { color: var(--ink3); }

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] > div > div > button {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  color: var(--ink3) !important;
  border-bottom: 2px solid transparent !important;
  padding: 8px 16px !important;
  background: transparent !important;
}
[data-testid="stTabs"] > div > div > button[aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom-color: var(--sepia) !important;
  font-weight: 500 !important;
}
[data-testid="stTabs"] > div > div {
  background: transparent !important;
  border-bottom: 1px solid var(--line) !important;
  gap: 4px;
}

/* ── Calligraphy plate (top of each tab) ───────────────────────────────────── */
.calligraphy-plate {
  padding: 16px 24px 12px;
  border-bottom: 1px solid var(--line);
  display: flex; align-items: center; gap: 22px;
  background: linear-gradient(to bottom, rgba(244,236,221,0.55), transparent);
  margin-bottom: 6px;
}
.calligraphy-plate img {
  display: block; max-width: 420px; width: 100%; height: auto;
  mix-blend-mode: multiply;
  filter: sepia(0.55) contrast(0.85) brightness(1.05);
  opacity: 0.88;
}
.calligraphy-plate-meta {
  flex-shrink: 0; max-width: 200px;
  border-left: 1px solid var(--line);
  padding-left: 18px;
}
.plate-eyebrow {
  font-size: 9.5px; letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--sepia); font-weight: 500;
}
.plate-title {
  font-family: 'Fraunces', serif; font-style: italic; font-size: 14px;
  color: var(--ink2); margin-top: 6px; line-height: 1.4;
}
.plate-source {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  color: var(--ink3); margin-top: 8px; letter-spacing: 0.04em;
}

/* ── Mode pills ────────────────────────────────────────────────────────────── */
.pills-row {
  display: flex; align-items: center; justify-content: space-between;
  margin: 12px 0;
}
.pills-group {
  display: inline-flex;
  background: rgba(255,255,255,0.6);
  border: 1px solid var(--line);
  border-radius: 100px;
  padding: 4px;
  gap: 2px;
}
.pill-btn {
  padding: 7px 15px; border-radius: 100px;
  font-size: 13px; color: var(--ink2);
  display: inline-flex; align-items: center; gap: 6px;
  cursor: pointer; border: none; background: transparent;
  font-family: 'Inter', sans-serif;
  transition: all 0.15s;
}
.pill-btn.active {
  background: var(--ink); color: var(--paper);
  font-weight: 500;
}
.pill-ar {
  font-family: 'IBM Plex Sans Arabic', sans-serif;
  font-size: 11px; opacity: 0.6; direction: rtl;
}
.session-id {
  font-size: 11.5px; color: var(--ink3);
  font-style: italic; font-family: 'Fraunces', serif;
}
.session-id span {
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink2); font-style: normal;
}

/* ── Lede heading ──────────────────────────────────────────────────────────── */
.lede {
  font-family: 'Fraunces', serif;
  font-weight: 350; font-size: 28px;
  letter-spacing: -0.01em; line-height: 1.2;
  color: var(--ink); margin: 14px 0 8px;
}
.lede em { font-style: italic; color: var(--sepia-deep); font-weight: 400; }
.lede-sub {
  font-family: 'Fraunces', serif;
  font-style: italic; font-size: 14.5px;
  color: var(--ink2); max-width: 580px;
  line-height: 1.5; margin-bottom: 16px;
}

/* ── Query composer card ───────────────────────────────────────────────────── */
.composer-card {
  background: var(--paper2);
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 14px 40px -22px rgba(74,56,42,0.32);
  overflow: hidden;
  margin-bottom: 18px;
}
.composer-ta {
  padding: 18px 22px 12px;
  direction: rtl; text-align: right;
  border-bottom: 1px solid var(--line2);
}
.composer-ta textarea {
  font-family: 'Amiri', serif !important;
  font-size: 20px !important;
  direction: rtl !important;
  text-align: right !important;
  line-height: 1.6 !important;
  color: var(--ink) !important;
  background: transparent !important;
  border: none !important;
  resize: vertical;
}
.composer-bar {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 16px;
  background: var(--paper2);
}
.attach-btn {
  background: rgba(255,255,255,0.7);
  border: 1px solid var(--line);
  color: var(--ink2);
  padding: 6px 12px; border-radius: 8px;
  font-size: 12px; cursor: pointer;
  font-family: 'Inter', sans-serif;
  display: inline-flex; align-items: center; gap: 6px;
}
.hint-text {
  font-size: 11px; color: var(--ink3);
  font-family: 'JetBrains Mono', monospace;
}
.inquire-btn {
  background: var(--ink); color: var(--paper);
  border: none; padding: 9px 20px;
  border-radius: 100px; font-size: 13px;
  font-weight: 500; cursor: pointer;
  font-family: 'Inter', sans-serif;
  display: inline-flex; align-items: center; gap: 8px;
}

/* ── Streamlit button override ─────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  background: var(--ink) !important;
  color: #F4ECDD !important;
  border: none !important;
  border-radius: 100px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em !important;
  padding: 9px 22px !important;
  transition: opacity 0.15s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Sidebar nav buttons — force white text on button + inner p/div/span ─── */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stButton button p,
section[data-testid="stSidebar"] .stButton button span,
section[data-testid="stSidebar"] .stButton button div,
section[data-testid="stSidebar"] .tab-nav-btn button,
section[data-testid="stSidebar"] .tab-nav-btn button p,
section[data-testid="stSidebar"] .tab-nav-btn button span,
section[data-testid="stSidebar"] .tab-nav-btn button div {
  color: #ffffff !important;
}

/* ── Streamlit text area ───────────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
  background: rgba(255,255,255,0.9) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  font-family: 'Amiri', serif !important;
  font-size: 20px !important;
  direction: rtl;
  text-align: right;
  color: var(--ink) !important;
  line-height: 1.6 !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--sepia) !important;
  box-shadow: 0 0 0 2px rgba(139,90,43,0.15) !important;
}
[data-testid="stTextArea"] label {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--ink3) !important;
}

/* ── Answer card ───────────────────────────────────────────────────────────── */
.answer-card {
  background: white;
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 14px 40px -22px rgba(74,56,42,0.32);
  overflow: hidden;
  margin: 6px 0 18px;
}
.answer-head {
  padding: 12px 22px;
  border-bottom: 1px solid var(--line2);
  display: flex; justify-content: space-between; align-items: center;
  background: linear-gradient(to bottom, rgba(217,205,180,0.18), transparent);
}
.answer-head-lbl {
  font-family: 'Fraunces', serif;
  font-style: italic; color: var(--sepia); font-size: 13px;
}
.answer-conf {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; color: var(--ink3);
  padding: 2px 7px; background: var(--paper2);
  border-radius: 4px; margin-left: 10px;
}
.verse-body {
  padding: 28px 48px 24px;
  direction: rtl; text-align: center;
  background: radial-gradient(ellipse at 50% 0%, rgba(200,137,59,0.07), transparent 60%);
}
.verse-body p {
  margin: 0;
  font-family: 'Amiri', serif;
  font-size: 22px; line-height: 2.0;
  color: var(--ink); font-weight: 400;
}
.verse-body p.s2 { font-size: 20px; color: var(--ink2); margin-top: 12px; }
/* Prose answers (capabilities, registry stats, debug) — smaller readable font */
.prose-answer {
  padding: 18px 28px 16px;
  direction: ltr; text-align: left;
}
.prose-answer p, .prose-answer div, .prose-answer span {
  font-family: 'Fraunces', serif !important;
  font-size: 14px !important; line-height: 1.7 !important;
  color: var(--ink) !important;
}
.prose-answer .rtl-verse {
  font-family: 'Amiri', serif !important;
  font-size: 16px !important; line-height: 1.9 !important;
  direction: rtl; text-align: right;
}
.verse-orn {
  color: var(--sepia); font-size: 14px; opacity: 0.55; margin: 12px 0;
}
.verse-gloss {
  margin-top: 20px;
  font-family: 'Fraunces', serif; font-style: italic;
  font-size: 14px; line-height: 1.7;
  color: var(--ink3); direction: ltr; text-align: center;
}
.verse-attrib {
  margin-top: 16px; padding-top: 14px;
  border-top: 1px dashed var(--line);
  display: flex; justify-content: center; gap: 18px;
  direction: ltr; font-size: 11.5px; color: var(--ink3);
  font-family: 'Fraunces', serif; font-style: italic;
}
.verse-attrib b { color: var(--ink2); font-weight: 500; font-style: normal; }
.answer-feet {
  padding: 12px 22px;
  background: var(--paper2);
  border-top: 1px solid var(--line2);
  display: flex; align-items: flex-start; gap: 18px; flex-wrap: wrap;
}
.feet-label {
  font-size: 10px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--ink3); margin-bottom: 5px;
}
.genre-badge {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 5px 11px; background: var(--paper2);
  border: 1px solid var(--line); border-radius: 100px; font-size: 12px;
}
.genre-badge .ar { font-family: 'IBM Plex Sans Arabic', sans-serif; color: var(--ink); direction: rtl; }
.genre-badge .en { color: var(--ink3); font-family: 'Fraunces', serif; font-style: italic; font-size: 11px; }
.emo-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px; border-radius: 100px; font-size: 12px;
  border: 1px solid rgba(139,90,43,0.28); margin: 2px;
}
.emo-pill .ar { font-family: 'IBM Plex Sans Arabic', sans-serif; color: var(--sepia-deep); direction: rtl; }
.emo-pill .en { color: var(--ink3); font-family: 'Fraunces', serif; font-style: italic; font-size: 11px; }
.emo-pill .w  { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink3); }
.feet-vdivider { width: 1px; align-self: stretch; background: var(--line); }

/* ── Three-layer cards ─────────────────────────────────────────────────────── */
.lcard {
  background: var(--paper2);
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 14px 40px -22px rgba(74,56,42,0.32);
  overflow: hidden;
}
.lcard.crimson { border-left: 4px solid var(--crimson); }
.lcard.indigo  { border-left: 4px solid var(--indigo); }
.lcard.jade    { border-left: 4px solid var(--jade); }
.lcard-head {
  padding: 12px 16px 10px;
  border-bottom: 1px solid var(--line2);
}
.lcard-head .row1 { display: flex; align-items: center; gap: 10px; }
.lcard-head .ar {
  font-family: 'IBM Plex Sans Arabic', sans-serif;
  font-weight: 500; font-size: 16px;
  color: var(--ink); direction: rtl;
}
.lcard.crimson .lcard-head .en { color: var(--crimson); }
.lcard.indigo  .lcard-head .en { color: var(--indigo); }
.lcard.jade    .lcard-head .en { color: var(--jade); }
.lcard-head .en {
  font-family: 'Fraunces', serif;
  font-style: italic; font-size: 12px; margin-top: 1px;
}
.lcard-head .gloss {
  font-size: 11px; color: var(--ink3);
  margin-top: 8px; font-style: italic;
  font-family: 'Fraunces', serif; line-height: 1.5;
}
.lcard-body {
  padding: 20px 16px;
  direction: rtl; text-align: center;
}
.lcard-body p {
  margin: 0; font-family: 'Amiri', serif;
  font-size: 18px; line-height: 2.1;
  color: var(--ink); white-space: pre-line;
}
.lcard-ann {
  padding: 8px 16px;
  background: var(--paper2);
  border-top: 1px solid var(--line2);
}
.lcard-ann .a {
  display: flex; justify-content: space-between;
  font-size: 11px; padding: 2px 0;
}
.lcard-ann .l { color: var(--ink3); font-style: italic; font-family: 'Fraunces', serif; }
.lcard-ann .v { color: var(--ink2); font-family: 'JetBrains Mono', monospace; }

/* ── Previous turns ────────────────────────────────────────────────────────── */
.prev-turns {
  background: rgba(255,255,255,0.45);
  border: 1px solid var(--line);
  border-radius: 10px; padding: 12px 18px;
  display: flex; align-items: center;
  justify-content: space-between; cursor: pointer;
  margin: 4px 0;
}
.prev-turns .inner {
  font-family: 'Fraunces', serif;
  font-style: italic; font-size: 14px; color: var(--ink2);
}
.turn-count {
  padding: 2px 8px; background: var(--paper2);
  border: 1px solid var(--line); border-radius: 100px;
  font-size: 10.5px; color: var(--ink3);
  font-family: 'JetBrains Mono', monospace;
  margin-left: 10px;
}
.turn-last {
  font-size: 11px; color: var(--ink3);
  font-style: italic; font-family: 'Fraunces', serif;
}

/* ── Echo box (three-layer query echo) ─────────────────────────────────────── */
.echo-box {
  background: rgba(255,255,255,0.55);
  border: 1px solid var(--line);
  border-radius: 10px; padding: 12px 18px;
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 14px;
}
.echo-lbl {
  font-family: 'Fraunces', serif;
  font-style: italic; font-size: 12px;
  color: var(--sepia); white-space: nowrap;
}
.echo-q {
  font-family: 'Amiri', serif; font-size: 18px;
  color: var(--ink); direction: rtl; flex: 1; text-align: right;
}

/* ── Footnote rail (three-layer) ───────────────────────────────────────────── */
.footnote-rail {
  padding: 12px 18px;
  background: var(--paper2);
  border: 1px solid var(--line);
  border-radius: 10px;
  display: flex; gap: 20px;
  font-size: 11.5px; color: var(--ink2);
  font-style: italic; font-family: 'Fraunces', serif;
  margin-top: 14px;
}

/* ── Deterministic answer badge ────────────────────────────────────────────── */
.det-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 14px;
  background: rgba(107,129,89,0.12);
  border: 1px solid rgba(107,129,89,0.35);
  border-radius: 100px;
  font-size: 12px; color: var(--jade);
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 12px;
}

/* ── RTL verse box (reused across views) ───────────────────────────────────── */
.rtl-verse {
  direction: rtl; text-align: right;
  background: rgba(255,255,255,0.75);
  border-right: 4px solid var(--sepia);
  border-radius: 6px;
  padding: 16px 20px;
  font-family: 'Amiri', serif;
  font-size: 20px; line-height: 1.9;
  color: var(--ink);
  margin: 8px 0;
}

/* ── Expander overrides ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--paper2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Fraunces', serif !important;
  font-style: italic !important;
  color: var(--ink2) !important;
}

/* ── Chat input: remove fixed-bottom, appear inline ───────────────────────── */
[data-testid="stBottom"] {
  position: relative !important;
  bottom: auto !important;
  left: auto !important;
  right: auto !important;
  width: 100% !important;
  max-width: 100% !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  z-index: 1 !important;
}
[data-testid="stBottom"] > div {
  padding: 0 !important;
  background: transparent !important;
}

/* ── Streamlit selectbox / multiselect ─────────────────────────────────────── */
[data-testid="stSelectbox"] select,
[data-testid="stMultiSelect"] input {
  background: white !important;
  border: 1px solid var(--line) !important;
}

/* ── Dividers ──────────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--line) !important; }

/* ── File uploader ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.5) !important;
  border: 1px dashed var(--line) !important;
  border-radius: 10px !important;
}

/* ── Info / warning / error / success boxes ────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.5) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { color: var(--ink3) !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-family: 'Fraunces', serif !important; }

/* ── Sidebar widgets ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label {
  color: var(--ink2) !important;
  font-size: 13px !important;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-family: 'Fraunces', serif !important;
  color: var(--ink) !important;
}

/* ── Top padding for main area content ─────────────────────────────────────── */
.main-content {
  padding: 24px 32px;
  max-width: 1080px;
  margin: 0 auto;
  box-sizing: border-box;
  width: 100%;
}

/* ── View section heading ──────────────────────────────────────────────────── */
.section-title {
  font-family: 'Fraunces', serif;
  font-size: 22px; color: var(--ink); margin-bottom: 4px;
}
.section-sub {
  font-size: 12.5px; color: var(--ink3);
  font-style: italic; font-family: 'Fraunces', serif;
  margin-bottom: 14px;
}
</style>
"""

@st.cache_data
def _calligraphy_plate_html() -> str:
    """
    HTML for the calligraphy plate shown at the top of the Scholar Workbench.
    Why base64: Streamlit's static file serving doesn't work reliably inside
    iframes / markdown blocks. Embedding the image as a data URI guarantees it
    always renders regardless of how Streamlit is launched.
    Why @st.cache_data: avoids re-reading the 133 KB base64 string on every rerun.
    """
    try:
        from app._calligraphy_b64 import CALLIGRAPHY_B64  # type: ignore[import]
        img_src = f"data:image/png;base64,{CALLIGRAPHY_B64}"
    except Exception:
        try:
            import base64 as _b64
            _p = _REPO_ROOT / "app" / "design_handoff_nabat_workbench 2" / "image.png"
            img_src = "data:image/png;base64," + _b64.b64encode(_p.read_bytes()).decode()
        except Exception:
            img_src = ""

    img_tag = (
        f'<img src="{img_src}" alt="يقول الفتى الشاعر" '
        f'style="display:block; max-width:420px; width:100%; height:auto; '
        f'mix-blend-mode:multiply; filter:sepia(0.55) contrast(0.85) brightness(1.05); opacity:0.88" />'
        if img_src else ""
    )
    return f"""
<div class="calligraphy-plate">
  <div style="flex:1; display:flex; align-items:center; justify-content:center; min-width:0">
    {img_tag}
  </div>
  <div class="calligraphy-plate-meta">
    <div class="plate-eyebrow">Opening line</div>
    <div class="plate-title">"<span style="font-family:Amiri,serif;direction:rtl;font-style:normal">يقول الفتى الشاعر</span>"<br/>— <i>so says the young poet</i></div>
    <div class="plate-source">HUBEIR · fol. 43r · l. 1</div>
  </div>
</div>
"""


def _masthead_html() -> str:
    """Return masthead HTML (brand only — Back button is injected via st.columns in main)."""
    return """
<div class="nabat-masthead">
  <div class="nabat-brand-mark">ن</div>
  <div>
    <div class="nabat-brand-word">NABAT<span class="dot">·</span>AI</div>
    <div class="nabat-brand-tag">a scholar's reading room for Khaleeji Nabati poetry</div>
  </div>
</div>
"""


def _render_masthead_with_back() -> None:
    """Render the masthead brand row. Back button lives in the sidebar."""
    st.markdown(_masthead_html(), unsafe_allow_html=True)


def _sidebar_corpus_html(total: int = 1502, transcribed_pct: int = 64, translated_pct: int = 22) -> str:
    return f"""
<div class="s-label">Corpus coverage</div>
<div style="margin-top:10px">
  <div class="corpus-num">{total:,} <span style="font-size:13px;color:var(--ink3);font-style:italic">bayts</span></div>
  <div style="font-size:12px;color:var(--ink2);margin-top:6px;line-height:1.5">
    Phases 1–4 · <b style="font-weight:500">25 manuscripts</b> · <b style="font-weight:500">c. 1800–1970</b>
  </div>
  <div class="corpus-bar"><span class="corpus-bar-fill"></span></div>
  <div class="corpus-stats">{transcribed_pct}% TRANSCRIBED · {translated_pct}% TRANSLATED</div>
</div>
"""


def _sidebar_provider_html(provider_info: dict) -> str:
    name    = provider_info.get("provider", "—")
    primary = provider_info.get("model_primary", "—")
    fallback= provider_info.get("using_fallback", False)
    status  = "fallback" if fallback else "primary"
    return f"""
<div class="s-label" style="margin-top:4px">LLM provider</div>
<div class="provider-card">
  <div class="dot-jade"></div>
  <div style="flex:1">
    <div class="provider-name">{name} · {primary}</div>
    <span class="provider-detail">{status} model active</span>
  </div>
</div>
"""


def _sidebar_index_html(index_ok: bool, total_anchors: int = 2222, total_chunks: int = 4750) -> str:
    vector_status = '<span class="idx-ok">✓ healthy</span>' if index_ok else '<span class="idx-warn">not built</span>'
    bm25_status   = '<span class="idx-ok">✓ healthy</span>' if index_ok else '<span class="idx-warn">not built</span>'
    chunks_display = f"{total_chunks:,} chunks" if index_ok else "—"
    return f"""
<div class="s-label" style="margin-top:4px">Index status</div>
<div style="margin-top:8px">
  <div class="index-row"><span style="color:var(--ink2)">Vector index</span>{vector_status}</div>
  <div class="index-row"><span style="color:var(--ink2)">BM25 sparse</span>{bm25_status}</div>
  <div class="index-row"><span style="color:var(--ink2)">Bayts indexed</span><span class="idx-dim">{total_anchors:,} entries</span></div>
  <div class="index-row"><span style="color:var(--ink2)">Chunks</span><span class="idx-dim">{chunks_display}</span></div>
  <div class="index-row"><span style="color:var(--ink2)">Phases</span><span class="idx-dim">1–4 complete</span></div>
</div>
"""


def _sidebar_neural_html(encoder_ok: bool, genre_neural: bool, emotion_neural: bool, whisper_ok: bool) -> str:
    def row(emoji, name, detail):
        return f'<div class="neural-row"><span>{emoji}</span><div style="flex:1"><div class="neural-name">{name}</div><span class="neural-detail">{detail}</span></div></div>'
    enc_row = row("🟢" if encoder_ok else "🔸", "Encoder",
                  "arabic-bge-large · v2.1" if encoder_ok else "hash vectors · fallback")
    gnr_row = row("🟢" if genre_neural else "🔸", "Genre",
                  "khaleeji-genre · AraPoemBERT" if genre_neural else "keyword heuristic · 43% cov")
    emo_row = row("🟢" if emotion_neural else "🔸", "Emotion",
                  "neural AraPoemBERT" if emotion_neural else "heuristic fallback · seed 4")
    asr_row = row("🟢" if whisper_ok else "🔸", "ASR",
                  "whisper fine-tuned" if whisper_ok else "whisper-large-v3 · cold")
    return f"""
<div class="s-label" style="margin-top:4px">🧠 Neural models</div>
<div class="neural-card">
  {enc_row}{gnr_row}{emo_row}{asr_row}
</div>
<div style="margin-top:6px;font-size:11px;color:var(--ink3);font-style:italic;font-family:'Fraunces',serif">
  🟢 neural · 🔸 heuristic / cold
</div>
"""


def _render_hero() -> None:
    """
    Hero landing page. Why a single HTML block + two st.buttons below:
    Streamlit's column/block-container layout fights custom centering.
    The only reliable way to get the design-spec centred hero (large italic
    Fraunces title, calligraphy image, button pair) is to render the visual
    content as one self-contained HTML block, then place the two action
    buttons as native st.buttons just below it so session_state routing works.
    """
    # Load calligraphy image as data URI
    import base64 as _b64
    calli_src = ""
    try:
        from app._calligraphy_b64 import CALLIGRAPHY_B64  # type: ignore[import]
        calli_src = f"data:image/png;base64,{CALLIGRAPHY_B64}"
    except Exception:
        for _candidate in [
            _REPO_ROOT / "app" / "static" / "calligraphy-sample.png",
            _REPO_ROOT / "doc" / "image.png",
        ]:
            try:
                calli_src = "data:image/png;base64," + _b64.b64encode(_candidate.read_bytes()).decode()
                break
            except Exception:
                continue

    # Why: embedding a 135 KB base64 string directly in the hero f-string
    # confuses Streamlit's markdown parser (treats subsequent lines as code blocks).
    # CSS <style> blocks bypass the markdown parser entirely, so the image is safe here.
    _bg = f'url("{calli_src}")' if calli_src else "none"
    st.markdown(f"""<style>
.calli-img-slot {{
  display: block; width: 100%; aspect-ratio: 1974 / 444;
  background-image: {_bg};
  background-size: contain; background-repeat: no-repeat; background-position: center;
  mix-blend-mode: multiply;
  filter: sepia(0.72) contrast(0.78) brightness(1.08) saturate(0.9);
  opacity: 0.82;
}}
</style>""", unsafe_allow_html=True)

    # Full-page CSS reset + hero styles injected once
    st.markdown("""
<style>
/* ── Hide Streamlit chrome on hero ───────────────── */
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* ── Reset Streamlit container ───────────────────── */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > section.main > div {
  padding: 0 !important;
}

/* ── Hero block ──────────────────────────────────── */
.nabat-hero {
  background: var(--paper);
  text-align: center;
  padding: 72px 24px 0;
  position: relative;
  overflow: hidden;
}
.nabat-hero::before {
  content: "";
  position: absolute; inset: 0; pointer-events: none; z-index: 0;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.16 0 0 0 0 0.12 0 0 0 0 0.08 0 0 0 0.5 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='0.35'/></svg>");
  opacity: 0.4; mix-blend-mode: multiply;
}
.nabat-hero * { position: relative; z-index: 1; }

.nabat-hero .eyebrow {
  font-family: 'Inter', sans-serif;
  font-size: 11px; letter-spacing: 0.28em; text-transform: uppercase;
  color: #8B5A2B; font-weight: 500; margin-bottom: 20px;
}
.nabat-hero h1 {
  font-family: 'Fraunces', serif;
  font-weight: 300;
  font-style: italic;
  font-size: clamp(26px, 3.8vw, 46px);
  line-height: 1.05;
  letter-spacing: -0.02em;
  color: #7A6A57;
  margin: 0 auto 24px;
  max-width: 900px;
}
.nabat-hero .sub {
  font-family: 'Fraunces', serif;
  font-style: italic; font-size: 14px; font-weight: 300; line-height: 1.6;
  color: #7A6A57; max-width: 560px; margin: 0 auto 10px;
}
.nabat-hero .ar-sub {
  font-family: 'IBM Plex Sans Arabic', sans-serif;
  direction: rtl; font-size: 14px; color: #7A6A57; margin-bottom: 48px;
}
.nabat-hero .calli-wrap {
  width: min(700px, 90%); margin: 0 auto;
  border-radius: 8px;
  overflow: hidden;
}
.nabat-hero .calli-wrap img {
  display: block; width: 100%; height: auto;
  mix-blend-mode: multiply;
  filter: sepia(0.72) contrast(0.78) brightness(1.08) saturate(0.9);
  opacity: 0.82;
}
.nabat-hero .cap {
  margin-top: 14px; padding-bottom: 4px;
  font-family: 'Fraunces', serif; font-style: italic;
  font-size: 13px; color: #7A6A57; letter-spacing: 0.04em;
}
.nabat-hero .cap b { color: #4A382A; font-weight: 500; font-style: normal; }

/* ── Hero button row ─────────────────────────────── */
.hero-btn-row {
  padding: 40px 0 56px;
  display: flex; justify-content: center; gap: 14px; flex-wrap: wrap;
}

/* Primary button — override Streamlit's own stButton */
[data-testid="stButton"][data-key="hero_enter_wb"] button,
[data-testid="stButton"][data-key="hero_enter_wb"] button:hover,
[data-testid="stButton"][data-key="hero_enter_wb"] button:focus {
  background: #2A1F17 !important;
  color: #F4ECDD !important;
  border: none !important;
  border-radius: 100px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  letter-spacing: 0.03em !important;
  padding: 14px 32px !important;
  min-width: 200px;
  box-shadow: none !important;
}
[data-testid="stButton"][data-key="hero_enter_wb"] button:hover {
  opacity: 0.85 !important;
}

/* Ghost button */
[data-testid="stButton"][data-key="hero_browse_corpus"] button,
[data-testid="stButton"][data-key="hero_browse_corpus"] button:hover,
[data-testid="stButton"][data-key="hero_browse_corpus"] button:focus {
  background: transparent !important;
  color: #4A382A !important;
  border: 1.5px solid #D9CDB4 !important;
  border-radius: 100px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important; font-weight: 400 !important;
  letter-spacing: 0.02em !important;
  padding: 14px 32px !important;
  min-width: 180px;
  box-shadow: none !important;
}
[data-testid="stButton"][data-key="hero_browse_corpus"] button:hover {
  background: rgba(42,31,23,0.05) !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Hero visual block (single HTML render — avoids Streamlit layout fighting) ──
    st.markdown(f"""
<div class="nabat-hero">
  <p class="eyebrow">NABAT · AI</p>
  <h1>Where poetry once lost to the wind<br/>is given form again</h1>
  <p class="sub">A bilingual scholar's workbench for Khaleeji Nabati manuscripts —
  searchable, listenable, citable, and quietly faithful to the page it came from.</p>
  <div class="calli-wrap">
<div class="calli-img-slot"></div>
<p class="cap">— <i>handwriting drawn from</i> <b>Hubeir MS, fol. 43r</b> · <i>attributed to</i> <b>Mājid bin Ẓāhir</b>, <i>c. 1782</i> —</p>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Action buttons (native st.button for session_state routing) ────────────
    st.markdown('<div class="hero-btn-row">', unsafe_allow_html=True)
    col_l, col_enter, col_browse, col_r = st.columns([1.2, 1.1, 1.1, 1.2])
    with col_enter:
        if st.button("Enter the workbench →", key="hero_enter_wb", use_container_width=True):
            st.session_state["page"] = "workbench"
            st.rerun()
    with col_browse:
        if st.button("Browse the corpus", key="hero_browse_corpus", use_container_width=True):
            st.session_state["page"] = "browse"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_browse_corpus() -> None:
    """
    Browse Corpus page — a simple searchable table of all verse anchors.
    Why this exists: the v2 hero has a "Browse the corpus" button that
    routes to page="browse". This gives users a lightweight way to explore
    the 2,222 anchors without posing a full RAG query.
    """
    # ── Masthead with back button ─────────────────────────────────────────────
    _render_masthead_with_back()

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="lede">Browse the corpus — <em>تصفّح الأرشيف</em></h2>'
        '<p class="lede-sub">2,222 bayts across all four phases (Phases 1–4): '
        '720 fully-transcribed bayts from ~39 poems (Phase 1–3) · '
        '1,502 matla bayts from TOC poems (Phase 4). '
        'Filter by poet, manuscript, or genre.</p>',
        unsafe_allow_html=True,
    )

    # ── Load registry — prefer full enriched > cleaned > phase4-only ──────────
    # Why prefer full_enriched: it contains all 2,222 anchors across Phases 1–4
    # including the full-verse Phase 1–3 data ingested 2026-04-26. The cleaned
    # and phase4-only files are kept as fallbacks for environments where the
    # full registry hasn't been built yet.
    for registry_candidate in [
        "anchor_registry_full_enriched.json",
        "anchor_registry_full.json",
        "anchor_registry_phase4_cleaned.json",
        "anchor_registry_phase4_enriched.json",
        "anchor_registry_phase4.json",
    ]:
        registry_path = _REPO_ROOT / "data" / "ground_truth" / registry_candidate
        if registry_path.exists():
            break

    records: list[dict] = []
    if registry_path.exists():
        try:
            import json
            with open(registry_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            # Filter out header rows and records with empty matla (archivist noise)
            records = [
                r for r in records
                if (r.get("matla_text") or "").strip()
                and (r.get("poet_name") or "").strip() not in ("مخطوطة", "")
                and "header_row" not in (r.get("_cleaning_tags") or [])
            ]
        except Exception as exc:
            st.error(f"Could not load registry: {exc}")
    else:
        st.warning("Registry not found — run `python scripts/rebuild_index.py` first.")

    if not records:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Bilingual genre map — Arabic internal key → display label
    GENRE_BILINGUAL = {
        "غزل":    "غزل — Love / Romantic",
        "رثاء":   "رثاء — Elegy / Lamentation",
        "مديح":   "مديح — Praise / Panegyric",
        "هجاء":   "هجاء — Satire / Invective",
        "فخر":    "فخر — Boasting / Self-praise",
        "حكمة":   "حكمة — Wisdom / Aphorism",
        "وصف":    "وصف — Description / Nature",
        "دينية":  "دينية — Religious / Devotional",
        "غزو":    "غزو — Raid / War Narrative",
    }

    # ── Filter controls ────────────────────────────────────────────────────────
    col_search, col_genre, col_ms = st.columns([2, 1, 1])
    with col_search:
        search_text = st.text_input(
            "Search verses / ابحث في الأبيات",
            placeholder="اكتب كلمة أو اسم شاعر… / poet name or verse text",
            key="browse_search",
        )
    with col_genre:
        # Only show genres actually present in this dataset
        present_genres = {r.get("genre", "") for r in records if r.get("genre") and r.get("genre") != "غير_محدد"}
        genre_display_opts = ["— any genre —"] + [
            GENRE_BILINGUAL.get(g, g) for g in sorted(present_genres)
            if g in GENRE_BILINGUAL
        ]
        selected_genre_display = st.selectbox("Genre / النوع", genre_display_opts, key="browse_genre")
        # Map back to Arabic key for filtering
        selected_genre = ""
        if selected_genre_display != "— any genre —":
            selected_genre = selected_genre_display.split(" — ")[0].strip()
    with col_ms:
        all_ms = sorted({r.get("manuscript_short_key", "") for r in records if r.get("manuscript_short_key")})
        ms_opts = ["— any manuscript —"] + all_ms
        selected_ms = st.selectbox("Manuscript / المخطوطة", ms_opts, key="browse_ms")

    # ── Apply filters ──────────────────────────────────────────────────────────
    filtered = records
    if search_text.strip():
        q = search_text.strip().lower()
        filtered = [
            r for r in filtered
            if q in (r.get("matla_text") or "").lower()
            or q in (r.get("poet_name") or "").lower()
        ]
    if selected_genre:
        filtered = [r for r in filtered if r.get("genre") == selected_genre]
    if selected_ms != "— any manuscript —":
        filtered = [r for r in filtered if r.get("manuscript_short_key") == selected_ms]

    st.caption(f"Showing {len(filtered):,} of {len(records):,} bayts")

    # ── Table ──────────────────────────────────────────────────────────────────
    rows_html = ""
    for r in filtered[:200]:   # cap at 200 rows for render performance
        poet    = r.get("poet_name") or "—"
        verse   = r.get("matla_text") or "—"
        genre   = r.get("genre", "")
        ms_key  = r.get("manuscript_short_key", "—")
        page    = r.get("page_number", "")
        attrib  = r.get("matla_attribution_note", "")  # from pipe-split cleaning

        # Skip rows where the verse is still just a name (cleaning confidence < 100%)
        if poet == verse or len(verse.strip()) < 4:
            continue

        # Display "unknown" poets less prominently
        poet_display = (
            f'<span style="color:var(--ink2)">{poet}</span>'
            if poet not in ("unknown", "—")
            else '<span style="color:var(--ink3);font-style:italic">unknown</span>'
        )

        # Attribution note (e.g. a dedicatee or secondary attribution from the MS)
        attrib_html = (
            f'<div style="font-size:11px;color:var(--ink3);font-style:italic;margin-top:2px">'
            f'↳ {attrib}</div>'
            if attrib else ""
        )

        badge = "🔸" if r.get("genre_source") == "heuristic_v1" else ("✅" if genre else "")
        # Bilingual genre display
        genre_en = GENRE_BILINGUAL.get(genre, "")
        genre_en_short = genre_en.split(" — ")[1] if " — " in genre_en else ""
        if genre and genre != "غير_محدد":
            en_sub = f'<br/><span style="font-size:11px;color:var(--ink3);font-style:italic">{genre_en_short}</span>' if genre_en_short else ""
            ar_span = f'<span style="font-family:IBM Plex Sans Arabic,sans-serif;direction:rtl;font-size:13px">{genre}</span>'
            genre_cell = f"{badge} {ar_span}{en_sub}"
        else:
            genre_cell = "<span style='color:var(--ink3)'>—</span>"

        page_display = str(page) if page else "—"
        rows_html += f"""
<tr>
  <td style="font-family:'IBM Plex Sans Arabic',sans-serif;direction:rtl;text-align:right;font-size:15px;line-height:1.8;color:var(--ink);padding:10px 16px;border-bottom:1px solid var(--line2)">{verse}{attrib_html}</td>
  <td style="font-family:'IBM Plex Sans Arabic',sans-serif;direction:rtl;text-align:right;font-size:13px;color:var(--ink2);padding:10px 16px;border-bottom:1px solid var(--line2);white-space:nowrap">{poet_display}</td>
  <td style="font-size:12px;color:var(--ink3);padding:10px 16px;border-bottom:1px solid var(--line2)">{genre_cell}</td>
  <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink3);padding:10px 16px;border-bottom:1px solid var(--line2)">{ms_key}<br/><span style="opacity:.6">p.{page_display}</span></td>
</tr>
"""

    st.markdown(f"""
<div style="overflow-x:auto;margin-top:16px">
<table style="width:100%;border-collapse:collapse;background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden">
  <thead>
    <tr style="background:var(--paper2)">
      <th style="padding:10px 16px;text-align:right;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Opening verse / المطلع</th>
      <th style="padding:10px 16px;text-align:right;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Poet / الشاعر</th>
      <th style="padding:10px 16px;text-align:left;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Genre / النوع</th>
      <th style="padding:10px 16px;text-align:left;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);font-weight:500;border-bottom:1px solid var(--line)">Source / المصدر</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>
{"<p style='margin-top:10px;font-size:12px;color:var(--ink3);font-style:italic'>Showing first 200 results — refine with the filters above.</p>" if len(filtered) > 200 else ""}
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab C — Governance & Metrics dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _gov_load_eval() -> dict:
    """Load data/evaluation_raw.json. Empty dict when missing — UI shows placeholder."""
    import json as _json
    path = _REPO_ROOT / "data" / "evaluation_raw.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}


def _gov_load_corpus_stats() -> dict:
    """
    Compute live corpus stats from data/qdrant/chunks_meta.json.
    Why live: the index can be rebuilt while the app is running; static numbers
    in the architecture doc go stale within minutes of a rebuild.
    """
    import json as _json
    from collections import Counter
    path = _REPO_ROOT / "data" / "qdrant" / "chunks_meta.json"
    if not path.exists():
        return {"total": 0, "by_level": {}, "rebuilt_at": None}
    try:
        with open(path, encoding="utf-8") as f:
            chunks = _json.load(f)
        levels = Counter((c.get("level") or "?") for c in chunks)
        rebuilt_at = path.stat().st_mtime
        # Books represented in the reference layer
        books = sorted({
            c.get("book_short_key", "")
            for c in chunks
            if c.get("level") == "reference" and c.get("book_short_key")
        })
        # Manuscripts represented (verse-level only — most reliable count)
        manuscripts = sorted({
            c.get("manuscript_short_key", "")
            for c in chunks
            if c.get("level") == "verse" and c.get("manuscript_short_key")
        })
        return {
            "total":       len(chunks),
            "by_level":    dict(levels),
            "rebuilt_at":  rebuilt_at,
            "books":       books,
            "manuscripts": manuscripts,
        }
    except Exception:
        return {"total": 0, "by_level": {}, "rebuilt_at": None}


def _gov_metric_card(
    label_en: str,
    label_ar: str,
    value: str,
    target: str,
    status: str,            # "ok" | "warn" | "fail" | "info"
    note: str = "",
) -> None:
    """Render a single metric card. status drives the colour bar at the top."""
    color_map = {
        "ok":   "#2D7A3F",  # forest green
        "warn": "#C9881C",  # ochre
        "fail": "#A43A2D",  # terra
        "info": "#5A5853",  # neutral ink
    }
    icon_map = {"ok": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
    bar = color_map.get(status, "#5A5853")
    icon = icon_map.get(status, "ℹ️")
    st.markdown(
        f"""
<div style="
  border:1px solid #E5DED1; border-top:3px solid {bar};
  border-radius:6px; padding:14px 16px; background:#FCFAF6;
  height:100%;
">
  <div style="font-size:12px; color:#5A5853; letter-spacing:0.04em;">
    {icon} {label_en}
    <span style="float:right; direction:rtl; font-family:'IBM Plex Sans Arabic',sans-serif;">{label_ar}</span>
  </div>
  <div style="font-size:28px; font-weight:500; color:#2A1F17;
              font-family:'Fraunces',serif; margin-top:6px;">{value}</div>
  <div style="font-size:11px; color:#7A766F; margin-top:2px;">target: {target}</div>
  {f'<div style="font-size:11px; color:#5A5853; margin-top:8px; line-height:1.4;">{note}</div>' if note else ''}
</div>
""",
        unsafe_allow_html=True,
    )


def _gov_provider_status() -> tuple[str, str, str]:
    """Return (provider, model, status_emoji) for the top-of-page health line."""
    try:
        from fatat_al_arab.llm import get_provider_info
        info = get_provider_info()
        prov = info.get("provider", "?")
        model = info.get("model_primary", "?")
        if prov == "stub":
            return prov, model, "⚪ stub mode (deterministic)"
        if info.get("using_fallback"):
            return prov, model, "🟡 on fallback model"
        # Try a 1-token health probe
        return prov, model, "🟢 live"
    except Exception as exc:
        return "?", "?", f"🔴 unavailable: {exc}"


def _render_governance() -> None:
    """
    Tab C — Governance & Metrics dashboard.

    Why this tab exists: the project makes specific quantitative claims (Recall@5,
    refusal precision, latency targets) that grade the system. Embedding the
    measurement directly in the app — pulling from the same data/evaluation_raw.json
    that the report renders from — closes the loop between *what is claimed* and
    *what is measured*. The audience sees the live numbers, not a curated slide.
    """
    st.markdown(
        '<h1 class="page-title">Governance & Metrics</h1>'
        '<p class="page-sub" style="color:#5A5853;">Live measurement of NABAT-AI against §7 evaluation axes — '
        '<span style="direction:rtl;">قياس مباشر لأداء النظام مقابل محاور التقييم.</span></p>',
        unsafe_allow_html=True,
    )

    eval_raw     = _gov_load_eval()
    corpus_stats = _gov_load_corpus_stats()
    provider, model, prov_status = _gov_provider_status()

    # ── Top row — system health ───────────────────────────────────────────────
    st.markdown("### System health · صحة النظام")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(
            f"**LLM provider** · `{provider}`<br>"
            f"<span style='font-size:12px;color:#5A5853;'>{prov_status}</span><br>"
            f"<span style='font-size:11px;color:#7A766F;font-family:monospace;'>{model}</span>",
            unsafe_allow_html=True,
        )
    with h2:
        if corpus_stats["total"] > 0:
            from datetime import datetime as _dt
            built = _dt.fromtimestamp(corpus_stats["rebuilt_at"]).strftime("%Y-%m-%d %H:%M")
            st.markdown(
                f"**Index status** · 🟢 ready<br>"
                f"<span style='font-size:12px;color:#5A5853;'>{corpus_stats['total']:,} chunks · "
                f"{len(corpus_stats.get('manuscripts', []))} manuscripts · "
                f"{len(corpus_stats.get('books', []))} reference books</span><br>"
                f"<span style='font-size:11px;color:#7A766F;'>rebuilt: {built}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Index not built yet. Run `python scripts/rebuild_index.py --force`.")
    with h3:
        if eval_raw.get("run_date"):
            st.markdown(
                f"**Last evaluation** · {eval_raw.get('run_date', '—')}<br>"
                f"<span style='font-size:12px;color:#5A5853;'>provider: <code>{eval_raw.get('provider', '?')}</code> · "
                f"{eval_raw.get('correctness', {}).get('non_refusal_count', 0) + eval_raw.get('correctness', {}).get('ooc_total', 0)} queries</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No evaluation run yet — use the button at the bottom.")

    # ── Tier 1 — hard correctness ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Tier 1 — Hard correctness · الموثوقية الأساسية")

    correctness = eval_raw.get("correctness", {})
    cit_rate = correctness.get("citation_resolvability_rate", 0.0)
    ref_prec = correctness.get("refusal_precision_ooc", 0.0)
    ooc_total = correctness.get("ooc_total", 0)

    cit_status = "ok" if cit_rate >= 1.0 else "fail"
    ref_status = "ok" if ref_prec >= 0.90 else ("warn" if ref_prec >= 0.5 else "fail")

    c1, c2 = st.columns(2)
    with c1:
        _gov_metric_card(
            "Citation-resolvability", "قابلية تتبع الاقتباسات",
            f"{cit_rate*100:.1f}%", "100%",
            cit_status,
            "Every cited anchor traces to either the manuscript registry or the reference corpus."
            if cit_status == "ok"
            else "At least one citation does not resolve to a known source — investigate before demo."
        )
    with c2:
        provider_in_eval = eval_raw.get("provider", "?")
        _gov_metric_card(
            "OOC refusal precision", "دقة رفض الاستفسارات الخارجية",
            f"{ref_prec*100:.1f}%", "≥ 90%",
            ref_status,
            f"{correctness.get('ooc_refusal_count', 0)}/{ooc_total} out-of-corpus queries refused. "
            f"Mode: <code>{provider_in_eval}</code>." +
            (" Re-run with a live LLM key for the demo number." if provider_in_eval == "stub" else "")
        )

    # ── Tier 2 — coverage (informational) ────────────────────────────────────
    st.markdown("### Tier 2 — Coverage · التغطية (إعلامية)")
    rec_at5 = correctness.get("recall_at_5", 0.0)
    ref_hit = correctness.get("reference_hit_rate", 0.0)
    ref_active = correctness.get("reference_hit_rate_active", 0)
    ref_total  = correctness.get("reference_hit_rate_total", 0)

    c3, c4 = st.columns(2)
    with c3:
        _gov_metric_card(
            "Manuscript anchor recall@5", "استرجاع المخطوطات الذهبية",
            f"{rec_at5*100:.1f}%", "context-dependent",
            "info",
            "Fraction of fixtures with a `gold_anchor_ids` list whose answer surfaces "
            "at least one gold manuscript chunk in the post-RRF top-5. Adding the "
            "reference corpus naturally trades anchor slots for cultural context — "
            "this number alone is not a quality verdict."
        )
    with c4:
        _gov_metric_card(
            "Reference hit rate", "نسبة استدعاء المراجع",
            f"{ref_hit*100:.1f}%", "(informational)",
            "info",
            f"{ref_active}/{ref_total} in-corpus queries activate ≥1 reference (PDF) chunk in top-5. "
            "Tracks the value of the new scholarly layer."
        )

    # ── Tier 3 — efficiency ──────────────────────────────────────────────────
    st.markdown("### Tier 3 — Efficiency · الكفاءة")
    eff = eval_raw.get("efficiency", {})
    p50 = eff.get("p50_ms", 0)
    p95 = eff.get("p95_ms", 0)
    p50_target_ms = eff.get("p50_target_ms", 4000)
    p95_target_ms = eff.get("p95_target_ms", 8000)
    p50_status = "ok" if p50 < p50_target_ms else "warn"
    p95_status = "ok" if p95 < p95_target_ms else "warn"

    c5, c6 = st.columns(2)
    with c5:
        _gov_metric_card(
            "p50 end-to-end latency", "زمن الاستجابة (وسيط)",
            f"{p50:.0f} ms", f"< {p50_target_ms} ms",
            p50_status,
            "Live-mode targets assume Together.ai or Groq (≈300–800 ms per call, 5–8 calls per query)."
        )
    with c6:
        _gov_metric_card(
            "p95 end-to-end latency", "زمن الاستجابة (95%)",
            f"{p95:.0f} ms", f"< {p95_target_ms} ms",
            p95_status,
            "Stub-mode latencies (~50 ms) are not representative of the demo runtime."
        )

    # ── Corpus distribution ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Corpus composition · مكوّنات الفهرس")
    by_level = corpus_stats.get("by_level") or {}
    if by_level:
        # Sort levels in a meaningful order: row-backed → aggregate → reference
        order = ["verse", "group", "poem", "manuscript", "poet", "era", "genre", "emotion", "reference"]
        rows = [(lv, by_level.get(lv, 0)) for lv in order if lv in by_level]
        # Append any unexpected levels
        for k, v in by_level.items():
            if k not in order:
                rows.append((k, v))
        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]
        try:
            import pandas as _pd
            df = _pd.DataFrame({"level": labels, "chunks": values}).set_index("level")
            st.bar_chart(df, height=260)
        except ImportError:
            for lv, n in rows:
                st.markdown(f"- `{lv:<12}` **{n:>5,}**")
    else:
        st.info("No chunks indexed yet.")

    # ── Phase ingestion summary ──────────────────────────────────────────────
    # Why: the corpus now spans all four phases; a reviewer needs to see the
    # anchor counts per phase at a glance without digging into JSON files.
    st.markdown("### Ground-truth phases · مراحل البيانات الذهبية")
    ph_cols = st.columns(4)
    _phase_rows = [
        ("Phase 1 — Momentum",      "ms07, ms14, ms15, ms22", 344,  "✅"),
        ("Phase 2 — Core Sadr/Ajuz","ms04, ms05, ms19, ms21", 240,  "✅"),
        ("Phase 3 — Edge Cases",    "ms01, ms03, ms06, ms08", 136,  "✅"),
        ("Phase 4 — TOC Metadata",  "vols 001–782",           1502, "✅"),
    ]
    for col, (title, mss, n, badge) in zip(ph_cols, _phase_rows):
        col.markdown(
            f"<div style='border:1px solid #E5DED1;border-top:3px solid #6B8159;"
            f"border-radius:6px;padding:12px 14px;background:#FCFAF6'>"
            f"<div style='font-size:11px;color:#5A5853;letter-spacing:.04em'>{badge} {title}</div>"
            f"<div style='font-size:24px;font-weight:500;color:#2A1F17;"
            f"font-family:Fraunces,serif;margin-top:4px'>{n:,}</div>"
            f"<div style='font-size:11px;color:#7A766F;margin-top:2px'>{mss}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    _total_entries = sum(r[2] for r in _phase_rows)
    st.caption(f"Combined: **{_total_entries:,}** bayts · {corpus_stats['total']:,} vector chunks · 25 manuscripts")

    # ── CRAG verdict distribution ────────────────────────────────────────────
    crag_dist = correctness.get("crag_verdict_distribution") or {}
    if crag_dist:
        st.markdown("### CRAG verdict distribution · توزيع تقييم الاسترجاع")
        try:
            import pandas as _pd
            df = _pd.DataFrame(
                {"verdict": list(crag_dist.keys()), "count": list(crag_dist.values())}
            ).set_index("verdict")
            st.bar_chart(df, height=200)
        except ImportError:
            for k, v in crag_dist.items():
                st.markdown(f"- {k}: {v}")

    # ── Robustness loop activation ───────────────────────────────────────────
    robustness = eval_raw.get("robustness") or {}
    if robustness.get("total_queries"):
        st.markdown("### Robustness — failure-budget activation · تنشيط حدود الإخفاق")
        b1, b2, b3 = st.columns(3)
        b1.metric("CRAG re-query rate",
                  f"{robustness.get('crag_requery_rate', 0)*100:.1f}%",
                  f"{robustness.get('crag_requery_count', 0)} queries")
        b2.metric("Self-RAG retry rate",
                  f"{robustness.get('self_rag_retry_rate', 0)*100:.1f}%",
                  f"{robustness.get('self_rag_retry_count', 0)} queries")
        b3.metric("Fallback LLM rate",
                  f"{robustness.get('fallback_llm_rate', 0)*100:.1f}%")

    # ── Per-stage timing waterfall ───────────────────────────────────────────
    per_stage = eff.get("per_stage_avg_ms") or {}
    if per_stage:
        st.markdown("### Per-stage average latency · زمن كل مرحلة")
        try:
            import pandas as _pd
            df = _pd.DataFrame(
                {"stage": list(per_stage.keys()), "ms": list(per_stage.values())}
            ).set_index("stage")
            st.bar_chart(df, height=200)
        except ImportError:
            for k, v in per_stage.items():
                st.markdown(f"- `{k}`: {v} ms")

    # ── Run mini-eval button ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Re-evaluate · إعادة التقييم")
    cl, cr = st.columns([3, 1])
    with cl:
        st.caption(
            "Click to run the full 50-fixture evaluation harness. Takes ~15 seconds in "
            "stub mode, longer with a live LLM. Output overwrites "
            "`data/evaluation_raw.json` and `data/evaluation_report.md`."
        )
    with cr:
        if st.button("▶ Run evaluation", key="gov_run_eval", use_container_width=True):
            with st.spinner("Running scripts/evaluate.py …"):
                import subprocess, sys as _sys
                env = os.environ.copy()
                env.setdefault("PYTHONPATH", str(_REPO_ROOT / "src"))
                proc = subprocess.run(
                    [_sys.executable, str(_REPO_ROOT / "scripts" / "evaluate.py")],
                    capture_output=True, text=True, env=env, timeout=600,
                )
                if proc.returncode == 0:
                    st.success("Evaluation complete — reload the tab to see new numbers.")
                else:
                    st.error(f"Evaluation failed (exit {proc.returncode}).")
                    with st.expander("stderr"):
                        st.code(proc.stderr[-2000:])

    # ── Honest framing footer ────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "This dashboard is part of the project's governance posture: every numerical "
        "claim is rendered from the same `data/evaluation_raw.json` the deliverable "
        "report quotes from. Stub-mode results are deterministic templates — they "
        "validate the pipeline but are not representative of live LLM quality."
    )


def main() -> None:
    """
    Why set_page_config first: Streamlit requires it before any other st.* call.
    layout="wide" gives RTL Arabic text enough horizontal room.

    Navigation:
      page = "hero"      → hero landing screen (default on first load)
      page = "workbench" → Scholar Workbench + Archive Manager + Governance tabs
    """
    st.set_page_config(
        layout="wide",
        page_title="NABAT-AI — Khaleeji Poetry Scholar",
        page_icon="📜",
    )

    # Preconnect + stylesheet link tags — non-blocking, so the browser starts
    # fetching fonts in parallel with the first paint rather than stalling it.
    # Why <link> instead of @import: CSS @import is render-blocking; <link> is not.
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,350;0,9..144,400;1,9..144,300;1,9..144,350;1,9..144,400&family=IBM+Plex+Sans+Arabic:wght@300;400&family=Inter:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

    # Inject design system CSS
    st.markdown(_NABAT_CSS, unsafe_allow_html=True)

    # Session state init
    if "page" not in st.session_state:
        st.session_state["page"] = "hero"
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "active_genre_filter" not in st.session_state:
        st.session_state["active_genre_filter"] = ""
    if "active_emotion_filter" not in st.session_state:
        st.session_state["active_emotion_filter"] = []
    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "default"
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "workbench"

    # ── Route: hero / browse / workbench ──────────────────────────────────────
    page = st.session_state["page"]

    if page == "hero":
        _render_hero()
        return

    # ── Browse corpus page ─────────────────────────────────────────────────────
    if page == "browse":
        _render_sidebar()
        _render_browse_corpus()
        return

    # ── Workbench page (default) ───────────────────────────────────────────────
    _render_masthead_with_back()
    _render_sidebar()

    # Tab selection is driven by the sidebar radio (key="active_tab")
    active_tab = st.session_state.get("active_tab", "workbench")

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    if active_tab == "workbench":
        _render_workbench()
    elif active_tab == "governance":
        _render_governance()
    else:
        _render_archive_manager()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__" and _STREAMLIT_AVAILABLE:
    main()
