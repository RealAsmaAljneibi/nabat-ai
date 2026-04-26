"""
src/fatat_al_arab/audio_input.py
=================================
Why this file exists: Khaleeji Nabati poetry is fundamentally an oral tradition.
Scholars and enthusiasts often *recite* or *hum* a verse before they can write it,
and searching for a half-remembered poem by voice is far more natural than typing
Arabic on a keyboard. This module converts a recorded or uploaded audio file into
a text query the RAG pipeline can process.

Model used: Whisper-small fine-tuned on Arabic poetry ASR, loaded from
~/poetry/models/local/whisper-small/ — the same checkpoint trained in the
MAAI7103 poetry project. The fine-tuned model handles:
  - Khaleeji dialectal phonology (gāf → qāf alternation, etc.)
  - Poetic metre breaks and pause patterns
  - Dialectal lexemes absent from standard Whisper's training data

Fallback chain:
  1. Local fine-tuned whisper-small (~/poetry/models/local/whisper-small/)
  2. HuggingFace whisper-small ("openai/whisper-small") — pulled on first use
  3. If both fail (no torch / no network) → returns None with a clear error msg

The module does NOT manage Streamlit widgets — that is streamlit_app.py's job.
It only owns: audio file → text string.

Graceful degradation is mandatory: the function must never raise.
Any failure returns (None, error_message) so the UI can show a helpful notice
rather than a crash.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import IO

# ── Model paths ────────────────────────────────────────────────────────────────
_POETRY_HOME    = Path(os.getenv("POETRY_HOME", str(Path.home() / "poetry")))
_LOCAL_WHISPER  = _POETRY_HOME / "models" / "local" / "whisper-small"
_HF_WHISPER     = "openai/whisper-small"

# Why 30 seconds: Whisper-small handles 30-second windows natively. A matla
# verse (the poem's opening couplet, which is what users will recite) is
# typically 5–15 seconds. Longer clips are silently truncated at 30 s.
_MAX_DURATION_SECONDS = 30

# Module-level singleton — loaded once per process.
_pipeline = None
_pipeline_loaded: bool = False
_pipeline_available: bool = False


def _load_pipeline() -> None:
    """
    Load the Whisper ASR pipeline once. Sets _pipeline_available flag.
    Why a pipeline wrapper: HuggingFace pipelines handle audio resampling,
    feature extraction, and beam search internally — no ffmpeg dependency
    needed for standard formats (wav, mp3, m4a via librosa/soundfile).
    """
    global _pipeline, _pipeline_loaded, _pipeline_available
    if _pipeline_loaded:
        return
    _pipeline_loaded = True
    try:
        from transformers import pipeline as hf_pipeline  # soft dependency

        model_id = (
            str(_LOCAL_WHISPER)
            if _LOCAL_WHISPER.exists()
            else _HF_WHISPER
        )
        _pipeline = hf_pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            chunk_length_s=_MAX_DURATION_SECONDS,
            generate_kwargs={"language": "ar", "task": "transcribe"},
        )
        _pipeline_available = True
    except Exception:
        _pipeline = None
        _pipeline_available = False


def is_whisper_available() -> bool:
    """
    True if the Whisper model loaded successfully.
    Used by Streamlit to decide whether to show the audio upload widget.
    """
    _load_pipeline()
    return _pipeline_available


def transcribe_audio(audio_source: str | Path | IO[bytes]) -> tuple[str | None, str | None]:
    """
    Transcribe an audio file to Arabic text suitable for RAG query input.

    Args:
        audio_source: One of:
          - str / Path  — path to a local audio file (wav, mp3, m4a, ogg, flac)
          - file-like   — bytes IO as returned by st.file_uploader

    Returns:
        (transcription, error_message)
        On success: (Arabic text string, None)
        On failure: (None, human-readable error string)

    Why return a tuple instead of raising: the Streamlit UI wraps this and must
    always have something to display. Raising would require try/except in the UI
    layer, which is harder to keep consistent across two entry points (file
    upload + mic recording path).
    """
    _load_pipeline()

    if not _pipeline_available:
        return None, (
            "نموذج التعرف على الصوت غير متوفر حالياً. "
            "تحقق من وجود ملفات نموذج Whisper في مجلد ~/poetry/models/local/whisper-small/ "
            "أو تثبيت مكتبة transformers.\n"
            "(Whisper ASR model is not available. Check that ~/poetry/models/local/whisper-small/ "
            "exists or that transformers is installed.)"
        )

    # ── Normalise input to a file path ─────────────────────────────────────
    tmp_path: Path | None = None
    try:
        if isinstance(audio_source, (str, Path)):
            audio_path = Path(audio_source)
            if not audio_path.exists():
                return None, f"ملف الصوت غير موجود: {audio_path}"
        else:
            # File-like (Streamlit UploadedFile) — write to a temp file so
            # the pipeline can seek it without requiring in-memory loading.
            suffix = _guess_suffix(audio_source)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source.read())
                tmp_path = Path(tmp.name)
            audio_path = tmp_path

        # ── Transcribe ─────────────────────────────────────────────────────
        result = _pipeline(str(audio_path))
        text: str = result.get("text", "").strip()

        if not text:
            return None, (
                "لم يتم التعرف على أي كلام في الملف الصوتي. "
                "تأكد من وضوح التسجيل وأنه يحتوي على كلام باللغة العربية.\n"
                "(No speech detected. Ensure the recording is clear and contains Arabic speech.)"
            )

        return text, None

    except Exception as exc:
        return None, (
            f"حدث خطأ أثناء معالجة الصوت: {exc}\n"
            f"(Audio processing error: {exc})"
        )
    finally:
        # Clean up temp file if we created one
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _guess_suffix(file_obj) -> str:
    """
    Infer file extension from the uploaded file object's name attribute.
    Why needed: Whisper relies on the file extension to pick the correct
    audio decoder. Streamlit's UploadedFile exposes .name with the original
    filename.
    """
    name = getattr(file_obj, "name", "") or ""
    suffix = Path(name).suffix.lower()
    return suffix if suffix in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"} else ".wav"


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> tuple[str | None, str | None]:
    """
    Convenience wrapper for raw bytes (e.g. from st.audio_input / mic recording).

    Args:
        audio_bytes: raw audio bytes
        filename:    original filename (used for extension detection)

    Returns same (transcription, error) tuple as transcribe_audio().
    """
    import io
    buf = io.BytesIO(audio_bytes)
    buf.name = filename   # type: ignore[attr-defined]  # mirrors UploadedFile API
    return transcribe_audio(buf)
