"""
src/al_nassikh/audio_corpus_ingest.py
======================================
⚡ SCOPE EXTENSION EXT-3 — Oral Tradition Audio Corpus Integration

Why this file exists: Nabati poetry is historically an oral tradition — the
manuscripts are the written record of what was primarily sung and recited.
The MAAI7103 project produced 3,340 Khaleeji Nabati audio clip transcriptions
(stored at ~/poetry/maai7103/ or provided as a data dump). This module ingests
those transcriptions into the same Qdrant index as the manuscript corpus,
carrying source_type="oral_tradition" so the UI can distinguish them.

Two ingestion modes:
  1. Transcription mode — if a JSON transcription file already exists
     (produced by the MAAI7103 pipeline), load it directly.
  2. Audio mode — if only .mp3/.wav/.m4a files exist, run Whisper transcription
     first, then ingest.

The resulting anchor entries have the same schema as manuscript entries
(anchor_id, poet_name, text, source_volume, source_page, etc.) with
additional fields:
  source_type:          "oral_tradition"
  audio_file:           original file path
  performer:            reciter name if known
  maai7103_clip_id:     clip ID from the MAAI7103 dataset

Cross-modal linking: if a verse text fuzzy-matches a manuscript entry
(rapidfuzz WRatio ≥ 80), the oral entry is cross-linked to that anchor_id
so retrieval can surface both the manuscript AND the oral tradition recording.

Architecture ref: Worker 1 (Al-Nassikh) ingestion — same pipeline as
ingest_phases_123.py. Outputs anchor entries compatible with index.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ORAL_DATA  = _REPO_ROOT / "data" / "oral_tradition"
_OUTPUT     = _ORAL_DATA / "oral_anchor_registry.json"

# Default location of the MAAI7103 transcription dump
_MAAI7103_DEFAULT_PATHS = [
    Path.home() / "poetry" / "maai7103" / "transcriptions.json",
    Path.home() / "poetry" / "maai7103" / "clips_metadata.json",
    _ORAL_DATA / "maai7103_transcriptions.json",
]

# ── Anchor schema ─────────────────────────────────────────────────────────────

def _make_oral_entry(
    clip_id: str,
    text: str,
    poet_name: str = "",
    performer: str = "",
    audio_file: str = "",
    duration_s: float = 0.0,
    manuscript_anchor_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build an anchor registry entry for an oral tradition clip.
    Compatible with the standard manuscript anchor schema used by index.py.
    """
    return {
        "anchor_id":            f"oral_{clip_id}",
        "poet_name":            poet_name or "غير معروف",
        "text":                 text,
        "source_volume":        "oral_tradition",
        "source_page":          clip_id,
        "source_image_path":    audio_file,   # re-used field to store audio path
        "manuscript_short_key": "oral",
        "source_type":          "oral_tradition",
        "maai7103_clip_id":     clip_id,
        "performer":            performer,
        "audio_file":           audio_file,
        "duration_s":           duration_s,
        "linked_manuscript_anchor": manuscript_anchor_id,  # cross-modal link
        "genre":                extra.get("genre", "غير_محدد") if extra else "غير_محدد",
        "genre_confidence":     extra.get("genre_confidence", 0.0) if extra else 0.0,
        "genre_source":         "oral_tradition_metadata",
        "emotions":             extra.get("emotions", []) if extra else [],
        # Transparency flags — distinguishes primary manuscript data from secondary sources
        "is_secondary_source":  True,
        "data_tier":            "secondary",
    }


# ── Transcription loading ─────────────────────────────────────────────────────

def _find_transcription_file() -> Optional[Path]:
    """Search default locations for the MAAI7103 transcription file."""
    for p in _MAAI7103_DEFAULT_PATHS:
        if p.exists():
            logger.info(f"Found MAAI7103 transcription file: {p}")
            return p
    return None


def load_maai7103_transcriptions(path: Optional[Path] = None) -> list[dict]:
    """
    Load MAAI7103 transcription data. Accepts two JSON formats:

    Format A (list of dicts with clip metadata):
      [{"clip_id": "001", "text": "...", "poet": "...", "audio_file": "..."}]

    Format B (flat dict keyed by clip_id):
      {"001": {"text": "...", "poet": "..."}, ...}

    Returns a list of normalised clip dicts.
    """
    if path is None:
        path = _find_transcription_file()
    if path is None or not path.exists():
        logger.warning("MAAI7103 transcription file not found. Returning empty list.")
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    clips = []

    if isinstance(raw, list):
        for item in raw:
            clips.append({
                "clip_id":    str(item.get("clip_id") or item.get("id") or len(clips)),
                "text":       item.get("text") or item.get("transcription") or "",
                "poet_name":  item.get("poet") or item.get("poet_name") or "",
                "performer":  item.get("performer") or item.get("reciter") or "",
                "audio_file": item.get("audio_file") or item.get("file") or "",
                "duration_s": float(item.get("duration") or item.get("duration_s") or 0.0),
                "genre":      item.get("genre") or "غير_محدد",
                "emotions":   item.get("emotions") or [],
            })
    elif isinstance(raw, dict):
        for clip_id, item in raw.items():
            clips.append({
                "clip_id":    str(clip_id),
                "text":       item.get("text") or item.get("transcription") or "",
                "poet_name":  item.get("poet") or item.get("poet_name") or "",
                "performer":  item.get("performer") or item.get("reciter") or "",
                "audio_file": item.get("audio_file") or "",
                "duration_s": float(item.get("duration") or 0.0),
                "genre":      item.get("genre") or "غير_محدد",
                "emotions":   item.get("emotions") or [],
            })

    logger.info(f"Loaded {len(clips)} MAAI7103 clips from {path}")
    return clips


def transcribe_audio_files(audio_dir: Path) -> list[dict]:
    """
    If only audio files exist (no pre-built transcriptions), run Whisper
    on each file and return clip dicts in the same format as load_maai7103_transcriptions.

    Requires: openai-whisper installed and the model downloaded.
    Gracefully logs and skips files that fail.
    """
    try:
        import whisper  # type: ignore[import]
        model = whisper.load_model("base")
        logger.info("Whisper model loaded for audio transcription")
    except ImportError:
        logger.error("openai-whisper not installed. Run: pip install openai-whisper")
        return []

    clips = []
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    audio_files = [
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    logger.info(f"Transcribing {len(audio_files)} audio files in {audio_dir}")

    for i, audio_file in enumerate(sorted(audio_files)):
        try:
            result = model.transcribe(str(audio_file), language="ar")
            text = result.get("text", "").strip()
            if not text:
                logger.warning(f"Empty transcription for {audio_file.name}")
                continue
            clips.append({
                "clip_id":    f"whisper_{i:04d}",
                "text":       text,
                "poet_name":  "",
                "performer":  "",
                "audio_file": str(audio_file),
                "duration_s": result.get("duration", 0.0),
                "genre":      "غير_محدد",
                "emotions":   [],
            })
            logger.info(f"Transcribed [{i+1}/{len(audio_files)}]: {audio_file.name}")
        except Exception as exc:
            logger.warning(f"Failed to transcribe {audio_file.name}: {exc}")

    return clips


# ── Cross-modal linking ────────────────────────────────────────────────────────

def _cross_link_to_manuscripts(
    clips: list[dict],
    manuscript_registry: list[dict],
    threshold: float = 80.0,
) -> list[dict]:
    """
    For each oral clip, find the closest matching manuscript anchor by
    fuzzy text similarity (rapidfuzz WRatio). If score ≥ threshold, set
    linked_manuscript_anchor in the clip dict.

    Why threshold 80: same acceptance gate as cross_link.py (Phase 1↔4 joining).
    """
    try:
        from rapidfuzz import fuzz  # type: ignore[import]
    except ImportError:
        logger.warning("rapidfuzz not installed — skipping cross-modal linking")
        return clips

    ms_texts = [(e.get("text", ""), e.get("anchor_id", "")) for e in manuscript_registry]

    for clip in clips:
        clip_text = clip.get("text", "")
        if not clip_text:
            continue
        best_score = 0.0
        best_anchor = None
        for ms_text, ms_anchor_id in ms_texts:
            score = fuzz.WRatio(clip_text, ms_text)
            if score > best_score:
                best_score = score
                best_anchor = ms_anchor_id
        if best_score >= threshold:
            clip["linked_manuscript_anchor"] = best_anchor
            clip["cross_link_score"] = best_score
            logger.debug(f"Clip {clip['clip_id']} → {best_anchor} (score={best_score:.1f})")

    return clips


# ── Main ingestion entry point ────────────────────────────────────────────────

def ingest_oral_corpus(
    transcription_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    manuscript_registry: Optional[list[dict]] = None,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """
    Full oral tradition ingestion pipeline.

    Priority:
      1. Use transcription_path if provided and exists.
      2. Use MAAI7103 default paths if available.
      3. Transcribe audio_dir with Whisper as last resort.

    Returns a list of anchor entries compatible with index.py.
    Writes to output_path (defaults to data/oral_tradition/oral_anchor_registry.json).
    """
    _ORAL_DATA.mkdir(parents=True, exist_ok=True)
    output_path = output_path or _OUTPUT

    # ── Step 1: Load clips ────────────────────────────────────────────────────
    if transcription_path and Path(transcription_path).exists():
        clips = load_maai7103_transcriptions(Path(transcription_path))
    else:
        clips = load_maai7103_transcriptions()   # tries default paths

    if not clips and audio_dir and Path(audio_dir).exists():
        logger.info("No transcriptions found — falling back to Whisper transcription")
        clips = transcribe_audio_files(Path(audio_dir))

    if not clips:
        logger.warning("No oral tradition data found. Returning empty registry.")
        return []

    # ── Step 2: Cross-link to manuscripts ─────────────────────────────────────
    if manuscript_registry:
        clips = _cross_link_to_manuscripts(clips, manuscript_registry)

    # ── Step 3: Build anchor entries ──────────────────────────────────────────
    entries = []
    for clip in clips:
        if not clip.get("text"):
            continue
        entry = _make_oral_entry(
            clip_id=clip["clip_id"],
            text=clip["text"],
            poet_name=clip.get("poet_name", ""),
            performer=clip.get("performer", ""),
            audio_file=clip.get("audio_file", ""),
            duration_s=clip.get("duration_s", 0.0),
            manuscript_anchor_id=clip.get("linked_manuscript_anchor"),
            extra={
                "genre":            clip.get("genre", "غير_محدد"),
                "genre_confidence": 0.6 if clip.get("genre") else 0.0,
                "emotions":         clip.get("emotions", []),
            },
        )
        entries.append(entry)

    # ── Step 4: Write output ──────────────────────────────────────────────────
    output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Wrote {len(entries)} oral tradition entries → {output_path}")
    return entries


def get_oral_corpus_stats(registry_path: Optional[Path] = None) -> dict:
    """Return summary statistics for the oral tradition corpus."""
    path = registry_path or _OUTPUT
    if not path.exists():
        return {"status": "not_built", "entries": 0}
    entries = json.loads(path.read_text(encoding="utf-8"))
    linked = sum(1 for e in entries if e.get("linked_manuscript_anchor"))
    poets  = len({e.get("poet_name") for e in entries if e.get("poet_name")})
    return {
        "status":           "ready",
        "entries":          len(entries),
        "linked_to_manuscript": linked,
        "unique_poets":     poets,
        "source_type":      "oral_tradition",
    }
