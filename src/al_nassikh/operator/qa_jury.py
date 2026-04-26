"""
src/al_nassikh/operator/qa_jury.py
====================================
Why this file exists: §2.3 Stage 8 — "The Jury" is the final quality gate
before a transcription enters the RAG index. It takes a PAGE-XML file (or
parsed dict from al_nassikh.parser) and surfaces every text line whose
annotator-assigned confidence is below the acceptability threshold. Lines that
pass the gate proceed to Phase-4 merge; lines that fail go back to the annotator
for correction or are marked as LOW confidence in the anchor registry.

Architecture ref: §2.3 Stage 8. §2.1 CER tiers: HIGH (<10%), MEDIUM (10-25%),
LOW (≥25%). The gate here is on annotator confidence, not CER, because CER
requires a reference — this stage runs before reference validation.

The function is deterministic and has no side effects — it reads the PAGE-XML
and returns a report dict. The Streamlit tab displays the report; the annotator
decides what to do with each flagged line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union


# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_HIGH_MIN   = 1.0    # exactly 1.0 → HIGH (HITL-confirmed)
CONFIDENCE_MEDIUM_MIN = 0.75   # 0.75-<1.0 → MEDIUM (model with some review)
CONFIDENCE_LOW_MAX    = 0.75   # below this → LOW → flag for re-review


def _tier(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH_MIN:
        return "HIGH"
    elif confidence >= CONFIDENCE_MEDIUM_MIN:
        return "MEDIUM"
    else:
        return "LOW"


def _parse_pagexml_lines(pagexml_source: Union[str, Path, dict]) -> list[dict]:
    """
    Extract text lines and their confidence scores from a PAGE-XML string or file.
    Returns list of {line_id, text, confidence, region_id, tier} dicts.

    Why regex (not lxml ElementTree): this function may be called from a test
    environment where lxml is not installed. The PAGE-XML structure is regular
    enough that regex is reliable for this narrow extraction task.
    """
    if isinstance(pagexml_source, dict):
        # Pre-parsed dict from al_nassikh.parser — iterate stanzas
        return _parse_parser_dict(pagexml_source)

    if isinstance(pagexml_source, Path):
        text = pagexml_source.read_text(encoding="utf-8")
    else:
        text = str(pagexml_source)

    lines = []
    # Match <TextLine ID="..." custom="readingOrder {index:N;} conf {...}">
    line_re = re.compile(
        r'<TextLine\s[^>]*ID="([^"]+)"[^>]*custom="([^"]*)"[^>]*>'
        r'.*?<Unicode>(.*?)</Unicode>',
        re.DOTALL,
    )
    conf_re = re.compile(r'conf\s*\{([^}]+)\}')
    region_re = re.compile(r'<TextRegion\s[^>]*ID="([^"]+)"')

    # Find region context for each line
    region_map: dict[str, str] = {}
    region_iter = re.finditer(
        r'<TextRegion\s[^>]*ID="([^"]+)".*?</TextRegion>', text, re.DOTALL
    )
    for rm in region_iter:
        region_id = rm.group(1)
        for lm in re.finditer(r'<TextLine\s[^>]*ID="([^"]+)"', rm.group(0)):
            region_map[lm.group(1)] = region_id

    for m in line_re.finditer(text):
        line_id     = m.group(1)
        custom_attr = m.group(2)
        unicode_text = m.group(3).strip()

        conf_match = conf_re.search(custom_attr)
        if conf_match:
            conf_str = conf_match.group(1).strip()
            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.5
        else:
            confidence = 1.0   # no conf attribute → assume HITL confirmed

        lines.append({
            "line_id":    line_id,
            "text":       unicode_text,
            "confidence": confidence,
            "region_id":  region_map.get(line_id, ""),
            "tier":       _tier(confidence),
        })

    return lines


def _parse_parser_dict(data: dict) -> list[dict]:
    """
    Extract lines from the dict structure produced by al_nassikh.parser.
    Handles both page-level dicts and stanza-level dicts.
    """
    lines = []
    # data might be keyed by page_id → {stanzas: [...]}
    pages = data.values() if isinstance(data, dict) else [data]
    for page in pages:
        stanzas = page.get("stanzas", [])
        page_id = page.get("poem_id", "")
        for stanza in stanzas:
            stanza_num = stanza.get("stanza_num", "?")
            for part in ["sadr", "ajuz"]:
                part_data = stanza.get(part, {})
                text = (
                    part_data.get("manuscript_reading") or
                    part_data.get("standard_reading") or
                    ""
                )
                # parser doesn't store per-line confidence; use poem-level
                confidence = page.get("confidence_score", 1.0)
                if text:
                    lines.append({
                        "line_id":    f"{page_id}_s{stanza_num}_{part}",
                        "text":       text,
                        "confidence": confidence,
                        "region_id":  page_id,
                        "tier":       _tier(confidence),
                    })
    return lines


def review(pagexml_source: Any) -> dict:
    """
    Stage 8 QA Jury — surface lines below the confidence threshold.

    Args:
        pagexml_source: PAGE-XML string, Path to a .xml file, or a
                        pre-parsed dict from al_nassikh.parser.

    Returns:
        {
          "total_lines":      int,
          "high_count":       int,
          "medium_count":     int,
          "low_count":        int,
          "flagged":          list[dict]   — only LOW lines
          "acceptance_rate":  float        — fraction of HIGH + MEDIUM lines
          "verdict":          "PASS" | "REVIEW" | "FAIL"
          "note":             str
        }

    Verdict:
      PASS   — all lines are HIGH or MEDIUM confidence
      REVIEW — some lines are LOW; annotator should inspect flagged list
      FAIL   — more than 30% of lines are LOW confidence
    """
    lines = _parse_pagexml_lines(pagexml_source)

    if not lines:
        return {
            "total_lines":    0,
            "high_count":     0,
            "medium_count":   0,
            "low_count":      0,
            "flagged":        [],
            "acceptance_rate": 1.0,
            "verdict":        "PASS",
            "note":           "No text lines found in source.",
        }

    high_lines   = [l for l in lines if l["tier"] == "HIGH"]
    medium_lines = [l for l in lines if l["tier"] == "MEDIUM"]
    low_lines    = [l for l in lines if l["tier"] == "LOW"]

    total          = len(lines)
    acceptance_rate = (len(high_lines) + len(medium_lines)) / total
    low_fraction   = len(low_lines) / total

    if low_fraction > 0.30:
        verdict = "FAIL"
        note = (
            f"{len(low_lines)}/{total} lines ({low_fraction:.0%}) are below confidence "
            f"threshold — exceeds 30% FAIL gate. The transcription needs substantial "
            f"correction before entering the RAG index."
        )
    elif low_lines:
        verdict = "REVIEW"
        note = (
            f"{len(low_lines)}/{total} lines flagged for review "
            f"({low_fraction:.0%}). Correct these lines in eScriptorium and re-run."
        )
    else:
        verdict = "PASS"
        note = (
            f"All {total} lines are HIGH or MEDIUM confidence. "
            f"Transcription is ready for Phase-4 merge."
        )

    return {
        "total_lines":    total,
        "high_count":     len(high_lines),
        "medium_count":   len(medium_lines),
        "low_count":      len(low_lines),
        "flagged":        low_lines,
        "acceptance_rate": round(acceptance_rate, 4),
        "verdict":        verdict,
        "note":           note,
    }
