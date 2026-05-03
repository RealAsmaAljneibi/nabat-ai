"""
creative_poet/nodes/muqayyim.py
=============================================
Why this node exists: Al-Muqayyim is the quality critic. A human poet (or a poet
using Al-Musharik's output) submits a poem; Al-Muqayyim evaluates it on four
dimensions and returns actionable, line-by-line feedback.

Four evaluation dimensions:
  1. Meter conformity    — does each bayt scan to the stated or apparent meter?
  2. Rhyme scheme        — is the qafiya (قافية) consistent across all verses?
  3. Khaleeji authenticity — vocabulary and imagery consistent with Nabati tradition?
  4. Occasion fit        — does the poem's tone and register suit the occasion?

This node works on submitted text only — it does not retrieve from the index.
The LLM's implicit knowledge of classical Arabic prosody and the Nabati tradition
is sufficient for grading. Al-Muqayyim can reference corpus examples by asking
Al-Mulhim first (separate turn); it does not call retrieve tools itself.

§5 failure budget: if the LLM grading call fails, return an honest error message.
There is no fallback grade — an empty critique is worse than an honest failure.

Tools used (from tools.py): SCORE_VERSE_QUALITY (no retrieval tools)
"""

from __future__ import annotations

import json
import logging

from fatat_al_arab.llm import chat
from creative_poet.personas import MUQAYYIM_PERSONA
from fatat_al_arab.state import trace_append

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CRITIQUE_DIMENSIONS = ("meter", "rhyme", "authenticity", "occasion")

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_MUQAYYIM = (
    MUQAYYIM_PERSONA
    + """
Task — Poem Critique (Stage M4):
Evaluate the submitted Nabati poem on four dimensions. Be specific and honest.

Return JSON with this exact shape:
{
  "overall_score": <1-5>,
  "scores": {
    "meter":        <1-5>,
    "rhyme":        <1-5>,
    "authenticity": <1-5>,
    "occasion":     <1-5>
  },
  "summary_ar": "<2-3 sentence overall assessment in Arabic>",
  "summary_en": "<2-3 sentence overall assessment in English>",
  "annotations": [
    {
      "line":       "<the verse line being annotated>",
      "dimension":  "meter|rhyme|authenticity|occasion",
      "verdict":    "strong|weak|error",
      "annotation": "<specific observation — one sentence>",
      "suggestion": "<concrete improvement, or empty string if verdict is strong>"
    }
  ],
  "strongest_verse": "<the line that works best — quote it>",
  "weakest_verse":   "<the line with the most critical issue — quote it>",
  "priority_fix":    "<the single most important correction the poet should make>"
}
Score guide: 5=excellent, 4=good, 3=acceptable, 2=weak, 1=significant error.
Return ONLY valid JSON.
"""
)


# ── Node ──────────────────────────────────────────────────────────────────────

def muqayyim_node(state: dict) -> dict:
    """
    LangGraph node — Al-Muqayyim poem quality critic.

    Reads:
      state["composition_context"]  fields: input_poem, occasion, genre
    Writes:
      state["critique_scores"]       {meter, rhyme, authenticity, occasion, overall}
      state["critique_annotations"]  per-line annotation list
      state["final_output"]          formatted critique markdown for the UI
    """
    ctx = state.get("composition_context") or {}
    input_poem: str = ctx.get("input_poem") or ""
    occasion:   str = ctx.get("occasion") or ctx.get("theme") or ""
    genre:      str = ctx.get("genre") or ""

    if not input_poem.strip():
        msg = "لم يُقدَّم نص القصيدة. (No poem text provided for critique.)"
        return {
            **state,
            "critique_scores":      {},
            "critique_annotations": [],
            "final_output":         msg,
            "guardrail_passed":     False,
            "guardrail_flags":      ["empty_poem"],
            "agent_trace": trace_append(
                state, stage="M4", icon="⚖️", label="Al-Muqayyim",
                summary="No poem provided",
            ),
        }

    prompt = (
        f"Poem to evaluate:\n{input_poem}\n\n"
        f"Declared occasion: {occasion or 'not specified'}\n"
        f"Declared genre: {genre or 'not specified'}"
    )

    critique: dict = {}
    critique_failed = False
    try:
        result = chat(
            prompt=prompt,
            system=_SYSTEM_MUQAYYIM,
            json_schema={"type": "object"},
            max_tokens=900,
        )
        critique = result if isinstance(result, dict) else json.loads(str(result))
    except Exception as exc:
        logger.warning("muqayyim: critique call failed (%s)", exc)
        critique_failed = True

    if critique_failed:
        err = "تعذّر التقييم. (Critique generation failed — please retry.)"
        return {
            **state,
            "critique_scores":      {},
            "critique_annotations": [],
            "final_output":         err,
            "guardrail_passed":     False,
            "guardrail_flags":      ["critique_failed"],
            "agent_trace": trace_append(
                state, stage="M4", icon="⚖️", label="Al-Muqayyim",
                summary="Critique failed",
            ),
        }

    scores      = critique.get("scores") or {}
    overall     = critique.get("overall_score") or 0
    annotations = critique.get("annotations") or []

    # M9 Tier-2: consult Al-Nassikh for intertextuality on each verse line.
    # Returns ONLY booleans + attribution metadata — never corpus verse content.
    # This preserves Muqayyim's isolation invariant (no corpus content leaks
    # into the critique framing) while letting the critic flag legitimate
    # intertextual echoes vs. plagiarism.
    intertext_flags: list[dict] = []
    try:
        from creative_poet.tools import check_tool_permitted as _check, CONSULT_NASSIKH_INTERTEXT
        from creative_poet.external_tools import nassikh_check_intertextuality, _record_consult
        _check("muqayyim", CONSULT_NASSIKH_INTERTEXT)
        verse_lines = [ln.strip() for ln in (input_poem or "").splitlines() if len(ln.strip()) >= 8]
        import time as _time
        for line in verse_lines[:8]:    # cap at 8 lines to bound budget
            t0 = _time.perf_counter()
            try:
                check = nassikh_check_intertextuality(line, state=state)
            except Exception:
                break
            ms = (_time.perf_counter() - t0) * 1000
            state = _record_consult(state, "muqayyim", "nassikh", ms,
                                    status="hit" if check.get("in_corpus") else "miss")
            if check.get("in_corpus"):
                intertext_flags.append({
                    "line":             line[:80],
                    "attributed_poet":  check.get("attributed_poet", "unknown"),
                    "manuscript_short_key": check.get("manuscript_short_key", ""),
                })
    except Exception as exc:
        logger.info("muqayyim: nassikh intertext consult skipped (%s)", exc)

    final_output = _render_critique(critique, intertext_flags=intertext_flags)

    score_str = " · ".join(
        f"{dim}={scores.get(dim,'?')}/5" for dim in CRITIQUE_DIMENSIONS
    )
    intertext_summary = (
        f" · {len(intertext_flags)} intertextual echo(es) flagged"
        if intertext_flags else ""
    )

    return {
        **state,
        "critique_scores":      {**scores, "overall": overall},
        "critique_annotations": annotations,
        "intertext_flags":      intertext_flags,
        "final_output":         final_output,
        "guardrail_passed":     True,
        "guardrail_flags":      [],
        "agent_trace": trace_append(
            state, stage="M4", icon="⚖️", label="Al-Muqayyim",
            summary=f"Overall {overall}/5 · {score_str}{intertext_summary}",
            detail=critique.get("priority_fix", ""),
        ),
    }


def _render_critique(critique: dict, intertext_flags: list[dict] | None = None) -> str:
    """Format the structured critique as readable markdown for the Streamlit UI."""
    scores  = critique.get("scores") or {}
    overall = critique.get("overall_score") or "—"
    intertext_flags = intertext_flags or []

    lines = [
        "## تقييم القصيدة — Poem Critique by Al-Muqayyim (المقيّم)",
        "",
        f"**Overall score: {overall}/5**",
        "",
        "| Dimension | Score |",
        "|---|:---:|",
    ]
    for dim in CRITIQUE_DIMENSIONS:
        lines.append(f"| {dim.title()} | {scores.get(dim,'—')}/5 |")

    lines += [
        "",
        f"**Summary (AR):** {critique.get('summary_ar','—')}",
        f"**Summary (EN):** {critique.get('summary_en','—')}",
        "",
        f"**Priority fix:** {critique.get('priority_fix','—')}",
        "",
        "### Line Annotations",
        "",
    ]

    _VERDICT_ICON = {"strong": "✅", "weak": "⚠️", "error": "❌"}
    for ann in critique.get("annotations") or []:
        icon = _VERDICT_ICON.get(ann.get("verdict", ""), "•")
        lines.append(f"{icon} **`{ann.get('line','—')}`**  _{ann.get('dimension','')}_")
        lines.append(f"   {ann.get('annotation','')}")
        if ann.get("suggestion"):
            lines.append(f"   *Suggestion: {ann['suggestion']}*")
        lines.append("")

    if critique.get("strongest_verse"):
        lines.append(f"**✨ Strongest line:** {critique['strongest_verse']}")
    if critique.get("weakest_verse"):
        lines.append(f"**🔧 Line to fix first:** {critique['weakest_verse']}")

    if intertext_flags:
        lines.extend(["", "### 🔗 Intertextual Echoes (via Al-Nassikh)",
                       "_Lines below echo verbatim or near-verbatim text in the indexed corpus._", ""])
        for f in intertext_flags:
            poet = f.get("attributed_poet") or "unknown"
            ms_key = f.get("manuscript_short_key") or ""
            ms_clause = f" · {ms_key}" if ms_key else ""
            lines.append(f"- `{f['line']}` → attributed to **{poet}**{ms_clause}")

    return "\n".join(lines)
