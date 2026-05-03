"""
creative_poet/nodes/composition_guardrails.py
===========================================================
Why this file exists: synthetic attribution is the ethical backbone of the creative
pipeline. A verse generated in a deceased poet's manner must NEVER appear without
explicit labeling that distinguishes it from the poet's actual work.

This is enforced structurally, not optionally:
  1. Al-Hafiz calls format_synthetic_badge() and wrap_with_badge() before returning.
  2. The LangGraph graph routes all Al-Hafiz output through enforce_attribution()
     as the final node before END — a non-bypassable gate.
  3. check_composition_attribution() verifies the badge sentinel is present.

In the Gulf cultural context, falsely attributing a verse to a poet — living or
deceased — is a serious ethical violation. The badge makes the chain of production
transparent. It cannot be removed by calling code.

No analogous guardrail exists for Al-Mulhim, Al-Musharik, or Al-Muqayyim because
their outputs are inherently labeled by context (scaffold, candidates, critique).
Only Al-Hafiz generates text that could be confused with an original poem.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Sentinel string that must appear in any Al-Hafiz output.
# check_composition_attribution() searches for this exact substring.
_BADGE_SENTINEL = "NABAT-AI Creative Output"  # unique phrase present in every badge


def format_synthetic_badge(
    agent_name: str,
    poet_name: str,
    anchor_ids: list[str],
) -> str:
    """
    Build the mandatory synthetic attribution badge.
    Always called by Al-Hafiz before returning output; also verified by
    enforce_attribution() as a defense-in-depth check.

    Args:
        agent_name:  e.g. "Al-Hafiz / الحافظ"
        poet_name:   the deceased poet whose style was used
        anchor_ids:  list of anchor_ids from the corpus that influenced the verse
    """
    anchors_str = (
        " · ".join(f"`{a}`" for a in anchor_ids[:5])
        if anchor_ids else "—"
    )
    return (
        f"---\n"
        f"🤖 **NABAT-AI Creative Output — {agent_name}**\n\n"
        f"**⚠️ NOT attributed to {poet_name}.** "
        f"Generated in their documented style using indexed verses.\n\n"
        f"**Source verses:** {anchors_str}\n\n"
        f"**Status:** Synthetic — produced by Al-Hafiz, not by the poet's hand\n"
        f"---"
    )


def wrap_with_badge(output: str, badge: str) -> str:
    """Prepend the badge above the verse text."""
    return f"{badge}\n\n{output}"


def check_composition_attribution(final_output: str) -> bool:
    """
    Return True if the synthetic attribution badge sentinel is present.
    Returns False if the output is empty or the badge is missing.
    """
    if not final_output:
        return False
    return _BADGE_SENTINEL in final_output


def enforce_attribution(state: dict) -> dict:
    """
    LangGraph node — final gate for all Al-Hafiz output paths.
    Blocks the output and substitutes a bilingual error message if the
    badge sentinel is absent. This node cannot be removed from the graph
    without breaking the ethical contract.

    Reads:  state["final_output"]
    Writes: state["final_output"]   (unchanged if badge present; blocked message if absent)
            state["guardrail_passed"]
            state["guardrail_flags"]
    """
    final_output: str = state.get("final_output") or ""

    if check_composition_attribution(final_output):
        logger.debug("composition_guardrails: attribution badge verified — output cleared.")
        return {**state, "guardrail_passed": True}

    logger.error(
        "composition_guardrails: synthetic badge MISSING in Al-Hafiz output — "
        "blocking output before it reaches the UI."
    )
    blocked = (
        "⛔ **تعذّر إصدار النص — شارة الإسناد التركيبي مفقودة.**\n\n"
        "Output blocked: the synthetic attribution badge was not found in the "
        "Al-Hafiz response. This is a required safeguard and cannot be bypassed.\n\n"
        "_If you are a developer: ensure hafiz_node calls format_synthetic_badge() "
        "and wrap_with_badge() before writing to state['final_output']._"
    )
    flags = list(state.get("guardrail_flags") or [])
    flags.append("missing_attribution_badge")

    from fatat_al_arab.state import trace_append
    return {
        **state,
        "final_output":    blocked,
        "guardrail_passed": False,
        "guardrail_flags":  flags,
        "agent_trace": trace_append(
            state, stage="G", icon="⛔", label="Attribution Guard",
            summary="Badge missing — output blocked",
            detail="Al-Hafiz output did not contain the mandatory synthetic attribution badge.",
        ),
    }
