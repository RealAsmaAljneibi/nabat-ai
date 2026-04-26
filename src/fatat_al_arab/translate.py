"""
src/fatat_al_arab/translate.py
==============================
Why this file exists: §2.4 Stage 1 (Bilingual Analyzer) and Stage 2 (HyDE +
Bilingual Expand) both need EN↔AR translation, but they must not import any LLM
SDK directly (§0 guiding decision 1). This module is the single translation entry
point: it accepts text in either language and returns both {ar, en} regardless of
the input language, reusing the same exponential-backoff logic inside llm.py.

Architecture refs: §2.4 Stage 1 (lang ID), §2.4 Stage 2 (bilingual expand),
§5 (Agent 1 translation budget: 0 retries, 3 s timeout — if this call fails the
raw query is passed to Agent 2 untranslated, see fallback behaviour below).
"""

from __future__ import annotations

import logging
from typing import Optional

from .llm import chat

logger = logging.getLogger(__name__)

# ── Translation prompt templates ──────────────────────────────────────────────
# Why explicit prompts per direction: a single bidirectional prompt causes the
# model to sometimes output the same language as the input. Separate templates
# keep the instruction unambiguous.

_SYSTEM_AR2EN = (
    "You are a scholarly translator specialising in classical and Gulf-dialect "
    "Arabic (Nabati poetry). Translate the user's Arabic input into fluent English. "
    "Preserve poetic register and Khaleeji cultural terms where English has no "
    "equivalent — transliterate rather than approximate. Return ONLY the English "
    "translation, no explanation."
)

_SYSTEM_EN2AR = (
    "أنت مترجم أكاديمي متخصص في الشعر النبطي والعربي الخليجي. "
    "ترجم النص الإنجليزي إلى عربية فصحى مع مراعاة المصطلحات الشعرية. "
    "أعد النص العربي فقط بدون أي شرح."
)


def _translate_ar_to_en(text: str, max_tokens: int = 256) -> str:
    """Why separate function: keeps the direction-specific system prompt in one place."""
    return chat(
        prompt=text,
        system=_SYSTEM_AR2EN,
        max_tokens=max_tokens,
    )


def _translate_en_to_ar(text: str, max_tokens: int = 256) -> str:
    """Why separate function: keeps the direction-specific system prompt in one place."""
    return chat(
        prompt=text,
        system=_SYSTEM_EN2AR,
        max_tokens=max_tokens,
    )


def translate(
    text: str,
    source_lang: str,
    max_tokens: int = 256,
) -> dict[str, str]:
    """
    Translate *text* from *source_lang* ("ar" | "en") to the other language.

    Why return {ar, en} always: every caller in the pipeline needs both sides
    (the Bilingual Analyzer stores both in QueryContext, the Bilingual Expander
    generates variants in both, the HyDE node embeds the Arabic version). Always
    returning the pair avoids callers having to remember which direction they called.

    Fallback (§5 budget): if the LLM call fails, the untranslated text is used for
    both sides and a warning is logged. This matches the §5 table entry:
    "Agent 1 translation: 0 retries, 3 s timeout — fallback: raw query passed to Agent 2".

    Args:
        text:        Input text in source_lang.
        source_lang: "ar" or "en".
        max_tokens:  Token budget for the translation.

    Returns:
        {"ar": <arabic text>, "en": <english text>}
    """
    if source_lang == "ar":
        try:
            en = _translate_ar_to_en(text, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("translate: AR→EN failed, using raw query as fallback: %s", exc)
            en = text  # §5 fallback: raw text on both sides
        return {"ar": text, "en": en}

    elif source_lang == "en":
        try:
            ar = _translate_en_to_ar(text, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("translate: EN→AR failed, using raw query as fallback: %s", exc)
            ar = text  # §5 fallback
        return {"ar": ar, "en": text}

    else:
        logger.warning(
            "translate: unknown source_lang '%s', returning text on both sides.", source_lang
        )
        return {"ar": text, "en": text}


def translate_query(query: str, source_lang: str) -> dict[str, str]:
    """
    Convenience alias used by agent1_query_understanding/tools.py registry.
    Named to match the §2.9 tool registry name 'translate_query'.
    """
    return translate(query, source_lang=source_lang)
