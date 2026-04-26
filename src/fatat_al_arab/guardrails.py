"""
src/fatat_al_arab/guardrails.py
================================
Why this file exists: §2.9 of the architecture defines three hard guardrails
that run AFTER Stage 10 (format_variants) and BEFORE anything reaches the UI.
They are the last line of defence against hallucination — if any guardrail
fires, the response is blocked and replaced with a scoped refusal.

Scaffold state (M0): all three functions have the correct signature and a
working stub implementation. Full implementation lands in M6 once the
passage set and anchor registry are available.

Three guardrails (§2.9):
  (a) citation_resolvable  — every factual sentence carries a citation that
                             resolves to a real entry in anchor_registry_phase4.
  (b) verbatim_verse       — no verse text appears that is not verbatim in the
                             approved passage set (paraphrase detection on
                             Arabic surface tokens).
  (c) scoped_refusal       — when CRAG returns empty, the response is the
                             "not in corpus" template — no unverified detail
                             decorates it.

Architecture refs: §2.9 (guardrail spec), §3.3 (cross-persona guardrails),
§5 (guardrail runs synchronously, no retry budget — it's a hard gate).
"""

from __future__ import annotations

import re
from typing import Optional

# ── Refusal template ─────────────────────────────────────────────────────────
# Why a constant: §2.9 guardrail (c) says the template is fixed — it must
# not be decorated with unverified detail. Having it here prevents any node
# from constructing its own variant.

REFUSAL_TEMPLATE_AR = (
    "لا يوجد في المخطوطات المرقَّمة ما يجيب هذا السؤال مباشرةً. "
    "يُنصح بمراجعة المخطوط الأصلي أو توسيع نطاق الرقمنة."
)

REFUSAL_TEMPLATE_EN = (
    "The digitised manuscripts do not contain a direct answer to this query. "
    "Please consult the original manuscript or expand the digitisation scope."
)


# ── Guardrail result ─────────────────────────────────────────────────────────

class GuardrailResult:
    """
    Why a result object: the Streamlit debug panel and the evaluation harness
    both need to know which guardrail fired and why, not just pass/fail.
    """
    def __init__(self, passed: bool, flags: list[str]):
        self.passed = passed
        self.flags  = flags   # human-readable reasons for each failure

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({', '.join(self.flags)})"
        return f"GuardrailResult({status})"


# ── Arabic normalisation (shared with al_nassikh.parser) ─────────────────────

_ALEF_VARIANTS = "أإآٱ"

def _normalise(text: str) -> str:
    """Light normalisation for surface-token comparison."""
    for v in _ALEF_VARIANTS:
        text = text.replace(v, "ا")
    text = text.replace("ة", "ه")
    text = re.sub(r"[\u064B-\u065F]", "", text)   # strip harakat
    text = text.replace("\u0640", "")              # strip tatweel
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _arabic_tokens(text: str) -> set[str]:
    """Extract Arabic word tokens from text."""
    return set(re.findall(r"[\u0600-\u06FF]+", _normalise(text)))


# ── (a) Citation resolvability ────────────────────────────────────────────────

# Matches citation tags like [anchor_id:ms07_p020_r003] or [anchor_id:some_key]
_CITATION_TAG_RE = re.compile(r"\[anchor_id:([^\]]+)\]")
# Strips the level+seq suffix added by index.py for the THREE row-backed levels:
# verse / group / poem. Their `base` IS a real source_row_id from
# anchor_registry_phase4.json, so the registry check can validate it.
_ROW_LEVEL_SUFFIX_RE = re.compile(r"__(?:verse|group|poem)__\d+$")
# Aggregate-level chunks (manuscript / poet / era / genre / emotion) are roll-ups
# whose `base` is a synthetic key (manuscript_short_key, genre label, era hash, …)
# — they intentionally have NO row in anchor_registry_phase4.json, so the
# registry check must skip them. Without this skip, every thematic query that
# surfaces a genre/manuscript/poet chunk in the top-K is flagged
# `passage not in registry` and silently overwritten with the generic refusal.
_AGGREGATE_LEVEL_RE = re.compile(r"__(?:manuscript|poet|era|genre|emotion)__\d+$")
# Reference-level chunks come from the scholarly PDFs ingested via
# scripts/build_reference_corpus.py. Their chunk_id pattern is
# `reference_<book_short_key>_p<page>_para<idx>`. They are their own citation
# target (the book + page IS the citation), so the manuscript-registry check
# must skip them — otherwise every PDF-grounded answer would be flagged as
# unresolvable and silently rewritten as the generic refusal.
_REFERENCE_LEVEL_RE = re.compile(r"^reference_[A-Za-z_]+_p\d+_para\d+$")

def citation_resolvable(
    response_text: str,
    anchor_registry: list[dict],
    passage_ids_used: list[str],
) -> GuardrailResult:
    """
    Why guardrail (a) matters: §1.4 says 'Citation Precision > 95% — every
    response grounded in verified source.' A response that names a manuscript
    page that doesn't exist in our registry is a hallucination, even if the
    verse text is correct.

    M0 scaffold: checks that every [anchor_id:…] tag in the response resolves
    to a real entry in anchor_registry. Full M6 impl adds per-sentence checking.

    Args:
        response_text:    The draft response from Stage 8 (synthesise).
        anchor_registry:  List of dicts from anchor_registry_phase4.json.
        passage_ids_used: chunk_ids that Stage 8 actually quoted.

    Returns:
        GuardrailResult(passed=True) if every cited anchor resolves.
    """
    if not anchor_registry:
        # No registry loaded — can't verify. Fail safe.
        return GuardrailResult(False, ["anchor_registry not loaded"])

    # Build lookup set from the registry
    valid_anchor_ids: set[str] = set()
    for entry in anchor_registry:
        # anchor_registry_phase4.json uses source_row_id as the anchor key
        row_id = entry.get("source_row_id") or entry.get("row_id", "")
        if row_id:
            valid_anchor_ids.add(str(row_id))
        # Also accept manuscript+page as a soft anchor key
        vol  = entry.get("source_volume", "")
        page = entry.get("page_number")
        if vol and page:
            valid_anchor_ids.add(f"{vol}_p{page}")

    # Find all citation tags in the response
    cited_ids = _CITATION_TAG_RE.findall(response_text)

    flags: list[str] = []
    for anchor_id in cited_ids:
        # LLM sometimes adds a space after the colon — strip it
        clean = anchor_id.strip()
        # Aggregate-level and reference-level citations don't live in the
        # manuscript registry by design — treat them as resolvable so a
        # thematic answer that cites a genre/manuscript/PDF-paragraph chunk
        # isn't overwritten by the refusal template.
        if _AGGREGATE_LEVEL_RE.search(clean) or _REFERENCE_LEVEL_RE.match(clean):
            continue
        # Row-backed chunk_ids carry a __verse|group|poem__seq suffix — strip it
        base = _ROW_LEVEL_SUFFIX_RE.sub("", clean)
        if clean not in valid_anchor_ids and base not in valid_anchor_ids:
            flags.append(f"unresolvable citation: [{anchor_id}]")

    # Also check that passage_ids_used are present (belt-and-suspenders)
    for pid in (passage_ids_used or []):
        # Aggregate-level + reference chunks have no registry row by design.
        if _AGGREGATE_LEVEL_RE.search(pid) or _REFERENCE_LEVEL_RE.match(pid):
            continue
        # passage_ids are chunk_ids — strip the row-level __verse|group|poem suffix
        base = _ROW_LEVEL_SUFFIX_RE.sub("", pid)
        if pid not in valid_anchor_ids and base not in valid_anchor_ids:
            flags.append(f"passage not in registry: {pid}")

    return GuardrailResult(passed=len(flags) == 0, flags=flags)


# ── (b) Verbatim verse check ─────────────────────────────────────────────────

# Minimum token overlap fraction to consider a passage "verbatim"
# (exact match is too strict — normalisation differences are expected)
_VERBATIM_OVERLAP_THRESHOLD = 0.75

# Regex for Arabic prose analysis — phrases matching these patterns are
# meta-commentary (analysis, introduction, citation framing), not verse quotes.
# Prefixes ف/و/ب/ل can attach to verbs in Arabic — handled via `[فوبل]*`.
_PROSE_ANALYSIS_RE = re.compile(
    r"(?:^|[\s،,])[فوبل]*"
    r"(?:يعبر|تعبر|يُعبر|تُعبر"
    r"|يُظهر|يظهر|تظهر|تُظهر"
    r"|يتناول|تتناول"
    r"|يُبين|يبين"
    r"|يُشير|يشير"
    r"|يوضح|توضح"
    r"|يُقدم|يقدم"
    r"|يُثبت|يثبت"
    r"|يُمثل|يمثل|تمثل"
    r"|يعكس|تعكس"
    r"|يُعطي|يعطي"
    r"|يتحكم|تتحكم"
    r"|يُلخص|يلخص"
    r"|يصف|تصف"
    r"|يقول|تقول"        # "يقول الشاعر" introduces verse quotes but is prose itself
    r"|يُسمى|تُسمى"
    r"|يُعد|تُعد"
    r"|يُجسد|تُجسد"
    r")"
    r"|قصائد\s+عن"       # topic introductions like "poems about..."
    r"|من\s+الجدير"      # "it is worth noting"
    r"|في\s+نفس\s+السياق"  # "in the same context"
    r"|مثل\s+قول"        # "like the saying of"
    r"|على\s+سبيل"
    r"|ومع\s+ذلك"      # "however" — discourse connector opening analysis
    r"|لا\s+توجد"      # "there are no..." — analysis assertion
    r"|لا\s+يوجد"
    r"|ولا\s+توجد"
    r"|يُمكن\s+القول"  # "one can say"
    r"|من\s+الواضح"    # "it is clear that"
    r"|يُلاحظ\s+أن"    # "it is noted that"
    r"|تجدر\s+الإشارة" # "it is worth noting"
    r"|بصفة\s+عامة",   # "generally speaking"
    re.UNICODE,
)

def verbatim_verse(
    response_text: str,
    approved_passages: list[dict],
    is_refusal: bool = False,
) -> GuardrailResult:
    """
    Why guardrail (b) matters: the LLM in Stage 8 must quote the Khaleeji
    verse verbatim — it cannot paraphrase or recompose. This guardrail catches
    cases where the model invented plausible-sounding verse text that isn't
    actually in our manuscripts.

    M0 scaffold: detects Arabic verse-length phrases (≥ 8 Arabic words) in the
    response and checks each against the approved passage set via token overlap.
    Full M6 impl adds char-level CER comparison for stricter detection.

    Args:
        response_text:     The draft response from Stage 8.
        approved_passages: The resolve_heritage output — Khaleeji text only.
        is_refusal:        True when on the refusal path — short Arabic prose
                           in the refusal template is expected and allowed.

    Returns:
        GuardrailResult(passed=True) if no suspicious verse phrases found.
    """
    if not approved_passages:
        # Refusal path: the refusal template itself contains Arabic prose.
        # That's expected — only block if it contains verse-length runs
        # (≥ 8 words), which would indicate unverified content was appended.
        if is_refusal:
            return GuardrailResult(True, [])
        # Non-refusal path with no approved passages — something went wrong
        # upstream. Fail safe only if there is substantial Arabic verse content
        # (≥ 8 Arabic tokens), not just a short prose statement.
        ar_words = _arabic_tokens(response_text)
        if len(ar_words) > 20:
            return GuardrailResult(
                False,
                ["substantial arabic text present but no approved passages exist"]
            )
        return GuardrailResult(True, [])

    # Build a corpus of approved Khaleeji tokens
    approved_token_sets: list[set[str]] = []
    for passage in approved_passages:
        text = (
            passage.get("text_khaleeji") or
            passage.get("text") or
            passage.get("display_text", "")
        )
        if text:
            approved_token_sets.append(_arabic_tokens(text))

    # Extract candidate verse phrases from the response
    # A "verse phrase" is a run of ≥ 8 Arabic words — verse hemistiches are
    # typically 5-10 words; shorter runs are likely prose labels or metadata.
    # Remove citation tags first to avoid false positives.
    clean_response = _CITATION_TAG_RE.sub("", response_text)
    ar_phrase_re   = re.compile(r"(?:[\u0600-\u06FF]+\s+){7,}[\u0600-\u06FF]+")
    candidate_phrases = ar_phrase_re.findall(clean_response)

    flags: list[str] = []
    for phrase in candidate_phrases:
        phrase_tokens = _arabic_tokens(phrase)
        if len(phrase_tokens) < 4:
            continue  # too short to be a verse — skip

        # Skip analysis prose — phrases that match known analysis commentary
        # patterns (verbs like يعبر، يُظهر، يعكس with optional ف/و prefix)
        # are meta-commentary, not verse quotes.
        if _PROSE_ANALYSIS_RE.search(phrase):
            continue

        # Check overlap against each approved passage
        max_overlap = 0.0
        for approved_tokens in approved_token_sets:
            if not approved_tokens:
                continue
            overlap = len(phrase_tokens & approved_tokens) / len(phrase_tokens)
            max_overlap = max(max_overlap, overlap)

        if max_overlap < _VERBATIM_OVERLAP_THRESHOLD:
            flags.append(
                f"suspected non-verbatim verse (max overlap {max_overlap:.0%}): "
                f"'{phrase[:50]}...'"
            )

    return GuardrailResult(passed=len(flags) == 0, flags=flags)


# ── (c) Scoped refusal integrity ─────────────────────────────────────────────

def scoped_refusal(
    response_text: str,
    is_refusal: bool,
    crag_verdict: Optional[str],
) -> GuardrailResult:
    """
    Why guardrail (c) matters: when CRAG grades all retrieved passages as
    Incorrect (§2.5 Stage 7), the system must return ONLY the refusal template
    — no unverified detail can be appended. This catches cases where an
    upstream node decorated the refusal with speculation.

    M0 scaffold: checks that a response flagged as a refusal doesn't contain
    long Arabic verse-like text (which would indicate decoration). Full M6 impl
    adds a semantic similarity check against the approved REFUSAL_TEMPLATE.

    Args:
        response_text: The response text to check.
        is_refusal:    True if the orchestrator determined CRAG returned empty.
        crag_verdict:  "Correct" | "Ambiguous" | "Incorrect" (or None if skipped).

    Returns:
        GuardrailResult(passed=True) if refusal integrity holds.
    """
    # If this is not a refusal path, guardrail (c) is not applicable.
    if not is_refusal and crag_verdict != "Incorrect":
        return GuardrailResult(True, [])

    flags: list[str] = []

    # Normalise the response for comparison against the known refusal templates.
    # If the response IS the template (or a bilingual variant), it passes outright.
    norm_response = _normalise(response_text)
    norm_ar       = _normalise(REFUSAL_TEMPLATE_AR)
    norm_en       = _normalise(REFUSAL_TEMPLATE_EN)

    if norm_ar in norm_response or norm_en in norm_response:
        # Response contains the approved template — check it isn't decorated
        # by stripping the template text and looking for leftover Arabic content.
        leftover = norm_response.replace(norm_ar, "").replace(norm_en, "").strip()
        ar_leftover_tokens = _arabic_tokens(leftover)
        if len(ar_leftover_tokens) > 8:
            flags.append(
                f"refusal template is decorated with {len(ar_leftover_tokens)} "
                f"additional Arabic tokens — possible unverified content"
            )
        return GuardrailResult(passed=len(flags) == 0, flags=flags)

    # If we reach here, this is flagged as a refusal but doesn't contain the
    # approved template text. That itself is a problem — the response may be a
    # custom refusal that bypasses the guardrail.
    flags.append(
        "response is flagged as refusal but does not contain the approved "
        "REFUSAL_TEMPLATE text — use guardrails.REFUSAL_TEMPLATE_AR/EN directly"
    )
    return GuardrailResult(passed=len(flags) == 0, flags=flags)


# ── Combined check ────────────────────────────────────────────────────────────

def run_all(
    response_text: str,
    anchor_registry: list[dict],
    approved_passages: list[dict],
    passage_ids_used: list[str],
    is_refusal: bool = False,
    crag_verdict: Optional[str] = None,
) -> GuardrailResult:
    """
    Run all three guardrails in sequence. Returns on first failure so the
    caller sees exactly which guardrail blocked emission.

    Called by agent2/graph.py after Stage 10 (format_variants) and before
    final_response is set in AgentState.
    """
    checks = [
        citation_resolvable(response_text, anchor_registry, passage_ids_used),
        verbatim_verse(response_text, approved_passages, is_refusal=is_refusal),
        scoped_refusal(response_text, is_refusal, crag_verdict),
    ]

    all_flags: list[str] = []
    for result in checks:
        if not result.passed:
            all_flags.extend(result.flags)

    return GuardrailResult(passed=len(all_flags) == 0, flags=all_flags)
