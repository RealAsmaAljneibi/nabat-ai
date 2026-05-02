"""
agent2_retrieval_synthesis/nodes/deterministic_answer.py
=========================================================
Why this file exists: When the intent_router (Agent 1, Stage 0.5) marks
`query_context.answer_source == "registry_lookup"`, the orchestrator calls
THIS node instead of the usual Stage 4-10 retrieval → synthesis path.
The node formats a bilingual answer from corpus_stats (which reads the
ground-truth JSON files directly) and emits a citation block that points
to the registry file rather than a verse anchor.

Why live in agent2 and not agent1: §2.9 of the architecture doc says the
final_response + formatted_response fields belong to Agent 2. Keeping that
contract means we don't have to invent a second output schema — the UI
renders a deterministic answer the same way it renders a RAG answer.

Architecture ref: doc/IMPLEMENTATION_PLAN.md M2c. §2.9 guardrails:
mandatory citation is satisfied because every number cites its source
file + field; §2.9 "scoped refusal" is satisfied because an intent we
cannot resolve still falls through to the normal RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from ...state import AgentState
# al_nassikh is a sibling package under src/ — use an absolute import to
# avoid crossing the fatat_al_arab package boundary with relative dots.
from al_nassikh import corpus_stats

logger = logging.getLogger(__name__)


# ── Bilingual answer templates ────────────────────────────────────────────────
# Why templates rather than free-form LLM text: a counting answer is fully
# determined by the number. An LLM would either (a) make up a different
# number (hallucination — the bug we are fixing) or (b) add waffle the user
# doesn't want. Templates keep it exact and auditable.

_AR = {
    "count_poems":
        "تحتوي المجموعة على **{bayts:,}** بيتاً مفهرساً في {mss_with_toc} مخطوطة "
        "(من أصل {mss_total} مخطوطة). "
        "منها **{toc_poems:,}** بيت مطلع من قصائد الفهرس، "
        "و**{full_poems:,}** قصيدة مكتملة التفريغ بيتاً بيتاً.",
    "count_poets":
        "عدد الشعراء المميزين في المجموعة: **{poets:,}** شاعراً.",
    "list_poets":
        "تضم المجموعة **{poets:,}** شاعراً مميزاً. من أكثر الشعراء وروداً في الفهرس:\n"
        "{top_poets_ar}",
    "count_child_grief_poems":
        "وجدت **{child_grief_count}** {child_grief_poem_word_ar} محتملة عن رثاء أو حزن على ابن/بنت/طفل. "
        "{child_grief_poets_ar}هذا عدّ موضوعي محافظ مبني على كلمات دالة في المتن/المناسبة "
        "مع وسم الرثاء الفضي، وليس تصنيفاً بشرياً نهائياً.",
    "count_manuscripts":
        "يحتوي السجل الرسمي على **{mss_total}** مخطوطة. "
        "منها **{mss_with_toc}** مخطوطة تحتوي على فهرس (جدول محتويات) قابل للقراءة، "
        "وتم استخراج بياناتها إلى طبقة الاسترجاع.",
    "count_pages":
        "تم فهرسة **{pages:,}** صفحة مصدرية عبر جميع المخطوطات المتاحة.",
    "corpus_overview":
        "ملخص المجموعة:\n"
        "- الأبيات المفهرسة: **{bayts:,}** بيتاً ({toc_poems:,} مطلع من فهرس + {full_poems} قصيدة كاملة)\n"
        "- الشعراء المميزون: **{poets:,}**\n"
        "- المخطوطات الكاملة: **{mss_total}** (منها {mss_with_toc} لها فهارس قابلة للقراءة)\n"
        "- الصفحات المصدرية: **{pages:,}**\n"
        "- النطاق الزمني: **{oldest}-{newest} م** (حوالي {span} سنة)",
    "age":
        "أقدم المخطوطات في المجموعة تعود إلى حوالي **{oldest} م**، "
        "وأحدثها إلى حوالي **{newest} م**، أي نطاق زمني يمتد **{span}** سنة تقريباً. "
        "({coverage})",
    "provenance":
        "المخطوطات في هذه المجموعة جُمعت من المناطق التالية:\n{regions_list}\n\n"
        "أبرز الجامعين/النسّاخ:\n{collectors_list}",
}

_EN = {
    "count_poems":
        "The corpus contains **{bayts:,}** indexed bayts (verse couplets) across {mss_with_toc} "
        "manuscripts (out of {mss_total} canonical manuscripts). "
        "Of these, **{toc_poems:,}** are opening bayts (matla) from TOC-listed poems, "
        "and **{full_poems}** poems are fully transcribed bayt-by-bayt.",
    "count_poets":
        "There are **{poets:,}** distinct poets represented in the corpus.",
    "list_poets":
        "The corpus contains **{poets:,}** distinct poets. The most frequently indexed poets include:\n"
        "{top_poets_en}",
    "count_child_grief_poems":
        "I found **{child_grief_count}** plausible poems about grieving a son/daughter/child. "
        "{child_grief_poets_en}This is a conservative topic count based on child-related terms plus grief/elegy signals, "
        "not a final human-labelled theme.",
    "count_manuscripts":
        "The canonical registry holds **{mss_total}** manuscripts. "
        "**{mss_with_toc}** of them have a legible table of contents and were "
        "merged into the retrieval layer.",
    "count_pages":
        "**{pages:,}** distinct source pages have been indexed across all available manuscripts.",
    "corpus_overview":
        "Corpus summary:\n"
        "- Bayts indexed: **{bayts:,}** ({toc_poems:,} matla from TOC + {full_poems} fully transcribed poems)\n"
        "- Distinct poets: **{poets:,}**\n"
        "- Total manuscripts: **{mss_total}** ({mss_with_toc} with readable TOC)\n"
        "- Source pages indexed: **{pages:,}**\n"
        "- Time span: **c. {oldest}-{newest} CE** (~{span} years)",
    "age":
        "The oldest manuscripts date to approximately **{oldest} CE**, the newest to "
        "approximately **{newest} CE** — a span of roughly **{span}** years. ({coverage})",
    "provenance":
        "The manuscripts in this corpus come from the following regions:\n{regions_list}\n\n"
        "Notable collectors and scribes:\n{collectors_list}",
}


# ── Answer builder ────────────────────────────────────────────────────────────

def _build_answer(intent: str, lang: str, s: dict) -> str:
    """Render the template for (intent, lang) against the corpus_summary dict."""
    tmpl_bank = _AR if lang == "ar" else _EN
    tmpl = tmpl_bank.get(intent)
    if tmpl is None:
        # Safety: unknown intent should never land here (router gates it),
        # but if it does, fall back to the overview.
        tmpl = tmpl_bank["corpus_overview"]

    regions_list = _bullet_list(s.get("regions") or [], lang)
    collectors_list = _bullet_list(s.get("collectors") or [], lang)
    unknown_labels = {"unknown", "مجهول"}
    top_poets = [
        (name, count)
        for name, count in corpus_stats.poet_counts_top_n(20)
        if name.strip().lower() not in unknown_labels
    ][:12]
    top_poets_ar = _poet_bullet_list(top_poets, "ar")
    top_poets_en = _poet_bullet_list(top_poets, "en")
    child_grief_count = corpus_stats.count_child_grief_poems()
    child_grief_poets = corpus_stats.sample_poets_child_grief(4)
    child_grief_poem_word_ar = "قصيدة" if child_grief_count == 1 else "قصائد"
    child_grief_poets_ar = (
        "من الشعراء الظاهرين في هذه المطابقة: " + "، ".join(child_grief_poets) + ". "
        if child_grief_poets else ""
    )
    child_grief_poets_en = (
        "Poets appearing in these matches include: " + ", ".join(child_grief_poets) + ". "
        if child_grief_poets else ""
    )

    return tmpl.format(
        bayts         = s.get("bayts", s.get("poems", 0)),   # new key; fall back to old for compat
        toc_poems     = s.get("toc_poems", 0),
        full_poems    = s.get("full_poems_transcribed", 0),
        poets         = s.get("poets", 0),
        mss_total     = s.get("manuscripts", 0),
        mss_with_toc  = s.get("manuscripts_with_toc", 0),
        pages         = s.get("pages_indexed", 0),
        oldest        = s.get("oldest_start", "?"),
        newest        = s.get("newest_end", "?"),
        span          = s.get("span_years", "?"),
        coverage      = _coverage_note(s, lang),
        regions_list  = regions_list,
        collectors_list = collectors_list,
        top_poets_ar  = top_poets_ar,
        top_poets_en  = top_poets_en,
        child_grief_count = child_grief_count,
        child_grief_poem_word_ar = child_grief_poem_word_ar,
        child_grief_poets_ar = child_grief_poets_ar,
        child_grief_poets_en = child_grief_poets_en,
    )


def _bullet_list(items: list[str], lang: str) -> str:
    """Render a bullet list — Arabic uses '• ' for RTL compatibility."""
    if not items:
        return "—"
    bullet = "• " if lang == "ar" else "- "
    return "\n".join(f"{bullet}{x}" for x in items)


def _poet_bullet_list(items: list[tuple[str, int]], lang: str) -> str:
    """Render top poet counts for a corpus-level list-poets answer."""
    if not items:
        return "—"
    bullet = "• " if lang == "ar" else "- "
    noun = "بيتاً" if lang == "ar" else "bayts"
    return "\n".join(f"{bullet}{name} — {count} {noun}" for name, count in items)


def _coverage_note(s: dict, lang: str) -> str:
    """Return the corpus_stats coverage note, falling back gracefully."""
    age = corpus_stats.manuscript_age_range()
    note = age.get("coverage")
    if note:
        return note
    return "coverage: registry" if lang == "en" else "التغطية: السجل"


# ── Citation block ────────────────────────────────────────────────────────────
# Why a structured citation: §2.9 (Capability Surface) requires every factual
# sentence to resolve to a citation. A registry-backed answer cites the JSON
# file and the specific field, not a verse anchor.

def _citation_block(intent: str, lang: str) -> list[dict]:
    """
    Return a list of citation dicts in the same schema the Multi-Variant
    formatter (format_variants) expects, so the Streamlit renderer can show
    the source without a special case.
    """
    field_map = {
        "count_poems":         ("anchor_registry_phase4.json", "len(entries)"),
        "count_poets":         ("anchor_registry_phase4.json", "distinct(normalised_poet)"),
        "list_poets":          ("anchor_registry_phase4.json", "top distinct(normalised_poet) by poem count"),
        "count_child_grief_poems": ("anchor_registry_phase4_enriched.json", "child/grief term match + genre=رثاء"),
        "count_manuscripts":   ("manuscript_registry.json",     "len(entries)"),
        "count_pages":         ("anchor_registry_phase4.json", "distinct(source_image_path)"),
        "corpus_overview":     ("manuscript_registry.json + anchor_registry_phase4.json", "aggregate"),
        "age":                 ("manuscript_registry.json",     "min(circa_date_start), max(circa_date_end)"),
        "provenance":          ("manuscript_registry.json",     "region_of_origin, collector"),
    }
    source, field = field_map.get(intent, ("manuscript_registry.json", "—"))
    label = "المصدر" if lang == "ar" else "Source"
    return [{
        "kind":   "registry_lookup",
        "label":  label,
        "source_file": source,
        "field":  field,
        "note":   (
            "Deterministic answer — computed directly from the ground-truth "
            "JSON, no LLM or retrieval used. This is why the count is exact."
            if lang == "en" else
            "إجابة حتمية — محسوبة مباشرة من ملفات البيانات المرجعية، "
            "دون استخدام نموذج لغوي أو استرجاع دلالي. لهذا السبب الرقم دقيق."
        ),
    }]


# ── Static explanation strings (instructor_debug subintents) ─────────────────
# Why static: these are architecture facts that should not be hallucinated by
# an LLM. Versioned strings here are auditable and testable.

_EXPLAIN = {
    "explain_rrf": """**Reciprocal Rank Fusion (RRF)** — `rrf.py:fuse(k=60)`

Score formula: `RRF(d) = Σ_r 1 / (k + rank_r(d))` where k=60.
Each retriever (BM25, Dense, ColBERT) ranks its top results independently.
RRF fuses them by summing reciprocal ranks — a doc ranked 1st by every
retriever gets `3×(1/61)≈0.049`; a doc ranked 5th by one retriever gets
`1/65≈0.015`. Ties broken by the dense retriever score.

Why k=60: Cormack & Buettcher (2009) showed k=60 is robust across many
retrieval settings. Lower k amplifies top-1 differences; higher k smooths them.
""",
    "explain_fallback": """**Mistral-7B fallback** — `llm.py:FALLBACK_THRESHOLD=2`

After `FALLBACK_THRESHOLD=2` consecutive Qwen2.5-7B failures, `llm.py`
switches to `mistralai/Mistral-7B-Instruct-v0.3` for all subsequent calls.
The counter resets on the next successful primary-model response.

Both models are served via Together.ai / Groq (same API key). If Together
is unavailable, `LLM_PROVIDER=stub` activates the offline canned responses.
""",
    "explain_triple_hybrid": """**Triple Hybrid Retrieval** — `retrieve.py`, 800 ms budget

Three retrievers run in parallel daemon threads:
  - **BM25** (`retrievers/bm25.py`): keyword match on normalised Arabic tokens
  - **Dense** (`retrievers/dense.py`): AraBERT 768-d cosine similarity
  - **ColBERT** (`retrievers/colbert.py`): token-level late interaction (stub)

Hard filters (genre, poet, manuscript) applied to the Qdrant payload first.
Any retriever that exceeds 800 ms is dropped and logged in
`state["retrieval_dropped_retrievers"]`. RRF fuses the surviving lists.
""",
    "explain_refusal_precision": """**Refusal and guardrail logic** — `guardrails.py`, `crag_grader.py`

A result is refused when:
  - CRAG grades all top-5 passages "Incorrect" (no relevant result found), OR
  - `should_refuse()` detects an out-of-corpus topic (non-Nabati poetry), OR
  - Agent 1 `needs_clarification=True` (ambiguous single-word query).

Refusal is bilingual (AR + EN) and never raises an exception — the UI always
receives a displayable `final_response`.
""",
    "explain_msa_to_khaleeji_swap": """**MSA → Khaleeji text swap** — `resolve_heritage.py` Stage 6

Retrieved chunks carry both `text_khaleeji` (dialectal) and
`text_msa_summary` (MSA) in the Qdrant payload. Stage 6 replaces MSA
summaries with the Khaleeji original so the synthesiser generates answers
grounded in the manuscript's actual language register rather than a
normalised MSA paraphrase.
""",
    "explain_self_query_schema": """**Self-Query filter schema** — `self_query.py:41–85`

Extracted fields (confidence ≥ 0.7 → hard filter; < 0.7 → soft boost):
  `poet`, `manuscript` (resolved via rapidfuzz), `genre`, `emotions_any`,
  `page_min`, `page_max`, `verse_min`, `verse_max`, `theme`

Hard filters become Qdrant payload predicates (exact match or range).
Soft filters adjust the RRF score of matching chunks by ×1.2.
""",
    "explain_a_vs_b_token_cost": """**Pure-Agentic vs Hybrid token cost** — §2.8 architecture doc

Pure-Agentic path (full Stages 1-10): ~2,800–4,200 tokens per turn
(bilingual_analyzer + HyDE + self_query + CRAG + synthesise + reflect).

Hybrid path (registry short-circuit, Stages 0.5–0.5b only): ~150–400 tokens.
Registry answers (counting, capabilities, debug) cost ~10× less than RAG.

The semantic router saves the full RAG cost (~3–5 s and ~3,000 tokens)
on every non-poetic query by routing it before bilingual_analyzer runs.
""",
    "verify_no_llm_transcription": """**Audio transcription is local, not LLM** — `audio_input.py`, `al_nassikh/operator/*`

Voice transcription uses Whisper-small (local checkpoint or HF fallback).
It never touches Together.ai / Groq. The transcribed text is passed to the
semantic router as a plain string — the router then makes one cloud call
to classify it, but the transcription itself is deterministic and offline.

Operator pipeline (`triage.py`, `bleed_suppress.py`, `standardise.py`) is
100% deterministic — no LLM is involved in any image pre-processing step.
""",
}

# ── Capabilities blurb ───────────────────────────────────────────────────────

_CAPABILITIES_AR = """**NABAT-AI** مساعد بحثي ثنائي اللغة للشعر النبطي الخليجي.

المجموعة: **{bayts:,}** بيتاً مفهرساً في **{mss_with_toc}** مخطوطة من أصل **{mss_total}** مخطوطة، تعود إلى حوالي **{oldest}–{newest} م**.

أخدم أربعة جمهور:
• **المؤسسات الثقافية** — تتبع المصدر، والناسخ، والبحث على مستوى المخطوطة.
• **الشعراء والباحثون** — بحث دلالي عبر الأبيات، وحل اللهجات، واقتباسات موثقة.
• **الطلاب والمعلمون** — استعراض حسب النوع والعاطفة والحقبة، مع النطق الخليجي.
• **عامة الجمهور** — إجابات بلغة بسيطة عن الأرشيف.

**الأبعاد القابلة للتصفية:** الشاعر · المخطوطة · الصفحة · المنطقة · الجامع · النوع · العاطفة · القرن
"""

_CAPABILITIES_EN = """**NABAT-AI** is a bilingual research assistant for the Khaleeji Nabati poetry corpus.

Corpus: **{bayts:,}** indexed bayts across **{mss_with_toc}** manuscripts (of {mss_total} canonical), c. **{oldest}–{newest} CE**.

I serve four audiences:
- **Cultural Institutions** — provenance, collector, manuscript-level lookups.
- **Poets & Researchers** — semantic search across bayts, dialect resolution, CRAG-graded citations.
- **Students & Educators** — guided tours by genre, emotion, era, plus al-Mantuq pronunciation.
- **General Public** — plain-language answers about the archive.

**Filterable dimensions:** poet · manuscript · page · region · collector · genre · emotion · century

Try: "How many bayts by Al-Hazani?" (registry) · "Bayts of longing in Najdi" (semantic) · "What was your CRAG verdict?" (instructor)
"""


def _answer_capabilities(state: AgentState) -> AgentState:
    """Return the bilingual capabilities blurb."""
    s = corpus_stats.corpus_summary()
    kwargs = {
        "bayts":       s.get("bayts", s.get("poems", 0)),
        "mss_with_toc": s.get("manuscripts_with_toc", 0),
        "mss_total":   s.get("manuscripts", 0),
        "oldest":      s.get("oldest_start", "?"),
        "newest":      s.get("newest_end", "?"),
    }
    ar_text = _CAPABILITIES_AR.format(**kwargs)
    en_text = _CAPABILITIES_EN.format(**kwargs)
    combined = f"{ar_text}\n\n---\n\n{en_text}"

    state["final_response"]   = combined
    state["formatted_response"] = {
        "al_maktub": combined, "orthographic": combined, "al_mantuq": "", "citations": [],
    }
    state["citations_used"]   = []
    state["passage_ids_used"] = []
    state["is_refusal"]       = False
    state["crag_verdict"]     = "Correct"
    state["self_rag_verdict"] = "pass"
    state["guardrail_passed"] = True
    state["guardrail_flags"]  = ["capabilities"]
    return state


# ── Instructor debug answer ───────────────────────────────────────────────────

def _answer_instructor_debug(state: AgentState) -> AgentState:
    """
    Render a pipeline-introspection card.
    subintent dispatches to either a snapshot render or a static explanation.
    """
    qc = state.get("query_context") or {}
    subintent = qc.get("intent_subintent") or "snapshot"
    lang = qc.get("query_lang", "en")

    # Static explanation subintents
    explanation = _EXPLAIN.get(subintent)
    if explanation:
        combined = (
            f"**تفسير تقني | Technical explanation: `{subintent}`**\n\n{explanation}"
        )
        state["final_response"]   = combined
        state["formatted_response"] = {
            "al_maktub": combined, "orthographic": combined, "al_mantuq": "", "citations": [],
        }
        state["citations_used"]   = []
        state["passage_ids_used"] = []
        state["is_refusal"]       = False
        state["crag_verdict"]     = "Correct"
        state["self_rag_verdict"] = "pass"
        state["guardrail_passed"] = True
        state["guardrail_flags"]  = ["instructor_debug"]
        return state

    # Snapshot render — reads from the prior-turn debug_snapshot
    snap = state.get("debug_snapshot") or {}
    if not snap:
        msg = (
            "لا توجد لقطة من الدورة السابقة بعد. يُرجى طرح سؤال أولاً ثم الاستفسار عن النتائج.\n\n"
            "No snapshot from a prior turn is available yet. "
            "Please ask a poetry question first, then inspect the results."
        )
        state["final_response"]   = msg
        state["formatted_response"] = {
            "al_maktub": msg, "orthographic": msg, "al_mantuq": "", "citations": [],
        }
        state["is_refusal"]       = False
        state["crag_verdict"]     = "Correct"
        state["self_rag_verdict"] = "pass"
        state["guardrail_passed"] = True
        state["guardrail_flags"]  = ["instructor_debug"]
        return state

    # Render snapshot as a Markdown debug card
    timings   = snap.get("stage_timings") or {}
    crag_v    = snap.get("crag_verdict", "—")
    crag_g    = snap.get("crag_grades") or []
    self_rag  = snap.get("self_rag_scores") or {}
    self_v    = snap.get("self_rag_verdict", "—")
    retr_t    = (snap.get("retriever_timings") or timings)
    dropped   = snap.get("retrieval_dropped_retrievers") or []

    grades_md = ""
    for g in crag_g[:5]:
        label = g.get("label", "?")
        conf  = g.get("confidence", 0.0)
        chunk = g.get("chunk_id", "")
        icon  = {"Correct": "🟢", "Ambiguous": "🟡", "Incorrect": "🔴"}.get(label, "⚪")
        grades_md += f"  - {icon} `{label}` (conf={conf:.2f}) — `{chunk}`\n"

    card = f"""**تقرير الدورة الأخيرة | Last-turn pipeline report**

| المرحلة / Stage | الزمن / Time |
|---|---|
| agent1_ms | {timings.get("agent1_ms", "—")} ms |
| agent2_ms | {timings.get("agent2_ms", "—")} ms |

**CRAG verdict:** {crag_v}
{grades_md}
**Self-RAG verdict:** {self_v}
- faithfulness: {self_rag.get("faithfulness", "—")}
- relevance: {self_rag.get("relevance", "—")}
- completeness: {self_rag.get("completeness", "—")}

**Dropped retrievers:** {", ".join(dropped) if dropped else "none"}
"""

    state["final_response"]   = card
    state["formatted_response"] = {
        "al_maktub": card, "orthographic": card, "al_mantuq": "", "citations": [],
    }
    state["citations_used"]   = []
    state["passage_ids_used"] = []
    state["is_refusal"]       = False
    state["crag_verdict"]     = "Correct"
    state["self_rag_verdict"] = "pass"
    state["guardrail_passed"] = True
    state["guardrail_flags"]  = ["instructor_debug"]
    return state


# ── Genre-aware count answer (deterministic count + LLM prose) ──────────────
# Why a hybrid path: the unfiltered count_poems template was reading "1,502
# poems" for every counting question, even "how many love poems" — the genre
# qualifier was silently dropped. This node keeps the count itself fully
# deterministic (corpus_stats.count_poems_by_genre, no hallucination risk) but
# delegates the *prose framing* to the LLM with the exact numbers as inputs,
# so the answer reads warmly without inventing data.

_GENRE_DESCRIPTIONS_EN = {
    "غزل":      "love and longing (ghazal)",
    "رثاء":     "elegy / lament for the dead (ritha)",
    "مديح":     "praise / panegyric (madih)",
    "فخر":      "boasting / self-praise (fakhr)",
    "غزو":      "raid and battle narrative (ghazw)",
    "حماسة":    "valour / war exhortation (hamasa)",
    "هجاء":     "satire / invective (hija)",
    "حكمة":     "wisdom / gnomic verse (hikma)",
    "وصف":      "descriptive verse on nature, camels, horses, landscape (wasf)",
    "دينية":    "religious / devotional verse (diniyya)",
}

_SYSTEM_GENRE_COUNT = (
    "You are Al-Nassikh (الناسخ — The Scribe), NABAT-AI's archival metadata agent. "
    "The user asked a counting question that has been resolved deterministically — "
    "the numbers below are ALREADY computed from the ground-truth registry. "
    "Your job is to write a warm, factual two-paragraph answer (Arabic first, then "
    "English) that incorporates the EXACT numbers given. You MUST NOT invent any "
    "number, poet name, or manuscript that is not in the inputs. Keep it under 120 "
    "words per language. Do not editorialise about the corpus's importance."
)


def _answer_count_by_genre(state: AgentState, genre: str) -> AgentState:
    """
    Deterministic count + LLM-prose framing for "how many <genre> poems".

    Why split deterministic-count from LLM-prose: the count itself is exact
    (no hallucination), but readers find templates stiff. Passing the exact
    count, the relative ranking, and a short poet sample to the LLM lets it
    write fluent prose that *cannot* drift on the number.
    """
    count_g = corpus_stats.count_poems_by_genre(genre)
    distribution = corpus_stats.genre_distribution()
    total_classified = sum(v for k, v in distribution.items() if k != "غير_محدد")
    sample_poets = corpus_stats.sample_poets_by_genre(genre, n=3)
    total = corpus_stats.count_poems()
    en_label = _GENRE_DESCRIPTIONS_EN.get(genre, genre)

    if count_g == 0:
        # No poems carry this label — be honest about it instead of refusing.
        ar_text = (
            f"لم يُصنَّف في المجموعة أي قصيدة تحت **{genre}** بعد. "
            f"النظام يستخدم تصنيفًا ذا قاعدة معرفية أولية (heuristic_v1) تغطي "
            f"~{total_classified} قصيدة من أصل {total:,}؛ يبقى الباقي ضمن "
            "**غير_محدد** في انتظار مراجعة بشرية."
        )
        en_text = (
            f"No poems are currently classified as **{genre}** ({en_label}). "
            f"The silver-baseline classifier (heuristic_v1) labelled "
            f"~{total_classified} of the {total:,} indexed poems; the rest "
            "remain in **غير_محدد** pending human review."
        )
        combined = f"{ar_text}\n\n---\n\n{en_text}"
        state["final_response"]    = combined
        state["formatted_response"] = {
            "al_maktub": combined, "orthographic": combined, "al_mantuq": "",
            "citations": _citation_block("count_poems", "en"),
        }
        state["citations_used"]   = ["anchor_registry_phase4_enriched.json"]
        state["passage_ids_used"] = []
        state["is_refusal"]       = False
        state["crag_verdict"]     = "Correct"
        state["self_rag_verdict"] = "pass"
        state["guardrail_passed"] = True
        state["guardrail_flags"]  = ["registry_lookup", f"genre:{genre}"]
        return state

    # Rank within classified genres so the prose can say "the largest" / "third"
    ranked = sorted(
        ((g, n) for g, n in distribution.items() if g != "غير_محدد"),
        key=lambda x: x[1], reverse=True,
    )
    rank_in_classified = next(
        (i + 1 for i, (g, _) in enumerate(ranked) if g == genre), None
    )
    rank_str = f"#{rank_in_classified} of {len(ranked)}" if rank_in_classified else "—"

    ar_text = (
        f"تحتوي المجموعة على **{count_g}** قصيدة مصنَّفة تحت **{genre}** "
        f"(من أصل {total:,} قصيدة مفهرسة). المرتبة بين الأنواع المصنَّفة: {rank_str}."
    )
    if sample_poets:
        ar_text += " من أبرز الشعراء في هذا النوع: " + "، ".join(sample_poets) + "."
    ar_text += " التصنيف هنا heuristic_v1 (خط أساس فضي) وليس حكماً بشرياً نهائياً."

    en_text = (
        f"The corpus carries **{count_g}** poems classified as **{en_label}** "
        f"(out of {total:,} indexed). Rank among classified genres: {rank_str}."
    )
    if sample_poets:
        en_text += " Representative poets for this genre include: " + ", ".join(sample_poets) + "."
    en_text += " Classifier: heuristic_v1 silver baseline, not human-verified."
    llm_text = f"{ar_text}\n\n---\n\n{en_text}"

    state["final_response"]   = llm_text
    state["formatted_response"] = {
        "al_maktub": llm_text, "orthographic": llm_text, "al_mantuq": "",
        "citations": _citation_block("count_poems", "en"),
    }
    state["citations_used"]   = ["anchor_registry_phase4_enriched.json"]
    state["passage_ids_used"] = []
    state["is_refusal"]       = False
    state["crag_verdict"]     = "Correct"
    state["self_rag_verdict"] = "pass"
    state["guardrail_passed"] = True
    state["guardrail_flags"]  = ["registry_lookup", f"genre:{genre}"]

    logger.info(
        "deterministic_answer: count_by_genre genre=%s count=%d sample_poets=%d",
        genre, count_g, len(sample_poets),
    )
    return state


# ── Unsupported dimension answer ─────────────────────────────────────────────

_UNSUPPORTED_DIM_META: dict[str, dict] = {
    "wasm": {
        "en_name": "Tribal-mark (wasm) annotations",
        "ar_name": "علامات الوسم القبلي",
    },
    "marginalia": {
        "en_name": "Margin notes / marginalia",
        "ar_name": "الحواشي والتعليقات الهامشية",
    },
    "secondary_scribe": {
        "en_name": "Secondary-scribe identification",
        "ar_name": "تحديد النسّاخ الثانويين",
    },
    "ink_bleed": {
        "en_name": "Ink-bleed / ghost-ink audit",
        "ar_name": "تدقيق نزيف الحبر",
    },
    "library_stamp": {
        "en_name": "Library-stamp bounding-box coordinates",
        "ar_name": "إحداثيات أختام المكتبة",
    },
    "handwriting_style": {
        "en_name": "Handwriting / calligraphic style (Naskh, Ruqʿah, etc.)",
        "ar_name": "أسلوب الخط (نسخ، رقعة، إلخ)",
    },
}

_INDEXED_DIMS = "poet · manuscript · page · region · collector · genre · emotion"
_INDEXED_DIMS_AR = "الشاعر · المخطوطة · الصفحة · المنطقة · الجامع · النوع · العاطفة"


def _answer_unsupported_dim(state: AgentState, dim: str) -> AgentState:
    """
    Bilingual "not indexed" note + a short LLM-generated educational context.
    The LLM call is a single bounded call (max_tokens=150, no retrieval).
    """
    meta = _UNSUPPORTED_DIM_META.get(dim, {
        "en_name": dim.replace("_", " ").title(),
        "ar_name": dim,
    })
    en_name = meta["en_name"]
    ar_name = meta["ar_name"]

    # Static not-indexed note
    en_note = (
        f"**{en_name}** are not yet indexed in this corpus. "
        f"Currently tracked dimensions: {_INDEXED_DIMS}. "
        "The note below is general knowledge, not corpus-specific."
    )
    ar_note = (
        f"**{ar_name}** غير مفهرس بعد في هذه المجموعة. "
        f"الأبعاد المتاحة حالياً: {_INDEXED_DIMS_AR}. "
        "الملاحظة أدناه معرفة عامة، وليست خاصة بهذا الأرشيف."
    )

    # One LLM call for 2-sentence educational context
    general_context = ""
    try:
        from ...llm import chat as llm_chat
        prompt = (
            f"In exactly two sentences, explain what '{en_name}' means in the "
            f"context of Gulf Nabati poetry manuscript studies. "
            f"Write for a non-specialist. Do not mention this corpus specifically."
        )
        resp = llm_chat(prompt=prompt, max_tokens=150)
        if isinstance(resp, str) and len(resp) > 10:
            general_context = resp.strip()
    except Exception as exc:
        logger.debug("unsupported_dim: LLM context call failed — %s", exc)

    if general_context:
        en_text = f"{en_note}\n\n> {general_context}"
        ar_text = f"{ar_note}\n\n> {general_context}"
    else:
        en_text = en_note
        ar_text = ar_note

    combined = f"{ar_text}\n\n---\n\n{en_text}"

    state["final_response"]   = combined
    state["formatted_response"] = {
        "al_maktub": combined, "orthographic": combined, "al_mantuq": "", "citations": [],
    }
    state["citations_used"]   = []
    state["passage_ids_used"] = []
    state["is_refusal"]       = False
    state["crag_verdict"]     = "Correct"
    state["self_rag_verdict"] = "pass"
    state["guardrail_passed"] = True
    state["guardrail_flags"]  = ["unsupported_dimension"]

    logger.info("deterministic_answer_node: unsupported_dim=%s answered.", dim)
    return state


# ── Node callable ────────────────────────────────────────────────────────────

def deterministic_answer_node(state: AgentState) -> AgentState:
    """
    Short-circuit answer path for non-poetic tracks and counting/metadata questions.

    Dispatch order:
      1. track="capabilities"     → bilingual capabilities blurb
      2. track="instructor_debug" → debug snapshot or static explanation
      3. deterministic_intent.startswith("unsupported_dim_") → educational note
      4. registry stats (existing counting / age / provenance path)

    Preconditions (set by intent_router or semantic_router):
        state["query_context"]["answer_source"] == "registry_lookup"
    """
    qc: dict[str, Any] = state.get("query_context") or {}
    track  = qc.get("track", "registry_lookup")
    intent = qc.get("deterministic_intent") or ""
    lang   = qc.get("query_lang", "en")

    # ── 0. Out-of-scope hard refusal ─────────────────────────────────────────
    # Named non-Nabati entities, sacred texts, non-Arabic epics detected by
    # intent_router before any LLM call. Return the approved refusal template
    # and set is_refusal=True so the evaluator counts it correctly.
    if track == "out_of_scope":
        logger.info("deterministic_answer_node: track=out_of_scope — returning hard refusal.")
        from fatat_al_arab.guardrails import REFUSAL_TEMPLATE_AR, REFUSAL_TEMPLATE_EN
        combined = f"{REFUSAL_TEMPLATE_AR}\n\n{REFUSAL_TEMPLATE_EN}"
        state["final_response"]      = combined
        state["formatted_response"]  = {
            "al_maktub": combined, "orthographic": combined,
            "al_mantuq": "", "citations": [],
        }
        state["citations_used"]      = []
        state["passage_ids_used"]    = []
        state["is_refusal"]          = True
        state["crag_verdict"]        = "Incorrect"
        state["self_rag_verdict"]    = "pass"
        state["guardrail_passed"]    = True
        state["guardrail_flags"]     = ["out_of_scope"]
        return state

    # ── 1. Capabilities ───────────────────────────────────────────────────────
    if track == "capabilities":
        logger.info("deterministic_answer_node: track=capabilities")
        return _answer_capabilities(state)

    # ── 2. Instructor debug ───────────────────────────────────────────────────
    if track == "instructor_debug":
        logger.info("deterministic_answer_node: track=instructor_debug subintent=%s", qc.get("intent_subintent"))
        return _answer_instructor_debug(state)

    # ── 2b. Image-grounded provenance ─────────────────────────────────────────
    # Delegated to the dedicated node — it touches the vector index, which
    # this file deliberately does not, so dispatch keeps the contract clean.
    if track == "image_grounded_provenance" or intent == "image_grounded_provenance":
        logger.info("deterministic_answer_node: track=image_grounded_provenance — delegating to image_provenance_node.")
        from .image_provenance import image_provenance_node
        return image_provenance_node(state)

    # ── 3. Unsupported dimension ──────────────────────────────────────────────
    if intent.startswith("unsupported_dim_"):
        dim = intent[len("unsupported_dim_"):]
        logger.info("deterministic_answer_node: unsupported_dim=%s", dim)
        return _answer_unsupported_dim(state, dim)

    # ── 3b. Genre-aware count (deterministic count + LLM prose) ──────────────
    if intent == "count_poems_by_genre":
        genre = qc.get("intent_genre") or ""
        logger.info("deterministic_answer_node: count_poems_by_genre genre=%s", genre)
        return _answer_count_by_genre(state, genre)

    # ── 4. Registry stats (existing path) ────────────────────────────────────
    if not intent:
        logger.warning("deterministic_answer_node called without deterministic_intent; leaving state unchanged.")
        return state

    s = corpus_stats.corpus_summary()
    ar_text = _build_answer(intent, "ar", s)
    en_text = _build_answer(intent, "en", s)
    combined = f"{ar_text}\n\n---\n\n{en_text}"
    citations = _citation_block(intent, lang)

    state["final_response"]  = combined
    state["formatted_response"] = {
        "al_maktub":    combined,
        "orthographic": combined,
        "al_mantuq":    "",
        "citations":    citations,
    }
    state["citations_used"]  = [c["source_file"] for c in citations]
    state["passage_ids_used"] = []
    state["is_refusal"]      = False
    state["crag_verdict"]    = "Correct"
    state["self_rag_verdict"] = "pass"
    state["guardrail_passed"] = True
    state["guardrail_flags"]  = ["registry_lookup"]

    logger.info(
        "deterministic_answer_node: answered intent=%s (lang=%s) with %d-char response",
        intent, lang, len(combined),
    )
    return state
