# NABAT-AI — Evaluation Report
## §7 Five-Axis Evaluation

**Generated:** 2026-05-02 20:13 UTC
**LLM provider:** `openai`
**Fixture counts:** 20 in-corpus · 20 OOC · 20 bilingual · 20 extension

---

## Axis 1 — Correctness

### CER Bucket Distribution (Al-Nassikh pipeline quality)

| Bucket | Count | Share |
|---|---|---|
| HIGH — CER < 10% (RAG-safe) | 0 | 0.0% |
| MEDIUM — CER 10-25% | 0 | 0.0% |
| LOW — CER ≥ 25% (gated) | 0 | 0.0% |
| **Total anchors** | **0** | 100% |

> **Why this matters:** anchors gated to LOW-CER are never emitted in synthesis.
> The HIGH bucket is the effective RAG corpus size.

### RAG Retrieval & Citation Quality

| Metric | Result | Target | Status |
|---|---|---|---|
| Citation-resolvability rate | 100.0% | 100% | ✅ |
| Recall@5 exact (gold anchor_id match) | 15.4% | ≥ 75% | ❌ |
| Recall@5 relaxed (correct manuscript) | 46.2% | ≥ 75% | ❌ |
| Refusal precision (OOC set, n=20) | 80.0% | ≥ 90% | ❌ |

**CRAG verdict distribution** (in-corpus queries):

| Verdict | Count |
|---|---|
| Correct | 10 |
| Ambiguous | 9 |
| Incorrect | 1 |

> **Recall@5 exact**: at least one gold anchor_id (exact page+row) appears in the top-20 retrieved chunks.
> **Recall@5 relaxed**: at least one retrieved chunk is from the correct manuscript (same anchor_id prefix).
> The exact metric measures page-level precision; the relaxed metric confirms the retriever found the right manuscript.
> Low exact / high relaxed = the retriever targets the right manuscript but ranks adjacent verses higher than the specific gold page.

---

## Axis 2 — Robustness

| Mechanism | Activation rate | Raw count / 60 queries |
|---|---|---|
| CRAG re-query (Loop A+B) | 10.0% | 6 |
| Self-RAG retry (Loop C) | 10.0% | 6 |
| Self-RAG budget exhaustion (2 retries used) | 0.0% | — |
| Fallback LLM (Qwen → Mistral switch) | 0.0% | — |
| Retriever drop (800 ms timeout) | 0.0% | — |

> All three conditional loops (§5 failure budgets) fired on zero queries in
> stub mode — expected, since the stub LLM responds instantaneously. In live
> API mode these rates reflect real network and model behaviour.

---

## Axis 3 — Efficiency

| Metric | Result | Target | Status |
|---|---|---|---|
| p50 end-to-end latency | 29576 ms | < 4,000 ms | ❌ |
| p95 end-to-end latency | 44707 ms | < 8,000 ms | ❌ |
| Mean latency | 22468 ms | — | — |
| Sample size | 60 queries | — | — |

### Per-stage average latency

| Stage | Avg latency |
|---|---|
| `agent1_ms` | 5896 ms |
| `agent2_ms` | 16572 ms |

> In stub mode all LLM calls return in < 1 ms (deterministic canned responses),
> so reported latencies reflect Python overhead only. Live API latencies will be
> dominated by network RTT and LLM generation time (typically 300–800 ms per call,
> 5-8 calls per query → 2–6 s end-to-end). The p50 < 4 s target is achievable
> with Groq (fast inference) but may require careful backoff tuning on Together.ai.

---

## Axis 4 — Human Judgment

*A 15-response sample scored on three 5-point Likert axes.*

### Likert axes

| Axis | Description | Target mean |
|---|---|---|
| Faithfulness | Every claim traces to a real manuscript passage | ≥ 4.0 / 5 |
| Dialect Fidelity | Khaleeji register is authentic (not MSA-flattened) | ≥ 3.5 / 5 |
| Usefulness | Response would satisfy a researcher or enthusiast | ≥ 3.5 / 5 |

### Sample responses (first 3 of 12)

| Query ID | Response snippet | Faithfulness | Dialect Fidelity | Usefulness |
|---|---|---|---|---|
| sq01 | تتضمن قصائد الخيل في الشعر النبطي الخليجي تعبيرات قوية عن الفخر والشجاعة. من بين هذه القصائد، نجد البيت الذي يقول:

"بال… | *(pending)* | *(pending)* | *(pending)* |
| sq05 | وجدت **35** قصائد محتملة عن رثاء أو حزن على ابن/بنت/طفل. من الشعراء الظاهرين في هذه المطابقة: غير محدد، حمود العبيد، ابن… | *(pending)* | *(pending)* | *(pending)* |
| sq06 | مطلع قصيدة المهادي في وصف الناقة هو:

"يقول المهادي والمهادي مهمل // له عبرة باقي الملا ما دروا بها" [anchor_id:manuscri… | *(pending)* | *(pending)* | *(pending)* |

> **Full sample:** see `data/evaluation_raw.json` → `human_judgment.responses`.

**Current aggregate:**

| Axis | Mean score |
|---|---|
| Faithfulness | *(pending)* |
| Dialect Fidelity | *(pending)* |
| Usefulness | *(pending)* |



## Axis 5 — Extensions · EXT-1…EXT-9

**Fixture set:** 20 queries
(`tests/fixtures/extension_queries.jsonl`)

| Metric | Result | Target | Status | n |
|---|---|---|---|---|
| Fast-path rate (EXT-8 registry_lookup) | 100.0% | ≥ 90% | ✅ | 3 |
| Dialect answer rate (EXT-1 bridge) | 75.0% | ≥ 50% | ✅ | 4 |
| Source routing accuracy (EXT-2/3/8) | 0.0% | ≥ 50% | ❌ | 10 |
| Genre filter answer rate (M3) | 25.0% | ≥ 50% | ❌ | 8 |
| Overall extension answer rate | 40.0% | — | — | 20 |

> **Fast-path rate:** registry_lookup queries (كم عدد, كم شاعراً…) must return
> via the deterministic fast-path — `guardrail_flags` contains `"registry_lookup"` and
> no LLM call is made. Target ≥ 90%.
>
> **Dialect answer rate:** queries using Khaleeji dialect terms (وش, شلون, يبون, الديرة…)
> should resolve to answers after EXT-1 normalisation, not be refused. Target ≥ 50%.
>
> **Source routing accuracy:** queries targeting `online_digitized` or `oral_tradition`
> sources should surface at least one chunk of the expected `source_type` in the
> RRF top-5. Requires the index to be built from `data/unified_registry.json`.
> If only the manuscript registry was indexed, source routing queries will score 0.
>
> **Genre filter answer rate:** queries naming an explicit genre (رثاء, فخر, مديح…)
> should be answered (genre filter activated by self_query → Qdrant payload filter).

---


## Summary: Architecture Claims vs Reality

| Architecture claim | Result | Honest verdict |
|---|---|---|
| Citation-resolvability 100% (§2.9 guardrail a) | 100.0% | ✅ Met in stub mode — live test pending |
| Recall@5 exact ≥ 75% (§5 scholar set) | 15.4% | ⚠️ Below target — relaxed (manuscript-level) = 46.2% |
| Recall@5 relaxed ≥ 75% (correct manuscript) | 46.2% | ⚠️ Below target |
| Refusal precision ≥ 90% (§2.9 guardrail c) | 80.0% | ⚠️ Below target — check OOC query routing |
| p50 < 4 s (§7 efficiency) | 29576 ms | ⚠️ Stub latency above 4 s — unexpected, check overhead |
| p95 < 8 s (§7 efficiency) | 44707 ms | ⚠️ Check p95 with live API |
| Fast-path ≥ 90% (EXT-8 registry_lookup) | 100.0% | ✅ Met |
| Dialect bridge ≥ 50% (EXT-1) | 75.0% | ✅ Met |
| Source routing ≥ 50% (EXT-2/3/8) | 0.0% | ⚠️ Rebuild index from unified_registry.json |

> **Note on stub mode:** the stub LLM returns deterministic canned responses that
> always trigger the is_refusal path. This means citation-resolvability and
> Recall@5 are measured on the guardrail-refusal template, not a real synthesis.
> Run `LLM_PROVIDER=together LLM_API_KEY=<key> python scripts/evaluate.py` for
> live numbers that reflect the actual pipeline quality.
>
> **Note on source routing (Axis 5):** source routing metrics require the Qdrant
> index to include oral_tradition and online_digitized chunks. Rebuild with:
> `python scripts/rebuild_index.py --registry data/unified_registry.json --force`

---

*NABAT-AI · MAAI1704 · Asma Salem Mubarak Najem Aljneibi*
*Evaluation generated by `scripts/evaluate.py`*
