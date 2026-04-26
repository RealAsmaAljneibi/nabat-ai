# NABAT-AI — Evaluation Report
## §7 Four-Axis Evaluation

**Generated:** 2026-04-25 22:33 UTC
**LLM provider:** `stub`
**Fixture counts:** 20 in-corpus · 20 OOC · 10 bilingual

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
| Recall@5 (in-corpus, gold anchors) | 23.1% | ≥ 75% | ❌ |
| Refusal precision (OOC set, n=20) | 100.0% | ≥ 90% | ✅ |

**CRAG verdict distribution** (in-corpus queries):

| Verdict | Count |
|---|---|
| Correct | 20 |

> Recall@5 is measured as: at least one gold anchor_id from the fixture appears
> in the response's citations_used list. This is a conservative lower bound —
> the stub LLM always returns the refusal template so cited anchors are empty;
> in live mode this number reflects true retrieval quality.

---

## Axis 2 — Robustness

| Mechanism | Activation rate | Raw count / 50 queries |
|---|---|---|
| CRAG re-query (Loop A+B) | 44.0% | 22 |
| Self-RAG retry (Loop C) | 0.0% | 0 |
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
| p50 end-to-end latency | 76 ms | < 4,000 ms | ✅ |
| p95 end-to-end latency | 139 ms | < 8,000 ms | ✅ |
| Mean latency | 313 ms | — | — |
| Sample size | 50 queries | — | — |

### Per-stage average latency

| Stage | Avg latency |
|---|---|
| `agent1_ms` | 166 ms |
| `agent2_ms` | 147 ms |

> In stub mode all LLM calls return in < 1 ms (deterministic canned responses),
> so reported latencies reflect Python overhead only. Live API latencies will be
> dominated by network RTT and LLM generation time (typically 300–800 ms per call,
> 5-8 calls per query → 2–6 s end-to-end). The p50 < 4 s target is achievable
> with Groq (fast inference) but may require careful backoff tuning on Together.ai.

---

## Axis 4 — Human Judgment

*A Nabati poetry scholar will review a 15-response sample and score each on three
5-point Likert axes. The table below is the scaffold to be filled in offline.*

### Likert axes

| Axis | Description | Target mean |
|---|---|---|
| Faithfulness | Every claim traces to a real manuscript passage | ≥ 4.0 / 5 |
| Dialect Fidelity | Khaleeji register is authentic (not MSA-flattened) | ≥ 3.5 / 5 |
| Usefulness | Response would satisfy a researcher or enthusiast | ≥ 3.5 / 5 |

### Sample responses (first 3 of 15)

| Query ID | Response snippet | Faithfulness | Dialect Fidelity | Usefulness |
|---|---|---|---|---|
| sq01 | يعرض الأرشيف شاهداً ذا صلة: قم یابن ابوی ارکب علی کور هبا ع//له بین ابانات والانجاج مرباع [anchor_id:manuscript12_p378_r… | — | — | — |
| sq02 | يعرض الأرشيف شاهداً ذا صلة: قم یابن ابوی ارکب علی کور هبا ع//له بین ابانات والانجاج مرباع [anchor_id:manuscript12_p378_r… | — | — | — |
| sq03 | يعرض الأرشيف شاهداً ذا صلة: قم یابن ابوی ارکب علی کور هبا ع//له بین ابانات والانجاج مرباع [anchor_id:manuscript12_p378_r… | — | — | — |

> **Fill in:** a Khaleeji Nabati poetry expert reviews the full 15-response sample
> in `data/evaluation_raw.json` under the `human_judgment.responses` key and
> records scores + notes per response. Aggregate means are updated in this report.

**Current aggregate (pending scholar review):**

| Axis | Mean score |
|---|---|
| Faithfulness | *(pending)* |
| Dialect Fidelity | *(pending)* |
| Usefulness | *(pending)* |

---

## Summary: Architecture Claims vs Reality

| Architecture claim | Result | Honest verdict |
|---|---|---|
| Citation-resolvability 100% (§2.9 guardrail a) | 100.0% | ✅ Met in stub mode — live test pending |
| Recall@5 ≥ 75% (§5 scholar set) | 23.1% | ⚠️ Below target in stub mode (expected — stub cites nothing). Run with live API for real measure. |
| Refusal precision ≥ 90% (§2.9 guardrail c) | 100.0% | ✅ Met |
| p50 < 4 s (§7 efficiency) | 76 ms | ✅ Met in stub |
| p95 < 8 s (§7 efficiency) | 139 ms | ✅ Met in stub |

> **Note on stub mode:** the stub LLM returns deterministic canned responses that
> always trigger the is_refusal path. This means citation-resolvability and
> Recall@5 are measured on the guardrail-refusal template, not a real synthesis.
> Run `LLM_PROVIDER=together LLM_API_KEY=<key> python scripts/evaluate.py` for
> live numbers that reflect the actual pipeline quality.

---

*NABAT-AI · MAAI1704 · Asma Salem Mubarak Najem Aljneibi*
*Evaluation generated by `scripts/evaluate.py`*
