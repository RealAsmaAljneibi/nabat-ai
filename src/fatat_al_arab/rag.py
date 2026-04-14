"""
Fatat Al Arab Agent — RAG Pipeline
=====================================
Worker 2: Retrieval-Augmented Generation for Khaleeji Nabati Poetry

Implements the full architecture from TASK_A_B_Plan.md §B.2:

  Ingestion:
    - Validation gate (confidence ≥ 0.90)
    - Metadata extraction
    - Image crop extraction

  Stanza-Aware Chunking (3 levels per architecture spec):
    - Level 1: Verse (بيت) — Sadr + Ajuz pair
    - Level 2: Stanza group (3-5 consecutive verses)
    - Level 3: Full poem with metadata

  Embedding:
    - CAMeL Tools normalization (normalize + dediacritize)
    - GATE-AraBERT-v1 (768-dim primary) — simulated with TF-IDF for MVP demo
      (production: sentence-transformers with CAMeL-BERT-base-MSA-AraPoemBERT)
    - Multilingual E5-Large (re-ranking) — simulated in MVP

  Storage:
    - Qdrant vector DB (dense + sparse BM25) — simulated with in-memory index for MVP
    - PostgreSQL (full poems) — simulated with JSON store
    - MinIO/S3 (image crops) — simulated with path references

  Retrieval:
    - Stage 1: Hybrid search — BM25 + dense vector via RRF fusion
    - Stage 2: Cross-encoder re-ranking (Arabic STS)
    - Stage 3: Contextual expansion (verse → stanza group)
    - Stage 4: Generation with citation injection

NOTE: In this initial implementation (MVP demo), the vector store is an in-memory
index using TF-IDF for BM25 simulation and cosine similarity for dense retrieval.
All module interfaces match the production architecture — only the model backends
are stubbed. Replace _embed() with GATE-AraBERT and _bm25_score() with actual
Qdrant BM25 to upgrade to production without changing any downstream code.
"""

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Arabic text utilities
# (mirrors al_nassikh_parser.py normalization)
# ─────────────────────────────────────────────

ALEF_VARIANTS = "أإآٱ"

def normalize_arabic(text: str, dediacritize: bool = True) -> str:
    """CAMeL Tools equivalent normalization pipeline."""
    if not text:
        return ""
    for v in ALEF_VARIANTS:
        text = text.replace(v, "ا")
    text = text.replace("ة", "ه")
    if dediacritize:
        text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace('\u0640', '')  # kashida
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    """Simple Arabic tokenizer — split on whitespace and punctuation."""
    text = normalize_arabic(text, dediacritize=True)
    tokens = re.findall(r'[\u0600-\u06FF]+', text)
    return [t for t in tokens if len(t) > 1]


# ─────────────────────────────────────────────
# Stanza-Aware Chunker (Architecture §B.2.2)
# ─────────────────────────────────────────────

def chunk_poem(poem: dict) -> list:
    """
    Produce 3 levels of chunks from a poem document.
    Returns flat list of chunk dicts, each with:
      - chunk_id, chunk_level, text, metadata, parent_poem_id, stanza_positions
    """
    chunks = []
    stanzas = poem.get("stanzas", [])
    poem_id = poem["poem_id"]
    poet = poem["poet"]["name"]
    volume = poem["source_volume"]
    page = poem["source_page"]
    occasion = poem.get("occasion")
    image_path = poem.get("source_image_path", "")

    base_meta = {
        "poem_id": poem_id,
        "poet": poet,
        "source_volume": volume,
        "source_page": page,
        "occasion": occasion,
        "source_image_path": image_path,
        "confidence_score": poem.get("confidence_score", 0.95),
        "region": poem["poet"].get("region", "unknown"),
    }

    # ── Level 1: Individual verses (بيت) ────────────────────────────
    for stanza in stanzas:
        verse_text = stanza.get("full_verse_manuscript", "")
        if not verse_text.strip():
            continue

        # Build search text from all 3 variants (per architecture: BM25 indexes all variants)
        sadr = stanza.get("sadr", {})
        ajuz = stanza.get("ajuz", {})
        variants_text = " ".join(filter(None, [
            sadr.get("manuscript_reading", ""),
            sadr.get("standard_reading", ""),
            sadr.get("dialectal_reading", ""),
            ajuz.get("manuscript_reading", ""),
            ajuz.get("standard_reading", ""),
            ajuz.get("dialectal_reading", ""),
        ]))

        chunks.append({
            "chunk_id": f"{poem_id}_v{stanza['stanza_num']:03d}",
            "chunk_level": 1,
            "text": verse_text,                  # Display text
            "search_text": variants_text,         # BM25 + dense embedding text (all variants)
            "stanza_positions": [stanza["stanza_num"]],
            "source_crop_bbox": stanza.get("source_crop_bbox", [0,0,0,0]),
            "sadr": sadr,
            "ajuz": ajuz,
            "metadata": {**base_meta, "stanza_num": stanza["stanza_num"]}
        })

    # ── Level 2: Stanza groups (3–5 consecutive verses) ─────────────
    GROUP_SIZE = 4  # architecture spec: 3-5 verses
    for i in range(0, len(stanzas), GROUP_SIZE):
        group = stanzas[i:i+GROUP_SIZE]
        group_lines = [s.get("full_verse_manuscript", "") for s in group if s.get("full_verse_manuscript")]
        if not group_lines:
            continue
        group_text = "\n".join(group_lines)
        positions = [s["stanza_num"] for s in group]

        # Aggregate variants for search
        all_variants = []
        for s in group:
            for part in ["sadr", "ajuz"]:
                d = s.get(part, {})
                all_variants.extend([
                    d.get("manuscript_reading", ""),
                    d.get("standard_reading", ""),
                    d.get("dialectal_reading", ""),
                ])

        chunks.append({
            "chunk_id": f"{poem_id}_g{i//GROUP_SIZE:03d}",
            "chunk_level": 2,
            "text": group_text,
            "search_text": " ".join(filter(None, all_variants)),
            "stanza_positions": positions,
            "source_crop_bbox": _merge_bboxes([s.get("source_crop_bbox", [0,0,0,0]) for s in group]),
            "metadata": {**base_meta,
                         "stanza_start": positions[0] if positions else 0,
                         "stanza_end": positions[-1] if positions else 0}
        })

    # ── Level 3: Full poem ───────────────────────────────────────────
    all_verse_text = "\n".join(
        s.get("full_verse_manuscript", "") for s in stanzas
        if s.get("full_verse_manuscript")
    )
    all_search_text_parts = []
    for s in stanzas:
        for part in ["sadr", "ajuz"]:
            d = s.get(part, {})
            all_search_text_parts.extend([
                d.get("manuscript_reading", ""),
                d.get("standard_reading", ""),
                d.get("dialectal_reading", ""),
            ])

    matla = poem.get("matla", {})
    poem_meta_text = " ".join(filter(None, [
        poet,
        matla.get("standard", ""),
        matla.get("dialectal", ""),
        matla.get("manuscript", ""),
        occasion or ""
    ]))

    chunks.append({
        "chunk_id": f"{poem_id}_full",
        "chunk_level": 3,
        "text": all_verse_text,
        "search_text": " ".join(filter(None, all_search_text_parts)) + " " + poem_meta_text,
        "stanza_positions": list(range(1, len(stanzas)+1)),
        "source_crop_bbox": None,
        "matla": matla,
        "verse_count": len(stanzas),
        "metadata": {**base_meta,
                     "matla_standard": matla.get("standard", ""),
                     "matla_dialectal": matla.get("dialectal", ""),
                     "verse_count": len(stanzas)}
    })

    return chunks


def _merge_bboxes(bboxes: list) -> list:
    """Merge multiple bounding boxes into one containing bbox."""
    valid = [b for b in bboxes if b and len(b) == 4 and any(b)]
    if not valid:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid)
    ]


# ─────────────────────────────────────────────
# Embedding Layer (Architecture §B.2.3)
# MVP: TF-IDF vectors stand in for GATE-AraBERT
# Production: replace _embed() with
#   sentence-transformers + CAMeL-AraBERT-base
# ─────────────────────────────────────────────

class ArabicEmbedder:
    """
    Arabic text embedder.
    MVP: TF-IDF sparse vectors (production-compatible interface).
    Production target: GATE-AraBERT-v1 (768-dim).
    """

    def __init__(self):
        self.vocab: dict = {}          # token → index
        self.idf: dict = {}            # token → IDF weight
        self.doc_count = 0
        self._built = False
        self.model_name = "tfidf-arabic-mvp (production: GATE-AraBERT-v1)"
        self.dimension = None          # Set after build

    def fit(self, corpus: list):
        """Build vocabulary and IDF from a list of text strings."""
        df = defaultdict(int)
        self.doc_count = len(corpus)
        for doc in corpus:
            tokens = set(tokenize(doc))
            for t in tokens:
                df[t] += 1
        # IDF with smoothing
        for token, count in df.items():
            self.idf[token] = math.log((1 + self.doc_count) / (1 + count)) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(self.idf.keys()))}
        self.dimension = len(self.vocab)
        self._built = True

    def embed(self, text: str) -> dict:
        """
        Return a sparse TF-IDF vector as {token_index: weight}.
        In production this returns a dense 768-dim float array from GATE-AraBERT.
        """
        if not self._built:
            raise RuntimeError("Call fit() before embed()")
        tokens = tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        vec = {}
        n = len(tokens) or 1
        for token, count in tf.items():
            if token in self.vocab:
                tfidf = (count / n) * self.idf.get(token, 1.0)
                vec[self.vocab[token]] = tfidf
        return vec

    def cosine_similarity(self, v1: dict, v2: dict) -> float:
        """Cosine similarity between two sparse vectors."""
        dot = sum(v1.get(k, 0) * v for k, v in v2.items())
        mag1 = math.sqrt(sum(x**2 for x in v1.values()))
        mag2 = math.sqrt(sum(x**2 for x in v2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)


# ─────────────────────────────────────────────
# In-Memory Vector Store
# (Production: Qdrant with multi-vector support)
# Architecture §B.2.4
# ─────────────────────────────────────────────

class NabatiVectorStore:
    """
    In-memory vector store simulating Qdrant's multi-vector points.
    Each point stores:
      - dense_vector (GATE-AraBERT / TF-IDF for MVP)
      - sparse_vector (BM25 weights)
      - payload (chunk metadata + text)

    Production: Replace with qdrant_client.QdrantClient
    with collections for dense + sparse hybrid search.
    """

    def __init__(self, embedder: ArabicEmbedder):
        self.embedder = embedder
        self.points: list = []         # list of {id, vector, bm25, payload}
        self._bm25_index: dict = {}    # token → list of (point_idx, weight)

    def upsert(self, chunk: dict):
        """Add a chunk as a vector point with payload."""
        search_text = chunk.get("search_text", chunk.get("text", ""))
        vector = self.embedder.embed(search_text)

        # BM25 sparse vector (token → TF-IDF weight for inverted index)
        bm25 = vector  # Same weights serve dual purpose in MVP

        point_idx = len(self.points)
        self.points.append({
            "id": chunk["chunk_id"],
            "vector": vector,
            "bm25": bm25,
            "payload": {
                "chunk_id": chunk["chunk_id"],
                "chunk_level": chunk["chunk_level"],
                "text": chunk["text"],
                "stanza_positions": chunk.get("stanza_positions", []),
                "source_crop_bbox": chunk.get("source_crop_bbox"),
                "metadata": chunk.get("metadata", {}),
                "matla": chunk.get("matla"),
                "sadr": chunk.get("sadr"),
                "ajuz": chunk.get("ajuz"),
            }
        })

        # Update BM25 inverted index
        for token_idx, weight in bm25.items():
            if token_idx not in self._bm25_index:
                self._bm25_index[token_idx] = []
            self._bm25_index[token_idx].append((point_idx, weight))

    def dense_search(self, query_vector: dict, top_k: int = 20) -> list:
        """Dense vector search (GATE-AraBERT semantic similarity)."""
        scored = []
        for i, point in enumerate(self.points):
            score = self.embedder.cosine_similarity(query_vector, point["vector"])
            scored.append((score, i))
        scored.sort(reverse=True)
        return [(self.points[i]["payload"], score) for score, i in scored[:top_k]]

    def bm25_search(self, query_tokens: list, top_k: int = 20) -> list:
        """BM25 sparse search across all 3 variant texts."""
        scores = defaultdict(float)
        for token in query_tokens:
            token_idx = self.embedder.vocab.get(token)
            if token_idx is None:
                continue
            idf = self.embedder.idf.get(token, 1.0)
            for point_idx, tf_weight in self._bm25_index.get(token_idx, []):
                scores[point_idx] += idf * tf_weight

        scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.points[i]["payload"], s) for i, s in scored[:top_k]]

    def hybrid_search_rrf(self, query_text: str, top_k: int = 10, k_rrf: int = 60) -> list:
        """
        Stage 1: Hybrid search — BM25 + Dense → Reciprocal Rank Fusion (RRF).
        Architecture spec §B.2.5, Stage 1.
        """
        query_norm = normalize_arabic(query_text, dediacritize=True)
        query_tokens = tokenize(query_norm)
        query_vector = self.embedder.embed(query_norm)

        dense_results = self.dense_search(query_vector, top_k=50)
        bm25_results = self.bm25_search(query_tokens, top_k=50)

        # Build ID → payload map
        id_to_payload = {}
        for payload, _ in dense_results + bm25_results:
            id_to_payload[payload["chunk_id"]] = payload

        # RRF fusion
        rrf_scores = defaultdict(float)
        for rank, (payload, _) in enumerate(dense_results):
            rrf_scores[payload["chunk_id"]] += 1.0 / (k_rrf + rank + 1)
        for rank, (payload, _) in enumerate(bm25_results):
            rrf_scores[payload["chunk_id"]] += 1.0 / (k_rrf + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(id_to_payload[cid], score) for cid, score in sorted_ids[:top_k]]

    def filter_by_level(self, results: list, level: int) -> list:
        """Filter results to a specific chunk level."""
        return [(p, s) for p, s in results if p.get("chunk_level") == level]

    def stats(self) -> dict:
        from collections import Counter
        levels = Counter(p["payload"]["chunk_level"] for p in self.points)
        return {
            "total_points": len(self.points),
            "by_level": dict(levels),
            "vocab_size": len(self.embedder.vocab),
            "model": self.embedder.model_name
        }


# ─────────────────────────────────────────────
# Anchor Registry (TOC Metadata Store)
# Architecture: Validation Gate + Anchor-Based Alignment
# ─────────────────────────────────────────────

class AnchorRegistry:
    """
    Stores TOC anchor entries for Anchor-Based First-Line Verification.
    Architecture spec §A.3 Step 1–2.
    """

    def __init__(self, anchors: list):
        self.anchors = anchors
        self._by_volume: dict = defaultdict(list)
        self._by_poet: dict = defaultdict(list)
        for a in anchors:
            vol = a.get("source_volume", "")
            poet = a.get("poet_name", "")
            self._by_volume[vol].append(a)
            if poet and poet != "unknown":
                self._by_poet[poet].append(a)

    def lookup_by_poet(self, poet_name: str) -> list:
        """Find all poems by a poet name (partial match supported)."""
        results = []
        for name, entries in self._by_poet.items():
            if poet_name in name or name in poet_name:
                results.extend(entries)
        return results

    def lookup_by_matla(self, matla_fragment: str) -> list:
        """Find poems whose Matla contains a given fragment."""
        results = []
        for a in self.anchors:
            if matla_fragment in a.get("matla_text", ""):
                results.append(a)
        return results

    def compute_cer(self, hypothesis: str, reference: str) -> float:
        """
        Character Error Rate for anchor-based verification.
        Architecture spec §A.3 Step 2.
        """
        if not reference:
            return 1.0
        ref_n = normalize_arabic(reference)
        hyp_n = normalize_arabic(hypothesis)
        d = _levenshtein(hyp_n, ref_n)
        return d / max(len(ref_n), 1)

    def classify_confidence(self, cer: float) -> str:
        """
        Architecture spec confidence thresholds:
          CER < 10%  → HIGH CONFIDENCE
          10–25%     → MEDIUM (route to Jury of Models)
          ≥ 25%      → LOW (route to HITL)
        """
        if cer < 0.10:
            return "HIGH"
        elif cer < 0.25:
            return "MEDIUM"
        else:
            return "LOW"


def _levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


# ─────────────────────────────────────────────
# Citation Injection (Architecture §B.2.5 Stage 4)
# ─────────────────────────────────────────────

def format_citation(chunk_payload: dict) -> str:
    """
    Format a mandatory inline citation per architecture spec §1.4:
    'Citation Precision > 95% — every response grounded in verified source.'
    """
    meta = chunk_payload.get("metadata", {})
    poet = meta.get("poet", "غير معروف")
    volume = meta.get("source_volume", "")
    page = meta.get("source_page", "")
    image_path = meta.get("source_image_path", "")
    confidence = meta.get("confidence_score", 0.0)
    stanza_positions = chunk_payload.get("stanza_positions", [])

    parts = [f"المصدر: {volume}"]
    if page:
        parts.append(f"صفحة {page}")
    if poet and poet != "unknown":
        parts.append(f"الشاعر: {poet}")
    if stanza_positions:
        if len(stanza_positions) == 1:
            parts.append(f"بيت {stanza_positions[0]}")
        else:
            parts.append(f"أبيات {stanza_positions[0]}–{stanza_positions[-1]}")
    if image_path:
        parts.append(f"[الصورة: {image_path}]")
    parts.append(f"[ثقة: {confidence:.0%}]")

    return " | ".join(parts)


# ─────────────────────────────────────────────
# RAG Query Engine (Architecture §B.2.5)
# ─────────────────────────────────────────────

class FatatAlArabAgent:
    """
    Fatat Al Arab — RAG Query Agent.
    Implements the 4-stage retrieval pipeline from architecture spec §B.2.5.
    """

    def __init__(self, vector_store: NabatiVectorStore, anchor_registry: AnchorRegistry):
        self.store = vector_store
        self.anchors = anchor_registry

    def query(self, user_query: str, top_k: int = 5, level_filter: Optional[int] = None) -> dict:
        """
        Full 4-stage retrieval + generation pipeline.

        Stage 1: Hybrid BM25 + Dense search → RRF fusion
        Stage 2: Cross-encoder re-ranking (simulated in MVP)
        Stage 3: Contextual expansion (verse → stanza group)
        Stage 4: Response generation with citation injection
        """

        # ── Stage 1: Hybrid Search (BM25 + Dense → RRF) ─────────────
        candidates = self.store.hybrid_search_rrf(user_query, top_k=top_k * 3)

        if level_filter:
            candidates = self.store.filter_by_level(candidates, level_filter)

        # ── Stage 2: Re-ranking ──────────────────────────────────────
        # Production: cross-encoder Arabic STS fine-tuned model
        # MVP: boost exact keyword matches in display text
        query_tokens = set(tokenize(normalize_arabic(user_query)))
        reranked = []
        for payload, rrf_score in candidates:
            text_tokens = set(tokenize(normalize_arabic(payload["text"])))
            exact_match_bonus = len(query_tokens & text_tokens) * 0.05
            reranked.append((payload, rrf_score + exact_match_bonus))
        reranked.sort(key=lambda x: x[1], reverse=True)

        top_results = reranked[:top_k]

        # ── Stage 3: Contextual Expansion ───────────────────────────
        # If top result is a Level 1 verse, optionally expand to Level 2 group
        expanded_results = []
        for payload, score in top_results:
            entry = {"payload": payload, "score": score, "citation": format_citation(payload)}
            # Expand: find the stanza-group (Level 2) chunk containing this verse
            if payload["chunk_level"] == 1:
                parent_poem_id = payload["metadata"]["poem_id"]
                stanza_num = payload["metadata"].get("stanza_num", 0)
                # Find the Level 2 chunk that contains this stanza
                group_chunk = self._find_parent_group(parent_poem_id, stanza_num)
                if group_chunk:
                    entry["expanded_context"] = group_chunk["text"]
                    entry["expanded_citation"] = format_citation(group_chunk)
            expanded_results.append(entry)

        # ── Stage 4: Response Generation ────────────────────────────
        response = self._generate_response(user_query, expanded_results)

        return {
            "query": user_query,
            "results": expanded_results,
            "response": response,
            "total_candidates": len(candidates),
            "retrieval_pipeline": "BM25+Dense→RRF→Rerank→Expand→Generate"
        }

    def _find_parent_group(self, poem_id: str, stanza_num: int) -> Optional[dict]:
        """Find the Level 2 chunk that contains a given stanza number."""
        for point in self.store.points:
            p = point["payload"]
            if (p["chunk_level"] == 2
                    and p["metadata"].get("poem_id") == poem_id
                    and stanza_num in p.get("stanza_positions", [])):
                return p
        return None

    def _generate_response(self, query: str, results: list) -> str:
        """
        Stage 4: Generate a cited response.
        Production: GPT-4o / Claude with Arabic poetry analysis prompt.
        MVP: Structured retrieval display with mandatory citations.
        (Architecture spec §1.4: Hallucination Rate < 2%)
        """
        if not results:
            return "لم يتم العثور على نتائج مطابقة للاستعلام."

        lines = []
        lines.append(f"🔍 نتائج البحث في ديوان الشعر النبطي الخليجي")
        lines.append(f"الاستعلام: {query}")
        lines.append("─" * 60)

        for i, entry in enumerate(results, 1):
            payload = entry["payload"]
            level = payload["chunk_level"]
            text = payload["text"]
            citation = entry["citation"]

            level_labels = {1: "بيت شعر", 2: "مجموعة أبيات", 3: "قصيدة كاملة"}
            level_label = level_labels.get(level, "نتيجة")

            lines.append(f"\n[{i}] {level_label} (درجة التطابق: {entry['score']:.4f})")

            if level == 1:
                # Show the verse text with Sadr/Ajuz if available
                sadr = payload.get("sadr", {})
                ajuz = payload.get("ajuz", {})
                if sadr.get("manuscript_reading") and ajuz.get("manuscript_reading"):
                    lines.append(f"  الصدر:  {sadr['manuscript_reading']}")
                    lines.append(f"  العجز:  {ajuz['manuscript_reading']}")
                    # Show standard variant if it differs
                    if sadr.get("standard_reading") != sadr.get("manuscript_reading"):
                        lines.append(f"  (المعيار: {sadr['standard_reading']} // {ajuz.get('standard_reading', '')})")
                else:
                    lines.append(f"  {text}")

                # Expanded context
                if entry.get("expanded_context"):
                    lines.append(f"  ─ السياق (الأبيات المحيطة):")
                    for vline in entry["expanded_context"].split("\n")[:4]:
                        if vline.strip():
                            lines.append(f"    {vline.strip()}")

            elif level == 2:
                for vline in text.split("\n")[:5]:
                    if vline.strip():
                        lines.append(f"  {vline.strip()}")

            elif level == 3:
                matla = payload.get("matla", {})
                lines.append(f"  مطلع القصيدة: {matla.get('manuscript', text[:100])}")
                lines.append(f"  عدد الأبيات: {payload.get('verse_count', '?')}")

            lines.append(f"  📜 {citation}")

        lines.append("\n─" * 60)
        lines.append("⚠️  كل نتيجة مستخرجة مباشرة من المخطوط المرقَّم — لا يوجد توليد لغوي خارج النص الأصلي.")
        lines.append("   (Hallucination Rate = 0% for MVP retrieval-only mode)")

        return "\n".join(lines)

    def anchor_search(self, query: str) -> list:
        """Search anchor registry (TOC metadata) for poets or Matla lines."""
        results = []
        # Try poet name search
        poet_results = self.anchors.lookup_by_poet(query)
        if poet_results:
            results.extend(poet_results)
        # Try Matla fragment search
        matla_results = self.anchors.lookup_by_matla(query)
        for r in matla_results:
            if r not in results:
                results.append(r)
        return results


# ─────────────────────────────────────────────
# Pipeline Builder (Ingestion + Indexing)
# ─────────────────────────────────────────────

def build_pipeline(poems_json_path: str, anchors_json_path: str) -> FatatAlArabAgent:
    """
    Build the full Fatat Al Arab RAG pipeline from Al-Nassikh outputs.
    This is the complete Worker 1 → Worker 2 handoff.
    """
    print("=" * 60)
    print("Fatat Al Arab Agent — Pipeline Initialization")
    print("=" * 60)

    # ── Load Al-Nassikh outputs ──────────────────────────────────────
    with open(poems_json_path, encoding="utf-8") as f:
        poems = json.load(f)
    with open(anchors_json_path, encoding="utf-8") as f:
        anchors_raw = json.load(f)

    print(f"Loaded {len(poems)} poem documents from Al-Nassikh")
    print(f"Loaded {len(anchors_raw)} TOC anchor entries")

    # ── Validation Gate (confidence ≥ 0.90) ─────────────────────────
    valid_poems = {pid: p for pid, p in poems.items()
                   if p.get("confidence_score", 0) >= 0.90}
    print(f"Validation gate: {len(valid_poems)}/{len(poems)} poems pass (confidence ≥ 0.90)")

    # ── Stanza-Aware Chunking ────────────────────────────────────────
    print("\nChunking poems (Level 1: verse | Level 2: group | Level 3: full)...")
    all_chunks = []
    for poem in valid_poems.values():
        chunks = chunk_poem(poem)
        all_chunks.extend(chunks)

    level_counts = {1: 0, 2: 0, 3: 0}
    for c in all_chunks:
        level_counts[c["chunk_level"]] = level_counts.get(c["chunk_level"], 0) + 1
    print(f"Total chunks: {len(all_chunks)}")
    print(f"  Level 1 (verses):       {level_counts[1]}")
    print(f"  Level 2 (stanza groups): {level_counts[2]}")
    print(f"  Level 3 (full poems):   {level_counts[3]}")

    # ── Embedding Layer (GATE-AraBERT / TF-IDF MVP) ─────────────────
    print("\nBuilding embedding model (TF-IDF MVP → production: GATE-AraBERT-v1)...")
    corpus = [c.get("search_text", c.get("text", "")) for c in all_chunks]
    embedder = ArabicEmbedder()
    embedder.fit(corpus)
    print(f"Vocabulary: {len(embedder.vocab)} Arabic tokens")
    print(f"Model: {embedder.model_name}")

    # ── Vector Store Ingestion ───────────────────────────────────────
    print("\nIngesting chunks into vector store (MVP: in-memory | production: Qdrant)...")
    store = NabatiVectorStore(embedder)
    for chunk in all_chunks:
        store.upsert(chunk)
    stats = store.stats()
    print(f"Vector store ready: {stats['total_points']} points indexed")
    print(f"  By level: {stats['by_level']}")

    # ── Anchor Registry ──────────────────────────────────────────────
    registry = AnchorRegistry(anchors_raw)
    print(f"\nAnchor registry: {len(anchors_raw)} TOC entries from {len(set(a['source_volume'] for a in anchors_raw))} volumes")

    # ── Agent Assembly ───────────────────────────────────────────────
    agent = FatatAlArabAgent(store, registry)
    print("\n✓ Fatat Al Arab Agent initialized — ready for queries.")
    return agent


# ─────────────────────────────────────────────
# Demo Queries (Initial Implementation Demo)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = Path("/sessions/fervent-sharp-faraday/mnt/handwritten-poems/al_nassikh_output")

    agent = build_pipeline(
        poems_json_path=str(OUTPUT_DIR / "phase1_poems.json"),
        anchors_json_path=str(OUTPUT_DIR / "anchor_registry.json")
    )

    demo_queries = [
        ("الناس", "Search for mentions of 'the people' across all manuscripts"),
        ("الله", "Search for verses invoking Allah"),
        ("ناصر بن حمد الهزاني", "Search for a specific poet by name (from TOC)"),
        ("الحمام", "Search for verses about doves"),
        ("الخيل", "Search for verses about horses — classical Nabati theme"),
    ]

    print("\n" + "=" * 60)
    print("DEMO QUERIES — NABAT-AI Initial Implementation")
    print("=" * 60)

    demo_output = []
    for query, description in demo_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"({description})")
        print()

        result = agent.query(query, top_k=3, level_filter=1)
        print(result["response"])

        # Also check anchor registry for this query
        anchor_hits = agent.anchor_search(query)
        if anchor_hits:
            print(f"\n📋 Anchor Registry Hits ({len(anchor_hits)}):")
            for ah in anchor_hits[:2]:
                print(f"  Poet: {ah.get('poet_name', '?')} | "
                      f"Page: {ah.get('page_number', '?')} | "
                      f"Matla: {ah.get('matla_text', '')[:50]}")

        demo_output.append({
            "query": query,
            "description": description,
            "result_count": len(result["results"]),
            "anchor_hits": len(anchor_hits),
            "top_result_text": result["results"][0]["payload"]["text"][:100] if result["results"] else "",
            "citation": result["results"][0]["citation"] if result["results"] else ""
        })

    # Save demo output for report
    demo_out_path = OUTPUT_DIR / "demo_query_results.json"
    with open(demo_out_path, "w", encoding="utf-8") as f:
        json.dump(demo_output, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Demo results saved → {demo_out_path}")
