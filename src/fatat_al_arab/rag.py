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
from typing import Optional, TypedDict, List

# LangGraph import — graceful fallback if not installed
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

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
# Governance: Constrained System Prompt
# Architecture §1.4 — Hallucination Rate < 2%
# Prevents LLM from generating modern Arabic or
# inventing verses not present in the manuscripts.
# ─────────────────────────────────────────────

SYSTEM_PROMPT_KHALEEJI = """\
أنتِ فتاة العرب — نظام استرجاع الشعر النبطي الخليجي.

قواعد صارمة لا تُخترق:
1. استخدمي فقط النصوص المُسترجعة من المخطوطات أدناه. لا تولّدي أي نص شعري من عندك.
2. لكل بيت تستشهدين به، أضيفي الاستشهاد الكامل (المخطوط | الصفحة | رقم البيت | مسار الصورة).
3. إذا لم يكن في المخطوطات ما يجيب السؤال مباشرةً، قولي:
   "لا يوجد في المخطوطات المرقَّمة ما يجيب هذا السؤال — يُنصح بمراجعة المخطوط الأصلي."
4. حافظي على اللهجة الخليجية النبطية — لا تستبدليها بالفصحى الحديثة أو تُعدّليها.
5. كل جملة تقولينها يجب أن تكون مُستنِدة بشكل مباشر إلى النصوص المُسترجعة.

النصوص المُسترجعة من المخطوطات:
{context}

سؤال المستخدم: {query}
"""


# ─────────────────────────────────────────────
# LangGraph State Schema
# Defines the shared state that flows through
# all nodes of the Fatat Al Arab graph.
# ─────────────────────────────────────────────

class RAGState(TypedDict):
    """Typed state for the Fatat Al Arab LangGraph agent (§5.2 Steps 1–9)."""
    # Step 1
    query: str                    # Original user query (Arabic or English)
    query_lang: str               # "ar" | "en" — detected language
    query_ar: str                 # Arabic version of query (translated if needed)
    # Step 2
    intent: str                   # "verse_search" | "poet_lookup" | "topic_search"
    metadata_filters: dict        # Extracted structured filters (poet, volume, region)
    # Step 3
    hyde_query: str               # HyDE-expanded Arabic query
    query_variants: List[str]     # 3–5 query variants for multi-query retrieval
    # Steps 4–5
    candidates: List[dict]        # RRF-fused retrieval results
    # Step 6
    resolved_results: List[dict]  # MSA match → Khaleeji original resolved
    # Step 7
    graded_docs: List[dict]       # CRAG-graded documents
    retrieval_grade: str          # "RELEVANT" | "PARTIALLY_RELEVANT" | "IRRELEVANT"
    rewrite_count: int            # Query rewrites attempted (max 2)
    # Step 8
    expanded_results: List[dict]  # Contextually expanded + cited results
    response: str                 # LLM-synthesised response
    # Step 9
    self_rag_verdict: str         # "GROUNDED" | "HALLUCINATION_RISK" | "NOT_SUPPORTED"
    self_rag_retries: int         # Self-RAG retry counter (max 2)
    # Meta
    fallback_triggered: bool
    conversation_history: List[dict]


# ─────────────────────────────────────────────
# Conversation Memory (Session State)
# Architecture: Memory for multi-turn queries.
# Stores the last N query/response pairs so the
# agent can resolve pronouns and follow-ups.
# ─────────────────────────────────────────────

class ConversationMemory:
    """Rolling window of past query/response pairs for multi-turn support."""

    MAX_TURNS = 5

    def __init__(self):
        self._history: List[dict] = []

    def add(self, query: str, response: str):
        self._history.append({"query": query, "response": response})
        if len(self._history) > self.MAX_TURNS:
            self._history.pop(0)

    def get_context_string(self) -> str:
        if not self._history:
            return ""
        turns = []
        for turn in self._history[-3:]:      # last 3 turns for context
            turns.append(f"س: {turn['query']}")
            turns.append(f"ج: {turn['response'][:120]}...")
        return "\n".join(turns)

    def resolve_query(self, query: str) -> str:
        """
        Pronoun resolution: if query contains a reference pronoun (هو/هي/هم/ذلك),
        prepend the last query subject so retrieval is not ambiguous.
        """
        PRONOUNS = {"هو", "هي", "هم", "هن", "ذلك", "تلك", "هذا", "هذه"}
        tokens = set(query.split())
        if tokens & PRONOUNS and self._history:
            last_query = self._history[-1]["query"]
            return f"{last_query} — {query}"
        return query

    @property
    def history(self) -> List[dict]:
        return list(self._history)


# ─────────────────────────────────────────────
# Step 2: Bilingual Analysis — Language Detection,
# EN→AR Translation, Intent Parsing, Self-Query
# Architecture §5.2 Step 2
# ─────────────────────────────────────────────

# Simple EN→AR translation table for common poetry search terms
EN_AR_TERMS = {
    "horse": "خيل", "horses": "خيل", "sword": "سيف", "love": "حب",
    "heart": "قلب", "people": "الناس", "god": "الله", "allah": "الله",
    "dove": "حمام", "poem": "شعر", "poetry": "قصيد", "tribe": "قوم",
    "tears": "دموع", "eyes": "عيون", "desert": "صحراء", "camel": "ابل",
    "poet": "شاعر", "verse": "بيت", "war": "حرب", "peace": "سلام",
    "praise": "مديح", "elegy": "رثاء", "nostalgia": "شوق",
}

def detect_language(text: str) -> str:
    """
    Detect whether the query is Arabic or English.
    Arabic = Unicode range U+0600–U+06FF.
    """
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars > len(text) * 0.3 else "en"

def translate_en_to_ar(text: str) -> str:
    """
    EN→AR translation.
    Production: Google Translate API / Helsinki-NLP/opus-mt-en-ar.
    MVP: token-level dictionary lookup for common poetry terms.
    """
    tokens = text.lower().split()
    ar_parts = []
    for token in tokens:
        clean = re.sub(r'[^a-z]', '', token)
        ar_parts.append(EN_AR_TERMS.get(clean, token))
    return " ".join(ar_parts)

def parse_intent(query_ar: str) -> str:
    """
    Intent classification for routing.
      verse_search  — searching for a specific verse text
      poet_lookup   — looking up a named poet
      topic_search  — thematic/semantic query
    """
    POET_INDICATORS = {"شاعر", "اشعار", "قال", "ابن", "بن", "ابو", "أبو"}
    VERSE_INDICATORS = {"بيت", "ابيات", "مطلع", "صدر", "عجز", "قافية"}

    tokens = set(tokenize(normalize_arabic(query_ar)))
    if tokens & POET_INDICATORS:
        return "poet_lookup"
    elif tokens & VERSE_INDICATORS:
        return "verse_search"
    else:
        return "topic_search"

def extract_metadata_filters(query_ar: str) -> dict:
    """
    Self-Query: extract structured metadata filters from natural language.
    Production: LLM-based metadata extraction with function calling.
    MVP: pattern matching for volume and confidence references.
    """
    filters = {}
    # Volume reference: "في المخطوط 07" or "manuscript07"
    vol_match = re.search(r'(manuscript\d+|مخطوط\s*\d+)', query_ar, re.IGNORECASE)
    if vol_match:
        filters["volume"] = vol_match.group(1).replace(" ", "").replace("مخطوط", "manuscript")
    # Confidence filter: "عالي الثقة" → min_confidence 0.9
    if "عالي" in query_ar or "موثق" in query_ar:
        filters["min_confidence"] = 0.95
    return filters


# ─────────────────────────────────────────────
# Step 3: Multi-Query Expansion
# Generates 3–5 query variants in Arabic for
# broader retrieval coverage before RRF fusion.
# Architecture §5.2 Step 3
# ─────────────────────────────────────────────

def expand_query_variants(query_ar: str, intent: str) -> List[str]:
    """
    Produce 3–5 query variants to maximise recall before RRF fusion.
    Production: LLM generates semantically diverse reformulations.
    MVP: rule-based expansion (dediacritised + synonym + root).
    """
    variants = [query_ar]  # Variant 1: original

    # Variant 2: fully dediacritised
    diac_stripped = normalize_arabic(query_ar, dediacritize=True)
    if diac_stripped not in variants:
        variants.append(diac_stripped)

    # Variant 3: HyDE dialectal synonym enrichment
    hyde_v = hyde_expand(query_ar)
    if hyde_v not in variants:
        variants.append(hyde_v)

    # Variant 4 (poet_lookup): expand to "أشعار {poet}"
    if intent == "poet_lookup":
        variants.append(f"أشعار {query_ar}")

    # Variant 5 (topic_search): add question framing for semantic retrieval
    if intent == "topic_search":
        variants.append(f"ابيات عن {query_ar}")

    return variants[:5]  # cap at 5


# ─────────────────────────────────────────────
# Step 4: ColBERT Retrieval Stub
# Token-level late interaction retrieval.
# Production: RAGatouille / pylate with
#   a fine-tuned Arabic ColBERT checkpoint.
# MVP: per-token BM25 overlap (same signal, explicit interface).
# Architecture §5.2 Step 4
# ─────────────────────────────────────────────

def colbert_search(
    query_ar: str,
    store: "NabatiVectorStore",
    top_k: int = 20,
) -> List[tuple]:
    """
    ColBERT late-interaction retrieval.
    Scores each document by summing per-token max-similarity scores.
    Production: replace with RAGatouille ColBERT index.
    MVP: per-token IDF-weighted overlap (same architectural interface).
    """
    query_tokens = tokenize(normalize_arabic(query_ar))
    if not query_tokens:
        return []

    scored = {}
    for point in store.points:
        payload = point["payload"]
        doc_tokens = tokenize(normalize_arabic(payload.get("text", "")))
        doc_token_set = set(doc_tokens)

        # Late interaction: for each query token, find max match in document
        token_scores = []
        for qt in query_tokens:
            if qt in doc_token_set:
                idf = store.embedder.idf.get(qt, 1.0)
                token_scores.append(idf)   # max sim = IDF weight when exact match
            else:
                token_scores.append(0.0)

        colbert_score = sum(token_scores) / max(len(query_tokens), 1)
        if colbert_score > 0:
            scored[point["id"]] = (payload, colbert_score)

    sorted_results = sorted(scored.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


# ─────────────────────────────────────────────
# Step 5: Three-Way RRF Fusion
# Merges BM25 + AraBERT + ColBERT lists.
# Architecture §5.2 Step 5
# ─────────────────────────────────────────────

def triple_rrf_fusion(
    bm25_results: List[tuple],
    dense_results: List[tuple],
    colbert_results: List[tuple],
    top_k: int = 10,
    k_rrf: int = 60,
) -> List[tuple]:
    """
    Three-way Reciprocal Rank Fusion over BM25 + AraBERT + ColBERT.
    Each retriever contributes 1/(k + rank) to the fused score.
    """
    id_to_payload = {}
    for payload, _ in bm25_results + dense_results + colbert_results:
        id_to_payload[payload["chunk_id"]] = payload

    rrf_scores: dict = defaultdict(float)
    for rank, (payload, _) in enumerate(bm25_results):
        rrf_scores[payload["chunk_id"]] += 1.0 / (k_rrf + rank + 1)
    for rank, (payload, _) in enumerate(dense_results):
        rrf_scores[payload["chunk_id"]] += 1.0 / (k_rrf + rank + 1)
    for rank, (payload, _) in enumerate(colbert_results):
        rrf_scores[payload["chunk_id"]] += 1.0 / (k_rrf + rank + 1)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(id_to_payload[cid], score) for cid, score in sorted_ids[:top_k]]


# ─────────────────────────────────────────────
# Step 6: Multi-Rep Resolution
# MSA standard match → return Khaleeji original.
# Architecture §5.2 Step 6
# ─────────────────────────────────────────────

def multi_rep_resolve(results: List[tuple], query_ar: str) -> List[tuple]:
    """
    Multi-Representation Resolution:
    Retrieval ran over MSA standard text; return the manuscript
    (Khaleeji dialectal) reading as the displayed result so users
    see the original verse, not the normalised search form.

    For each result, ensure payload["text"] is the manuscript reading
    and attach the standard/dialectal variants for the response layer.
    """
    resolved = []
    for payload, score in results:
        # The chunk already stores manuscript text as "text" and
        # all variants in sadr/ajuz — expose them explicitly here.
        sadr = payload.get("sadr", {})
        ajuz = payload.get("ajuz", {})
        payload["display_text"] = payload.get("text", "")          # manuscript (Khaleeji)
        payload["standard_text"] = (                                # MSA normalised
            f"{sadr.get('standard_reading','')} // {ajuz.get('standard_reading','')}"
            if sadr.get("standard_reading") else payload.get("text", "")
        )
        payload["dialectal_text"] = (                              # dialectal form
            f"{sadr.get('dialectal_reading','')} // {ajuz.get('dialectal_reading','')}"
            if sadr.get("dialectal_reading") else payload.get("text", "")
        )
        resolved.append((payload, score))
    return resolved


# ─────────────────────────────────────────────
# Step 8: LLM Synthesis with Constrained Prompt
# Formats the system prompt with retrieved context
# and calls the LLM (stubbed for MVP).
# Architecture §5.2 Step 8
# ─────────────────────────────────────────────

def llm_synthesise(query_ar: str, results: List[dict]) -> str:
    """
    Stage 8 — LLM synthesis with mandatory citations.
    Production: call Claude / Qwen2.5 via API with SYSTEM_PROMPT_KHALEEJI.
    MVP: fills the prompt template and returns structured retrieval text
         (zero hallucination — no generative model called).

    To upgrade: replace the return block with an Anthropic/OpenAI API call
    using `prompt` as the user message and SYSTEM_PROMPT_KHALEEJI as system.
    """
    # Build context block from CRAG-approved results
    context_lines = []
    for i, entry in enumerate(results, 1):
        payload = entry["payload"]
        display  = payload.get("display_text", payload.get("text", ""))
        standard = payload.get("standard_text", "")
        citation = entry.get("citation", "")
        context_lines.append(
            f"[{i}] النص الأصلي (خليجي): {display}\n"
            f"    النص المعياري (MSA):   {standard}\n"
            f"    {citation}"
        )
    context = "\n\n".join(context_lines)

    # Filled prompt (ready to send to LLM in production)
    prompt = SYSTEM_PROMPT_KHALEEJI.format(context=context, query=query_ar)

    # ── MVP: return structured retrieval (no LLM call) ───────────────
    # In production replace this block with:
    #   from anthropic import Anthropic
    #   client = Anthropic()
    #   msg = client.messages.create(
    #       model="claude-opus-4-6",
    #       max_tokens=1024,
    #       system=SYSTEM_PROMPT_KHALEEJI.split("{context}")[0],
    #       messages=[{"role": "user", "content": prompt}]
    #   )
    #   return msg.content[0].text
    return prompt   # returns filled prompt as MVP response


# ─────────────────────────────────────────────
# HyDE — Hypothetical Document Embedding
# Architecture §B.2.5 — bridges modern Arabic
# queries to Khaleeji Nabati dialect in the index.
# Production: LLM generates a plausible Nabati verse.
# MVP: dialectal synonym enrichment.
# ─────────────────────────────────────────────

# Khaleeji dialectal synonym table (query → dialectal equivalents)
DIALECTAL_SYNONYMS = {
    "حصان":  "خيل عيس جواد مهر",
    "قلب":   "فؤاد صدر جنان",
    "عيون":  "محاجر اعيان",
    "الله":  "الرحمن الرحيم رب",
    "الناس": "الخلق البشر العباد",
    "حب":    "هوى وجد شوق غرام",
    "بكاء":  "دموع عبرات",
    "سيف":   "حسام صارم",
    "شعر":   "قصيد قصيدة ابيات",
    "قوم":   "اهل حي عشيرة",
}

def hyde_expand(query: str) -> str:
    """
    HyDE: enrich the user query with Khaleeji dialectal synonyms so that
    the TF-IDF / AraBERT embedding better matches manuscript vocabulary.
    Production: replace with LLM call that generates a hypothetical Nabati verse.
    """
    tokens = tokenize(normalize_arabic(query, dediacritize=False))
    enriched_parts = [query]
    for token in tokens:
        if token in DIALECTAL_SYNONYMS:
            enriched_parts.append(DIALECTAL_SYNONYMS[token])
    return " ".join(enriched_parts)


# ─────────────────────────────────────────────
# CRAG — Corrective RAG Document Grader
# Architecture §B.2.5 — Detector stage.
# Grades each retrieved document for query
# relevance. Irrelevant docs trigger query rewrite.
# ─────────────────────────────────────────────

# Score thresholds (tuned against Phase 1 data)
CRAG_RELEVANT_SCORE   = 0.025   # RRF score above this → RELEVANT
CRAG_PARTIAL_SCORE    = 0.008   # RRF score above this → PARTIAL
MIN_TOKEN_OVERLAP     = 1       # At least 1 shared token required


def crag_grade_documents(query: str, candidates: List[tuple]) -> tuple:
    """
    Grade retrieved documents for relevance to the query.

    Returns:
        graded_docs: list of (payload, score, grade) triples
        overall_grade: "RELEVANT" | "PARTIALLY_RELEVANT" | "IRRELEVANT"

    Grading logic (per CRAG paper):
      - RELEVANT: score > threshold AND token overlap > 0
      - PARTIAL:  score > lower threshold OR has token overlap
      - IRRELEVANT: fails both
    """
    query_tokens = set(tokenize(normalize_arabic(query)))
    graded = []

    for payload, score in candidates:
        text_tokens = set(tokenize(normalize_arabic(payload.get("text", ""))))
        overlap = len(query_tokens & text_tokens)

        if score >= CRAG_RELEVANT_SCORE and overlap >= MIN_TOKEN_OVERLAP:
            grade = "RELEVANT"
        elif score >= CRAG_PARTIAL_SCORE or overlap >= MIN_TOKEN_OVERLAP:
            grade = "PARTIAL"
        else:
            grade = "IRRELEVANT"

        graded.append((payload, score, grade))

    relevant   = sum(1 for _, _, g in graded if g == "RELEVANT")
    partial    = sum(1 for _, _, g in graded if g == "PARTIAL")

    if relevant >= 2:
        overall = "RELEVANT"
    elif relevant + partial >= 1:
        overall = "PARTIALLY_RELEVANT"
    else:
        overall = "IRRELEVANT"

    return graded, overall


def crag_rewrite_query(original_query: str, attempt: int) -> str:
    """
    Query rewriting for CRAG retry.
    Production: LLM-based query decomposition.
    MVP: progressively broader keyword extraction.
    """
    tokens = tokenize(normalize_arabic(original_query))
    if attempt == 1:
        # First retry: remove stop-words and search with content words only
        STOP = {"في", "من", "على", "ان", "الى", "مع", "عن", "لا", "ما", "كان"}
        content = [t for t in tokens if t not in STOP]
        return " ".join(content) if content else original_query
    else:
        # Second retry: use only the longest token (most distinctive word)
        return max(tokens, key=len) if tokens else original_query


# ─────────────────────────────────────────────
# Self-RAG — Reflection / Grounding Verifier
# Architecture §B.2.5 — Evaluation stage.
# Verifies the generated response is grounded
# in the retrieved context (not hallucinated).
# ─────────────────────────────────────────────

def self_rag_reflect(response: str, results: List[dict]) -> str:
    """
    Self-RAG reflection: check that the response is grounded.

    Returns:
        "GROUNDED"           — response cites verified manuscript sources
        "HALLUCINATION_RISK" — response is long but cites no retrievable source
        "NOT_SUPPORTED"      — no results to ground the response

    Production: LLM judges whether each claim in the response
    is supported by a retrieved passage (using NLI or self-critique prompt).
    MVP: structural citation check.
    """
    if not results:
        return "NOT_SUPPORTED"

    # Collect citation anchors from actual retrieved payloads
    valid_anchors = set()
    for entry in results:
        meta = entry["payload"].get("metadata", {})
        vol  = meta.get("source_volume", "")
        page = meta.get("source_page", "")
        if vol:
            valid_anchors.add(vol)
        if page:
            valid_anchors.add(str(page))

    # Check response references at least one real anchor
    anchors_found = sum(1 for anchor in valid_anchors if anchor in response)

    if anchors_found >= 1:
        return "GROUNDED"
    elif len(response) > 200:
        # Long response with no traceable citation — hallucination risk
        return "HALLUCINATION_RISK"
    else:
        return "NOT_SUPPORTED"


# ─────────────────────────────────────────────
# Metadata Self-Query Filter
# Architecture: PostgreSQL metadata filtering.
# Allows structured filtering by poet, volume,
# region, confidence level before vector search.
# ─────────────────────────────────────────────

def metadata_filter(
    candidates: List[tuple],
    poet: Optional[str] = None,
    volume: Optional[str] = None,
    region: Optional[str] = None,
    min_confidence: float = 0.0,
    chunk_level: Optional[int] = None,
) -> List[tuple]:
    """
    Self-Query: apply structured metadata filters to retrieval candidates.
    Simulates PostgreSQL WHERE clause filtering on the poem metadata store.

    In production this runs BEFORE vector search via Qdrant's payload filter API,
    reducing the search space before ANN lookup.
    """
    filtered = []
    for payload, score in candidates:
        meta = payload.get("metadata", {})

        if poet and poet.lower() not in meta.get("poet", "").lower():
            continue
        if volume and volume not in meta.get("source_volume", ""):
            continue
        if region and region != meta.get("region", "unknown"):
            continue
        if meta.get("confidence_score", 1.0) < min_confidence:
            continue
        if chunk_level is not None and payload.get("chunk_level") != chunk_level:
            continue

        filtered.append((payload, score))
    return filtered


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
        self.memory = ConversationMemory()

    def query(
        self,
        user_query: str,
        top_k: int = 5,
        level_filter: Optional[int] = None,
        # Self-Query metadata filters (§B.2.5 PostgreSQL filtering)
        filter_poet: Optional[str] = None,
        filter_volume: Optional[str] = None,
        filter_region: Optional[str] = None,
        filter_min_confidence: float = 0.0,
    ) -> dict:
        """
        Fatat Al Arab — full LangGraph-compatible 6-node pipeline.

        Node 1 — hyde:      HyDE query expansion (dialectal synonym enrichment)
        Node 2 — retrieve:  Hybrid BM25 + Dense → RRF fusion
        Node 3 — grade:     CRAG document grading (Detector stage)
        Node 4 — rerank:    Cross-encoder re-ranking + contextual expansion
        Node 5 — generate:  Constrained response generation with citation injection
        Node 6 — reflect:   Self-RAG grounding verification (Evaluation stage)

        Fallback chain: IRRELEVANT retrieval → query rewrite → retry → anchor search.
        """

        # ── Initialise LangGraph state ───────────────────────────────
        state: RAGState = {
            "query": self.memory.resolve_query(user_query),
            "hyde_query": "",
            "candidates": [],
            "graded_docs": [],
            "retrieval_grade": "",
            "rewrite_count": 0,
            "expanded_results": [],
            "response": "",
            "self_rag_verdict": "",
            "fallback_triggered": False,
            "conversation_history": self.memory.history,
        }

        # ╔══════════════════════════════════════════════════╗
        # ║  NODE 1 — HyDE: Hypothetical Document Embedding ║
        # ╚══════════════════════════════════════════════════╝
        state["hyde_query"] = hyde_expand(state["query"])

        MAX_CRAG_RETRIES = 2
        while state["rewrite_count"] <= MAX_CRAG_RETRIES:

            # ╔══════════════════════════════════════════════╗
            # ║  NODE 2 — RETRIEVE: Hybrid BM25 + Dense RRF ║
            # ╚══════════════════════════════════════════════╝
            raw_candidates = self.store.hybrid_search_rrf(
                state["hyde_query"], top_k=top_k * 4
            )

            # Self-Query metadata filter (simulates Qdrant payload filter)
            if any([filter_poet, filter_volume, filter_region, filter_min_confidence]):
                raw_candidates = metadata_filter(
                    raw_candidates,
                    poet=filter_poet,
                    volume=filter_volume,
                    region=filter_region,
                    min_confidence=filter_min_confidence,
                )

            if level_filter:
                raw_candidates = self.store.filter_by_level(raw_candidates, level_filter)

            state["candidates"] = raw_candidates

            # ╔══════════════════════════════════════════════╗
            # ║  NODE 3 — GRADE: CRAG Document Grading      ║
            # ╚══════════════════════════════════════════════╝
            graded_docs, overall_grade = crag_grade_documents(
                state["query"], state["candidates"]
            )
            state["graded_docs"] = graded_docs
            state["retrieval_grade"] = overall_grade

            # ── CRAG routing decision ────────────────────────────────
            # RELEVANT/PARTIALLY_RELEVANT → proceed to reranking
            # IRRELEVANT → rewrite query and retry (up to MAX_CRAG_RETRIES)
            if overall_grade != "IRRELEVANT":
                break

            if state["rewrite_count"] < MAX_CRAG_RETRIES:
                state["rewrite_count"] += 1
                rewritten = crag_rewrite_query(state["query"], state["rewrite_count"])
                state["hyde_query"] = hyde_expand(rewritten)
            else:
                # All retries exhausted → trigger anchor-registry fallback
                state["fallback_triggered"] = True
                break

        # ╔══════════════════════════════════════════════════╗
        # ║  FALLBACK: Anchor Registry Search               ║
        # ║  Triggered when CRAG grades all docs IRRELEVANT ║
        # ╚══════════════════════════════════════════════════╝
        if state["fallback_triggered"] or not state["candidates"]:
            anchor_hits = self.anchor_search(state["query"])
            fallback_response = self._format_anchor_fallback(state["query"], anchor_hits)
            result = {
                "query": user_query,
                "results": [],
                "response": fallback_response,
                "total_candidates": 0,
                "retrieval_grade": "IRRELEVANT",
                "self_rag_verdict": "NOT_SUPPORTED",
                "fallback_triggered": True,
                "rewrite_count": state["rewrite_count"],
                "retrieval_pipeline": "BM25+Dense→RRF→CRAG(FAIL)→AnchorFallback",
            }
            self.memory.add(user_query, fallback_response)
            return result

        # ╔══════════════════════════════════════════════════╗
        # ║  NODE 4 — RERANK + CONTEXTUAL EXPANSION         ║
        # ╚══════════════════════════════════════════════════╝
        # Keep only RELEVANT and PARTIAL docs after CRAG
        passing = [(p, s) for p, s, g in state["graded_docs"] if g != "IRRELEVANT"]

        # Cross-encoder re-ranking (MVP: exact-token match bonus)
        query_tokens = set(tokenize(normalize_arabic(state["query"])))
        reranked = []
        for payload, rrf_score in passing:
            text_tokens = set(tokenize(normalize_arabic(payload["text"])))
            exact_match_bonus = len(query_tokens & text_tokens) * 0.05
            reranked.append((payload, rrf_score + exact_match_bonus))
        reranked.sort(key=lambda x: x[1], reverse=True)

        top_results = reranked[:top_k]

        # Contextual expansion: Level 1 verse → Level 2 stanza group
        expanded_results = []
        for payload, score in top_results:
            entry = {
                "payload": payload,
                "score": score,
                "citation": format_citation(payload),
                "crag_grade": next(
                    (g for p, s, g in state["graded_docs"] if p["chunk_id"] == payload["chunk_id"]),
                    "UNKNOWN"
                ),
            }
            if payload["chunk_level"] == 1:
                parent_poem_id = payload["metadata"]["poem_id"]
                stanza_num = payload["metadata"].get("stanza_num", 0)
                group_chunk = self._find_parent_group(parent_poem_id, stanza_num)
                if group_chunk:
                    entry["expanded_context"] = group_chunk["text"]
                    entry["expanded_citation"] = format_citation(group_chunk)
            expanded_results.append(entry)

        state["expanded_results"] = expanded_results

        # ╔══════════════════════════════════════════════════╗
        # ║  NODE 5 — GENERATE: Constrained Response        ║
        # ╚══════════════════════════════════════════════════╝
        response = self._generate_response(state["query"], expanded_results)
        state["response"] = response

        # ╔══════════════════════════════════════════════════╗
        # ║  NODE 6 — REFLECT: Self-RAG Grounding Check     ║
        # ╚══════════════════════════════════════════════════╝
        verdict = self_rag_reflect(response, expanded_results)
        state["self_rag_verdict"] = verdict

        # Append a governance footer if hallucination risk is detected
        if verdict == "HALLUCINATION_RISK":
            response += (
                "\n\n⚠️ تحذير الحوكمة: لم يتم التحقق من ارتباط كامل هذه النتائج بالمخطوط الأصلي. "
                "يُرجى مراجعة الصور المرفقة."
            )
            state["response"] = response

        self.memory.add(user_query, response)

        return {
            "query": user_query,
            "results": expanded_results,
            "response": response,
            "total_candidates": len(state["candidates"]),
            "retrieval_grade": state["retrieval_grade"],
            "self_rag_verdict": state["self_rag_verdict"],
            "fallback_triggered": False,
            "rewrite_count": state["rewrite_count"],
            "retrieval_pipeline": "HyDE→BM25+Dense→RRF→CRAG→Rerank→Expand→Generate→SelfRAG",
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
        poet_results = self.anchors.lookup_by_poet(query)
        if poet_results:
            results.extend(poet_results)
        matla_results = self.anchors.lookup_by_matla(query)
        for r in matla_results:
            if r not in results:
                results.append(r)
        return results

    def _format_anchor_fallback(self, query: str, anchor_hits: list) -> str:
        """
        Fallback response when CRAG grades all vector-search results IRRELEVANT
        after MAX_CRAG_RETRIES attempts. Falls back to TOC anchor registry.
        This is the hard governance floor — the system never returns an empty
        or hallucinated response; it always has a structured fallback.
        """
        lines = [
            f"🔍 البحث في فهرس المخطوطات (Anchor Registry Fallback)",
            f"الاستعلام: {query}",
            "─" * 60,
        ]
        if anchor_hits:
            lines.append(f"وُجدت {len(anchor_hits)} نتيجة في فهرس المخطوطات (TOC Anchors):\n")
            for i, hit in enumerate(anchor_hits[:5], 1):
                poet     = hit.get("poet_name", "غير معروف")
                page     = hit.get("page_number", "?")
                matla    = hit.get("matla_text", "")[:80]
                volume   = hit.get("source_volume", "")
                occasion = hit.get("occasion", "")
                lines.append(f"[{i}] الشاعر: {poet} | المجلد: {volume} | صفحة: {page}")
                if matla:
                    lines.append(f"     المطلع: {matla}")
                if occasion:
                    lines.append(f"     المناسبة: {occasion}")
        else:
            lines.append(
                "لا يوجد في المخطوطات المرقَّمة ما يجيب هذا السؤال مباشرةً. "
                "يُنصح بمراجعة المخطوط الأصلي أو توسيع نطاق الرقمنة."
            )
        lines += [
            "\n─" * 60,
            "ℹ️  هذه النتائج مستخرجة من فهرس المحتويات (TOC) — وليس من نص الأبيات.",
            "    للبحث النصي الكامل، يجب استكمال رقمنة المخطوطات.",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# LangGraph Graph Builder
# Compiles the 6-node state machine when LangGraph
# is installed. Falls back to FatatAlArabAgent.query()
# when running in MVP mode without the dependency.
#
# Graph edges:
#   START → hyde → retrieve → grade →
#     IRRELEVANT + retries_left → rewrite → retrieve
#     IRRELEVANT + no retries   → anchor_fallback → END
#     RELEVANT / PARTIAL        → rerank → generate → reflect → END
# ─────────────────────────────────────────────

def build_langgraph(agent: "FatatAlArabAgent"):
    """
    Build and compile the Fatat Al Arab LangGraph state machine.
    Returns a compiled graph if LangGraph is available, else None.
    """
    if not LANGGRAPH_AVAILABLE:
        print("ℹ️  LangGraph not installed — using FatatAlArabAgent.query() directly.")
        print("   Install with: pip install langgraph")
        return None

    def node_hyde(state: RAGState) -> RAGState:
        state["hyde_query"] = hyde_expand(state["query"])
        return state

    def node_retrieve(state: RAGState) -> RAGState:
        raw = agent.store.hybrid_search_rrf(state["hyde_query"], top_k=20)
        state["candidates"] = raw
        return state

    def node_grade(state: RAGState) -> RAGState:
        graded, overall = crag_grade_documents(state["query"], state["candidates"])
        state["graded_docs"] = graded
        state["retrieval_grade"] = overall
        return state

    def node_rewrite(state: RAGState) -> RAGState:
        state["rewrite_count"] += 1
        rewritten = crag_rewrite_query(state["query"], state["rewrite_count"])
        state["hyde_query"] = hyde_expand(rewritten)
        return state

    def node_anchor_fallback(state: RAGState) -> RAGState:
        hits = agent.anchor_search(state["query"])
        state["response"] = agent._format_anchor_fallback(state["query"], hits)
        state["fallback_triggered"] = True
        state["self_rag_verdict"] = "NOT_SUPPORTED"
        return state

    def node_generate(state: RAGState) -> RAGState:
        # Build expanded results from graded docs
        passing = [(p, s) for p, s, g in state["graded_docs"] if g != "IRRELEVANT"]
        expanded = []
        for payload, score in passing[:5]:
            entry = {"payload": payload, "score": score, "citation": format_citation(payload)}
            if payload["chunk_level"] == 1:
                pid = payload["metadata"]["poem_id"]
                snum = payload["metadata"].get("stanza_num", 0)
                grp = agent._find_parent_group(pid, snum)
                if grp:
                    entry["expanded_context"] = grp["text"]
            expanded.append(entry)
        state["expanded_results"] = expanded
        state["response"] = agent._generate_response(state["query"], expanded)
        return state

    def node_reflect(state: RAGState) -> RAGState:
        verdict = self_rag_reflect(state["response"], state["expanded_results"])
        state["self_rag_verdict"] = verdict
        if verdict == "HALLUCINATION_RISK":
            state["response"] += (
                "\n\n⚠️ تحذير الحوكمة: النتائج غير مُتحقق منها بالكامل. "
                "راجع صور المخطوط الأصلي."
            )
        return state

    def route_after_grade(state: RAGState) -> str:
        """CRAG routing: relevant → generate | irrelevant + retries → rewrite | else → fallback."""
        if state["retrieval_grade"] != "IRRELEVANT":
            return "generate"
        if state["rewrite_count"] < 2:
            return "rewrite"
        return "anchor_fallback"

    graph = StateGraph(RAGState)
    graph.add_node("hyde",            node_hyde)
    graph.add_node("retrieve",        node_retrieve)
    graph.add_node("grade",           node_grade)
    graph.add_node("rewrite",         node_rewrite)
    graph.add_node("anchor_fallback", node_anchor_fallback)
    graph.add_node("generate",        node_generate)
    graph.add_node("reflect",         node_reflect)

    graph.set_entry_point("hyde")
    graph.add_edge("hyde",     "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {
        "generate":        "generate",
        "rewrite":         "rewrite",
        "anchor_fallback": "anchor_fallback",
    })
    graph.add_edge("rewrite",         "retrieve")
    graph.add_edge("anchor_fallback", END)
    graph.add_edge("generate",        "reflect")
    graph.add_edge("reflect",         END)

    return graph.compile()


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

    # ── LangGraph Compilation ────────────────────────────────────────
    graph = build_langgraph(agent)
    if graph:
        print("✓ LangGraph state machine compiled — 6 nodes, CRAG routing active.")
    else:
        print("✓ Fatat Al Arab Agent initialized (MVP mode — no LangGraph).")

    print("\n✓ Fatat Al Arab Agent initialized — ready for queries.")
    return agent


# ─────────────────────────────────────────────
# Demo Queries (Initial Implementation Demo)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "ground_truth"

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

        # Pipeline diagnostics (maps to 4-Stage rubric)
        print(f"\n  📊 Pipeline Diagnostics:")
        print(f"     CRAG Grade:      {result.get('retrieval_grade', 'N/A')}")
        print(f"     Self-RAG:        {result.get('self_rag_verdict', 'N/A')}")
        print(f"     Rewrite Count:   {result.get('rewrite_count', 0)}")
        print(f"     Fallback:        {result.get('fallback_triggered', False)}")
        print(f"     Pipeline:        {result.get('retrieval_pipeline', '')}")

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
            "retrieval_grade": result.get("retrieval_grade", ""),
            "self_rag_verdict": result.get("self_rag_verdict", ""),
            "rewrite_count": result.get("rewrite_count", 0),
            "fallback_triggered": result.get("fallback_triggered", False),
            "top_result_text": result["results"][0]["payload"]["text"][:100] if result["results"] else "",
            "citation": result["results"][0]["citation"] if result["results"] else "",
        })

    # Save demo output for report
    demo_out_path = OUTPUT_DIR / "demo_query_results.json"
    with open(demo_out_path, "w", encoding="utf-8") as f:
        json.dump(demo_output, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Demo results saved → {demo_out_path}")
