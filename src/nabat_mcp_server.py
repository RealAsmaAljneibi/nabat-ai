"""
src/nabat_mcp_server.py
========================
Why this file exists: Exposes NABAT-AI as a Model Context Protocol (MCP) server
so Claude Desktop and Claude Code can query the poetry corpus as a first-class
tool — no Streamlit required. Any MCP client (Claude Desktop, other agents) can
call search_nabati_poetry, get_corpus_stats, get_poet_bio, inspect_qdrant, and
enrich_poet as structured tools.

Register in Claude Desktop:  ~/Library/Application Support/Claude/claude_desktop_config.json
Register in Claude Code:      .claude/settings.json  (already done — see mcpServers)

Run locally to test:
    PYTHONPATH=src LLM_PROVIDER=stub python src/nabat_mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("nabat-ai")


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_nabati_poetry",
            description=(
                "Search 4,031 Khaleeji Nabati poetry entries across manuscripts, oral tradition, and online sources. "
                "Accepts Arabic or English queries. Runs the full NABAT-AI pipeline "
                "(intent routing → retrieval → CRAG grading → synthesis → Self-RAG). "
                "Returns a grounded bilingual answer with citations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Your question in Arabic or English",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_corpus_stats",
            description=(
                "Return a summary of the NABAT-AI corpus: total bayts, poets, "
                "manuscripts, pages indexed, and time span. Deterministic — no LLM."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_poet_bio",
            description=(
                "Look up biographical information for a named Gulf Nabati poet. "
                "Checks the local poets_bio.json first; falls back to Wikipedia "
                "if the name is not found locally."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "poet_name": {
                        "type": "string",
                        "description": "Poet name in Arabic (preferred) or English",
                    },
                },
                "required": ["poet_name"],
            },
        ),
        Tool(
            name="inspect_qdrant",
            description=(
                "Direct vector-index inspection tool for development. "
                "Runs a semantic search against the Qdrant index and returns "
                "raw chunk metadata (scores, levels, anchor IDs, genres). "
                "Useful for debugging retrieval without running the full pipeline."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (Arabic or English)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 20)",
                        "default": 5,
                    },
                    "level": {
                        "type": "string",
                        "description": "Filter by chunk level: verse, group, poem, manuscript, poet, genre, emotion, reference",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="enrich_poet",
            description=(
                "Fetch biographical data for a poet from Wikipedia (Arabic first, "
                "then English) or Brave Search. Returns birth year, region, and a "
                "bio snippet. Results are cached locally so the same poet is never "
                "fetched twice."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "poet_name": {
                        "type": "string",
                        "description": "Poet name in Arabic",
                    },
                },
                "required": ["poet_name"],
            },
        ),
        # ── M9 inter-agent consultation tools (mirror of external_tools.py) ──
        Tool(
            name="hafiz_voice_blend",
            description=(
                "Two-call voice retrieval blend for a target poet: (1) any-topic "
                "voice samples to teach style, (2) topical samples to ground "
                "occasion. Returns up to 20 deduplicated exemplars with metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target_poet": {"type": "string"},
                    "occasion":    {"type": "string"},
                    "genre":       {"type": "string"},
                },
                "required": ["target_poet"],
            },
        ),
        Tool(
            name="grade_ajuz_candidates",
            description=(
                "Al-Muqayyim batched grading of ajuz (second hemistich) "
                "candidates. Returns each candidate with meter / rhyme / "
                "authenticity scores, sorted by overall."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sadr":       {"type": "string", "description": "First hemistich"},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"ajuz": {"type": "string"}}},
                    },
                },
                "required": ["sadr", "candidates"],
            },
        ),
        Tool(
            name="voice_fingerprint",
            description=(
                "Compact voice fingerprint for a poet: opening phrases, "
                "signature imagery, key vocabulary, dialect register. "
                "Useful when scaffolding 'in poet X's voice'."
            ),
            inputSchema={
                "type": "object",
                "properties": {"target_poet": {"type": "string"}},
                "required": ["target_poet"],
            },
        ),
        Tool(
            name="check_intertextuality",
            description=(
                "Boolean intertextuality check: does this hemistich appear "
                "verbatim in the indexed corpus? Returns ONLY {in_corpus, "
                "attributed_poet, manuscript_short_key} — never verse content. "
                "Designed for critique workflows where corpus content must NOT "
                "leak into the framing."
            ),
            inputSchema={
                "type": "object",
                "properties": {"verse_line": {"type": "string"}},
                "required": ["verse_line"],
            },
        ),
    ]


# ── Tool implementations ──────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── search_nabati_poetry ──────────────────────────────────────────────────
    if name == "search_nabati_poetry":
        try:
            from fatat_al_arab.orchestrator import run
            result = run(arguments["query"])
            response = result.get("final_response") or "No result returned."
            crag     = result.get("crag_verdict", "—")
            self_rag = result.get("self_rag_verdict", "—")
            footer   = f"\n\n---\n*CRAG: {crag} · Self-RAG: {self_rag}*"
            return [TextContent(type="text", text=response + footer)]
        except Exception as exc:
            return [TextContent(type="text", text=f"Pipeline error: {exc}")]

    # ── get_corpus_stats ──────────────────────────────────────────────────────
    if name == "get_corpus_stats":
        try:
            from al_nassikh import corpus_stats
            s = corpus_stats.corpus_summary()
            text = json.dumps(s, ensure_ascii=False, indent=2)
            return [TextContent(type="text", text=text)]
        except Exception as exc:
            return [TextContent(type="text", text=f"Stats error: {exc}")]

    # ── get_poet_bio ──────────────────────────────────────────────────────────
    if name == "get_poet_bio":
        poet_name = arguments["poet_name"].strip()
        try:
            bio_path = _REPO / "data" / "ground_truth" / "poets_bio.json"
            bios     = json.loads(bio_path.read_text(encoding="utf-8"))
            match    = next(
                (b for b in bios if poet_name in b.get("poet_name", "")), None
            )
            if match:
                text = (
                    f"**{match['poet_name']}**\n\n"
                    f"{match.get('bio_en', '')}\n\n---\n\n{match.get('bio_ar', '')}\n\n"
                    f"Sources: {', '.join(match.get('sources', []))}"
                )
                return [TextContent(type="text", text=text)]
        except Exception:
            pass

        # Wikipedia fallback
        try:
            from al_nassikh.poet_enrichment import PoetEnrichmentClient
            client = PoetEnrichmentClient()
            result = client.enrich(poet_name)
            if result:
                text = (
                    f"**{poet_name}** *(enriched via {result['enriched_via']})*\n\n"
                    f"{result.get('bio_ar') or result.get('bio_en', 'No bio text.')}\n\n"
                    f"Birth year: {result.get('birth_year_approx', '—')}  "
                    f"Region: {result.get('region', '—')}\n"
                    f"Source: {', '.join(result.get('sources', []))}"
                )
                return [TextContent(type="text", text=text)]
        except Exception as exc:
            pass

        return [TextContent(
            type="text",
            text=f"No bio found for '{poet_name}' in the local registry or Wikipedia.",
        )]

    # ── inspect_qdrant ────────────────────────────────────────────────────────
    if name == "inspect_qdrant":
        query  = arguments["query"]
        top_k  = min(int(arguments.get("top_k", 5)), 20)
        level  = arguments.get("level")
        try:
            from fatat_al_arab.embed import embed_text
            from fatat_al_arab.index import load_index
            from fatat_al_arab.retrievers.dense import DenseRetriever

            qdrant_path = _REPO / "data" / "qdrant"
            collection  = load_index(str(qdrant_path))
            retriever   = DenseRetriever(collection)
            results     = retriever.retrieve(query, top_k=top_k * 2)

            if level:
                results = [r for r in results if r.get("level") == level]
            results = results[:top_k]

            lines = [f"Top {len(results)} chunks for: '{query}'", ""]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. [{r.get('level','?')}] score={r.get('score',0):.4f}  "
                    f"anchor={r.get('anchor_id','?')}  "
                    f"poet={r.get('poet_name','?')}  "
                    f"genre={r.get('genre','?')}"
                )
                text_preview = (r.get("text") or "")[:80].replace("\n", " ")
                lines.append(f"   text: {text_preview}…")
            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as exc:
            return [TextContent(type="text", text=f"Qdrant inspection error: {exc}")]

    # ── enrich_poet ───────────────────────────────────────────────────────────
    if name == "enrich_poet":
        poet_name = arguments["poet_name"].strip()
        try:
            from al_nassikh.poet_enrichment import PoetEnrichmentClient, is_valid_poet_name
            if not is_valid_poet_name(poet_name):
                return [TextContent(type="text", text=f"'{poet_name}' looks like an OCR artifact — skipped.")]
            client = PoetEnrichmentClient()
            result = client.enrich(poet_name)
            if result:
                text = (
                    f"**{poet_name}** — enriched via {result['enriched_via']}\n\n"
                    f"{result.get('bio_ar') or result.get('bio_en', '')}\n\n"
                    f"Birth year (approx): {result.get('birth_year_approx', '—')}\n"
                    f"Region: {result.get('region', '—')}\n"
                    f"Source: {', '.join(result.get('sources', []))}"
                )
            else:
                text = f"No Wikipedia or Brave Search result found for '{poet_name}'."
            return [TextContent(type="text", text=text)]
        except Exception as exc:
            return [TextContent(type="text", text=f"Enrichment error: {exc}")]

    # ── M9: hafiz_voice_blend ─────────────────────────────────────────────────
    if name == "hafiz_voice_blend":
        try:
            from creative_poet.external_tools import hafiz_retrieve_voice_blend
            exemplars = hafiz_retrieve_voice_blend(
                target_poet=arguments["target_poet"],
                occasion=arguments.get("occasion"),
                genre=arguments.get("genre"),
            )
            text = json.dumps(
                [{"text": (e.get("matla_text") or e.get("text", ""))[:200],
                  "poet": e.get("poet_name", ""),
                  "anchor_id": e.get("anchor_id", "")}
                 for e in exemplars],
                ensure_ascii=False, indent=2,
            )
            return [TextContent(type="text", text=text)]
        except Exception as exc:
            return [TextContent(type="text", text=f"Voice blend error: {exc}")]

    # ── M9: grade_ajuz_candidates ─────────────────────────────────────────────
    if name == "grade_ajuz_candidates":
        try:
            from creative_poet.external_tools import muqayyim_grade_ajuz_candidates
            ranked = muqayyim_grade_ajuz_candidates(
                sadr=arguments["sadr"],
                candidates=arguments["candidates"],
            )
            return [TextContent(type="text", text=json.dumps(ranked, ensure_ascii=False, indent=2))]
        except Exception as exc:
            return [TextContent(type="text", text=f"Grading error: {exc}")]

    # ── M9: voice_fingerprint ─────────────────────────────────────────────────
    if name == "voice_fingerprint":
        try:
            from creative_poet.external_tools import hafiz_voice_fingerprint
            fp = hafiz_voice_fingerprint(arguments["target_poet"])
            return [TextContent(type="text", text=json.dumps(fp, ensure_ascii=False, indent=2))]
        except Exception as exc:
            return [TextContent(type="text", text=f"Fingerprint error: {exc}")]

    # ── M9: check_intertextuality ─────────────────────────────────────────────
    if name == "check_intertextuality":
        try:
            from creative_poet.external_tools import nassikh_check_intertextuality
            check = nassikh_check_intertextuality(arguments["verse_line"])
            return [TextContent(type="text", text=json.dumps(check, ensure_ascii=False, indent=2))]
        except Exception as exc:
            return [TextContent(type="text", text=f"Intertext check error: {exc}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Entry point ───────────────────────────────────────────────────────────────
# Why async with: stdio_server is a context manager in MCP SDK 1.x, not a
# plain coroutine. It sets up stdin/stdout streams, then hands them to app.run.

async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(_main())
