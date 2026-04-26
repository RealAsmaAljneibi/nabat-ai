"""
src/al_nassikh/parser.py
========================
Why this file exists: Al-Nassikh Worker 1 — the PAGE-XML parser is the
first stage of the new-manuscript ingestion pipeline. It converts eScriptorium
PAGE-XML exports into the multi-variant JSON format consumed by the rest of the
pipeline.

Why it is not yet implemented: the 50-page Phase-1 ground truth corpus was
produced manually and is already stored in:
    data/ground_truth/anchor_registry_phase4.json (1,502 entries)
    data/ground_truth/anchor_registry_phase4_enriched.json (+ genre/emotions)

The parser is only needed when NEW manuscripts are added via the eScriptorium
ingest pipeline (Tab B → Archive Manager → Operator Tools). That is planned
for Phase 2 / post-MVP.

Status: PLACEHOLDER — not blocking the current RAG pipeline.

When implementing, this module should expose:
    parse_pagexml(xml_path: str | Path) -> list[dict]
        Parse a single PAGE-XML file → list of verse dicts matching the
        anchor_registry schema (source_row_id, poet_name, matla_text, etc.)

    parse_pagexml_batch(xml_dir: str | Path) -> list[dict]
        Batch variant for a directory of PAGE-XML files.

The Arabic normaliser shared with embed.py lives in embed.py for now;
once this module is implemented, normalise_arabic() should be moved here
and imported from embed.py with:
    from al_nassikh.parser import normalise_arabic

Architecture ref: §2.3 Al-Nassikh Worker 1 ingest pipeline.
"""
# Not yet implemented — see module docstring above.
