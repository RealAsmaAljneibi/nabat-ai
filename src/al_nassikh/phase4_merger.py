"""
src/al_nassikh/phase4_merger.py
================================
Why this file exists: Al-Nassikh Worker 1 — the Phase 4 merger was the script
that built the 1,502-entry TOC dictionary by merging manually transcribed
Kraken line outputs with the digitised TOC (فهرس) pages.

Why it is not implemented here: the Phase 4 merge is COMPLETE. The output
lives at:
    data/ground_truth/anchor_registry_phase4.json     (1,502 anchors)
    data/ground_truth/anchor_registry_phase4_enriched.json  (+ M2d genre)

The merge was done in a Jupyter notebook during data preparation and is not
needed to run the live system. This file is kept as a placeholder so the
CLAUDE.md directory map and architecture references remain consistent.

Status: PLACEHOLDER — merge complete; re-running is a one-off operation
if the TOC source data changes.

When implementing (e.g. for Phase 2 manuscripts), this module should expose:
    merge_kraken_and_toc(kraken_dir, toc_csv, output_path) -> list[dict]
        Merge Kraken HTR line transcriptions with the TOC spreadsheet,
        apply cross-links, write anchor_registry JSON.

Architecture ref: §2.3 Al-Nassikh Worker 1, Phase 4 TOC dictionary.
"""
# Merge complete — see module docstring above.
