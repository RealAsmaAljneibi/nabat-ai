# NABAT-AI — command runner
# Install: brew install just

# List available recipes
default:
    @just --list

# Install Python dependencies
install:
    pip install -r requirements.txt

# Copy env template (won't overwrite existing .env)
env:
    cp -n .env.example .env

# Build / rebuild the Qdrant vector index
index:
    PYTHONPATH=src python scripts/rebuild_index.py \
        --registry data/ground_truth/anchor_registry_full_enriched.json \
        --force

# Launch the Streamlit app
app:
    streamlit run app/streamlit_app.py

# Run the evaluation harness (stub LLM — no API key needed)
eval:
    LLM_PROVIDER=stub PYTHONPATH=src python scripts/evaluate.py

# Run all tests
test:
    PYTHONPATH=src pytest tests/ -v

# Run a single test file — usage: just test-file tests/test_rrf.py
test-file FILE:
    PYTHONPATH=src pytest {{ FILE }} -v

# Enrich genre + emotion tags over all anchors
enrich:
    PYTHONPATH=src python scripts/enrich_genre_heuristic.py

# Dry-run genre enrichment (no writes)
enrich-dry:
    PYTHONPATH=src python scripts/enrich_genre_heuristic.py --dry-run

# Ingest Phase 1/2/3 PAGE-XML into anchor registry
ingest:
    PYTHONPATH=src python scripts/ingest_phases_123.py

# Dry-run ingest (no writes)
ingest-dry:
    PYTHONPATH=src python scripts/ingest_phases_123.py --dry-run

# Full setup from scratch: install → env → index
setup: install env index
