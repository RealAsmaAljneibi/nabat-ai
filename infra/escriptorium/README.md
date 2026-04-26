# eScriptorium — Self-Hosted Setup

Why self-hosted: §1.6.3 of the architecture requires eScriptorium to run locally
so the Streamlit Archive Manager (Tab A) can drive it through its REST API without
the archivist opening a separate browser window. The Streamlit tab iframes
eScriptorium's own annotation editor for the manual work, then pulls the PAGE-XML
back automatically.

## One-command bring-up

```bash
cd infra/escriptorium

# 1. Configure
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and SECRET_KEY

# 2. Start services
docker compose up -d

# 3. Initialise database (first run only)
docker compose exec app python manage.py migrate

# 4. Create admin user (first run only)
docker compose exec app python manage.py createsuperuser

# 5. Get API token (copy into root .env as ESCR_API_TOKEN)
docker compose exec app python manage.py drf_create_token admin
```

eScriptorium will be accessible at **http://localhost:8080** (or the port set in `ESCR_PORT`).

## Pre-seed Khaleeji zone labels

After bring-up, run this once to add the five Khaleeji zone types to eScriptorium's
ontology (so they appear as named options in the annotator's zone-label dropdown):

```python
from al_nassikh.escriptorium_client import ensure_project, create_document, set_ontology

project = ensure_project("NABAT-AI Phase 1")
doc = create_document(project["project_id"], "manuscript07")
set_ontology(doc["document_id"])   # seeds TWO_COLUMN_POETRY, PROSE_ATTRIBUTION, etc.
```

## Stop / restart

```bash
docker compose down          # stop (data preserved in data/escriptorium/)
docker compose up -d         # restart
docker compose logs -f app   # follow Django logs
docker compose logs -f celery  # follow Kraken worker logs
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `EScriptoriumUnavailableError: ESCR_API_TOKEN is not set` | Run step 5 above and add to root `.env` |
| Port 8080 already in use | Change `ESCR_PORT=8081` in `.env` and `ESCR_BASE_URL=http://localhost:8081` in root `.env` |
| Kraken job stuck | Check `docker compose logs -f celery`; restart with `docker compose restart celery` |
| Database connection refused | Check `docker compose ps db`; wait for health check to pass |
