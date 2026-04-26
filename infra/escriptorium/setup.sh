#!/usr/bin/env bash
# infra/escriptorium/setup.sh
# ============================
# One-command eScriptorium setup for NABAT-AI.
# Run this ONCE the first time. Subsequent starts: docker compose up -d
#
# Usage:
#   cd infra/escriptorium
#   bash setup.sh
#
# What it does:
#   1. Clones eScriptorium source (GitLab) into ./escriptorium-src
#   2. Creates .env from .env.example with a generated SECRET_KEY
#   3. Builds the Docker image (~10-15 min first time)
#   4. Starts all services (db, redis, app, celery, nginx)
#   5. Runs Django migrations
#   6. Creates an admin superuser (admin / nabatai2024)
#   7. Mints an API token and writes it to .env and ../../.env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         NABAT-AI — eScriptorium Setup                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Clone eScriptorium source ────────────────────────────────────────
if [ ! -d "escriptorium-src" ]; then
  echo "▶ Cloning eScriptorium from GitLab (this is a one-time download ~60 MB)…"
  git clone --depth 1 https://gitlab.com/scripta/escriptorium.git escriptorium-src
  echo "✅ Source cloned."
else
  echo "✅ escriptorium-src already exists — skipping clone."
fi

# ── Step 2: Create .env ───────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "▶ Generating .env from template…"
  cp .env.example .env
  # Generate a random SECRET_KEY
  SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")
  # Replace the placeholder value (cross-platform sed)
  sed -i.bak "s|change-me-in-production|$SECRET|g" .env && rm -f .env.bak
  echo "✅ .env created with generated SECRET_KEY."
else
  echo "✅ .env already exists — skipping."
fi

# ── Step 3: Build + start services ───────────────────────────────────────────
echo ""
echo "▶ Building Docker image and starting services…"
echo "  (First build takes 10–15 minutes — grab a coffee ☕)"
echo ""
docker compose up --build -d

# ── Step 4: Wait for app to be ready ─────────────────────────────────────────
echo ""
echo "▶ Waiting for the app to respond at http://localhost:8080…"
echo "  (amd64 image running under Rosetta 2 on Apple Silicon — may take 60–90s)"
MAX_WAIT=180
WAITED=0
until curl -sf http://localhost:8080/login/ -o /dev/null 2>/dev/null; do
  if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠️  App did not respond in ${MAX_WAIT}s."
    echo "   Check logs: docker compose logs app --tail 40"
    break
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  printf "  Still waiting… (%ds)\r" "$WAITED"
done
if curl -sf http://localhost:8080/login/ -o /dev/null 2>/dev/null; then
  echo "✅ App is up."
fi

echo ""
echo "▶ Creating admin superuser (username: admin, password: nabatai2024)…"
docker compose exec -T app python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@nabat.ai', 'nabatai2024')
    print('Created admin user.')
else:
    print('Admin user already exists.')
" 2>/dev/null || echo "⚠️  Could not create superuser — may already exist."
echo "✅ Superuser ready (admin / nabatai2024)."

# ── Step 7: Mint API token ────────────────────────────────────────────────────
echo ""
echo "▶ Minting API token for admin…"
TOKEN=$(docker compose exec -T app python manage.py drf_create_token admin 2>/dev/null \
  | grep -oE '[0-9a-f]{40}' | head -1)

if [ -z "$TOKEN" ]; then
  echo "⚠️  Could not extract token automatically."
  echo "    Run manually: docker compose exec app python manage.py drf_create_token admin"
else
  echo "✅ Token: $TOKEN"

  # Write token to infra .env
  if grep -q "^ESCR_API_TOKEN=" .env; then
    sed -i.bak "s|^ESCR_API_TOKEN=.*|ESCR_API_TOKEN=$TOKEN|" .env && rm -f .env.bak
  else
    echo "ESCR_API_TOKEN=$TOKEN" >> .env
  fi

  # Write token + base URL to root .env
  ROOT_ENV="$ROOT_DIR/.env"
  if [ ! -f "$ROOT_ENV" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_ENV" 2>/dev/null || touch "$ROOT_ENV"
  fi
  if grep -q "^ESCR_API_TOKEN=" "$ROOT_ENV"; then
    sed -i.bak "s|^ESCR_API_TOKEN=.*|ESCR_API_TOKEN=$TOKEN|" "$ROOT_ENV" && rm -f "$ROOT_ENV.bak"
  else
    echo "ESCR_API_TOKEN=$TOKEN" >> "$ROOT_ENV"
  fi
  if ! grep -q "^ESCR_BASE_URL=" "$ROOT_ENV"; then
    echo "ESCR_BASE_URL=http://localhost:8080" >> "$ROOT_ENV"
  fi
  echo "✅ Token written to .env and ../../.env"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅  eScriptorium is ready                              ║"
echo "║                                                          ║"
echo "║  URL:      http://localhost:8080                         ║"
echo "║  Username: admin                                         ║"
echo "║  Password: nabatai2024                                   ║"
echo "║                                                          ║"
echo "║  Restart the Streamlit app to pick up the new token.     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
