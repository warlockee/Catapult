#!/bin/bash
set -e

# =============================================================================
# Catapult "Grab and Go" Deployment
# =============================================================================

# 1. Configuration Check
if [ ! -f .env ]; then
    echo "⚠️  .env not found. Creating from default template..."
    cp .env.example .env
    echo "✅ Created .env"
fi

# 2. Set Environment Variables
# Load .env to get custom configurations (if any)
set -a
source .env
set +a

# Default STORAGE_ROOT if not set (Self-contained)
export STORAGE_ROOT="${STORAGE_ROOT:-./storage}"

# 3. Prepare Directories
# Create directories as current user to avoid Root ownership issues from Docker
echo "📂 Preparing storage at ${STORAGE_ROOT}..."
mkdir -p "${STORAGE_ROOT}/artifacts" \
         "${STORAGE_ROOT}/dockerbuild_jobs" \
         "${STORAGE_ROOT}/docker_logs" \
         "${STORAGE_ROOT}/snapshots" \
         "${STORAGE_ROOT}/models" \
         "${STORAGE_ROOT}/deployments" \
         "${STORAGE_ROOT}/deployment_logs" \
         "${STORAGE_ROOT}/benchmark_logs" \
         "infrastructure/nginx/ssl"

# 4. Launch Services
echo "🚀 Starting services..."
docker-compose up -d --build --remove-orphans

# 5. Wait for Backend Health
echo "⏳ Waiting for backend to be healthy..."
max_attempts=30
attempt=0
until docker-compose exec -T backend curl -sf http://localhost:8000/api/health > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ Backend failed to become healthy after ${max_attempts} attempts"
        docker-compose logs backend --tail=50
        exit 1
    fi
    echo "   Attempt $attempt/$max_attempts - waiting..."
    sleep 2
done
echo "✅ Backend is healthy"

# 6. Post-Deployment Setup
echo "🛠  Running migrations..."
docker-compose exec -T backend alembic upgrade head

echo "🔑 Ensuring admin access..."
# This script is idempotent, will just print key if exists
docker-compose exec -T backend python scripts/create_api_key.py --name "admin" --role admin --value "admin.admin" --reset || true

echo "================================================="
echo "✅ Catapult Deployment Active!"
echo "   Web UI:            http://localhost:8080"
echo "   API:               http://localhost:8080/api"
echo "================================================="
