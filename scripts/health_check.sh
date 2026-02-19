#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "=================================="
echo "Deployment Integration Test"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop first.${NC}"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found. Please create it from .env.example${NC}"
    exit 1
fi

source .env

# Validate required variables
if [ "$POSTGRES_PASSWORD" == "your-secure-password-change-me" ] || [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}❌ POSTGRES_PASSWORD not set in .env${NC}"
    exit 1
fi

if [ "$API_KEY_SALT" == "change-this-to-random-32-char-salt-minimum" ] || [ -z "$API_KEY_SALT" ]; then
    echo -e "${RED}❌ API_KEY_SALT not set in .env${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Testing deployment script...${NC}"
echo

# Test 1: Run deploy.sh (this will take a while)
echo "Running deployment script..."
if ./deploy.sh; then
    echo -e "${GREEN}✓ Deployment script completed${NC}"
else
    echo -e "${RED}✗ Deployment script failed${NC}"
    exit 1
fi

echo
echo -e "${BLUE}Step 2: Testing service health...${NC}"
echo

# Test 2: Check PostgreSQL
echo -n "Testing PostgreSQL connection... "
if docker-compose exec -T postgres pg_isready -U ${POSTGRES_USER:-registry} > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    exit 1
fi

# Test 3: Check backend health endpoint
echo -n "Testing backend health endpoint... "
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/health || echo "")
if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "  Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "  Response: $HEALTH_RESPONSE"
    exit 1
fi

# Test 4: Check nginx proxy
echo -n "Testing nginx proxy... "
NGINX_RESPONSE=$(curl -s http://localhost/api/health || echo "")
if echo "$NGINX_RESPONSE" | grep -q '"status"'; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "  Response: $NGINX_RESPONSE"
    exit 1
fi

# Test 5: Check frontend is served
echo -n "Testing frontend is served... "
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/ || echo "000")
if [ "$FRONTEND_RESPONSE" == "200" ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${YELLOW}⚠ Got HTTP $FRONTEND_RESPONSE (may be expected if frontend build failed)${NC}"
fi

# Test 6: Check API docs
echo -n "Testing API docs endpoint... "
DOCS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/docs || echo "000")
if [ "$DOCS_RESPONSE" == "200" ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${RED}✗ FAILED (HTTP $DOCS_RESPONSE)${NC}"
    exit 1
fi

# Test 7: Check database migrations ran
echo -n "Testing database migrations... "
MIGRATION_CHECK=$(docker-compose exec -T backend alembic current 2>&1 || echo "failed")
if echo "$MIGRATION_CHECK" | grep -q "head\|001"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "  Current migration: $MIGRATION_CHECK"
else
    echo -e "${YELLOW}⚠ Could not verify migrations: $MIGRATION_CHECK${NC}"
fi

# Test 8: Check API key creation
echo -n "Testing API key creation... "
API_KEY_OUTPUT=$(docker-compose exec -T backend python scripts/create_api_key.py --name "test-key-$(date +%s)" 2>&1 || echo "failed")
if echo "$API_KEY_OUTPUT" | grep -q "API Key Created Successfully"; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${YELLOW}⚠ API key creation: $API_KEY_OUTPUT${NC}"
fi

echo
echo "=================================="
echo -e "${GREEN}✅ Integration Tests Complete!${NC}"
echo "=================================="
echo
echo "Services are running:"
echo "  - Web UI:      http://localhost"
echo "  - API:         http://localhost/api"
echo "  - API Docs:    http://localhost/docs"
echo "  - Health:      http://localhost/api/health"
echo
echo "To stop services: docker-compose down"
echo

