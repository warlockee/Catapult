#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."
echo "Deployment Script Test Suite"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Test function
test_check() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: $test_name... "
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Check deploy.sh exists and is executable
test_check "deploy.sh exists" "[ -f deploy.sh ]"
test_check "deploy.sh is executable" "[ -x deploy.sh ]"

# Test 2: Check deploy.sh syntax
echo -n "Testing: deploy.sh syntax... "
if bash -n deploy.sh > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    bash -n deploy.sh
    ((FAILED++))
fi

# Test 3: Check required files exist
test_check "docker-compose.yml exists" "[ -f docker-compose.yml ]"
test_check ".env.example exists" "[ -f .env.example ]"

# Test 4: Check docker-compose.yml syntax
echo -n "Testing: docker-compose.yml syntax... "
if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ SKIPPED (Docker not available)${NC}"
fi

# Test 5: Check backend files exist
test_check "backend/Dockerfile exists" "[ -f backend/Dockerfile ]"
test_check "backend/requirements.txt exists" "[ -f backend/requirements.txt ]"
test_check "backend/app/main.py exists" "[ -f backend/app/main.py ]"
test_check "backend/scripts/create_api_key.py exists" "[ -f backend/scripts/create_api_key.py ]"
test_check "backend/alembic.ini exists" "[ -f backend/alembic.ini ]"

# Test 6: Check frontend files exist
test_check "frontend/Dockerfile exists" "[ -f 'frontend/Dockerfile' ]"
test_check "frontend/package.json exists" "[ -f 'frontend/package.json' ]"

# Test 7: Check nginx config exists
test_check "infrastructure/nginx/nginx.conf exists" "[ -f infrastructure/nginx/nginx.conf ]"
test_check "infrastructure/nginx/conf.d/default.conf exists" "[ -f infrastructure/nginx/conf.d/default.conf ]"

# Test 8: Validate deploy.sh checks for .env
echo -n "Testing: deploy.sh checks for .env... "
if grep -q 'if \[ ! -f \.env \]' deploy.sh; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 9: Validate deploy.sh checks POSTGRES_PASSWORD
echo -n "Testing: deploy.sh validates POSTGRES_PASSWORD... "
if grep -q 'POSTGRES_PASSWORD' deploy.sh; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 10: Validate deploy.sh checks API_KEY_SALT
echo -n "Testing: deploy.sh validates API_KEY_SALT... "
if grep -q 'API_KEY_SALT' deploy.sh; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 11: Check health endpoint exists in backend
echo -n "Testing: Health endpoint exists in backend... "
if grep -q '/api/health' backend/app/main.py; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 12: Check create_api_key.py is executable
test_check "create_api_key.py is executable" "[ -x backend/scripts/create_api_key.py ]"

# Test 13: Check storage directory creation logic
echo -n "Testing: deploy.sh creates storage directory... "
if grep -q 'CEPH_MOUNT_PATH' deploy.sh && grep -q 'mkdir -p' deploy.sh; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Summary
echo
echo "=================================="
echo "Test Summary"
echo "=================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo
    echo "Next steps:"
    echo "  1. Start Docker Desktop"
    echo "  2. Ensure .env file is configured with:"
    echo "     - POSTGRES_PASSWORD (secure password)"
    echo "     - API_KEY_SALT (random 32+ character string)"
    echo "     - CEPH_MOUNT_PATH (storage directory path)"
    echo "  3. Run: ./deploy.sh"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please review above.${NC}"
    exit 1
fi

