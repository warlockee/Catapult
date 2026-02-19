#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "=================================================="
echo "Full Lifecycle End-to-End Test"
echo "=================================================="
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Cleanup
echo -e "${YELLOW}Step 1: Cleaning up existing environment...${NC}"
docker compose down -v
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo

# Step 2: Deploy
echo -e "${YELLOW}Step 2: Deploying application...${NC}"
./deploy.sh
echo -e "${GREEN}✓ Deployment complete${NC}"
echo

# Step 3: Create Test API Key
echo -e "${YELLOW}Step 3: Creating test API key...${NC}"
# We need to wait a bit to ensure backend is fully ready to accept commands if deploy.sh didn't wait enough
# deploy.sh waits for health check, so we should be good.

TEST_KEY_NAME="lifecycle-test-$(date +%s)"
API_KEY_OUTPUT=$(docker compose exec -T backend python scripts/create_api_key.py --name "$TEST_KEY_NAME" 2>&1)

if echo "$API_KEY_OUTPUT" | grep -q "Key:"; then
    TEST_API_KEY=$(echo "$API_KEY_OUTPUT" | grep "Key:" | awk '{print $2}')
    echo -e "${GREEN}✓ Created API key: ${TEST_API_KEY:0:10}...${NC}"
else
    echo -e "${RED}❌ Failed to create API key${NC}"
    echo "$API_KEY_OUTPUT"
    exit 1
fi
echo

# Step 4: Install SDK
echo -e "${YELLOW}Step 4: Installing Python SDK...${NC}"
if ! python3 -c "import bso" 2>/dev/null; then
    cd sdk/python
    pip install -e . -q
    cd ../..
    echo -e "${GREEN}✓ SDK installed${NC}"
else
    echo -e "${GREEN}✓ SDK already installed${NC}"
fi
echo

# Step 5: Run Tests
echo -e "${YELLOW}Step 5: Running E2E tests...${NC}"
export TEST_API_KEY=$TEST_API_KEY
python3 tests/e2e_test.py

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}==================================================${NC}"
    echo -e "${GREEN}✅ FULL LIFECYCLE TEST PASSED${NC}"
    echo -e "${GREEN}==================================================${NC}"
else
    echo -e "${RED}==================================================${NC}"
    echo -e "${RED}❌ FULL LIFECYCLE TEST FAILED${NC}"
    echo -e "${RED}==================================================${NC}"
fi

exit $EXIT_CODE
