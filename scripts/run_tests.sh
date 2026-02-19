#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "============================================"
echo "Docker Release Registry - Test Runner"
echo "============================================"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if services are running
if ! docker compose ps | grep -q "registry-backend.*Up"; then
    echo -e "${RED}❌ Backend is not running!${NC}"
    echo
    echo "Please start the application first:"
    echo "  ./deploy.sh"
    exit 1
fi

# Check if API key is set
if [ -z "$TEST_API_KEY" ]; then
    echo -e "${YELLOW}⚠  TEST_API_KEY not set${NC}"
    echo
    echo "Getting existing API keys..."
    echo
    docker compose exec backend python -c "
import asyncio
from app.core.database import async_session_maker
from app.models.api_key import ApiKey
from sqlalchemy import select

async def get_keys():
    async with async_session_maker() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.is_active == True))
        keys = result.scalars().all()
        if keys:
            print('Available API keys:')
            for key in keys:
                print(f'  - {key.name} (ID: {key.id})')
            print()
            print('You need to get the actual key value. Options:')
            print('  1. Use the key from deployment output')
            print('  2. Create a new key: docker compose exec backend python scripts/create_api_key.py --name test-key')
        else:
            print('No API keys found. Creating one...')

asyncio.run(get_keys())
" 2>/dev/null || true

    echo
    echo "Then set it and run again:"
    echo "  export TEST_API_KEY='your-key-here'"
    echo "  ./scripts/run_tests.sh"
    exit 1
fi

echo -e "${GREEN}✓${NC} Services are running"
echo -e "${GREEN}✓${NC} API key is set"
echo

# Install SDK if needed
if ! python3 -c "import bso" 2>/dev/null; then
    echo "Installing Python SDK..."
    cd sdk/python
    pip install -e . -q
    cd ../..
    echo -e "${GREEN}✓${NC} SDK installed"
    echo
fi

# Run tests
echo "Running end-to-end tests..."
echo "API URL: http://localhost:8080/api"
echo "API Key: ${TEST_API_KEY:0:20}..."
echo
echo "============================================"
echo

python3 tests/e2e_test.py

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}============================================${NC}"
else
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo -e "${RED}============================================${NC}"
fi

exit $EXIT_CODE
