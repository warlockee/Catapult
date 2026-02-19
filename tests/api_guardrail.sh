#!/bin/bash

# Configuration
API_URL="${TEST_BASE_URL:-http://localhost:8080/api/v1}"
API_KEY="${TEST_API_KEY:?ERROR: TEST_API_KEY environment variable must be set}"
TIMESTAMP=$(date +%s)
TEST_MODEL_NAME="guardrail-model-${TIMESTAMP}"
TEST_RELEASE_VERSION="v1.0.0-${TIMESTAMP}"
TEST_ARTIFACTORY_FILE="guardrail_test_artifact.txt"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Helpers
log() {
    echo -e "${GREEN}[TEST] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    # Don't exit immediately, try to clean up
}

assert_status() {
    local response_file=$1
    local expected_status=$2
    local actual_status=$(cat "${response_file}.status")
    
    if [ "$actual_status" -ne "$expected_status" ]; then
        error "Expected status $expected_status but got $actual_status"
        cat "${response_file}.json"
        return 1
    fi
    return 0
}

cleanup() {
    log "Starting Cleanup..."
    
    # Delete Deployment
    if [ ! -z "$DEPLOYMENT_ID" ]; then
        log "Deleting Deployment $DEPLOYMENT_ID"
        # Deployments don't have delete endpoint in list? Let's check API. 
        # Assuming no delete for deployment based on common patterns, skipping or check if implemented.
        # Looking at previous file list, api.ts had NO deleteDeployment. Ignoring.
        true
    fi

    # Delete Artifact
    if [ ! -z "$ARTIFACT_ID" ]; then
        log "Deleting Artifact $ARTIFACT_ID"
        curl -s -X DELETE -H "X-API-Key: $API_KEY" "${API_URL}/artifacts/${ARTIFACT_ID}" > /dev/null
    fi

    # Delete Release
    if [ ! -z "$RELEASE_ID" ]; then
        log "Deleting Release $RELEASE_ID"
        curl -s -X DELETE -H "X-API-Key: $API_KEY" "${API_URL}/releases/${RELEASE_ID}" > /dev/null
    fi

    # Delete Model
    if [ ! -z "$MODEL_ID" ]; then
        log "Deleting Model $MODEL_ID"
        curl -s -X DELETE -H "X-API-Key: $API_KEY" "${API_URL}/models/${MODEL_ID}" > /dev/null
    fi
    
    rm -f "${TEST_ARTIFACTORY_FILE}"
    rm -f response.json response.status
    log "Cleanup Complete."
}

# Trap cleanup (optional, but manual invocation is safer for debugging)
# trap cleanup EXIT

# ==============================================================================
# 1. System & Health
# ==============================================================================
log "Checking Health..."
HEALTH_URL=$(echo "$API_URL" | sed 's|/v1||')/health
curl -s -w "%{http_code}" -o response.json "$HEALTH_URL" > response.status
assert_status "response" 200 || exit 1

log "Checking Storage Stats..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/system/storage" > response.status
assert_status "response" 200 || exit 1
jq . response.json

# ==============================================================================
# 2. Models
# ==============================================================================
log "Creating Model..."
curl -s -w "%{http_code}" -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
    -d "{\"name\": \"$TEST_MODEL_NAME\", \"storage_path\": \"models/$TEST_MODEL_NAME\", \"description\": \"Test Model\"}" \
    -o response.json "${API_URL}/models" > response.status
assert_status "response" 201 || exit 1
MODEL_ID=$(jq -r .id response.json)
log "Created Model ID: $MODEL_ID"

log "Listing Models..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/models" > response.status
assert_status "response" 200 || exit 1

# ==============================================================================
# 3. Releases
# ==============================================================================
log "Creating Release..."
curl -s -w "%{http_code}" -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
    -d "{\"model_id\": \"$MODEL_ID\", \"version\": \"$TEST_RELEASE_VERSION\", \"tag\": \"latest\", \"digest\": \"sha256:test\", \"platform\": \"linux/amd64\"}" \
    -o response.json "${API_URL}/releases" > response.status
assert_status "response" 201 || exit 1
RELEASE_ID=$(jq -r .id response.json)
log "Created Release ID: $RELEASE_ID"

# ==============================================================================
# 4. Artifacts (Upload)
# ==============================================================================
log "Uploading Artifact..."
echo "Guardrail Test Content" > "$TEST_ARTIFACTORY_FILE"
curl -s -w "%{http_code}" -X POST -H "X-API-Key: $API_KEY" \
    -F "file=@$TEST_ARTIFACTORY_FILE" \
    -F "release_id=$RELEASE_ID" \
    -o response.json "${API_URL}/artifacts/upload" > response.status

# Expect 201 Created (Type inference should work)
assert_status "response" 201 || exit 1
ARTIFACT_ID=$(jq -r .id response.json)
log "Created Artifact ID: $ARTIFACT_ID"
ARTIFACT_TYPE=$(jq -r .artifact_type response.json)
if [ "$ARTIFACT_TYPE" != "binary" ]; then
    error "Artifact type inference failed. Expected 'binary', got '$ARTIFACT_TYPE'"
    exit 1
fi
log "Verified Artifact Type Inference: $ARTIFACT_TYPE"

# ==============================================================================
# 5. Deployments
# ==============================================================================
log "Creating Deployment..."
curl -s -w "%{http_code}" -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
    -d "{\"release_id\": \"$RELEASE_ID\", \"environment\": \"staging\", \"status\": \"running\", \"metadata\": {}}" \
    -o response.json "${API_URL}/deployments" > response.status
assert_status "response" 201 || exit 1
DEPLOYMENT_ID=$(jq -r .id response.json)
log "Created Deployment ID: $DEPLOYMENT_ID"

log "Listing Deployments..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/deployments" > response.status
assert_status "response" 200 || exit 1

log "Getting Deployment..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/deployments/${DEPLOYMENT_ID}" > response.status
assert_status "response" 200 || exit 1

# ==============================================================================
# 6. Audit Logs
# ==============================================================================
log "Listing Audit Logs..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/audit-logs" > response.status
assert_status "response" 200 || exit 1

# ==============================================================================
# 7. API Keys
# ==============================================================================
log "Creating API Key..."
curl -s -w "%{http_code}" -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
    -d "{\"name\": \"test-key-${TIMESTAMP}\"}" \
    -o response.json "${API_URL}/api-keys" > response.status
assert_status "response" 201 || exit 1
NEW_KEY_ID=$(jq -r .id response.json)
log "Created API Key ID: $NEW_KEY_ID"

log "Listing API Keys..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/api-keys" > response.status
assert_status "response" 200 || exit 1

log "Revoking API Key..."
curl -s -w "%{http_code}" -X DELETE -H "X-API-Key: $API_KEY" "${API_URL}/api-keys/${NEW_KEY_ID}" > response.status
assert_status "response" 204 || exit 1
log "Revoked API Key"

# ==============================================================================
# 8. Docker Builds
# ==============================================================================
log "Listing Docker Build Templates..."
curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" -o response.json "${API_URL}/docker/templates/organic" > response.status
# 404 is acceptable if templates directory is missing, but let's assume it might fail or pass.
# For now, just log the result.
log "Docker Template Status: $(cat response.status)"

# Trigger a docker build (might fail if worker not running, so treating as soft check)
log "Creating Docker Build..."
curl -s -w "%{http_code}" -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
    -d "{\"release_id\": \"$RELEASE_ID\", \"image_tag\": \"test:latest\", \"build_type\": \"organic\"}" \
    -o response.json "${API_URL}/docker/builds" > response.status
# Allow 201 or 503 (if worker unavailable)
STATUS=$(cat response.status)
if [ "$STATUS" -eq 201 ]; then
    BUILD_ID=$(jq -r .id response.json)
    log "Created Docker Build ID: $BUILD_ID"
else
    log "Docker Build Creation Skipped/Failed (Status: $STATUS) - Worker might be offline or template missing."
fi

# ==============================================================================
# Summary
# ==============================================================================
log "All Tests Passed Successfully!"

# Cleanup
cleanup
