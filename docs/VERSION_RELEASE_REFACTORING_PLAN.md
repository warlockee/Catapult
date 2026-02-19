# Version/Release Terminology Refactoring Plan

## ✅ IMPLEMENTATION STATUS: COMPLETED

**Completed Date:** 2026-01-01

All phases have been implemented with backward compatibility. Key changes:
- Database: `releases` table renamed to `versions`
- Backend: All models, schemas, repositories, and API endpoints updated
- Frontend: API client, types, and Dashboard updated
- Python SDK: Updated with backward-compatible methods
- Documentation: README and all docs updated

## Overview

This document outlines the complete plan to fix the conceptual confusion between "model version" and "release" throughout the codebase.

### Problem Statement

The dashboard showed 191 total releases, but the release tab showed none. This is because:
- **Dashboard**: Queries all items (no filter) → shows 191
- **Release Tab**: Defaults to `is_release=true` filter → shows 0

The root cause is terminology confusion where "release" is used to mean "version" throughout the codebase.

### Correct Conceptual Model

| Term | Definition | `is_release` value |
|------|------------|-------------------|
| **Version** | Any version of a model (base entity) | `false` (default) |
| **Release** | A promoted/verified version | `true` |

---

## Part 1: Database Layer

### 1.1 Table Rename

```sql
ALTER TABLE releases RENAME TO versions;
```

### 1.2 Index Renames

```sql
ALTER INDEX idx_releases_image_version_quant RENAME TO idx_versions_model_version_quant;
ALTER INDEX idx_releases_created_at RENAME TO idx_versions_created_at;
ALTER INDEX idx_releases_status RENAME TO idx_versions_status;
ALTER INDEX idx_releases_is_release RENAME TO idx_versions_is_release;
```

### 1.3 Foreign Key Columns

| Table | Current Column | Decision |
|-------|----------------|----------|
| `deployments` | `release_id` | **Keep** (backward compat) - update FK target only |
| `artifacts` | `release_id` | **Keep** (backward compat) - update FK target only |
| `docker_builds` | `release_id` | **Keep** (backward compat) - update FK target only |

**Note:** FK constraints must be updated to reference `versions.id` instead of `releases.id`.

### 1.4 Migration File

Create new migration: `backend/alembic/versions/xxxx_rename_releases_to_versions.py`

```python
"""Rename releases table to versions.

Revision ID: xxxx
"""
from alembic import op

def upgrade():
    # Rename table
    op.rename_table('releases', 'versions')

    # Rename indexes
    op.execute('ALTER INDEX idx_releases_image_version_quant RENAME TO idx_versions_model_version_quant')
    op.execute('ALTER INDEX idx_releases_created_at RENAME TO idx_versions_created_at')
    # ... other indexes

def downgrade():
    op.rename_table('versions', 'releases')
    # Reverse index renames
```

---

## Part 2: Backend Layer

### 2.1 Models

#### File Renames

| Current | New |
|---------|-----|
| `backend/app/models/release.py` | `backend/app/models/version.py` |

#### Class Changes (`version.py`)

```python
# Before
class Release(Base):
    __tablename__ = "releases"

# After
class Version(Base):
    __tablename__ = "versions"
```

#### Relationship Updates

| File | Current | New |
|------|---------|-----|
| `models/model.py` | `releases = relationship("Release", ...)` | `versions = relationship("Version", ...)` |
| `models/deployment.py` | `release = relationship("Release", ...)` | `version = relationship("Version", ...)` |
| `models/artifact.py` | `release = relationship("Release", ...)` | `version = relationship("Version", ...)` |
| `models/docker_build.py` | `release = relationship("Release", ...)` | `version = relationship("Version", ...)` |

### 2.2 Schemas

#### File Renames

| Current | New |
|---------|-----|
| `backend/app/schemas/release.py` | `backend/app/schemas/version.py` |

#### Class Renames

| Current | New |
|---------|-----|
| `ReleaseBase` | `VersionBase` |
| `ReleaseCreate` | `VersionCreate` |
| `ReleaseUpdate` | `VersionUpdate` |
| `ReleaseResponse` | `VersionResponse` |
| `ReleaseWithImage` | `VersionWithModel` |
| `ReleaseOption` | `VersionOption` |

#### Model Schema Updates (`model.py`)

| Current | New |
|---------|-----|
| `ModelWithReleases` | `ModelWithVersions` |
| `release_count: int` | `version_count: int` |

### 2.3 Repository

#### File Renames

| Current | New |
|---------|-----|
| `backend/app/repositories/release_repository.py` | `backend/app/repositories/version_repository.py` |

#### Class/Method Renames

| Current | New |
|---------|-----|
| `ReleaseRepository` | `VersionRepository` |
| `list_releases()` | `list_versions()` |
| `create_release()` | `create_version()` |
| `update_release()` | `update_version()` |
| `check_version_exists()` | Keep name (already correct) |
| `get_latest_for_model()` | Keep name |

### 2.4 Exceptions

#### Updates to `backend/app/core/exceptions.py`

| Current | New |
|---------|-----|
| `ReleaseNotFoundError` | `VersionNotFoundError` |
| `ReleaseAlreadyExistsError` | `VersionAlreadyExistsError` |
| Message: `"Release not found: {id}"` | `"Version not found: {id}"` |
| Message: `"Release already exists: {version}"` | `"Version already exists: {version}"` |

### 2.5 API Endpoints

#### File Renames

| Current | New |
|---------|-----|
| `backend/app/api/v1/endpoints/releases.py` | `backend/app/api/v1/endpoints/versions.py` |

#### Endpoint Changes

| Current | New |
|---------|-----|
| `POST /v1/releases` | `POST /v1/versions` |
| `GET /v1/releases` | `GET /v1/versions` |
| `GET /v1/releases/{id}` | `GET /v1/versions/{id}` |
| `PUT /v1/releases/{id}` | `PUT /v1/versions/{id}` |
| `DELETE /v1/releases/{id}` | `DELETE /v1/versions/{id}` |
| `GET /v1/releases/latest` | `GET /v1/versions/latest` |
| `GET /v1/releases/options` | `GET /v1/versions/options` |
| `GET /v1/releases/{id}/deployments` | `GET /v1/versions/{id}/deployments` |
| `GET /v1/models/{id}/releases` | `GET /v1/models/{id}/versions` |

#### Router Update (`router.py`)

```python
# Before
from app.api.v1.endpoints import releases
api_router.include_router(releases.router, prefix="/releases", tags=["releases"])

# After
from app.api.v1.endpoints import versions
api_router.include_router(versions.router, prefix="/versions", tags=["versions"])
```

### 2.6 Audit Log Actions

| Current Action | New Action | Historical Data |
|----------------|------------|-----------------|
| `create_release` | `create_version` | Keep old entries unchanged |
| `update_release` | `update_version` | Keep old entries unchanged |
| `delete_release` | `delete_version` | Keep old entries unchanged |
| `resource_type="release"` | `resource_type="version"` | Keep old entries unchanged |

---

## Part 3: Frontend Layer

### 3.1 Types (`frontend/src/lib/api.ts`)

#### Interface Renames

| Current | New |
|---------|-----|
| `interface Release` | `interface Version` |
| `interface ReleaseOption` | `interface VersionOption` |

#### Field Updates

```typescript
// In Image interface
release_count?: number;  // → version_count?: number;
```

#### API Method Renames

| Current | New |
|---------|-----|
| `listReleases()` | `listVersions()` |
| `getRelease()` | `getVersion()` |
| `createRelease()` | `createVersion()` |
| `updateRelease()` | `updateVersion()` |
| `deleteRelease()` | `deleteVersion()` |
| `promoteRelease()` | `promoteVersion()` |
| `getReleaseDeployments()` | `getVersionDeployments()` |
| `getLatestRelease()` | `getLatestVersion()` |
| `listReleaseOptions()` | `listVersionOptions()` |
| `getImageReleases()` | `getImageVersions()` |

#### Endpoint URL Updates

```typescript
// All /v1/releases → /v1/versions
fetchApi<PaginatedResponse<Version>>('/v1/versions', ...)
```

### 3.2 Components

#### File Renames

| Current | New |
|---------|-----|
| `ReleaseList.tsx` | `VersionList.tsx` |
| `ReleaseDetail.tsx` | `VersionDetail.tsx` |
| `CreateReleaseDialog.tsx` | `CreateVersionDialog.tsx` |

#### Component Name Updates

| Current | New |
|---------|-----|
| `export function ReleaseList()` | `export function VersionList()` |
| `export function ReleaseDetail()` | `export function VersionDetail()` |
| `export function CreateReleaseDialog()` | `export function CreateVersionDialog()` |

#### Internal Variable Updates

All occurrences of `release`/`releases` variables should be renamed to `version`/`versions`.

#### UI Labels in VersionList

```tsx
// Page title
<h1>Versions</h1>
<p>View all model versions and releases</p>

// Button
<Button>Create Version</Button>

// Filter tabs (KEEP "Releases" tab!)
<Button onClick={() => handleFilterChange('all')}>All</Button>
<Button onClick={() => handleFilterChange('releases')}>Releases</Button>  // is_release=true
<Button onClick={() => handleFilterChange('versions')}>Drafts</Button>    // is_release=false

// Actions
<Button>Promote to Release</Button>  // Sets is_release=true
<Button>Demote</Button>              // Sets is_release=false
```

### 3.3 Routes (`frontend/src/App.tsx`)

#### Route Updates

```tsx
// Before
<Route path="/releases" element={<ReleaseList />} />
<Route path="/releases/:releaseId" element={<ReleaseDetail />} />
<Route path="/models/:modelId/releases/:releaseId" element={<ReleaseDetail />} />

// After
<Route path="/versions" element={<VersionList />} />
<Route path="/versions/:versionId" element={<VersionDetail />} />
<Route path="/models/:modelId/versions/:versionId" element={<VersionDetail />} />

// Backward compatibility redirects
<Route path="/releases" element={<Navigate to="/versions" replace />} />
<Route path="/releases/:id" element={<Navigate to="/versions/:id" replace />} />
```

#### Lazy Load Updates

```tsx
const VersionList = lazy(() => import('./components/VersionList').then(m => ({ default: m.VersionList })));
const VersionDetail = lazy(() => import('./components/VersionDetail').then(m => ({ default: m.VersionDetail })));
```

### 3.4 Sidebar (`frontend/src/components/Sidebar.tsx`)

```tsx
// Before
{ path: '/releases', icon: Package, label: 'Releases', prefetchKey: 'releases' }

// After
{ path: '/versions', icon: Package, label: 'Versions', prefetchKey: 'versions' }
```

### 3.5 Dashboard Fix (`frontend/src/components/Dashboard.tsx`)

**This fixes the original bug!**

```tsx
// Before - counts ALL versions as "releases"
const { data: releasesData } = useQuery({
  queryKey: ['releases', 'dashboard'],
  queryFn: () => api.listReleases({ size: 20 }),
});
const totalReleases = releasesData?.total || 0;

// After - separate counts for versions and releases
const { data: versionsData } = useQuery({
  queryKey: ['versions', 'dashboard'],
  queryFn: () => api.listVersions({ size: 20 }),
});
const { data: officialReleasesData } = useQuery({
  queryKey: ['versions', 'dashboard', 'releases-only'],
  queryFn: () => api.listVersions({ is_release: true, size: 1 }),
});
const totalVersions = versionsData?.total || 0;
const totalReleases = officialReleasesData?.total || 0;  // Only is_release=true
```

#### Dashboard Card Updates

```tsx
// Before
<CardTitle>Total Releases</CardTitle>
<div>{totalReleases}</div>  // Shows 191 (all versions)

// After
<CardTitle>Total Versions</CardTitle>
<div>{totalVersions}</div>  // Shows 191 (all versions)

// Add new card
<CardTitle>Official Releases</CardTitle>
<div>{totalReleases}</div>  // Shows count of is_release=true only
```

---

## Part 4: Python SDK

### 4.1 Package Structure

| Current | Decision |
|---------|----------|
| `sdk/python/catapult/` | Renamed to `catapult` |

### 4.2 Client Methods (`sdk/python/catapult/client.py`)

| Current | New |
|---------|-----|
| `create_release()` | `create_version()` |
| `list_releases()` | `list_versions()` |
| `get_release()` | `get_version()` |
| `delete_release()` | `delete_version()` |
| `get_latest_release()` | `get_latest_version()` |

#### Endpoint URL Updates

```python
# All /v1/releases → /v1/versions
response = self.client.post("/v1/versions", json=payload)
response = self.client.get("/v1/versions", params=params)
```

### 4.3 Models (`sdk/python/catapult/models.py`)

| Current | New |
|---------|-----|
| `class Release` | `class Version` |

### 4.4 Backward Compatibility Aliases

```python
# At end of client.py
class Registry:
    # ... new methods ...

    # Deprecated aliases for backward compatibility
    create_release = create_version
    list_releases = list_versions
    get_release = get_version
    delete_release = delete_version
    get_latest_release = get_latest_version

# At end of models.py
Release = Version  # Deprecated alias
```

---

## Part 5: Tests

### 5.1 Test Files to Update

| File | Changes |
|------|---------|
| `tests/e2e_test.py` | Update endpoints, variable names |
| `tests/verify_release_filtering.py` | Rename to `verify_version_filtering.py` |
| `tests/verify_sdk_features.py` | Update SDK method calls |
| `tests/verify_multiple_artifacts.py` | Update release references |
| `tests/verify_optimized_build_e2e.py` | Update release references |

### 5.2 Search and Replace

```bash
# In all test files
find tests/ -name "*.py" -exec sed -i 's/release/version/g' {} \;
find tests/ -name "*.py" -exec sed -i 's/Release/Version/g' {} \;
find tests/ -name "*.py" -exec sed -i 's/\/releases/\/versions/g' {} \;
```

---

## Part 6: Documentation

### 6.1 Files to Update

| File | Changes |
|------|---------|
| `README.md` | API docs, schema docs, examples |
| `frontend/src/components/Help.tsx` | FAQ content |
| `sdk/python/README.md` | Usage examples |
| `sdk/python/example.py` | Code examples |

### 6.2 Key Documentation Updates

- Database schema section: `releases` table → `versions` table
- API endpoint documentation
- SDK usage examples
- FAQ: "How do I create a version?" / "How do I promote to release?"

---

## Verification Section

### Phase 1: Backend Verification

| Check | Command/Action | Expected Result |
|-------|----------------|-----------------|
| Migration applies | `alembic upgrade head` | No errors, table renamed |
| Backend starts | `uvicorn app.main:app` | No import errors |
| API docs accessible | `GET /docs` | Shows `/v1/versions` endpoints |
| List versions | `GET /v1/versions` | Returns paginated versions |
| List with filter | `GET /v1/versions?is_release=true` | Returns only promoted versions |
| Get single version | `GET /v1/versions/{id}` | Returns version details |
| Create version | `POST /v1/versions` | Creates with `is_release=false` |
| Promote version | `PUT /v1/versions/{id}` + `{"is_release": true}` | Version promoted |
| Model versions | `GET /v1/models/{id}/versions` | Returns versions for model |
| Old endpoint gone | `GET /v1/releases` | Returns 404 |

### Phase 2: Frontend Verification

| Check | Action | Expected Result |
|-------|--------|-----------------|
| Build succeeds | `npm run build` | No TypeScript errors |
| App loads | Navigate to `/` | Dashboard renders |
| Sidebar updated | Check sidebar | Shows "Versions" |
| Versions page | Navigate to `/versions` | VersionList renders |
| Old route redirects | Navigate to `/releases` | Redirects to `/versions` |
| Filter tabs work | Click "Releases" tab | Shows only `is_release=true` |
| Create version | Click "Create Version" | Dialog opens, creates version |
| Promote action | Click "Promote to Release" | Sets `is_release=true` |
| Detail page | Click on a version | `/versions/{id}` shows details |

### Phase 3: Data Integrity Verification

| Check | Query | Expected |
|-------|-------|----------|
| Table exists | `\d versions` | Table structure shown |
| Old table gone | `\d releases` | "Did not find any relation" |
| Row count preserved | `SELECT COUNT(*) FROM versions;` | Same as before |
| is_release preserved | `SELECT COUNT(*) FROM versions WHERE is_release = true;` | Same as before |
| FKs work | `SELECT * FROM deployments d JOIN versions v ON d.release_id = v.id LIMIT 1;` | Join succeeds |

### Phase 4: Dashboard Count Verification (Original Bug Fix)

| Check | Expected Result |
|-------|-----------------|
| Dashboard "Total Versions" | Shows count of ALL versions (191) |
| Dashboard "Official Releases" | Shows count where `is_release=true` |
| Versions page "All" tab | Matches "Total Versions" |
| Versions page "Releases" tab | Matches "Official Releases" |

### Phase 5: SDK Verification

| Check | Command | Expected |
|-------|---------|----------|
| SDK imports | `from catapult import Registry` | No errors |
| List versions | `registry.list_versions()` | Returns versions |
| Create version | `registry.create_version(...)` | Creates version |
| Backward compat | `registry.list_releases()` | Works (deprecated) |

### Phase 6: E2E Verification Script

```bash
#!/bin/bash
# scripts/verify_version_refactor.sh

API_URL="${API_URL:-http://localhost:8000/api}"
API_KEY="${API_KEY:-admin}"

echo "=== Version/Release Refactoring Verification ==="

# 1. New endpoints exist
echo -n "GET /v1/versions: "
curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $API_KEY" "$API_URL/v1/versions"
echo ""

# 2. Old endpoints are gone
echo -n "GET /v1/releases (should 404): "
curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $API_KEY" "$API_URL/v1/releases"
echo ""

# 3. Filtering works
echo -n "GET /v1/versions?is_release=true: "
RESULT=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/v1/versions?is_release=true")
TOTAL=$(echo $RESULT | jq '.total')
echo "Found $TOTAL official releases"

# 4. All versions count
echo -n "GET /v1/versions (all): "
RESULT=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/v1/versions")
TOTAL=$(echo $RESULT | jq '.total')
echo "Found $TOTAL total versions"

# 5. Model versions endpoint
MODEL_ID=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/v1/models?size=1" | jq -r '.items[0].id')
if [ "$MODEL_ID" != "null" ] && [ -n "$MODEL_ID" ]; then
  echo -n "GET /v1/models/$MODEL_ID/versions: "
  curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $API_KEY" "$API_URL/v1/models/$MODEL_ID/versions"
  echo ""
fi

# 6. Deployments FK still works
echo -n "GET /v1/deployments: "
curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $API_KEY" "$API_URL/v1/deployments"
echo ""

echo "=== Verification Complete ==="
```

---

## Recovery Verification Section

### Pre-Migration Backup

```bash
#!/bin/bash
# scripts/backup_before_refactor.sh

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)_pre_version_refactor"
mkdir -p "$BACKUP_DIR"

echo "=== Pre-Refactoring Backup ==="

# 1. Database backup
echo "Backing up database..."
pg_dump -h localhost -U postgres model_registry > "$BACKUP_DIR/database.sql"

# 2. Record current counts
echo "Recording counts..."
psql -h localhost -U postgres model_registry -c "
  SELECT
    (SELECT COUNT(*) FROM releases) as total_releases,
    (SELECT COUNT(*) FROM releases WHERE is_release = true) as promoted,
    (SELECT COUNT(*) FROM deployments) as deployments,
    (SELECT COUNT(*) FROM artifacts) as artifacts,
    (SELECT COUNT(*) FROM docker_builds) as docker_builds
" > "$BACKUP_DIR/counts_before.txt"

# 3. Git state
echo "Recording git state..."
git rev-parse HEAD > "$BACKUP_DIR/git_commit.txt"

# 4. Alembic version
cd backend && alembic current > "$BACKUP_DIR/alembic_version.txt"

echo "Backup complete: $BACKUP_DIR"
```

### Scenario A: Fresh Database from Migrations

```bash
#!/bin/bash
# scripts/verify_recovery_migrations.sh

echo "=== Scenario A: Recovery from Migrations ==="

# 1. Create fresh database
psql -h localhost -U postgres -c "DROP DATABASE IF EXISTS model_registry_test;"
psql -h localhost -U postgres -c "CREATE DATABASE model_registry_test;"

# 2. Run all migrations
cd backend
DATABASE_URL="postgresql://postgres:postgres@localhost/model_registry_test" alembic upgrade head

# 3. Verify schema
echo "Verifying 'versions' table exists..."
psql -h localhost -U postgres model_registry_test -c "\d versions"

echo "Verifying 'releases' table does NOT exist..."
psql -h localhost -U postgres model_registry_test -c "\d releases" 2>&1 | grep -q "Did not find" && echo "PASS" || echo "FAIL"

# 4. Verify FK constraints
echo "Verifying FK constraints..."
psql -h localhost -U postgres model_registry_test -c "
  SELECT conname, confrelid::regclass
  FROM pg_constraint
  WHERE conrelid = 'deployments'::regclass AND contype = 'f';
"

echo "=== Migration Recovery: Complete ==="
```

### Scenario B: Recovery from Snapshot

```bash
#!/bin/bash
# scripts/verify_recovery_snapshot.sh

SNAPSHOT_FILE="$1"
if [ -z "$SNAPSHOT_FILE" ]; then
  echo "Usage: $0 <snapshot.sql>"
  exit 1
fi

echo "=== Scenario B: Recovery from Snapshot ==="

# 1. Create fresh database
psql -h localhost -U postgres -c "DROP DATABASE IF EXISTS model_registry_restored;"
psql -h localhost -U postgres -c "CREATE DATABASE model_registry_restored;"

# 2. Restore from snapshot
psql -h localhost -U postgres model_registry_restored < "$SNAPSHOT_FILE"

# 3. Verify versions table exists
echo "Checking 'versions' table..."
TABLE_EXISTS=$(psql -h localhost -U postgres model_registry_restored -tAc "
  SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'versions');
")

if [ "$TABLE_EXISTS" = "t" ]; then
  echo "PASS: 'versions' table exists"
else
  echo "FAIL: 'versions' table missing"

  # Check if old table exists (pre-refactor snapshot)
  OLD_EXISTS=$(psql -h localhost -U postgres model_registry_restored -tAc "
    SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'releases');
  ")
  if [ "$OLD_EXISTS" = "t" ]; then
    echo "WARNING: Snapshot is from before refactoring (has 'releases' table)"
  fi
  exit 1
fi

# 4. Verify row counts
echo "Verifying data..."
psql -h localhost -U postgres model_registry_restored -c "
  SELECT
    (SELECT COUNT(*) FROM versions) as versions,
    (SELECT COUNT(*) FROM deployments) as deployments,
    (SELECT COUNT(*) FROM artifacts) as artifacts;
"

# 5. Verify FKs work
echo "Verifying FK joins..."
psql -h localhost -U postgres model_registry_restored -c "
  SELECT COUNT(*) FROM deployments d JOIN versions v ON d.release_id = v.id;
"

echo "=== Snapshot Recovery: Complete ==="
```

### Scenario C: Docker Volume Recovery

```bash
#!/bin/bash
# scripts/verify_recovery_docker.sh

echo "=== Scenario C: Docker Volume Recovery ==="

# 1. Capture pre-reboot state
echo "Capturing pre-reboot state..."
PRE_COUNT=$(docker exec catapult-db psql -U postgres -d model_registry -tAc "SELECT COUNT(*) FROM versions;")
echo "Pre-reboot version count: $PRE_COUNT"

# 2. Stop containers
echo "Stopping containers..."
docker-compose down

# 3. Wait
sleep 5

# 4. Start containers
echo "Starting containers..."
docker-compose up -d

# 5. Wait for DB
echo "Waiting for database..."
sleep 10
until docker exec catapult-db pg_isready -U postgres 2>/dev/null; do
  echo "Waiting..."
  sleep 2
done

# 6. Verify data persisted
POST_COUNT=$(docker exec catapult-db psql -U postgres -d model_registry -tAc "SELECT COUNT(*) FROM versions;")
echo "Post-reboot version count: $POST_COUNT"

if [ "$PRE_COUNT" = "$POST_COUNT" ]; then
  echo "PASS: Data persisted correctly"
else
  echo "FAIL: Data mismatch! Pre: $PRE_COUNT, Post: $POST_COUNT"
  exit 1
fi

# 7. Verify API works
echo "Verifying API..."
sleep 5
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/versions)
if [ "$STATUS" = "200" ]; then
  echo "PASS: API responding"
else
  echo "FAIL: API returned $STATUS"
  exit 1
fi

echo "=== Docker Recovery: Complete ==="
```

### Scenario D: Cold Start Verification

```bash
#!/bin/bash
# scripts/verify_cold_start.sh

echo "=== Scenario D: Cold Start ==="

# 1. Kill existing processes
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

# 2. Clear Python cache
echo "Clearing caches..."
find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 3. Start backend
echo "Starting backend..."
cd backend
timeout 30 uvicorn app.main:app --host 0.0.0.0 --port 8000 &
APP_PID=$!
sleep 5

# 4. Health check
echo "Health check..."
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null)
if [ "$HEALTH" = "healthy" ]; then
  echo "PASS: Backend healthy"
else
  echo "FAIL: Backend unhealthy"
  kill $APP_PID 2>/dev/null
  exit 1
fi

# 5. Versions endpoint
echo "Testing /v1/versions..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: admin" http://localhost:8000/api/v1/versions)
if [ "$STATUS" = "200" ]; then
  echo "PASS: Versions endpoint working"
else
  echo "FAIL: Versions endpoint returned $STATUS"
  kill $APP_PID 2>/dev/null
  exit 1
fi

# 6. Old endpoint gone
echo "Verifying /v1/releases is gone..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: admin" http://localhost:8000/api/v1/releases)
if [ "$STATUS" = "404" ]; then
  echo "PASS: Old endpoint removed"
else
  echo "WARNING: Old endpoint still exists ($STATUS)"
fi

kill $APP_PID 2>/dev/null
echo "=== Cold Start: Complete ==="
```

### Recovery Verification Checklist

Run after any reboot, restore, or deployment:

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Database accessible | `pg_isready -h localhost -U postgres` | Ready |
| 2 | `versions` table exists | `\d versions` | Structure shown |
| 3 | `releases` table gone | `\d releases` | "Did not find" |
| 4 | Alembic version correct | `alembic current` | Latest hash |
| 5 | Row counts match backup | Compare with backup | Match |
| 6 | FK joins work | `SELECT ... JOIN versions ...` | Success |
| 7 | Backend starts | `uvicorn app.main:app` | No errors |
| 8 | `/v1/versions` works | `curl /api/v1/versions` | 200 + data |
| 9 | Frontend builds | `npm run build` | Success |
| 10 | UI loads `/versions` | Browser | Renders |

---

## Rollback Procedures

### Per-Phase Rollback

| Phase | Rollback Command |
|-------|------------------|
| Database | `alembic downgrade -1` OR restore SQL backup |
| Backend model | `git checkout -- backend/app/models/` |
| Backend schemas | `git checkout -- backend/app/schemas/` |
| Backend repository | `git checkout -- backend/app/repositories/` |
| Backend exceptions | `git checkout -- backend/app/core/exceptions.py` |
| Backend API | `git checkout -- backend/app/api/` |
| Frontend types | `git checkout -- frontend/src/lib/api.ts` |
| Frontend components | `git checkout -- frontend/src/components/` |
| Frontend routes | `git checkout -- frontend/src/App.tsx` |
| SDK | `git checkout -- sdk/` |

### Full Rollback

```bash
# 1. Restore database
psql -h localhost -U postgres model_registry < backups/XXXXXX/database.sql

# 2. Restore code
git revert HEAD

# 3. Rebuild
cd backend && pip install -e .
cd frontend && npm run build
```

### Rollback Decision Matrix

| Symptom | Likely Cause | Rollback Scope |
|---------|--------------|----------------|
| Backend won't start | Import error | Code only (backend) |
| API returns 500 | Schema mismatch | Database + Code |
| Frontend won't build | TypeScript error | Frontend only |
| Data missing | Migration issue | Database restore |
| Wrong counts | Query logic | Dashboard.tsx only |

---

## Decision Points

| Decision | Options | Recommendation |
|----------|---------|----------------|
| FK column names (`release_id`) | Keep vs rename to `version_id` | **Keep** (less breaking) |
| SDK package name | Renamed to `catapult` | **Renamed** |
| Old `/v1/releases` endpoint | Remove vs keep deprecated | **Remove** (clean break) |
| Frontend route `/releases` | Remove vs redirect | **Redirect** to `/versions` |
| Audit log actions | Update historical vs keep | **Keep old**, change new only |

---

## Execution Order

1. **Pre-work**
   - [ ] Create backup
   - [ ] Review plan with team
   - [ ] Create feature branch

2. **Database** (Phase 1)
   - [ ] Create migration file
   - [ ] Test on staging
   - [ ] Apply to production

3. **Backend** (Phases 2-6)
   - [ ] Rename model file and class
   - [ ] Rename schema file and classes
   - [ ] Rename repository file and class
   - [ ] Update exceptions
   - [ ] Rename endpoint file and update router
   - [ ] Update all imports
   - [ ] Run backend tests

4. **Frontend** (Phases 7-10)
   - [ ] Update types in api.ts
   - [ ] Rename component files
   - [ ] Update routes in App.tsx
   - [ ] Fix Dashboard.tsx counts
   - [ ] Update Sidebar
   - [ ] Run frontend build

5. **SDK** (Phase 11)
   - [ ] Update client methods
   - [ ] Update models
   - [ ] Add backward compat aliases
   - [ ] Update examples

6. **Tests** (Phase 12)
   - [ ] Update all test files
   - [ ] Run full test suite

7. **Documentation** (Phase 13)
   - [ ] Update README.md
   - [ ] Update Help.tsx
   - [ ] Update SDK docs

8. **Verification**
   - [ ] Run all verification scripts
   - [ ] Run recovery verification
   - [ ] Manual UI testing

9. **Deployment**
   - [ ] Deploy to staging
   - [ ] Verify on staging
   - [ ] Deploy to production
   - [ ] Monitor for errors

---

## Appendix: Files Changed Summary

### Backend Files

```
backend/
├── alembic/versions/
│   └── xxxx_rename_releases_to_versions.py  # NEW
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── releases.py → versions.py    # RENAMED
│   │   │   └── models.py                    # MODIFIED (endpoint)
│   │   └── router.py                        # MODIFIED
│   ├── core/
│   │   └── exceptions.py                    # MODIFIED
│   ├── models/
│   │   ├── release.py → version.py          # RENAMED
│   │   ├── model.py                         # MODIFIED (relationship)
│   │   ├── deployment.py                    # MODIFIED (relationship)
│   │   ├── artifact.py                      # MODIFIED (relationship)
│   │   └── docker_build.py                  # MODIFIED (relationship)
│   ├── repositories/
│   │   └── release_repository.py → version_repository.py  # RENAMED
│   └── schemas/
│       ├── release.py → version.py          # RENAMED
│       └── model.py                         # MODIFIED
```

### Frontend Files

```
frontend/src/
├── App.tsx                                  # MODIFIED (routes)
├── components/
│   ├── ReleaseList.tsx → VersionList.tsx    # RENAMED
│   ├── ReleaseDetail.tsx → VersionDetail.tsx # RENAMED
│   ├── CreateReleaseDialog.tsx → CreateVersionDialog.tsx # RENAMED
│   ├── Dashboard.tsx                        # MODIFIED (counts fix)
│   ├── Sidebar.tsx                          # MODIFIED (nav)
│   ├── ModelDetail.tsx                      # MODIFIED
│   └── Help.tsx                             # MODIFIED (docs)
└── lib/
    └── api.ts                               # MODIFIED (types, methods)
```

### SDK Files

```
sdk/python/catapult/
├── client.py                                # MODIFIED
├── models.py                                # MODIFIED
└── example.py                               # MODIFIED
```

### Test Files

```
tests/
├── verify_release_filtering.py → verify_version_filtering.py  # RENAMED
├── e2e_test.py                              # MODIFIED
├── verify_sdk_features.py                   # MODIFIED
└── ...                                      # MODIFIED
```

### Documentation

```
docs/
├── VERSION_RELEASE_REFACTORING_PLAN.md      # THIS FILE
README.md                                    # MODIFIED
```
