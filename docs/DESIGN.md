# Catapult - Design Document

## Overview

MLOps platform for ML Engineers and Release Engineers to track models, releases, artifacts, and deployments.

**Use Case**: Single organization, internal use only - no multi-tenancy needed.

---

## System Architecture

### Core Concepts

1. **Model** - Metadata about an ML model (name, company it's built for, storage location)
2. **Release** - A specific Docker image version of a model (with quantization, digest)
3. **Artifact** - Pre-built packages/binaries baked into a release (wheels, binaries)
4. **Deployment** - A live deployment of a release to infrastructure (with endpoint URL)

### Flow

```
Model Checkpoint (S3/Ceph)
    ↓
Model Metadata (Registry)
    ↓
Docker Build (bakes in model + artifacts)
    ↓
Release (Docker Image in Registry)
    ↓
Deploy to K8s/Cloud
    ↓
Deployment (Live Endpoint)
```

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL (images)                            │
├─────────────────────────────────────────────────────────────────┤
│ PK  id              UUID                                         │
│     name            String(255)  UNIQUE                          │
│     company         String(255)  (customer it's built for)      │
│     base_model      String(100)  ("llama-3", "gpt-4")          │
│     parameter_count String(50)   ("7B", "70B", "405B")         │
│     storage_path    String(1000) (S3/Ceph path to model files) │
│     description     Text                                         │
│     tags            JSONB        (["production", "experimental"])│
│     metadata        JSONB        (flexible key-value)            │
│     created_at      DateTime                                     │
│     updated_at      DateTime                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N (One-to-Many)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RELEASE (releases)                         │
├─────────────────────────────────────────────────────────────────┤
│ PK  id              UUID                                         │
│ FK  image_id        UUID  → images.id                           │
│     version         String(100)  ("1.0.0", "2.0.0")            │
│     tag             String(100)  ("v1.0-fp16", "latest")        │
│     digest          String(255)  (Docker sha256:...)            │
│     quantization    String(50)   ("fp16", "int8", "int4", "fp8")│
│     size_bytes      BigInteger                                   │
│     platform        String(50)   ("linux/amd64", etc.)         │
│     architecture    String(50)   ("amd64", "arm64")             │
│     os              String(50)   ("linux", "darwin")            │
│     status          String(50)   ("active", "deprecated")       │
│     release_notes   Text                                         │
│     metadata        JSONB                                        │
│     created_at      DateTime                                     │
│                                                                   │
│ UNIQUE: (image_id, version, quantization)                        │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    │ 1:N                       │ 1:N
                    │                           │
                    ▼                           ▼
┌──────────────────────────┐      ┌─────────────────────────────┐
│   ARTIFACT (artifacts)   │      │  DEPLOYMENT (deployments)   │
├──────────────────────────┤      ├─────────────────────────────┤
│ PK  id          UUID     │      │ PK  id          UUID        │
│ FK  release_id  UUID     │      │ FK  release_id  UUID        │
│     name        String   │      │     environment String(100) │
│     type        String   │      │     cluster     String(255) │
│     file_path   String   │      │     k8s_namespace String    │
│     size_bytes  BigInt   │      │     endpoint_url String     │
│     checksum    String   │      │     replicas    Integer     │
│     platform    String   │      │     status      String(50)  │
│     python_ver  String   │      │     deployed_by String(255) │
│     metadata    JSONB    │      │     deployed_at DateTime    │
│     created_at  DateTime │      │     terminated_at DateTime  │
│     uploaded_by String   │      │     metadata    JSONB       │
└──────────────────────────┘      └─────────────────────────────┘
```

### Relationships

| Parent | Child | Type | Example |
|--------|-------|------|---------|
| **Model** | **Release** | 1:N | llama-3-7b → [v1.0-fp16, v1.0-int8, v2.0-fp16] |
| **Release** | **Artifact** | 1:N | v1.0-fp16 → [inference-1.0.0.whl, cuda-kernels.so] |
| **Release** | **Deployment** | 1:N | v1.0-fp16 → [production, staging, development] |

---

## Data Model Details

### Model (images table)

Metadata about an ML model.

**Key Fields:**
- `name` - Unique model name (e.g., "llama-3-7b-companyA")
- `company` - Which customer/company the model is built for
- `base_model` - Base architecture (e.g., "llama-3", "gpt-4")
- `parameter_count` - Model size (e.g., "7B", "70B", "405B")
- `storage_path` - S3/Ceph path to model checkpoint files (e.g., "s3://ml-models/companyA/llama-3-7b/" or "/mnt/ceph/models/llama-3-7b/")
- `tags` - JSONB array for flexible tagging (e.g., ["production", "chatbot"])
- `metadata` - JSONB for additional custom fields

**Example:**
```json
{
  "name": "llama-3-7b-companyA",
  "company": "CompanyA",
  "base_model": "llama-3",
  "parameter_count": "7B",
  "storage_path": "s3://ml-models/companyA/llama-3-7b/",
  "tags": ["production", "chatbot", "customer-service"],
  "metadata": {
    "training_date": "2024-01-15",
    "use_case": "customer service chatbot"
  }
}
```

### Release (releases table)

A specific Docker image version containing the model + artifacts.

**Key Fields:**
- `version` - Semantic version (e.g., "1.0.0", "2.0.0")
- `tag` - Docker tag (e.g., "v1.0-fp16", "latest")
- `digest` - Docker image SHA256 digest
- `quantization` - Quantization type (fp16, int8, int4, fp8)
- `status` - Release status (active, deprecated)
- `release_notes` - What changed in this release

**Naming Convention:**
```
Docker images are built separately and pushed to a registry.
Tag format: v{version}-{quantization}
Example: v1.0-fp16, v2.0-int8
```

**Example:**
```json
{
  "image_id": "uuid-model-123",
  "version": "1.0.0",
  "tag": "v1.0-fp16",
  "digest": "sha256:abc123...",
  "quantization": "fp16",
  "size_bytes": 15000000000,
  "status": "active",
  "release_notes": "Initial release with fp16 quantization"
}
```

### Artifact (artifacts table)

Pre-built packages/binaries baked into the release Docker image.

**Types:**
- `wheel` - Python wheel files (.whl)
- `binary` - Compiled binaries
- `sdist` - Source distributions
- `tarball` - Compressed archives

**Example:**
```json
{
  "release_id": "uuid-release-456",
  "name": "inference-server-1.0.0-py311-linux_x86_64.whl",
  "type": "wheel",
  "file_path": "s3://artifacts/inference-server-1.0.0.whl",
  "size_bytes": 50000000,
  "checksum": "sha256:def456...",
  "platform": "linux_x86_64",
  "python_version": "3.11"
}
```

### Deployment (deployments table)

A live deployment of a release to infrastructure.

**Key Fields:**
- `environment` - Target environment (production, staging, development)
- `cluster` - Kubernetes cluster name
- `k8s_namespace` - Kubernetes namespace
- `endpoint_url` - **CRITICAL**: The actual API endpoint URL
- `replicas` - Number of replicas running
- `status` - Deployment status (deploying, success, failed, terminated)

**Example:**
```json
{
  "release_id": "uuid-release-456",
  "environment": "production",
  "cluster": "k8s-prod-us-west-2",
  "k8s_namespace": "ml-prod",
  "endpoint_url": "https://api.companyA.prod/v1/llama",
  "replicas": 3,
  "status": "success",
  "deployed_by": "alice@company.com",
  "deployed_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "gpu_type": "nvidia-a100",
    "deployed_via": "argocd",
    "helm_release": "llama-prod-v1"
  }
}
```

---

## Common Queries

### For ML Engineers

```sql
-- Find all models for a customer
SELECT * FROM images WHERE company = 'CompanyA';

-- Find all 7B models
SELECT * FROM images WHERE parameter_count = '7B';

-- Find models with specific tag
SELECT * FROM images WHERE tags ? 'production';

-- Find all fp16 releases
SELECT * FROM releases WHERE quantization = 'fp16';

-- Get latest release for a model
SELECT * FROM releases
WHERE image_id = :model_id
ORDER BY created_at DESC
LIMIT 1;
```

### For Release Engineers

```sql
-- Find all production deployments
SELECT d.*, m.name, r.version
FROM deployments d
JOIN releases r ON d.release_id = r.id
JOIN images m ON r.image_id = m.id
WHERE d.environment = 'production' AND d.status = 'success';

-- Find all deployments for a customer
SELECT d.*, m.name, r.version
FROM deployments d
JOIN releases r ON d.release_id = r.id
JOIN images m ON r.image_id = m.id
WHERE m.company = 'CompanyA';

-- Get deployment history for a model
SELECT d.*
FROM deployments d
JOIN releases r ON d.release_id = r.id
WHERE r.image_id = :model_id
ORDER BY d.deployed_at DESC;

-- Find failed deployments
SELECT * FROM deployments
WHERE status = 'failed'
ORDER BY deployed_at DESC;
```

---

## API Endpoints

### Models

```
POST   /v1/images              - Register a new model
GET    /v1/images              - List all models (with filters)
GET    /v1/images/{id}         - Get model details
PUT    /v1/images/{id}         - Update model metadata
DELETE /v1/images/{id}         - Delete model
```

### Releases

```
POST   /v1/releases            - Create a new release
GET    /v1/releases            - List all releases (with filters)
GET    /v1/releases/{id}       - Get release details
PUT    /v1/releases/{id}       - Update release (e.g., deprecate)
DELETE /v1/releases/{id}       - Delete release
```

### Artifacts

```
POST   /v1/artifacts           - Upload artifact metadata
GET    /v1/artifacts           - List artifacts for a release
GET    /v1/artifacts/{id}      - Get artifact details
DELETE /v1/artifacts/{id}      - Delete artifact
```

### Deployments

```
POST   /v1/deployments         - Record a deployment
GET    /v1/deployments         - List deployments (with filters)
GET    /v1/deployments/{id}    - Get deployment details
PUT    /v1/deployments/{id}    - Update deployment status
DELETE /v1/deployments/{id}    - Mark deployment as terminated
```

---

## Example Workflow

### 1. Register Model

```bash
curl -X POST http://localhost/api/v1/images \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama-3-7b-companyA",
    "company": "CompanyA",
    "base_model": "llama-3",
    "parameter_count": "7B",
    "storage_path": "s3://ml-models/companyA/llama-3-7b/",
    "tags": ["production", "chatbot"]
  }'
```

### 2. Build Docker Image (External Process)

```bash
# Pull model from storage
aws s3 sync s3://ml-models/companyA/llama-3-7b/ ./model/

# Build Docker image with model + artifacts
docker build -t myregistry.com/llama-3-7b-companyA:v1.0-fp16 .

# Push to registry
docker push myregistry.com/llama-3-7b-companyA:v1.0-fp16

# Get digest
DIGEST=$(docker inspect --format='{{.RepoDigests}}' myregistry.com/llama-3-7b-companyA:v1.0-fp16)
```

### 3. Create Release

```bash
curl -X POST http://localhost/api/v1/releases \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "uuid-model-123",
    "version": "1.0.0",
    "tag": "v1.0-fp16",
    "digest": "sha256:abc123...",
    "quantization": "fp16",
    "platform": "linux/amd64"
  }'
```

### 4. Record Artifacts

```bash
curl -X POST http://localhost/api/v1/artifacts \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "release_id": "uuid-release-456",
    "name": "inference-1.0.0-py311.whl",
    "artifact_type": "wheel",
    "file_path": "s3://artifacts/inference-1.0.0.whl",
    "size_bytes": 50000000,
    "checksum": "sha256:def456...",
    "platform": "linux_x86_64",
    "python_version": "3.11"
  }'
```

### 5. Deploy to Production

```bash
# Deploy via kubectl/helm (external process)
kubectl apply -f deployment.yaml

# Record deployment in registry
curl -X POST http://localhost/api/v1/deployments \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "release_id": "uuid-release-456",
    "environment": "production",
    "cluster": "k8s-prod-us-west-2",
    "k8s_namespace": "ml-prod",
    "endpoint_url": "https://api.companyA.prod/v1/llama",
    "replicas": 3,
    "status": "success",
    "metadata": {
      "gpu_type": "nvidia-a100",
      "deployed_via": "argocd"
    }
  }'
```

---

## Schema Changes from Current

### images Table (now "models")

**Current Implementation (2025-11-20):**
- `storage_path` (String(1000), NOT NULL) - **REQUIRED** - S3/Ceph path to model files
- `repository` (String(500), NULLABLE) - **OPTIONAL** - Legacy field, kept for backward compatibility
- `company` (String(255)) - Which customer the model is for
- `tags` (JSONB) - Flexible tagging array
- `base_model` (String(100)) - Base model architecture
- `parameter_count` (String(50)) - Model size
- `metadata` (JSONB) - Additional custom fields

**Migration Applied:** Table renamed from `images` to `models`, `storage_path` now required field

### releases Table

**Add:**
- `quantization` (String(50)) - Quantization type
- `release_notes` (Text) - What changed
- `status` (String(50)) - Release status

**Keep:**
- All existing fields

**Update Unique Constraint:**
- From: `(image_id, version)`
- To: `(image_id, version, quantization)`

### artifacts Table

**No changes needed** - Already has all required fields

### deployments Table

**Add:**
- `cluster` (String(255)) - K8s cluster name
- `k8s_namespace` (String(255)) - K8s namespace (rename from `namespace`)
- `endpoint_url` (String(500)) - **CRITICAL**: The actual endpoint
- `replicas` (Integer) - Number of replicas
- `terminated_at` (DateTime, nullable) - When deployment was terminated

**Rename:**
- `environment` already exists (keep)
- Move metadata fields to `metadata` JSONB

---

## Future Considerations

### Things We're NOT Implementing (YAGNI)

- ❌ Multi-tenancy / Organizations
- ❌ User accounts / Complex permissions
- ❌ Namespaces
- ❌ Resource quotas
- ❌ Usage billing
- ❌ Complex audit logging (API key logs are enough)

### Things We MIGHT Add Later

- Health check monitoring for deployments
- Automatic deployment status updates
- Model performance metrics tracking
- Rollback functionality
- A/B testing metadata
- Cost tracking per deployment

---

## Authentication

Using existing API Key system:
- ML Engineers get API keys with full access
- Release Engineers get API keys with full access
- No complex role-based access control needed (internal team)

---

## Frontend Features

### Dashboard
- List of all models (filterable by company, tags, parameter count)
- Search functionality
- Model cards with key info

### Model Detail Page
- Model metadata
- Release timeline (all versions)
- Active deployments with endpoint URLs
- Artifacts list

### Deployment View
- All active deployments across environments
- Filter by environment, status, customer
- Endpoint URLs prominently displayed
- Deployment history

### Search/Filter
- By company
- By tags (production, experimental, etc.)
- By base model (llama-3, gpt-4, etc.)
- By parameter count (7B, 70B, etc.)
- By environment (production, staging, dev)

---

## Migration Plan

1. Create new columns in `images` table
2. Create new columns in `releases` table
3. Create new columns in `deployments` table
4. Update unique constraint on `releases`
5. Update API schemas
6. Update frontend to use new fields
7. Backfill existing data (if needed)

---

## Summary

**Simple, focused design for internal ML/Release Engineering use:**

- 4 core tables: Models, Releases, Artifacts, Deployments
- Tag-based flexible organization (no complex namespaces)
- Track actual deployment endpoints (the key use case)
- No multi-tenancy overhead
- Easy to query and understand
