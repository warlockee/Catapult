# Docker-Based Release System Design

**Date:** 2025-11-20
**Status:** Design Phase
**Version:** 1.0

---

## Executive Summary

Transform the release creation process from metadata-only to a full Docker image build pipeline. Users upload Dockerfiles, submit build jobs to an async task queue, monitor build progress in real-time, and receive a published Docker image URL upon completion.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Release Form     │  │ Build Monitor    │  │ Build Logs    │ │
│  │ - Model select   │  │ - Progress bar   │  │ - Live stream │ │
│  │ - Dockerfile up  │  │ - Status updates │  │ - Error msgs  │ │
│  │ - Registry pick  │  │ - Cancel button  │  │ - Completion  │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ POST /v1/releases/build                                  │   │
│  │  1. Validate inputs                                      │   │
│  │  2. Save Dockerfile to temp storage                      │   │
│  │  3. Create BuildJob record (status: pending)             │   │
│  │  4. Enqueue Celery task                                  │   │
│  │  5. Return job_id + WebSocket URL                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ GET /v1/builds/{job_id}                                  │   │
│  │  - Get build status, progress, logs                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ WebSocket /v1/builds/{job_id}/stream                     │   │
│  │  - Real-time build log streaming                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Celery + Redis Queue                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Task: build_and_push_release                             │   │
│  │                                                          │   │
│  │ 1. Update status → "downloading_artifacts"               │   │
│  │ 2. Download model checkpoint from S3/Ceph                │   │
│  │ 3. Download linked artifacts                             │   │
│  │ 4. Update status → "building"                            │   │
│  │ 5. Run docker build with streaming logs                  │   │
│  │ 6. Update status → "pushing"                             │   │
│  │ 7. Push to target registry (Docker Hub, etc.)            │   │
│  │ 8. Update status → "completed"                           │   │
│  │ 9. Save image URL + digest to Release record             │   │
│  │                                                          │   │
│  │ On Error:                                                │   │
│  │  - Update status → "failed"                              │   │
│  │  - Save error logs                                       │   │
│  │  - Cleanup temp files                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Docker Engine                              │
│  - Builds images in isolated containers                         │
│  - Streams build logs back to Celery task                       │
│  - Tags images with registry/image:tag                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Container Registry                            │
│  - Docker Hub (docker.io)                                       │
│  - Private Registry (configurable)                              │
│  - AWS ECR, Google GCR, Azure ACR, etc.                         │
│  - Returns: image digest (sha256:...)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema Changes

### New Table: `build_jobs`

```sql
CREATE TABLE build_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    release_id UUID REFERENCES releases(id) ON DELETE CASCADE,

    -- Build configuration
    dockerfile_path VARCHAR(1000) NOT NULL,
    build_context JSONB DEFAULT '{}',  -- Build args, labels, etc.
    target_registry VARCHAR(500) NOT NULL,
    image_name VARCHAR(255) NOT NULL,
    image_tag VARCHAR(100) NOT NULL,

    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    -- pending, downloading_artifacts, building, pushing, completed, failed, cancelled
    progress INTEGER DEFAULT 0,  -- 0-100

    -- Results
    image_url VARCHAR(1000),
    image_digest VARCHAR(255),  -- sha256:...
    size_bytes BIGINT,

    -- Logs and errors
    build_logs TEXT,
    error_message TEXT,

    -- Metadata
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Audit
    created_by VARCHAR(255),  -- API key name

    INDEX idx_build_jobs_release_id (release_id),
    INDEX idx_build_jobs_status (status),
    INDEX idx_build_jobs_created_at (created_at DESC)
);
```

### Updated Table: `releases`

Add fields to track Docker image information:

```sql
ALTER TABLE releases ADD COLUMN build_job_id UUID REFERENCES build_jobs(id);
ALTER TABLE releases ADD COLUMN docker_image_url VARCHAR(1000);
ALTER TABLE releases ADD COLUMN docker_image_digest VARCHAR(255);
```

---

## API Endpoints

### 1. Create Build Job

**POST** `/v1/releases/build`

**Request Body:**
```json
{
  "model_id": "uuid",
  "version": "1.0.0",
  "dockerfile": "base64_encoded_dockerfile_content",
  "target_registry": "docker.io",
  "image_name": "myorg/mymodel",
  "image_tag": "1.0.0-fp16",
  "build_args": {
    "PYTHON_VERSION": "3.11",
    "CUDA_VERSION": "12.1"
  },
  "quantization": "fp16",
  "release_notes": "Initial release",
  "platform": "linux/amd64",
  "artifact_ids": ["uuid1", "uuid2"]  // Optional: specific artifacts to include
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "release_id": "uuid",
  "status": "pending",
  "websocket_url": "ws://localhost/api/v1/builds/{job_id}/stream",
  "created_at": "2025-11-20T10:00:00Z"
}
```

**Validation:**
- Model exists
- Dockerfile is valid base64
- Target registry is allowed (whitelist)
- Image name follows naming convention
- Storage path accessible

---

### 2. Get Build Status

**GET** `/v1/builds/{job_id}`

**Response:**
```json
{
  "id": "uuid",
  "release_id": "uuid",
  "status": "building",
  "progress": 45,
  "started_at": "2025-11-20T10:00:05Z",
  "current_step": "Building layer 5/12",
  "image_url": null,
  "image_digest": null,
  "error_message": null
}
```

---

### 3. Stream Build Logs

**WebSocket** `/v1/builds/{job_id}/stream`

**Messages:**
```json
// Progress update
{
  "type": "progress",
  "status": "building",
  "progress": 45,
  "message": "Step 5/12 : RUN pip install -r requirements.txt"
}

// Log line
{
  "type": "log",
  "timestamp": "2025-11-20T10:00:30Z",
  "line": "Collecting torch>=2.0.0"
}

// Completion
{
  "type": "complete",
  "status": "completed",
  "image_url": "docker.io/myorg/mymodel:1.0.0-fp16",
  "image_digest": "sha256:abc123...",
  "size_bytes": 5368709120
}

// Error
{
  "type": "error",
  "status": "failed",
  "error": "Failed to push image: authentication required"
}
```

---

### 4. Cancel Build

**DELETE** `/v1/builds/{job_id}`

**Response:**
```json
{
  "message": "Build job cancelled",
  "status": "cancelled"
}
```

---

### 5. List Build Jobs

**GET** `/v1/builds`

**Query Params:**
- `release_id` (optional)
- `status` (optional)
- `limit` / `offset` (pagination)

**Response:**
```json
[
  {
    "id": "uuid",
    "release_id": "uuid",
    "model_name": "llama-3-70b",
    "version": "1.0.0",
    "status": "completed",
    "progress": 100,
    "image_url": "docker.io/myorg/llama-3-70b:1.0.0",
    "created_at": "2025-11-20T10:00:00Z",
    "completed_at": "2025-11-20T10:15:30Z"
  }
]
```

---

## Celery Task Implementation

### Task: `build_and_push_release`

**File:** `backend/app/tasks/build_tasks.py`

```python
import docker
import boto3
import base64
from celery import Task
from app.core.celery_app import celery_app
from app.services.build_service import (
    update_build_status,
    save_build_logs,
    download_model_files,
    download_artifacts,
)

@celery_app.task(bind=True, max_retries=3)
def build_and_push_release(
    self: Task,
    job_id: str,
    model_id: str,
    storage_path: str,
    dockerfile_content: str,
    target_registry: str,
    image_name: str,
    image_tag: str,
    build_args: dict,
    artifact_ids: list,
) -> dict:
    """
    Build and push Docker image for a model release.

    Steps:
    1. Download model checkpoint files from S3/Ceph
    2. Download associated artifacts
    3. Create build context directory
    4. Run docker build with log streaming
    5. Push to target registry
    6. Update release record with image URL and digest
    """

    client = docker.from_env()
    build_context_path = f"/tmp/builds/{job_id}"

    try:
        # Step 1: Update status to downloading
        update_build_status(job_id, "downloading_artifacts", progress=10)

        # Step 2: Download model files
        model_path = download_model_files(storage_path, build_context_path)
        save_build_logs(job_id, f"Downloaded model files to {model_path}\n")
        update_build_status(job_id, "downloading_artifacts", progress=20)

        # Step 3: Download artifacts
        for artifact_id in artifact_ids:
            artifact_path = download_artifacts(artifact_id, build_context_path)
            save_build_logs(job_id, f"Downloaded artifact {artifact_id}\n")
        update_build_status(job_id, "downloading_artifacts", progress=30)

        # Step 4: Write Dockerfile
        dockerfile_path = f"{build_context_path}/Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(base64.b64decode(dockerfile_content).decode('utf-8'))

        # Step 5: Build image
        update_build_status(job_id, "building", progress=40)

        full_image_name = f"{target_registry}/{image_name}:{image_tag}"

        # Stream build logs
        build_logs = []
        for line in client.api.build(
            path=build_context_path,
            tag=full_image_name,
            buildargs=build_args,
            decode=True,
            rm=True,  # Remove intermediate containers
            forcerm=True,  # Always remove intermediate containers
        ):
            if 'stream' in line:
                log_line = line['stream'].strip()
                if log_line:
                    build_logs.append(log_line)
                    save_build_logs(job_id, log_line + "\n", append=True)

            # Update progress during build (40-80%)
            if 'aux' in line:
                # Build complete
                update_build_status(job_id, "building", progress=80)

        # Step 6: Get image digest
        image = client.images.get(full_image_name)
        image_digest = image.attrs.get('RepoDigests', [''])[0].split('@')[-1]
        size_bytes = image.attrs.get('Size', 0)

        # Step 7: Push to registry
        update_build_status(job_id, "pushing", progress=85)

        for line in client.api.push(
            full_image_name,
            stream=True,
            decode=True,
        ):
            if 'status' in line:
                save_build_logs(job_id, f"{line['status']}\n", append=True)
            if 'error' in line:
                raise Exception(f"Push failed: {line['error']}")

        # Step 8: Complete
        update_build_status(
            job_id,
            "completed",
            progress=100,
            image_url=full_image_name,
            image_digest=image_digest,
            size_bytes=size_bytes,
        )

        return {
            "status": "completed",
            "image_url": full_image_name,
            "image_digest": image_digest,
            "size_bytes": size_bytes,
        }

    except Exception as e:
        # Update status to failed
        update_build_status(
            job_id,
            "failed",
            error_message=str(e),
        )
        save_build_logs(job_id, f"\nERROR: {str(e)}\n", append=True)
        raise

    finally:
        # Cleanup build context
        import shutil
        if os.path.exists(build_context_path):
            shutil.rmtree(build_context_path)
```

---

## Frontend Implementation

### Updated Release Form

**File:** `Docker Release Registry/src/components/CreateReleaseForm.tsx`

**Key Changes:**

1. **Add Dockerfile Upload**
```tsx
<FormField label="Dockerfile">
  <FileUpload
    accept=".dockerfile,Dockerfile"
    onChange={(file) => setDockerfile(file)}
    required
  />
</FormField>
```

2. **Add Registry Selection**
```tsx
<FormField label="Target Registry">
  <Select
    options={[
      { value: 'docker.io', label: 'Docker Hub' },
      { value: 'ghcr.io', label: 'GitHub Container Registry' },
      { value: 'gcr.io', label: 'Google Container Registry' },
      { value: 'custom', label: 'Custom Registry' },
    ]}
    value={targetRegistry}
    onChange={setTargetRegistry}
  />
</FormField>

{targetRegistry === 'custom' && (
  <FormField label="Registry URL">
    <Input
      placeholder="registry.example.com"
      value={customRegistry}
      onChange={(e) => setCustomRegistry(e.target.value)}
    />
  </FormField>
)}
```

3. **Add Build Args**
```tsx
<FormField label="Build Arguments (Optional)">
  <KeyValueEditor
    pairs={buildArgs}
    onChange={setBuildArgs}
    placeholder={{ key: 'ARG_NAME', value: 'value' }}
  />
</FormField>
```

4. **Submit Handler**
```tsx
const handleSubmit = async (e: FormEvent) => {
  e.preventDefault();

  // Read Dockerfile as base64
  const dockerfileBase64 = await fileToBase64(dockerfile);

  const response = await api.createBuildJob({
    model_id: modelId,
    version,
    dockerfile: dockerfileBase64,
    target_registry: targetRegistry,
    image_name: imageName,
    image_tag: imageTag,
    build_args: buildArgs,
    quantization,
    release_notes: releaseNotes,
    artifact_ids: selectedArtifacts,
  });

  // Navigate to build monitor page
  navigate(`/builds/${response.job_id}`);
};
```

---

### Build Monitor Component

**File:** `Docker Release Registry/src/components/BuildMonitor.tsx`

```tsx
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';

interface BuildStatus {
  id: string;
  status: string;
  progress: number;
  current_step: string;
  image_url?: string;
  error_message?: string;
}

export function BuildMonitor() {
  const { jobId } = useParams();
  const [status, setStatus] = useState<BuildStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Fetch initial status
    api.getBuildJob(jobId).then(setStatus);

    // Connect to WebSocket for live updates
    const websocket = new WebSocket(
      `ws://localhost/api/v1/builds/${jobId}/stream`
    );

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        setStatus(prev => ({
          ...prev!,
          status: data.status,
          progress: data.progress,
          current_step: data.message,
        }));
      } else if (data.type === 'log') {
        setLogs(prev => [...prev, data.line]);
      } else if (data.type === 'complete') {
        setStatus(prev => ({
          ...prev!,
          status: 'completed',
          progress: 100,
          image_url: data.image_url,
        }));
      } else if (data.type === 'error') {
        setStatus(prev => ({
          ...prev!,
          status: 'failed',
          error_message: data.error,
        }));
      }
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, [jobId]);

  const handleCancel = async () => {
    await api.cancelBuildJob(jobId);
  };

  return (
    <div className="build-monitor">
      <h1>Build Status</h1>

      {/* Progress Bar */}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${status?.progress || 0}%` }}
        />
      </div>
      <p>{status?.current_step}</p>

      {/* Status Badge */}
      <StatusBadge status={status?.status} />

      {/* Logs */}
      <div className="build-logs">
        <h2>Build Logs</h2>
        <pre>
          {logs.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </pre>
      </div>

      {/* Actions */}
      {status?.status === 'pending' || status?.status === 'building' ? (
        <button onClick={handleCancel}>Cancel Build</button>
      ) : null}

      {/* Success Result */}
      {status?.status === 'completed' && (
        <div className="build-result">
          <h2>Build Successful! 🎉</h2>
          <p>Image URL: <code>{status.image_url}</code></p>
          <button onClick={() => navigator.clipboard.writeText(status.image_url!)}>
            Copy Docker Pull Command
          </button>
        </div>
      )}

      {/* Error */}
      {status?.status === 'failed' && (
        <div className="build-error">
          <h2>Build Failed ❌</h2>
          <p>{status.error_message}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Infrastructure Requirements

### 1. Add Celery + Redis to Docker Compose

**File:** `docker-compose.yml`

```yaml
services:
  # ... existing services ...

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  celery_worker:
    build: ./backend
    command: celery -A app.core.celery_app worker --loglevel=info --concurrency=2
    volumes:
      - ./backend:/app
      - /var/run/docker.sock:/var/run/docker.sock  # Docker socket for builds
      - build_cache:/tmp/builds
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379/0
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - DOCKER_REGISTRY_USERNAME=${DOCKER_REGISTRY_USERNAME}
      - DOCKER_REGISTRY_PASSWORD=${DOCKER_REGISTRY_PASSWORD}
    depends_on:
      - db
      - redis
    restart: unless-stopped

  celery_beat:
    build: ./backend
    command: celery -A app.core.celery_app beat --loglevel=info
    volumes:
      - ./backend:/app
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

volumes:
  redis_data:
  build_cache:
```

---

### 2. Celery Configuration

**File:** `backend/app/core/celery_app.py`

```python
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "model_registry",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.build_tasks"],
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per build
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks
)
```

---

### 3. WebSocket Support

**File:** `backend/app/api/v1/endpoints/builds.py`

```python
from fastapi import WebSocket, WebSocketDisconnect
from app.services.build_service import get_build_logs_stream

@router.websocket("/{job_id}/stream")
async def build_log_stream(
    websocket: WebSocket,
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream build logs via WebSocket.
    """
    await websocket.accept()

    try:
        # Send initial backlog
        initial_logs = await get_build_logs(db, job_id)
        await websocket.send_json({
            "type": "backlog",
            "logs": initial_logs,
        })

        # Stream new logs
        async for log_entry in get_build_logs_stream(db, job_id):
            await websocket.send_json(log_entry)

            # Stop streaming if build completed or failed
            if log_entry.get('type') in ['complete', 'error']:
                break

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
```

---

## Security Considerations

1. **Registry Authentication**
   - Store registry credentials in environment variables
   - Support multiple authentication methods (token, username/password)
   - Use Docker credential helpers

2. **Dockerfile Validation**
   - Scan for malicious commands
   - Whitelist base images
   - Limit build resource usage (CPU, memory, disk)

3. **Access Control**
   - Only allow authenticated users to create builds
   - Audit all build jobs
   - Rate limit build submissions

4. **Network Isolation**
   - Run builds in isolated network namespace
   - Limit outbound network access during builds
   - Use private registries when possible

---

## Monitoring & Observability

1. **Metrics**
   - Build success/failure rate
   - Average build duration
   - Queue depth
   - Storage usage

2. **Alerts**
   - Build failures
   - Long-running builds
   - Queue backlog
   - Disk space

3. **Logging**
   - Structured logs for all build events
   - Centralized log aggregation
   - Log retention policy

---

## Migration Plan

### Phase 1: Infrastructure Setup (Week 1)
- [ ] Add Redis and Celery to docker-compose
- [ ] Create build_jobs table migration
- [ ] Implement basic Celery task
- [ ] Set up Docker client in worker

### Phase 2: Backend Implementation (Week 2)
- [ ] Build job creation endpoint
- [ ] Celery build task with artifact download
- [ ] WebSocket streaming
- [ ] Status tracking and error handling

### Phase 3: Frontend Implementation (Week 2-3)
- [ ] Update release form with Dockerfile upload
- [ ] Registry selection UI
- [ ] Build monitor component
- [ ] Real-time log viewer

### Phase 4: Testing & Refinement (Week 3-4)
- [ ] Integration tests
- [ ] Load testing
- [ ] Security review
- [ ] Documentation

### Phase 5: Deployment (Week 4)
- [ ] Staging deployment
- [ ] Production rollout
- [ ] Monitoring setup
- [ ] User training

---

## Open Questions

1. **Resource Limits**: What limits should we set for build jobs (CPU, memory, timeout)?
2. **Storage**: Where should we store build contexts and logs long-term?
3. **Cleanup**: How long should we retain build logs and temp files?
4. **Concurrency**: How many parallel builds should we allow?
5. **Retries**: Should failed builds auto-retry? How many times?

---

## Future Enhancements

- [ ] Multi-architecture builds (linux/amd64, linux/arm64)
- [ ] Build caching optimization
- [ ] Vulnerability scanning (Trivy, Clair)
- [ ] Image signing (Sigstore)
- [ ] Integration with CI/CD (GitHub Actions, GitLab CI)
- [ ] Build templates library
- [ ] Scheduled rebuilds
- [ ] Build analytics dashboard
