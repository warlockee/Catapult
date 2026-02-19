# Controller-Executor Architecture

## Overview

Catapult uses a decoupled architecture that separates the **control plane** (triggering and monitoring) from the **executor plane** (running evaluations) and the **inference plane** (GPU model serving).

This document covers:
1. Runtime architecture (evaluation/benchmark execution)
2. Build pipeline (Docker image building for inference)
3. Deployment pipeline (getting images to inference nodes)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE (Local Host)                          │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Frontend   │────▶│   Backend    │────▶│    Redis     │                │
│  │   (React)    │     │  (FastAPI)   │     │   (Queue)    │                │
│  │              │     │              │     │              │                │
│  │  - Trigger   │     │  - REST API  │     │  - Task      │                │
│  │  - Monitor   │     │  - DB CRUD   │     │    Queue     │                │
│  │  - Display   │     │  - Auth      │     │  - Pub/Sub   │                │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                │
│                                                    │                        │
│                              ┌─────────────────────┘                        │
│                              ▼                                              │
│                       ┌──────────────┐                                      │
│                       │    Celery    │                                      │
│                       │   Worker(s)  │                                      │
│                       │              │                                      │
│                       │  - Load data │                                      │
│                       │  - Send HTTP │                                      │
│                       │  - Calc WER  │                                      │
│                       └──────┬───────┘                                      │
│                              │                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │ HTTP Requests
                               │ (audio data)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PLANE (Remote GPU Host)                      │
│                                                                             │
│  Production Endpoint: http://gpu-node-1.example.com:26007                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    ASR Model Server (vLLM/TGI)                        │  │
│  │                                                                       │  │
│  │  Endpoints:                                                           │  │
│  │  - POST /v1/audio/transcriptions  (audio → text)                      │  │
│  │  - POST /v1/chat/completions      (multimodal inference)              │  │
│  │  - GET  /v1/models                (model discovery)                   │  │
│  │  - GET  /health                   (health check)                      │  │
│  │                                                                       │  │
│  │  GPU: H100/A100                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Control Plane

| Component | Responsibility | Stateless? |
|-----------|---------------|------------|
| Frontend | UI for triggering evaluations, displaying progress/results | Yes |
| Backend API | REST API, database operations, authentication | Yes |
| PostgreSQL | Persistent storage for benchmarks, deployments, results | No |
| Redis | Task queue, caching, pub/sub for real-time updates | No |

### Executor Plane

| Component | Responsibility | Stateless? |
|-----------|---------------|------------|
| Celery Worker | Execute benchmark/evaluation tasks, orchestrate HTTP calls | Yes |

### Inference Plane

| Component | Responsibility | Stateless? |
|-----------|---------------|------------|
| Model Server | GPU inference, audio transcription, model serving | Yes* |

*Model weights are loaded but inference is stateless per request.

## Data Flow

### Benchmark/Evaluation Flow

```
1. User clicks "Run Evaluation" in UI
   └──▶ POST /api/v1/benchmarks/async
        {
          endpoint_url: "http://gpu-node-1:26007",
          production_endpoint_id: 367,
          asr_eval_dataset: "/fsx/data/.../common_voice.arrow",
          evaluation_only: true
        }

2. Backend creates Benchmark record (status: pending)
   └──▶ Queues Celery task: run_benchmark_task(benchmark_id)

3. Celery Worker picks up task
   └──▶ Loads audio samples from dataset
   └──▶ For each sample:
        └──▶ POST http://gpu-node-1:26007/v1/audio/transcriptions
             Body: { audio: base64, model: "..." }
        └──▶ Receive transcription response
        └──▶ Compare with ground truth, accumulate WER

4. Worker updates Benchmark record with results
   └──▶ status: completed, wer: 0.144, cer: 0.089

5. Frontend polls for status updates
   └──▶ GET /api/v1/benchmarks/{id}
   └──▶ Displays progress and final results
```

## Scalability Considerations

### Current State (Single Host)

- Control plane and executor on same host
- Single Celery worker process
- Suitable for: Development, small-scale evaluations

### Future: Horizontal Scaling

#### Option 1: Multiple Workers (Same Host)
```yaml
# docker-compose.yml
services:
  worker:
    deploy:
      replicas: 4  # Scale workers
```

**Pros**: Simple, shared filesystem access
**Cons**: Limited by single host resources

#### Option 2: Distributed Workers (Multiple Hosts)
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Control Plane  │     │  Worker Host 1  │     │  Worker Host 2  │
│  (API + Redis)  │────▶│  Celery Worker  │     │  Celery Worker  │
│                 │────▶│                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────────────────────────┐
                        │    Shared Storage (NFS/S3/FSx)      │
                        │    /fsx/data/audio_eval/            │
                        └─────────────────────────────────────┘
```

**Implementation**:
```python
# Worker can run on any host with:
celery -A app.worker worker --concurrency=4 -Q benchmarks

# Redis connection string points to control plane
REDIS_URL=redis://control-plane-host:6379
```

**Pros**: True horizontal scaling, fault tolerance
**Cons**: Requires shared storage, network configuration

#### Option 3: Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-worker
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: worker
        image: model-registry-worker:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
        volumeMounts:
        - name: data
          mountPath: /fsx/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: fsx-pvc
```

**Pros**: Auto-scaling, self-healing, cloud-native
**Cons**: Kubernetes complexity, infrastructure cost

#### Option 4: Serverless Execution
```
┌─────────────────┐     ┌─────────────────┐
│  Control Plane  │────▶│  AWS Lambda /   │
│                 │     │  Cloud Run      │
└─────────────────┘     └─────────────────┘
                               │
                        Triggered per evaluation
                        Scales to zero when idle
```

**Pros**: Pay-per-use, infinite scale, no infra management
**Cons**: Cold starts, execution time limits, data access patterns

### Future: Multi-Region Inference

```
                    ┌──────────────────────────────────────────┐
                    │            Control Plane                  │
                    │         (Single Region: US-East)          │
                    └──────────────────┬───────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   Inference Plane   │   │   Inference Plane   │   │   Inference Plane   │
│   (US-East: H100)   │   │   (EU-West: A100)   │   │   (Asia: H100)      │
│   Latency: 10ms     │   │   Latency: 80ms     │   │   Latency: 150ms    │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

### Scaling Recommendations

| Scale | Workers | Architecture | Use Case |
|-------|---------|--------------|----------|
| Small | 1-2 | Single host | Development, testing |
| Medium | 4-8 | Single host, multiple workers | Team usage |
| Large | 10-50 | Distributed workers | Production, CI/CD |
| Enterprise | 50+ | Kubernetes + auto-scaling | Multi-tenant SaaS |

## Configuration

### Environment Variables

```bash
# Control Plane
DATABASE_URL=postgresql://user:pass@localhost:5432/registry
REDIS_URL=redis://localhost:6379

# Worker Configuration
CELERY_BROKER_URL=redis://localhost:6379
CELERY_RESULT_BACKEND=redis://localhost:6379
CELERY_WORKER_CONCURRENCY=4

# For distributed workers, point to control plane
CELERY_BROKER_URL=redis://control-plane-host:6379
```

### Worker Queues

```python
# Separate queues for different workloads
CELERY_TASK_ROUTES = {
    'app.worker.run_benchmark_task': {'queue': 'benchmarks'},
    'app.worker.run_deployment_task': {'queue': 'deployments'},
}

# Start workers for specific queues
celery -A app.worker worker -Q benchmarks --concurrency=4
celery -A app.worker worker -Q deployments --concurrency=2
```

## Monitoring

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `celery_tasks_pending` | Tasks waiting in queue | > 100 |
| `celery_tasks_active` | Currently executing tasks | - |
| `benchmark_duration_seconds` | Time to complete evaluation | > 3600 |
| `inference_latency_p95` | 95th percentile inference time | > 5000ms |
| `worker_memory_usage` | Worker memory consumption | > 80% |

### Health Checks

```bash
# Check worker status
celery -A app.worker inspect active

# Check queue length
celery -A app.worker inspect reserved

# Check worker stats
celery -A app.worker inspect stats
```

## Security Considerations

1. **Network Isolation**: Control plane should be in private subnet
2. **API Authentication**: All endpoints require API key
3. **mTLS**: Consider mutual TLS for worker ↔ inference communication
4. **Secrets Management**: Use vault/secrets manager for credentials
5. **Audit Logging**: Log all benchmark triggers and results

## Docker Build Pipeline

### Current State

Docker images for inference are built separately from the control plane:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BUILD PIPELINE                                    │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  Dockerfile  │────▶│   Docker     │────▶│   Registry   │                │
│  │  (local)     │     │   Build      │     │   (Harbor/   │                │
│  │              │     │              │     │    ECR)      │                │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                │
│                                                    │                        │
│  Examples:                                         │                        │
│  - kb/dockers/Dockerfile.asr                       │                        │
│  - Self-contained vLLM + model weights             │                        │
│                                                    │                        │
└────────────────────────────────────────────────────┼────────────────────────┘
                                                     │
                                                     │ docker pull
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PLANE (GPU Nodes)                            │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   docker run -d --gpus all \                                          │  │
│  │     -p 26007:8000 \                                                   │  │
│  │     registry.example.com/asr-model:v3-2b-svad                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Build Locations

| Build Location | Pros | Cons |
|----------------|------|------|
| **Local machine** | Fast iteration, easy debugging | Manual, not reproducible |
| **CI/CD (GitHub Actions)** | Automated, reproducible | Slower, needs GPU for testing |
| **Build server (dedicated)** | Fast, has GPU access | Infrastructure cost |
| **Control plane** | Centralized, integrated | Resource contention |

### Current Dockerfile Structure

```dockerfile
# kb/dockers/Dockerfile.asr
FROM your-vllm:latest

# Self-contained: includes model weights
COPY model_weights/ /models/

# Server configuration
ENV MODEL_PATH=/models/audio-model-v3-2b
ENV PORT=8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", ...]
```

### Future: Integrated Build Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE                                       │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Frontend   │────▶│   Backend    │────▶│    Build     │                │
│  │   (UI)       │     │   (API)      │     │   Service    │                │
│  │              │     │              │     │   (Celery)   │                │
│  │  "Build      │     │  POST /v1/   │     │              │                │
│  │   Image"     │     │  builds      │     │  - Clone     │                │
│  └──────────────┘     └──────────────┘     │  - Build     │                │
│                                            │  - Push      │                │
│                                            │  - Test      │                │
│                                            └──────┬───────┘                │
│                                                   │                        │
└───────────────────────────────────────────────────┼────────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────┐
                    │                               │                   │
                    ▼                               ▼                   ▼
           ┌──────────────┐              ┌──────────────┐     ┌──────────────┐
           │   Registry   │              │  Build Node  │     │  GPU Test    │
           │   (Harbor)   │◀─────────────│  (Docker)    │────▶│  Node        │
           │              │   push       │              │     │  (Validate)  │
           └──────────────┘              └──────────────┘     └──────────────┘
```

### Build Task Schema (Future)

```python
class DockerBuild(Base):
    id: UUID
    status: str  # pending, building, pushing, testing, completed, failed

    # Source
    dockerfile_path: str
    build_context: str
    git_ref: str

    # Target
    image_name: str
    image_tag: str
    registry_url: str

    # Build config
    build_args: Dict[str, str]
    target_platform: str  # linux/amd64, linux/arm64

    # Results
    image_digest: str
    image_size_bytes: int
    build_duration_seconds: float

    # Validation
    health_check_passed: bool
    inference_test_passed: bool

    created_at: datetime
    completed_at: datetime
```

### Build + Deploy + Evaluate Flow

```
1. User uploads/selects Dockerfile
   └──▶ POST /api/v1/builds
        { dockerfile: "...", image_name: "asr-model", tag: "v3" }

2. Build service builds and pushes image
   └──▶ docker build -t registry/asr-model:v3 .
   └──▶ docker push registry/asr-model:v3

3. User deploys to production node
   └──▶ POST /api/v1/production-endpoints
        { image: "registry/asr-model:v3", host: "gpu-node-1", port: 26007 }

4. User triggers evaluation
   └──▶ POST /api/v1/benchmarks/async
        { production_endpoint_id: 367, evaluation_only: true }

5. Results stored and displayed
   └──▶ WER: 14.4%, CER: 8.9%
```

### Remote Build Execution

For large models, building on the control plane may not be feasible:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Control Plane  │────▶│  Build Queue    │────▶│  Build Agent    │
│                 │     │  (Redis)        │     │  (GPU Node)     │
│  Trigger build  │     │                 │     │                 │
│                 │     │                 │     │  - Has GPU      │
│                 │◀────│  Status updates │◀────│  - Has storage  │
│                 │     │                 │     │  - Runs docker  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

Build agent can run on:
- Same node as inference (if idle)
- Dedicated build server with GPU
- Cloud build service (with GPU support)

## Future Enhancements

1. **Priority Queues**: Urgent evaluations skip the queue
2. **Resource Quotas**: Limit concurrent evaluations per user/team
3. **Result Caching**: Cache evaluation results for identical configs
4. **Webhook Notifications**: Notify external systems on completion
5. **Cost Attribution**: Track GPU-hours per team/project
6. **Integrated Build Pipeline**: Build Docker images from control plane
7. **Build Caching**: Layer caching for faster rebuilds
8. **Multi-arch Builds**: Support ARM64 for cost optimization
9. **Image Scanning**: Security vulnerability scanning before deploy
