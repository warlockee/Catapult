# Catapult

**The open-source MLOps platform for teams that actually ship models to production.**

Most model registries stop at tracking metadata. Catapult picks up where they leave off — it builds your Docker images, deploys your models, benchmarks performance, evaluates quality, and promotes to production. One platform, one `deploy.sh`, zero glue scripts.

---

## The Problem

You trained a great model. Now what?

- You write a Dockerfile. Then another for a different GPU. Then another for ARM. Then one for audio models. Each one is slightly different and lives in someone's home directory.
- You `ssh` into machines to deploy, then forget which GPU is running what.
- Benchmarking is a Jupyter notebook someone ran once and lost.
- Your "model registry" is a shared Google Sheet with columns like "status" and "notes" that nobody updates.
- Promoting to production means pinging three people on Slack and hoping the deploy script still works.

Sound familiar?

## The Solution

Catapult is a self-hosted platform that manages the full lifecycle of ML models — from registration through deployment to production promotion. Built by an ML team that got tired of duct-taping together shell scripts, spreadsheets, and manual processes.

```bash
git clone https://github.com/warlockee/Catapult
cd Catapult
./deploy.sh
```

That's it. Backend, frontend, worker, database, reverse proxy — all running.

---

## What It Does

### Model & Version Management
Register models, track versions, promote releases. Every version carries its metadata, training metrics, quantization details, and lineage — stored as flexible JSONB, not rigid schemas you have to fight.

```python
from catapult import Registry

registry = Registry(base_url="http://localhost/api", api_key="your-key")

model = registry.create_model(name="myorg/llama-3-8b", server_type="vllm")
version = registry.create_version(
    model_name=model.name,
    version="1.0.0",
    metadata={"accuracy": 0.95, "framework": "pytorch-2.1"}
)
registry.promote_version(version.id)  # Mark as official release
```

### Docker Build System
Trigger Docker builds from the UI with real-time log streaming. Choose from **25+ specialized Dockerfile templates** covering vLLM, ASR, TTS, embedding, and multimodal models — across CUDA (A100/H100), CPU, ROCm, ARM, Neuron (AWS Inferentia), TPU, and Gaudi (HPU) platforms. Stuck builds recover automatically. Old images get garbage-collected.

### One-Click Deployments
Deploy containers directly from the registry with GPU-aware scheduling. Monitor health, tail logs, restart — all from the dashboard. The system auto-detects available GPUs and allocates them to deployments.

### Performance Benchmarking
Run latency and throughput benchmarks against any deployment. Track TTFT, p50/p95/p99 latencies, tokens/second, and error rates. Execute inline or in isolated Docker containers. Compare across versions to catch regressions before they hit production.

### Quality Evaluation
Pluggable evaluation framework with factory-based registration. Ships with ASR evaluation (WER, CER) out of the box — add your own evaluators for LLM quality, vision accuracy, or any domain-specific metric. Quality and performance tracked separately because a fast model that hallucinates isn't a good model.

### Artifact Management
Upload wheels, checkpoints, configs, and binaries. Browse shared filesystems (Ceph, NFS, FSx). Attach multiple artifacts to Docker builds. SHA256 integrity verification included.

### GPU Fleet Visibility
See which models run on which machines, which GPUs are allocated, and what's available. No more `ssh`-ing into boxes to figure out what's running where.

### MLflow Bridge
Already using MLflow for experiment tracking? Catapult isn't a replacement — it's the next step. Link versions to MLflow runs and experiments, with automatic metadata sync.

---

## Architecture

```
                    ┌─────────────┐
                    │    Nginx    │ :8080
                    │  (reverse   │
                    │   proxy)    │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────┴──────┐          ┌──────┴──────┐
       │   React UI  │          │   FastAPI   │ :8000
       │  TypeScript  │          │   Backend   │
       │   Vite SPA   │          │   (async)   │
       └─────────────┘          └──────┬──────┘
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                  ┌──────┴──────┐ ┌───┴────┐ ┌──────┴──────┐
                  │   Celery    │ │ Redis  │ │  PostgreSQL │
                  │   Worker    │ │ Queue  │ │     (DB)    │
                  │+Docker sock │ │        │ │             │
                  └─────────────┘ └────────┘ └─────────────┘
```

**Backend**: FastAPI with async SQLAlchemy 2.0, Pydantic v2, SSE log streaming, domain event system
**Frontend**: React 18, TypeScript, Shadcn/ui, TanStack Query v5, Recharts
**Worker**: Celery with Docker socket access for builds, deployments, and benchmarks
**Database**: PostgreSQL 15 with JSONB columns and GIN indexes for flexible metadata
**Queue**: Redis 7 for task brokering and result caching

### What Makes the Architecture Good

- **Repository pattern** — Clean separation between API endpoints, business logic, and data access. Every model has a dedicated repository with shared base.
- **Domain exceptions** — 25+ typed exceptions (`NotFoundError`, `AlreadyExistsError`, `OperationError`) mapped to HTTP codes. No raw `HTTPException` in business logic.
- **Task dispatcher protocol** — `CeleryTaskDispatcher` for production, `NoOpTaskDispatcher` for testing. Adding new async task types means implementing one interface.
- **Event-driven** — Domain events (`ReleaseCreated`, `DockerBuildCompleted`, `DeploymentStatusChanged`) enable loose coupling. Auto-trigger builds on version creation, notifications on deployment failures.
- **Pluggable evaluators** — Register new evaluation types via factory pattern. No core changes needed.

---

## How It Compares

| Feature | MLflow | DVC | Weights & Biases | **Catapult** |
|---|---|---|---|---|
| Model tracking | Yes | Yes | Yes | **Yes** |
| Docker builds from UI | No | No | No | **25+ templates** |
| Deploy from UI | No | No | No | **GPU-aware** |
| Benchmarking | No | No | No | **TTFT, p50-p99, TPS** |
| Quality evaluation | No | No | No | **Pluggable framework** |
| GPU fleet management | No | No | No | **Yes** |
| Filesystem integration | No | Partial | No | **Ceph, NFS, FSx** |
| Self-hosted, single command | Partial | Yes | No | **Yes** |
| Audit logging | Enterprise | No | Enterprise | **Included** |
| RBAC (API keys) | Enterprise | No | Enterprise | **Included** |
| MLflow integration | N/A | No | No | **Bridge, not replace** |

MLflow tracks experiments. DVC versions data. W&B visualizes training runs. **Catapult handles everything after training** — the part where most teams lose weeks to shell scripts and tribal knowledge.

---

## Built For

- **ML teams shipping models to production** — not just tracking experiments
- **vLLM / PyTorch serving teams** — first-class LLM serving support with inference benchmarking
- **Multi-platform teams** — CUDA, ROCm, ARM, Neuron, TPU, Gaudi templates out of the box
- **Teams on shared storage** — native Ceph, NFS, and FSx integration, not just S3
- **Teams that value simplicity** — one repo, one deploy command, no Kubernetes required

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- A machine with GPU access (for model serving — Catapult itself runs on CPU)

### Deploy

```bash
git clone https://github.com/warlockee/Catapult
cd Catapult
cp .env.example .env    # Configure DB, storage paths
./deploy.sh
```

Open `http://localhost:8080` — your registry is ready.

The script will:
- Build Docker images
- Start all services (PostgreSQL, Redis, Backend, Worker, Frontend, Nginx)
- Run database migrations
- Create an initial admin API key

### Register Your First Model

```bash
# Via API
curl -X POST http://localhost:8080/api/v1/models \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "myorg/llama-3-8b", "storage_path": "/models/llama-3-8b", "server_type": "vllm"}'
```

```python
# Via Python SDK
pip install catapult-sdk

from catapult import Registry

registry = Registry.from_env()
model = registry.create_model(name="myorg/llama-3-8b", server_type="vllm")
version = registry.create_version(model_name=model.name, version="1.0.0")
```

### Build & Deploy

```python
# Build a Docker image from a template
build = registry.create_docker_build(
    version_id=version.id,
    template="vllm_gpu",
    tag="myorg/llama-3-8b:1.0.0"
)

# Deploy it
deployment = registry.create_deployment(
    version_id=version.id,
    image=build.image_tag,
    environment="development",
    gpu_count=1
)

# Benchmark it
benchmark = registry.run_benchmark(
    deployment_id=deployment.id,
    benchmark_type="inference",
    duration_seconds=60
)
```

---

## Python SDK

### Installation

```bash
cd sdk/python
pip install -e .

# Or from PyPI (when published)
pip install catapult-sdk
```

### Integration with Training Scripts

```python
import torch
from catapult import Registry

# Initialize from environment variables
# Set: REGISTRY_URL and REGISTRY_API_KEY
registry = Registry.from_env()

# Your training code
model = train_model()
accuracy = evaluate_model(model, test_loader)

# Register the version with training metadata
version = registry.create_version(
    model_name="myorg/sentiment-model",
    version="2.1.0",
    tag="v2.1.0",
    digest="sha256:...",
    metadata={
        "model_type": "BERT-base",
        "accuracy": accuracy,
        "training_samples": len(train_dataset),
        "pytorch_version": torch.__version__,
        "git_commit": get_git_commit(),
    }
)

print(f"Registered version: {version.id}")
```

---

## API Reference

### Health & Info
- `GET /api/health` — Health check
- `GET /api/v1/info` — API version and system info

### Models
- `GET /api/v1/models` — List models
- `POST /api/v1/models` — Create model
- `GET /api/v1/models/{id}` — Get model
- `PUT /api/v1/models/{id}` — Update model
- `DELETE /api/v1/models/{id}` — Delete model

### Versions
- `POST /api/v1/versions` — Create version
- `GET /api/v1/versions` — List versions (use `is_release=true` for official releases)
- `GET /api/v1/versions/latest` — Get latest version
- `GET /api/v1/versions/{id}` — Get version
- `PUT /api/v1/versions/{id}` — Update version (including promote/demote via `is_release`)
- `DELETE /api/v1/versions/{id}` — Delete version
- `GET /api/v1/versions/{id}/deployments` — List deployments for version

### Deployments
- `POST /api/v1/deployments` — Record / create deployment
- `GET /api/v1/deployments` — List deployments
- `GET /api/v1/deployments/{id}` — Get deployment

### Docker Builds
- `POST /api/v1/docker/builds` — Start a Docker build
- `GET /api/v1/docker/builds` — List builds
- `GET /api/v1/docker/builds/{id}` — Get build status
- `GET /api/v1/docker/builds/{id}/logs/stream` — SSE log stream
- `GET /api/v1/docker/templates/{type}` — Get Dockerfile template
- `GET /api/v1/docker/disk-usage` — Docker disk usage

### Benchmarks
- `POST /api/v1/benchmarks` — Run benchmark (sync)
- `POST /api/v1/benchmarks/async` — Run benchmark (async via Celery)
- `GET /api/v1/benchmarks/{id}` — Get benchmark result
- `GET /api/v1/benchmarks/deployment/{id}` — Benchmarks for a deployment
- `GET /api/v1/benchmarks/deployment/{id}/summary` — Benchmark summary

### Evaluations
- `POST /api/v1/evaluations` — Run evaluation
- `GET /api/v1/evaluations/{id}` — Get evaluation result
- `GET /api/v1/evaluations/deployment/{id}` — Evaluations for a deployment

### Artifacts
- `GET /api/v1/artifacts` — List artifacts
- `POST /api/v1/artifacts` — Upload artifact

### API Keys
- `POST /api/v1/api-keys` — Create API key
- `GET /api/v1/api-keys` — List API keys
- `DELETE /api/v1/api-keys/{id}` — Revoke API key

### Audit Logs
- `GET /api/v1/audit-logs` — List audit logs

---

## Extending Catapult

The architecture is designed for extensibility:

- **Custom Dockerfile templates** — Drop a `Dockerfile.<name>` into `kb/dockers/` and it appears in the build UI
- **Custom evaluators** — Implement the `BaseEvaluator` interface, register via factory, and your evaluation type is available across the platform
- **Custom event handlers** — Subscribe to domain events to trigger external workflows (CI/CD, notifications, data pipelines)

---

## Management Commands

```bash
# Create API key
docker-compose exec backend python scripts/create_api_key.py --name "my-key"

# View logs
docker-compose logs -f backend

# Run migrations
docker-compose exec backend alembic upgrade head

# Backup database
docker-compose exec postgres pg_dump -U registry registry > backup.sql

# Restore database
docker-compose exec -T postgres psql -U registry registry < backup.sql
```

---

## Development

```bash
# 1. Start database only
docker-compose up -d postgres redis

# 2. Backend
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# 3. Frontend (in another terminal)
cd frontend
npm install
npm run dev

# 4. SDK development
cd sdk/python
pip install -e .
```

---

## Troubleshooting

### Backend can't connect to database
```bash
docker-compose ps postgres
docker-compose logs postgres
docker-compose exec postgres psql -U registry -d registry -c "SELECT 1;"
```

### Frontend can't reach API
```bash
docker-compose logs nginx
curl http://localhost:8080/api/health
```

### Storage path not accessible
```bash
# Use local directory for development
STORAGE_ROOT=./storage  # default — no external mount needed
```

---

## Project Structure

```
Catapult/
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Config, database, security
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   ├── alembic/           # Database migrations
│   └── scripts/           # Utility scripts
├── frontend/              # React frontend
│   └── src/
│       ├── components/    # React components
│       └── lib/           # API client, utilities
├── sdk/python/            # Python SDK
│   └── catapult/
├── kb/dockers/            # 25+ Dockerfile templates
├── infrastructure/        # Nginx configs
├── tests/                 # E2E tests
├── docker-compose.yml     # Main deployment config
└── deploy.sh              # One-command deployment
```

---

## Roadmap

- [ ] Kubernetes-native deployment mode
- [ ] Multi-node distributed Docker builds
- [ ] Model comparison and diff views
- [ ] Webhook integrations (Slack, Discord, PagerDuty)
- [ ] ONNX and TensorRT optimization pipeline
- [ ] HuggingFace Hub bidirectional sync
- [ ] Cost tracking per deployment
- [ ] Web-based Dockerfile template editor

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache 2.0](LICENSE)

---

*Built by ML engineers who got tired of the gap between "model trained" and "model in production." We hope it saves you the same weeks it saved us.*
