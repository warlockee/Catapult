# Catapult

[![CI](https://github.com/warlockee/Catapult/actions/workflows/ci.yml/badge.svg)](https://github.com/warlockee/Catapult/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Ship models to production, not just track them.**

Register models, build Docker images, deploy with GPU awareness, benchmark latency, evaluate quality — one platform, one command.

```bash
git clone https://github.com/warlockee/Catapult && cd Catapult && ./deploy.sh
```

<!-- TODO: Add hero screenshot showing the dashboard -->
<!-- ![Dashboard](docs/screenshots/dashboard.png) -->

---

## How It Works

### Register → Build → Deploy → Benchmark

<!-- TODO: Add screenshots for each step -->
<!-- ![Register a model](docs/screenshots/model-register.png) -->
<!-- ![Docker build with live logs](docs/screenshots/docker-build.png) -->
<!-- ![One-click deploy](docs/screenshots/deploy.png) -->
<!-- ![Benchmark results](docs/screenshots/benchmark.png) -->

| Step | What happens |
|------|-------------|
| **Register** | Track models and versions with flexible JSONB metadata |
| **Build** | 25+ Dockerfile templates — vLLM, ASR, TTS, embedding across CUDA/ROCm/ARM/Neuron/TPU/Gaudi |
| **Deploy** | GPU-aware container deployment with health monitoring and log tailing |
| **Benchmark** | TTFT, p50/p95/p99 latency, tokens/sec, error rates |
| **Evaluate** | Pluggable quality framework (ships with ASR WER/CER) |

---

## Install

### Deploy (Docker)

```bash
git clone https://github.com/warlockee/Catapult
cd Catapult
./deploy.sh
# → Web UI at http://localhost:8080
# → API at http://localhost:8080/api
```

Starts PostgreSQL, Redis, FastAPI backend, Celery worker, React frontend, and Nginx. Creates an admin API key on first run. API docs at [localhost:8080/docs](http://localhost:8080/docs).

### Python SDK

```bash
cd sdk/python && pip install -e .
```

### MCP Server (for Claude / AI assistants)

```bash
cd mcp && pip install -e .
```

Add to your MCP config (`claude_desktop_config.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "catapult": {
      "command": "catapult-mcp",
      "env": {
        "REGISTRY_URL": "http://localhost:8080/api",
        "REGISTRY_API_KEY": "your-api-key"
      }
    }
  }
}
```

42 tools across models, versions, deployments, Docker builds, benchmarks, evaluations, artifacts, and system health.

---

## Usage

### Python SDK

```python
from catapult import Registry

registry = Registry(base_url="http://localhost:8080/api", api_key="your-key")

# Register
model = registry.create_model(name="myorg/llama-3-8b", server_type="vllm")
version = registry.create_version(model_name=model.name, version="1.0.0",
                                   metadata={"accuracy": 0.95})

# Build
build = registry.create_build(release_id=version.id, template_type="default",
                               image_tag="myorg/llama-3-8b:1.0.0")
for line in registry.stream_build_logs(build.id):
    print(line)

# Deploy
deployment = registry.deploy(release_id=version.id, environment="staging")

# Promote
registry.promote_version(version.id)
```

### curl

```bash
API="http://localhost:8080/api/v1"
KEY="your-api-key"

# Create model
curl -X POST $API/models \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"name": "llama-3-8b", "storage_path": "/models/llama-3-8b"}'

# Create version
curl -X POST $API/versions \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"image_id": "<model-id>", "version": "v1.0.0", "tag": "latest"}'

# List models
curl $API/models -H "X-API-Key: $KEY"
```

### MCP (Claude / AI assistants)

With the MCP server connected, your AI assistant can:

```
User: "Deploy llama-3-8b to staging with 1 GPU"

Claude: I'll deploy that for you.
  → get_model(name="llama-3-8b")
  → get_latest_version(model_name="llama-3-8b")
  → execute_deployment(version_id="...", environment="staging", gpu_count=1)
  → get_deployment_health(deployment_id="...")
  ✓ Deployed at http://localhost:9001, health: passing
```

```
User: "Benchmark it — 100 concurrent requests"

Claude: Running benchmark now.
  → run_benchmark(deployment_id="...", concurrent_requests=100, total_requests=1000)
  → get_benchmark(benchmark_id="...")
  ✓ p50: 45ms, p95: 120ms, p99: 210ms, throughput: 850 tok/s
```

```
User: "What's running on our GPUs?"

Claude:
  → list_deployments(status="running")
  → get_docker_disk_usage()
  ✓ 3 active deployments across 4 GPUs, 1.2TB Docker storage used
```

---

## How It Compares

MLflow tracks experiments. DVC versions data. W&B visualizes training. **Catapult handles everything after training.**

| | MLflow | DVC | W&B | **Catapult** |
|---|---|---|---|---|
| Model tracking | Yes | Yes | Yes | Yes |
| Docker builds | No | No | No | **25+ templates** |
| Deploy from UI | No | No | No | **GPU-aware** |
| Benchmarking | No | No | No | **TTFT, p50-p99** |
| Evaluation | No | No | No | **Pluggable** |
| GPU fleet view | No | No | No | **Yes** |
| Audit / RBAC | Enterprise | No | Enterprise | **Included** |
| MCP integration | No | No | No | **42 tools** |
| Self-hosted | Partial | Yes | No | **One command** |
| MLflow bridge | N/A | No | No | **Yes** |

---

## Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ React UI │────▶│  Nginx   │────▶│ FastAPI  │
│ Vite SPA │     │  :8080   │     │  :8000   │
└──────────┘     └──────────┘     └────┬─────┘
                                       │
                        ┌──────────────┼──────────────┐
                        │              │              │
                   ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
                   │ Celery  │    │  Redis  │    │ Postgres│
                   │ Worker  │    │  Queue  │    │   DB    │
                   │+Docker  │    └─────────┘    └─────────┘
                   └─────────┘
```

FastAPI + async SQLAlchemy · React 18 + Shadcn/ui · Celery + Docker socket · PostgreSQL JSONB · Redis

---

## Extending

- **Dockerfile templates** — drop `Dockerfile.<name>` in `kb/dockers/`, it appears in the build UI
- **Evaluators** — implement `BaseEvaluator`, register via factory
- **Event handlers** — subscribe to domain events for CI/CD triggers

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Found a vulnerability? See [SECURITY.md](SECURITY.md). Apache 2.0 licensed.

---

*Built by ML engineers who got tired of the gap between "model trained" and "model in production."*
