# Catapult MCP Server - Design Document

**Status: IMPLEMENTED** - See `/mcp/` directory for implementation.

## Overview

MCP (Model Context Protocol) server for Catapult, enabling AI agents to interact with the registry system programmatically.

**Three Interfaces - One Backend:**
- **WebUI** - Human operators, visual interface
- **SDK** - Scripts, CI/CD pipelines, programmatic access
- **MCP** - AI agents, natural language interaction, autonomous workflows

All three interfaces share the same Registry Server (FastAPI backend).

---

## Architecture

```
                                Catapult System

         ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
         │    WebUI     │   │     SDK      │   │     MCP      │
         │   (React)    │   │   (Python)   │   │  (FastMCP)   │
         │              │   │              │   │              │
         │  Humans      │   │  Scripts     │   │  AI Agents   │
         │  Interactive │   │  CI/CD       │   │  Autonomous  │
         └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                            X-API-Key auth
                                   │
                                   ▼
         ┌─────────────────────────────────────────────────────────┐
         │                  Registry Server                         │
         │                    (FastAPI)                             │
         │                                                          │
         │  /api/v1/models    /api/v1/versions   /api/v1/deployments │
         │  /api/v1/docker    /api/v1/artifacts  /api/v1/benchmarks │
         │  /api/v1/release-configs            /api/v1/system      │
         └─────────────────────────────────────────────────────────┘
                                   │
                                   ▼
         ┌─────────────────────────────────────────────────────────┐
         │                   Data Sources                           │
         │                                                          │
         │  PostgreSQL    Docker    Ceph/Storage    Release Configs │
         │  (metadata)    (builds)  (model files)   (prod configs)  │
         └─────────────────────────────────────────────────────────┘
```

---

## Design Principles

1. **Parity with SDK** - Every SDK capability has an MCP equivalent
2. **AI-First Descriptions** - Tool docs explain *when* and *why*, not just *what*
3. **Composable Primitives** - Small, focused tools that chain together
4. **Safe by Default** - Read operations default; writes are explicit and marked
5. **DRY** - MCP wraps SDK, doesn't reimplement HTTP logic

---

## Key Concept: Two Deployment Domains

The registry manages **two distinct deployment domains** that must not be confused:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT DOMAINS                                   │
├───────────────────────────────────┬─────────────────────────────────────────┤
│      LOCAL DEPLOYMENTS            │      PRODUCTION (GitOps)                │
│      (Registry-Managed)           │      (GitOps-Managed)                   │
├───────────────────────────────────┼─────────────────────────────────────────┤
│                                   │                                         │
│  Where: Registry host (Docker)    │  Where: Remote GPU clusters             │
│  Purpose: Dev, staging, testing   │  Purpose: Production workloads          │
│  Lifecycle: Start/stop via API    │  Lifecycle: PR merge triggers deploy    │
│  Data: PostgreSQL deployments     │  Data: Release config Git repo          │
│  table                            │                                         │
│                                   │                                         │
│  MCP Tools:                       │  MCP Tools:                             │
│  ├─ list_deployments              │  ├─ list_release_configs                │
│  ├─ execute_deployment ⚠️         │  ├─ get_machine_topology                │
│  ├─ stop_deployment ⚠️            │  ├─ propose_release_config ⚠️           │
│  ├─ get_deployment_logs           │  │   (creates PR, human approval)       │
│  └─ get_deployment_status         │  └─ (NO stop/start - read-only view)    │
│                                   │                                         │
│  Can control: YES                 │  Can control: NO (PR workflow only)     │
│                                   │                                         │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

**Critical Distinction:**
- `execute_deployment` starts a **local Docker container** on the registry host for testing
- `propose_release_config` creates a **PR** for production - it does NOT deploy directly
- `stop_deployment` only works on **local deployments**, never production
- Production deployments appear in `list_release_configs`, NOT in `list_deployments`

---

## Design Decision: Tool Granularity

**Why 42 tools instead of fewer consolidated tools?**

We chose granular tools over consolidated ones for these reasons:

1. **Composability** - AI can chain small tools to build complex workflows
2. **Clear Intent** - Each tool does one thing, making audit logs meaningful
3. **Parallel Execution** - Multiple independent reads can run simultaneously
4. **Error Isolation** - Failures are scoped to specific operations
5. **SDK Parity** - Tools map 1:1 to SDK methods for predictability

**Trade-off acknowledged:** More tools means more context for the AI to process.
Mitigation: Resources provide bulk data without tool calls, and Prompts guide common workflows.

---

## Tool Inventory

### Summary

| Category | Read | Write | Total |
|----------|------|-------|-------|
| Models | 2 | 3 | 5 |
| Versions | 3 | 2 | 5 |
| Deployments | 6 | 4 | 10 |
| Docker Builds | 4 | 1 | 5 |
| Benchmarks | 3 | 1 | 4 |
| Release Configs | 5 | 1 | 6 |
| Artifacts | 2 | 0 | 2 |
| System | 4 | 0 | 4 |
| Search | 1 | 0 | 1 |
| **Total** | **30** | **12** | **42** |

### Tool Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MCP TOOLS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MODELS (5)                          VERSIONS (5)                           │
│  ├─ list_models                      ├─ list_versions                       │
│  ├─ get_model                        ├─ get_version                         │
│  ├─ create_model ⚠️                  ├─ get_latest_version                  │
│  ├─ update_model ⚠️                  ├─ create_version ⚠️                   │
│  └─ delete_model ⚠️                  └─ promote_version ⚠️                  │
│                                                                              │
│  DEPLOYMENTS (10)                    DOCKER BUILDS (5)                      │
│  ├─ list_deployments                 ├─ list_docker_builds                  │
│  ├─ get_deployment                   ├─ get_docker_build                    │
│  ├─ get_deployment_status            ├─ get_docker_build_logs               │
│  ├─ get_deployment_logs              ├─ trigger_docker_build ⚠️             │
│  ├─ get_deployment_health            └─ get_docker_disk_usage               │
│  ├─ discover_api_spec                                                       │
│  ├─ execute_deployment ⚠️            BENCHMARKS (4)                         │
│  ├─ stop_deployment ⚠️               ├─ list_benchmarks                     │
│  ├─ start_deployment ⚠️              ├─ get_benchmark                       │
│  └─ restart_deployment ⚠️            ├─ run_benchmark ⚠️                    │
│                                      └─ get_benchmark_summary               │
│  ARTIFACTS (2)                                                              │
│  ├─ list_artifacts                   RELEASE CONFIGS (6)                    │
│  └─ get_artifact                     ├─ list_release_configs                │
│                                      ├─ get_release_config                  │
│  SYSTEM (4)                          ├─ get_machine_topology                │
│  ├─ health_check                     ├─ get_models_by_deployment            │
│  ├─ get_audit_logs                   ├─ list_docker_templates               │
│  ├─ get_storage_usage                └─ propose_release_config ⚠️           │
│  └─ browse_model_storage                                                    │
│                                                                              │
│  SEARCH (1)                          ⚠️ = Write operation                   │
│  └─ search                                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tool Specifications

### Models

```python
list_models(search, server_type, source, limit) -> list[dict]
    """List models in the registry."""

get_model(model_id) -> dict
    """Get detailed model information."""

create_model(name, storage_path, server_type, description, tags, requires_gpu) -> dict  # ⚠️
    """Register a new model."""

update_model(model_id, description, tags, metadata) -> dict  # ⚠️
    """Update model metadata."""

delete_model(model_id) -> dict  # ⚠️
    """Delete a model and all its versions."""
```

### Versions

```python
list_versions(model_name, is_release, limit) -> list[dict]
    """List model versions."""

get_version(version_id) -> dict
    """Get detailed version information."""

get_latest_version(model_name) -> dict
    """Get the most recent version of a model."""

create_version(model_name, version, tag, digest, ceph_path, quantization, is_release, auto_build) -> dict  # ⚠️
    """Create a new model version."""

promote_version(version_id, is_release) -> dict  # ⚠️
    """Promote or demote a version to/from release status."""
```

### Deployments (Local/Staging Only)

These tools manage **local Docker deployments** on the registry host.
They do NOT affect production deployments (use Release Configs for production).

```python
list_deployments(status, environment, model_name, limit) -> list[dict]
    """List LOCAL deployments (Docker containers on registry host)."""

get_deployment(deployment_id) -> dict
    """Get detailed deployment information."""

get_deployment_status(deployment_id) -> dict
    """Get container runtime status (quick health check)."""

get_deployment_logs(deployment_id, lines) -> str
    """Get container logs."""

get_deployment_health(deployment_id) -> dict
    """Perform active health check on deployment endpoint."""

discover_api_spec(deployment_id) -> dict
    """Probe deployment to discover its API capabilities.

    Behavior:
    1. Attempts GET /openapi.json for OpenAPI spec
    2. Probes common endpoints: /health, /v1/models, /v1/chat/completions
    3. Returns detected API type: 'openai', 'fastapi', 'audio', 'unknown'
    4. Lists discovered endpoints with methods
    """

execute_deployment(version_id, environment, gpu_enabled, deployment_type) -> dict  # ⚠️
    """Start a LOCAL Docker container for testing/staging.

    ⚠️ This is NOT production deployment. For production, use propose_release_config.

    Starts a container on the registry host with the model version.
    GPU is auto-detected if not specified.
    """

stop_deployment(deployment_id) -> dict  # ⚠️
    """Stop a LOCAL deployment. Only works on registry-managed containers."""

start_deployment(deployment_id) -> dict  # ⚠️
    """Restart a stopped LOCAL deployment."""

restart_deployment(deployment_id) -> dict  # ⚠️
    """Restart a LOCAL deployment (stop + start)."""
```

### Docker Builds

```python
list_docker_builds(version_id, status, limit) -> list[dict]
    """List Docker builds."""

get_docker_build(build_id) -> dict
    """Get Docker build details."""

get_docker_build_logs(build_id) -> str
    """Get Docker build logs."""

trigger_docker_build(version_id, build_type, image_tag) -> dict  # ⚠️
    """Trigger a Docker image build."""

get_docker_disk_usage() -> dict
    """Get Docker disk usage statistics."""
```

### Benchmarks

```python
list_benchmarks(deployment_id, limit) -> list[dict]
    """List benchmark results."""

get_benchmark(benchmark_id) -> dict
    """Get detailed benchmark results."""

run_benchmark(deployment_id, endpoint_path, concurrent_requests, total_requests, request_body) -> dict  # ⚠️
    """Run performance benchmark on a deployment."""

get_benchmark_summary(deployment_id) -> dict
    """Get latest benchmark summary for a deployment."""
```

### Release Configs (Production Topology - Read-Only + PR)

These tools provide **visibility** into production deployments managed via GitOps.
The registry reads from the release config Git repo but does NOT directly control production.

```python
list_release_configs(machine, model_name) -> list[dict]
    """List production deployment configurations.

    Returns configs from the release config Git submodule.
    These represent what's deployed on remote GPU clusters.
    """

get_release_config(machine, port) -> dict
    """Get specific production deployment config."""

get_machine_topology() -> dict
    """Get GPU allocation across all production machines.

    Returns: {machine: {total_gpus_used, gpu_allocation, deployments}}
    Useful for capacity planning and finding available resources.
    """

get_models_by_deployment() -> dict
    """Get production deployments grouped by model path."""

list_docker_templates() -> list[str]
    """List available docker-compose templates for production."""

propose_release_config(machine, port, template_name, model_name, model_path, gpu_ids, tensor_parallel, description) -> dict  # ⚠️
    """Create a Pull Request to deploy to production.

    ⚠️ This creates a PR in the release config repo - it does NOT deploy directly.
    A human must review and merge the PR for deployment to happen.

    Returns: {pr_url, status: "pending_review"}
    """
```

### Artifacts

```python
list_artifacts(version_id, model_id, artifact_type, limit) -> list[dict]
    """List artifacts (wheels, binaries, etc)."""

get_artifact(artifact_id) -> dict
    """Get artifact details."""
```

### System

```python
health_check() -> dict
    """Check registry system health."""

get_audit_logs(action, resource_type, limit) -> list[dict]
    """Get audit log entries."""

get_storage_usage() -> dict
    """Get storage usage statistics."""

browse_model_storage(relative_path) -> list[dict]
    """Browse model storage filesystem.

    Security: Path is SANDBOXED to the storage root (STORAGE_ROOT env var).
    - Accepts relative paths only (leading '/' is stripped)
    - Cannot escape storage root (path traversal blocked by backend)
    - Returns files/directories within model storage only

    Example:
        browse_model_storage("models/llama-70b")  # List model files
        browse_model_storage("")  # List storage root
    """
```

### Search

```python
search(query, types, limit) -> dict
    """Search across models, versions, and deployments."""
```

---

## MCP Resources

Resources provide read-only data access:

```python
# Models
model://{model_id}              # Model metadata as JSON
model://{model_id}/versions     # All versions of a model

# Versions
version://{version_id}          # Version metadata as JSON

# Deployments
deployment://{deployment_id}    # Deployment metadata as JSON
deployment://{deployment_id}/logs  # Recent deployment logs

# Production Topology
topology://machines             # Full cluster topology
topology://machine/{machine}    # Single machine's deployments

# System
health://status                 # Current system health
```

---

## MCP Prompts

Workflow templates for common operations:

### deploy_workflow

End-to-end deployment from model to running container:
1. Find model
2. Get latest version
3. Check/trigger Docker build
4. Execute deployment
5. Verify health
6. Run benchmark

### troubleshoot_workflow

Deployment troubleshooting:
1. Check status
2. Analyze logs
3. Verify health
4. Check build status
5. Diagnose common issues

### production_deployment_workflow

Production deployment via PR (human approval required):
1. Check current topology
2. Find available resources
3. Verify model
4. Select template
5. Create PR

### capacity_planning_workflow

Cluster capacity analysis:
1. Get cluster overview
2. Analyze per-machine allocation
3. Check storage status
4. Identify opportunities

### model_comparison_workflow

Compare two models:
1. Get model details
2. Compare versions
3. Check deployments
4. Compare benchmarks
5. Review production status

---

## Project Structure

```
mcp/
├── pyproject.toml
├── README.md
└── src/
    └── model_registry_mcp/
        ├── __init__.py
        ├── server.py                 # FastMCP server entry point
        ├── config.py                 # Settings (env vars)
        ├── client.py                 # Registry client (wraps SDK)
        │
        ├── tools/
        │   ├── __init__.py           # Registers all tools
        │   ├── models.py             # 5 tools
        │   ├── versions.py           # 5 tools
        │   ├── deployments.py        # 10 tools
        │   ├── docker_builds.py      # 5 tools
        │   ├── benchmarks.py         # 4 tools
        │   ├── release_configs.py    # 6 tools
        │   ├── artifacts.py          # 2 tools
        │   ├── system.py             # 4 tools
        │   └── search.py             # 1 tool
        │
        ├── resources/
        │   ├── __init__.py
        │   └── resources.py          # All MCP resources
        │
        └── prompts/
            ├── __init__.py
            └── prompts.py            # All MCP prompts
```

---

## Configuration

```python
class MCPConfig(BaseSettings):
    # Registry connection (same env vars as SDK)
    registry_url: str = "http://localhost/api"
    registry_api_key: str

    # Operation modes
    read_only: bool = False              # Disable all writes
    allow_production_changes: bool = True # Allow propose_release_config

    # Performance
    cache_ttl_seconds: int = 60
    request_timeout_seconds: int = 30

    # Limits
    max_log_lines: int = 5000
    max_results: int = 100

    class Config:
        env_prefix = "REGISTRY_"  # REGISTRY_URL, REGISTRY_API_KEY
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REGISTRY_URL` | No | `http://localhost/api` | Registry API URL |
| `REGISTRY_API_KEY` | Yes | - | API key for authentication |
| `REGISTRY_READ_ONLY` | No | `false` | Disable write operations |
| `REGISTRY_ALLOW_PRODUCTION_CHANGES` | No | `true` | Allow PR creation |

---

## Package Definition

```toml
[project]
name = "model-registry-mcp"
version = "1.0.0"
description = "MCP server for Catapult"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
    "catapult-sdk>=2.0.0",
    "pydantic-settings>=2.0",
]

[project.scripts]
model-registry-mcp = "model_registry_mcp.server:main"
```

---

## Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                       SECURITY LAYERS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: API Key Authentication                                │
│  ├─ MCP uses REGISTRY_API_KEY                                   │
│  ├─ Key has role: viewer | operator | admin                     │
│  └─ Role determines allowed operations                          │
│                                                                  │
│  Layer 2: MCP Read-Only Mode                                    │
│  ├─ REGISTRY_READ_ONLY=true disables all write tools           │
│  └─ Safe for exploration/monitoring use cases                   │
│                                                                  │
│  Layer 3: Production Writes Gate                                │
│  ├─ REGISTRY_ALLOW_PRODUCTION_CHANGES=false                    │
│  ├─ Blocks propose_release_config                              │
│  └─ Must explicitly enable for production changes               │
│                                                                  │
│  Layer 4: Human Approval (Production)                           │
│  ├─ Production changes create PRs, not direct deployments      │
│  ├─ Requires human review and merge                             │
│  └─ Audit trail in Git history                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Strategy

### Phase 1: Core Infrastructure

1. Set up project structure (`mcp/`)
2. Implement `config.py` with settings
3. Implement `client.py` wrapping SDK
4. Create `server.py` with FastMCP setup

### Phase 2: Read-Only Tools

1. Models: `list_models`, `get_model`
2. Versions: `list_versions`, `get_version`, `get_latest_version`
3. Deployments: `list_deployments`, `get_deployment`, `get_deployment_status`, `get_deployment_logs`
4. System: `health_check`, `get_audit_logs`
5. Search: `search`

### Phase 3: Write Tools

1. Model management: `create_model`, `update_model`, `delete_model`
2. Version management: `create_version`, `promote_version`
3. Deployment lifecycle: `execute_deployment`, `stop_deployment`, `start_deployment`, `restart_deployment`

### Phase 4: Advanced Features

1. Docker builds: all 5 tools
2. Benchmarks: all 4 tools
3. Release configs: all 6 tools
4. Artifacts: all 2 tools
5. Additional deployment tools: `get_deployment_health`, `discover_api_spec`

### Phase 5: Resources & Prompts

1. Implement all MCP resources
2. Implement all MCP prompts
3. Documentation and examples

---

## SDK Dependency

MCP wraps the existing SDK (`catapult`) to avoid duplicating HTTP logic:

```python
# client.py
from catapult import Registry, RegistryError
from catapult.models import Model, Version, Deployment

class RegistryClientAdapter:
    """Wraps catapult.Registry for MCP use."""

    def __init__(self, config: MCPConfig):
        self._registry = Registry(
            base_url=config.registry_url,
            api_key=config.registry_api_key,
        )
        self._read_only = config.read_only

    def list_models(self, **kwargs) -> list[dict]:
        models = self._registry.list_models(**kwargs)
        return [self._model_to_dict(m) for m in models]

    def execute_deployment(self, version_id: str, **kwargs) -> dict:
        if self._read_only:
            raise MCPError("Write operations disabled in read-only mode")
        # Call registry API...
```

---

## Example Usage

### Starting the MCP Server

```bash
# Set environment variables
export REGISTRY_URL="http://localhost/api"
export REGISTRY_API_KEY="your-api-key"

# Run MCP server
model-registry-mcp
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "model-registry": {
      "command": "model-registry-mcp",
      "env": {
        "REGISTRY_URL": "http://registry.internal/api",
        "REGISTRY_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Example AI Interaction

```
User: "Deploy the latest llama-70b model to staging"

AI uses:
1. list_models(search="llama-70b")
2. get_latest_version("llama-70b-v2")
3. list_docker_builds(version_id="abc-123")
4. execute_deployment(version_id="abc-123", environment="staging")
5. get_deployment_status("def-456")
6. get_deployment_logs("def-456", lines=50)

AI responds: "Deployed llama-70b-v2 to staging. Container is running
and healthy at http://localhost:9001. Logs show successful model loading."
```

---

## Testing Strategy

### Unit Tests

- Test each tool function in isolation
- Mock SDK client responses
- Verify error handling

### Integration Tests

- Test against running registry instance
- Verify end-to-end workflows
- Test authentication and authorization

### MCP Protocol Tests

- Verify tool registration
- Test resource access
- Validate prompt rendering

---

## Future Considerations

### Potential Additions

- Streaming log support (MCP streaming)
- Webhook notifications
- Custom tool plugins
- Multi-registry support

### Not Planned

- Direct SSH deployment (separate concern)
- API key management (admin-only)
- Database operations (internal)
- File uploads (use SDK)

---

## Summary

| Aspect | Specification |
|--------|---------------|
| **Tools** | 42 total (30 read, 12 write) |
| **Resources** | 8 resource types |
| **Prompts** | 5 workflow templates |
| **SDK Dependency** | Wraps `catapult` package (DRY) |
| **Auth** | Same as WebUI/SDK (X-API-Key) |
| **Config** | Environment variables (REGISTRY_*) |
| **Entry Point** | `model-registry-mcp` CLI |
