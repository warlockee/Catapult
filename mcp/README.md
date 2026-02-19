# Catapult MCP Server

MCP (Model Context Protocol) server for Catapult. Enables AI assistants to manage models, versions, deployments, and production configurations.

## Installation

Prerequisites:
- Python 3.10+
- `pip`

```bash
# 1. Install the SDK first
cd ../sdk/python
pip install -e .

# 2. Install the MCP server
cd ../../mcp
pip install -e .
```

## Configuration

To use with Claude Desktop or Antigravity, add the following to your config file (`~/.config/Claude/claude_desktop_config.json`).

**Important**: Use the **absolute path** to the `model-registry-mcp` executable to avoid PATH issues.

```json
{
  "mcpServers": {
    "model-registry": {
      "command": "/home/your-user/.local/bin/model-registry-mcp",
      "env": {
        "REGISTRY_URL": "http://localhost/api",
        "REGISTRY_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Quick Start Tutorial

Once configured, you can interact with the registry using natural language. Here are common workflows:

### Scenario 1: Explore Models
> **User**: "List existing models and find the latest version of llama-3."

**Tools Used**:
1. `list_models(search="llama-3")` -> Returns model list.
2. `get_latest_version(model_name="llama-3-70b")` -> Returns version details.

### Scenario 2: Deploy to Staging
> **User**: "Deploy the latest version of 'audio-model-v3' to the staging environment."

**Tools Used**:
1. `get_latest_version(model_name="audio-model-v3")` -> Gets version ID (e.g., `v123`).
2. `execute_deployment(version_id="v123", environment="staging")` -> Starts Docker container.
3. `get_deployment_status(deployment_id="...")` -> Confirms it's running.

### Scenario 3: production Deployment (GitOps)
> **User**: "Draft a release config to deploy 'llama-3-70b' on machine 'gpu-worker-01'."

**Tools Used**:
1. `get_machine_topology()` -> checks available GPUs.
2. `list_docker_templates()` -> shows available templates (e.g., `vllm.yml`).
3. `propose_release_config(machine="gpu-worker-01", ...)` -> **Creates a GitHub PR**.

### Scenario 4: Troubleshooting
> **User**: "Why is the deployment on port 8000 failing?"

**Tools Used**:
1. `list_deployments()` -> Finds deployment ID.
2. `get_deployment_health(deployment_id="...")` -> Returns error.
3. `get_deployment_logs(deployment_id="...", lines=50)` -> Fetches logs for analysis.


## Tools (42 total)

### Models (5 tools)
- `list_models` - List models with filtering
- `get_model` - Get model details
- `create_model` - Register new model
- `update_model` - Update model metadata
- `delete_model` - Delete model and versions

### Versions (5 tools)
- `list_versions` - List versions with filtering
- `get_version` - Get version details
- `get_latest_version` - Get most recent version
- `create_version` - Create new version
- `promote_version` - Mark as release

### Local Deployments (10 tools)
- `list_deployments` - List local deployments
- `get_deployment` - Get deployment details
- `get_deployment_status` - Quick container status
- `get_deployment_logs` - Get container logs
- `get_deployment_health` - Active health check
- `discover_api_spec` - Probe API capabilities
- `execute_deployment` - Start local deployment
- `stop_deployment` - Stop deployment
- `start_deployment` - Restart stopped deployment
- `restart_deployment` - Stop + start

### Docker Builds (5 tools)
- `list_docker_builds` - List builds
- `get_docker_build` - Get build details
- `get_docker_build_logs` - Get build logs
- `trigger_docker_build` - Start new build
- `get_docker_disk_usage` - Check disk space

### Benchmarks (4 tools)
- `list_benchmarks` - List benchmark results
- `get_benchmark` - Get detailed results
- `run_benchmark` - Run performance test
- `get_benchmark_summary` - Get latest summary

### Release Configs (6 tools) - Production Topology
- `list_release_configs` - List production deployments
- `get_release_config` - Get specific config
- `get_machine_topology` - GPU allocation view
- `get_models_by_deployment` - Deployments by model
- `list_docker_templates` - Available templates
- `propose_release_config` - Create PR for production change

### Artifacts (2 tools)
- `list_artifacts` - List artifacts
- `get_artifact` - Get artifact details

### System (4 tools)
- `health_check` - Check system health
- `get_audit_logs` - View audit trail
- `get_storage_usage` - Storage statistics
- `browse_model_storage` - Browse model files

### Search (1 tool)
- `search` - Unified search across all resources

## Resources (8 types)

- `registry://models` - All models
- `registry://models/{id}` - Specific model
- `registry://versions/{id}` - Specific version
- `registry://deployments` - Local deployments
- `registry://deployments/{id}` - Specific deployment
- `registry://production` - Production topology
- `registry://production/{machine}` - Machine config
- `registry://health` - System health

## Prompts (5 workflows)

- `deploy_model` - Local deployment workflow
- `propose_production_deployment` - Production PR workflow
- `troubleshoot_deployment` - Debug issues
- `benchmark_model` - Performance testing
- `register_model` - New model registration

## Security Model

1. **API Key**: Required for all operations
2. **Read-Only Mode**: `REGISTRY_READ_ONLY=true` disables all writes
3. **Production Gate**: `REGISTRY_ALLOW_PRODUCTION_CHANGES=true` required for PRs
4. **Human Approval**: Production changes via PR require manual review/merge

## Two Deployment Domains

### Local Deployments
- Docker containers on the registry host
- For testing and staging
- Managed by `execute_deployment`, `stop_deployment`, etc.
- Direct control, no approval needed

### Production Deployments
- Defined in release configs (GitOps)
- Read via `list_release_configs`, `get_machine_topology`
- Changes via `propose_release_config` create PRs
- Require human review and merge

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Claude    │────▶│  MCP Server │────▶│  Catapult   │
│  Desktop    │     │  (this pkg) │     │   API       │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Release   │
                    │   (GitOps)  │
                    └─────────────┘
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy src/
```
