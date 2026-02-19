"""MCP Prompts for Catapult.

Prompts provide guided workflows for common tasks.
"""

from mcp.server.fastmcp import FastMCP


def register_prompts(mcp: FastMCP) -> None:
    """Register prompt handlers."""

    @mcp.prompt()
    def deploy_model() -> str:
        """Guided workflow for deploying a model locally."""
        return """# Deploy Model Workflow

You are helping the user deploy a model locally for testing/staging.

## Steps:

1. **Find the model**: Use `list_models` or `search` to find the model
2. **Check versions**: Use `list_versions` with model_name to see available versions
3. **Verify build**: Use `list_docker_builds` to ensure a Docker image exists
   - If no build exists, suggest using `trigger_docker_build`
4. **Deploy**: Use `execute_deployment` with the version_id
   - Default environment is "staging"
   - GPU is auto-detected
5. **Verify**: Use `get_deployment_status` to check it's running
6. **Test**: Use `get_deployment_health` to verify the endpoint responds

## Important Notes:
- This creates a LOCAL deployment on the registry host
- For PRODUCTION deployment, use the `propose_production_deployment` prompt instead
- Local deployments are for testing only

## Example Commands:
```
list_models(search="llama")
list_versions(model_name="llama-3")
execute_deployment(version_id="<uuid>", environment="staging")
get_deployment_status(deployment_id="<uuid>")
```
"""

    @mcp.prompt()
    def propose_production_deployment() -> str:
        """Guided workflow for proposing a production deployment."""
        return """# Production Deployment Workflow

You are helping the user propose a production deployment change.

## IMPORTANT: Production changes require human approval via PR review.

## Steps:

1. **Review current topology**: Use `get_machine_topology` to see GPU allocation
2. **Find available resources**: Look for free GPUs on target machine
3. **Check templates**: Use `list_docker_templates` to see available configs
4. **Prepare the model**: Ensure version is marked as release with `promote_version`
5. **Propose change**: Use `propose_release_config` with:
   - machine: target hostname
   - port: available port
   - template_name: from templates list
   - model_name: model to deploy
   - model_path: path on Ceph
   - gpu_ids: comma-separated GPU IDs
   - tensor_parallel: for multi-GPU inference
6. **Share PR URL**: The response includes the PR URL for review

## Important Notes:
- Changes are NOT immediate - they require PR approval and merge
- Always provide a description explaining the change
- Verify GPU availability before proposing
- The PR will be reviewed by humans before deployment

## Example:
```
get_machine_topology()
list_docker_templates()
propose_release_config(
    machine="gpu-server-01",
    port=8001,
    template_name="vllm.yml",
    model_name="llama-3-70b",
    model_path="/models/llama-3-70b",
    gpu_ids="4,5,6,7",
    tensor_parallel=4,
    description="Add llama-3-70b to gpu-server-01"
)
```
"""

    @mcp.prompt()
    def troubleshoot_deployment() -> str:
        """Guided workflow for troubleshooting deployment issues."""
        return """# Troubleshoot Deployment Workflow

You are helping the user debug a deployment issue.

## Diagnostic Steps:

1. **Check status**: Use `get_deployment_status` for container state
   - Look at: running, healthy, exit_code, error
2. **View logs**: Use `get_deployment_logs` to see container output
   - Increase lines (up to 5000) for more history
3. **Check health**: Use `get_deployment_health` to test the endpoint
   - Note response time and any errors
4. **Discover API**: Use `discover_api_spec` to verify endpoints
5. **Check resources**: Use `get_docker_disk_usage` for disk space

## Common Issues:

### Container Won't Start
- Check logs for error messages
- Verify the Docker image exists
- Check disk space

### Container Running But Unhealthy
- Check endpoint URL is correct
- View recent logs for errors
- Try `restart_deployment`

### Slow Performance
- Run `run_benchmark` to measure latency
- Check GPU allocation
- Review container logs for warnings

### API Errors
- Use `discover_api_spec` to verify endpoints
- Check request format matches API type

## Example Commands:
```
get_deployment_status(deployment_id="<uuid>")
get_deployment_logs(deployment_id="<uuid>", lines=500)
get_deployment_health(deployment_id="<uuid>")
restart_deployment(deployment_id="<uuid>")
```
"""

    @mcp.prompt()
    def benchmark_model() -> str:
        """Guided workflow for benchmarking a model deployment."""
        return """# Benchmark Model Workflow

You are helping the user benchmark a model deployment.

## Prerequisites:
- A running local deployment (use `deploy_model` prompt first if needed)

## Steps:

1. **Verify deployment**: Use `get_deployment_status` to confirm it's running
2. **Check current metrics**: Use `get_benchmark_summary` for existing data
3. **Run benchmark**: Use `run_benchmark` with:
   - deployment_id: the deployment to test
   - endpoint_path: default is "/v1/chat/completions"
   - concurrent_requests: parallel load (default 10)
   - total_requests: total requests to send (default 100)
4. **Poll results**: Benchmark runs async - use `get_benchmark` to check status
5. **Review metrics**: Look at latency percentiles and throughput

## Key Metrics:
- **latency_p50_ms**: Median response time
- **latency_p99_ms**: 99th percentile (worst case)
- **requests_per_second**: Throughput
- **error_rate**: Percentage of failed requests

## Example:
```
get_deployment_status(deployment_id="<uuid>")
get_benchmark_summary(deployment_id="<uuid>")
run_benchmark(
    deployment_id="<uuid>",
    concurrent_requests=10,
    total_requests=100
)
# Wait, then:
get_benchmark(benchmark_id="<returned-id>")
```
"""

    @mcp.prompt()
    def register_model() -> str:
        """Guided workflow for registering a new model."""
        return """# Register Model Workflow

You are helping the user register a new model in the registry.

## Prerequisites:
- Model files uploaded to Ceph storage

## Steps:

1. **Verify storage path**: Use `browse_model_storage` to confirm files exist
2. **Check for duplicates**: Use `list_models(search="<name>")` to avoid conflicts
3. **Create model**: Use `create_model` with:
   - name: unique model name
   - storage_path: Ceph path to model files
   - server_type: vllm, audio, whisper, tts, embedding
   - description: what the model does
   - tags: searchable keywords
   - requires_gpu: True for most ML models
4. **Create version**: Use `create_version` with:
   - model_id: from created model
   - version: semantic version (e.g., "1.0.0")
   - tag: Docker tag
   - digest: content hash
   - ceph_path: version-specific path if different
5. **Build Docker image**: Use `trigger_docker_build` if needed
6. **Test locally**: Use `execute_deployment` to verify

## Naming Conventions:
- Model names: lowercase with hyphens (e.g., "llama-3-8b")
- Versions: semantic versioning (e.g., "1.0.0", "1.1.0-beta")
- Tags: match version or use descriptive tags

## Example:
```
browse_model_storage("models/my-new-model")
list_models(search="my-new-model")
create_model(
    name="my-new-model",
    storage_path="/models/my-new-model",
    server_type="vllm",
    description="Fine-tuned LLaMA model for code generation",
    tags=["llama", "code", "fine-tuned"]
)
create_version(
    model_id="<uuid>",
    version="1.0.0",
    tag="v1.0.0",
    digest="sha256:abc123...",
    auto_build=True
)
```
"""
