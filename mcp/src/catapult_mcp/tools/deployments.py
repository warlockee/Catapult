"""Deployment tools for MCP.

These tools manage LOCAL Docker deployments on the registry host.
They do NOT affect production deployments (use Release Configs for production).
"""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_deployment_tools(mcp: FastMCP) -> None:
    """Register deployment-related tools."""

    @mcp.tool()
    def list_deployments(
        status: str | None = None,
        environment: str | None = None,
        model_name: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List LOCAL deployments (Docker containers on registry host).

        These are NOT production deployments. For production topology,
        use list_release_configs.

        Args:
            status: Filter by: running, stopped, failed, pending, deploying
            environment: Filter by: staging, production, development
            model_name: Filter by model name
            limit: Maximum results (default 50)

        Returns:
            List of deployments with: id, model_name, version, status,
            health_status, environment, endpoint_url, host_port,
            gpu_enabled, deployed_at
        """
        client = get_client()
        return client.list_deployments(
            status=status,
            environment=environment,
            model_name=model_name,
            limit=limit,
        )

    @mcp.tool()
    def get_deployment(deployment_id: str) -> dict:
        """Get detailed deployment information.

        Args:
            deployment_id: Deployment UUID

        Returns:
            Full deployment with: id, release_id, model_name, version,
            environment, status, health_status, container_id, endpoint_url,
            host_port, deployment_type, gpu_enabled, started_at, stopped_at,
            deployed_by, metadata
        """
        client = get_client()
        return client.get_deployment(deployment_id)

    @mcp.tool()
    def get_deployment_status(deployment_id: str) -> dict:
        """Get container runtime status (quick health check).

        Use for quick checks without fetching full deployment details.

        Args:
            deployment_id: Deployment UUID

        Returns:
            Status: {running: bool, healthy: bool, exit_code: int|null,
            started_at: str|null, error: str|null}
        """
        client = get_client()
        return client.get_deployment_status(deployment_id)

    @mcp.tool()
    def get_deployment_logs(deployment_id: str, lines: int = 100) -> str:
        """Get container logs.

        Args:
            deployment_id: Deployment UUID
            lines: Number of lines (default 100, max 5000)

        Returns:
            Log text
        """
        client = get_client()
        return client.get_deployment_logs(deployment_id, lines)

    @mcp.tool()
    def get_deployment_health(deployment_id: str) -> dict:
        """Perform active health check on deployment endpoint.

        Probes the deployment's HTTP endpoint to verify it's responding.

        Args:
            deployment_id: Deployment UUID

        Returns:
            {healthy: bool, endpoint_url: str, response_time_ms: int|null}
        """
        client = get_client()
        return client.get_deployment_health(deployment_id)

    @mcp.tool()
    def discover_api_spec(deployment_id: str) -> dict:
        """Probe deployment to discover its API capabilities.

        Attempts to find OpenAPI spec and detect endpoint types.

        Behavior:
        1. Attempts GET /openapi.json for OpenAPI spec
        2. Probes common endpoints: /health, /v1/models, /v1/chat/completions
        3. Returns detected API type: 'openai', 'fastapi', 'audio', 'unknown'
        4. Lists discovered endpoints with methods

        Args:
            deployment_id: Deployment UUID

        Returns:
            {api_type: str, endpoints: [{path, method}],
            detected_capabilities: [str]}
        """
        client = get_client()
        return client.discover_api_spec(deployment_id)

    @mcp.tool()
    def execute_deployment(
        version_id: str,
        environment: str = "staging",
        gpu_enabled: bool | None = None,
        deployment_type: str = "local",
    ) -> dict:
        """Start a LOCAL Docker container for testing/staging.

        This is a WRITE operation.

        IMPORTANT: This is NOT production deployment.
        For production, use propose_release_config.

        Starts a container on the registry host with the model version.
        GPU is auto-detected if not specified.

        Args:
            version_id: Version UUID to deploy
            environment: Target environment (default "staging")
            gpu_enabled: Force GPU on/off (auto-detected if None)
            deployment_type: "local" or "metadata" (default "local")

        Returns:
            {id, status, endpoint_url, container_id}
        """
        client = get_client()
        return client.execute_deployment(
            version_id=version_id,
            environment=environment,
            gpu_enabled=gpu_enabled,
            deployment_type=deployment_type,
        )

    @mcp.tool()
    def stop_deployment(deployment_id: str) -> dict:
        """Stop a LOCAL deployment.

        This is a WRITE operation.

        Only works on registry-managed containers.
        Does NOT affect production deployments.

        Args:
            deployment_id: Deployment UUID

        Returns:
            Updated deployment status
        """
        client = get_client()
        return client.stop_deployment(deployment_id)

    @mcp.tool()
    def start_deployment(deployment_id: str) -> dict:
        """Restart a stopped LOCAL deployment.

        This is a WRITE operation.

        Restarts with the same configuration as before.

        Args:
            deployment_id: Deployment UUID

        Returns:
            Updated deployment status
        """
        client = get_client()
        return client.start_deployment(deployment_id)

    @mcp.tool()
    def restart_deployment(deployment_id: str) -> dict:
        """Restart a LOCAL deployment (stop + start).

        This is a WRITE operation.

        Args:
            deployment_id: Deployment UUID

        Returns:
            Updated deployment status
        """
        client = get_client()
        return client.restart_deployment(deployment_id)
