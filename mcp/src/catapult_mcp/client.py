"""Registry client adapter for MCP.

Wraps HTTP calls to the registry API. Uses httpx directly rather than the SDK
to ensure all endpoints are available and to add MCP-specific error handling.
"""

from typing import Any
import httpx

from .config import MCPConfig, get_config


class MCPError(Exception):
    """Base exception for MCP errors."""

    pass


class RegistryClient:
    """HTTP client for Registry API.

    Provides methods matching the MCP tool requirements.
    """

    def __init__(self, config: MCPConfig | None = None):
        """Initialize the client.

        Args:
            config: Optional config. Uses global config if not provided.
        """
        self._config = config or get_config()
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            # Backend uses /api/v1 prefix for all API endpoints
            # Assumes config.url ends with /api (e.g. http://localhost/api)
            base_url = self._config.url.rstrip("/") + "/v1"
            self._client = httpx.Client(
                base_url=base_url,
                headers={"X-API-Key": self._config.api_key},
                timeout=self._config.request_timeout_seconds,
            )
        return self._client

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle HTTP response and raise errors if needed."""
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:
                error_detail = response.text
            raise MCPError(f"API error ({response.status_code}): {error_detail}")
        return response.json()

    def _check_write_allowed(self) -> None:
        """Check if write operations are allowed."""
        if self._config.read_only:
            raise MCPError("Write operations disabled in read-only mode")

    def _check_production_write_allowed(self) -> None:
        """Check if production write operations are allowed."""
        self._check_write_allowed()
        if not self._config.allow_production_changes:
            raise MCPError("Production changes disabled. Set REGISTRY_ALLOW_PRODUCTION_CHANGES=true")

    # =========================================================================
    # Models
    # =========================================================================

    def list_models(
        self,
        search: str | None = None,
        server_type: str | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List models in the registry."""
        params: dict[str, Any] = {"limit": min(limit, self._config.max_results)}
        if search:
            params["search"] = search
        if server_type:
            params["server_type"] = server_type
        if source:
            params["source"] = source

        response = self.client.get("/models", params=params)
        data = self._handle_response(response)
        return data.get("items", data) if isinstance(data, dict) else data

    def get_model(self, model_id: str) -> dict:
        """Get model by ID or name."""
        response = self.client.get(f"/models/{model_id}")
        return self._handle_response(response)

    def create_model(
        self,
        name: str,
        storage_path: str,
        server_type: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        requires_gpu: bool = True,
        metadata: dict | None = None,
    ) -> dict:
        """Create a new model."""
        self._check_write_allowed()
        payload: dict[str, Any] = {
            "name": name,
            "storage_path": storage_path,
            "requires_gpu": requires_gpu,
        }
        if server_type:
            payload["server_type"] = server_type
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        response = self.client.post("/models", json=payload)
        return self._handle_response(response)

    def update_model(
        self,
        model_id: str,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Update model metadata."""
        self._check_write_allowed()
        payload: dict[str, Any] = {}
        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata

        response = self.client.put(f"/models/{model_id}", json=payload)
        return self._handle_response(response)

    def delete_model(self, model_id: str) -> dict:
        """Delete a model."""
        self._check_write_allowed()
        response = self.client.delete(f"/models/{model_id}")
        self._handle_response(response)
        return {"success": True, "deleted_id": model_id}

    # =========================================================================
    # Versions
    # =========================================================================

    def list_versions(
        self,
        model_name: str | None = None,
        is_release: bool | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List versions."""
        params: dict[str, Any] = {"size": min(limit, self._config.max_results)}
        if model_name:
            params["model_name"] = model_name
        if is_release is not None:
            params["is_release"] = is_release

        response = self.client.get("/versions", params=params)
        data = self._handle_response(response)
        return data.get("items", data) if isinstance(data, dict) else data

    def get_version(self, version_id: str) -> dict:
        """Get version by ID."""
        response = self.client.get(f"/versions/{version_id}")
        return self._handle_response(response)

    def get_latest_version(self, model_name: str) -> dict:
        """Get latest version of a model."""
        response = self.client.get("/versions/latest", params={"model_name": model_name})
        return self._handle_response(response)

    def create_version(
        self,
        model_id: str,
        version: str,
        tag: str,
        digest: str,
        ceph_path: str | None = None,
        quantization: str | None = None,
        is_release: bool = False,
        auto_build: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        """Create a new version."""
        self._check_write_allowed()
        payload: dict[str, Any] = {
            "model_id": model_id,
            "version": version,
            "tag": tag,
            "digest": digest,
            "is_release": is_release,
            "auto_build": auto_build,
        }
        if ceph_path:
            payload["ceph_path"] = ceph_path
        if quantization:
            payload["quantization"] = quantization
        if metadata:
            payload["metadata"] = metadata

        response = self.client.post("/versions", json=payload)
        return self._handle_response(response)

    def promote_version(self, version_id: str, is_release: bool = True) -> dict:
        """Promote or demote a version."""
        self._check_write_allowed()
        response = self.client.put(f"/versions/{version_id}", json={"is_release": is_release})
        return self._handle_response(response)

    # =========================================================================
    # Deployments (Local)
    # =========================================================================

    def list_deployments(
        self,
        status: str | None = None,
        environment: str | None = None,
        model_name: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List local deployments."""
        params: dict[str, Any] = {"size": min(limit, self._config.max_results)}
        if status:
            params["status"] = status
        if environment:
            params["environment"] = environment
        # Note: model_name filter may need to be done client-side

        response = self.client.get("/deployments", params=params)
        data = self._handle_response(response)
        items = data.get("items", data) if isinstance(data, dict) else data

        # Client-side filter by model_name if needed
        if model_name and items:
            items = [d for d in items if d.get("image_name", "").lower() == model_name.lower()]

        return items

    def get_deployment(self, deployment_id: str) -> dict:
        """Get deployment by ID."""
        response = self.client.get(f"/deployments/{deployment_id}")
        return self._handle_response(response)

    def get_deployment_status(self, deployment_id: str) -> dict:
        """Get deployment container status."""
        response = self.client.get(f"/deployments/{deployment_id}/status")
        return self._handle_response(response)

    def get_deployment_logs(self, deployment_id: str, lines: int = 100) -> str:
        """Get deployment logs."""
        lines = min(lines, self._config.max_log_lines)
        response = self.client.get(f"/deployments/{deployment_id}/logs", params={"tail": lines})
        data = self._handle_response(response)
        return data.get("logs", "")

    def get_deployment_health(self, deployment_id: str) -> dict:
        """Check deployment health."""
        response = self.client.get(f"/deployments/{deployment_id}/health")
        return self._handle_response(response)

    def discover_api_spec(self, deployment_id: str) -> dict:
        """Discover API spec from deployment."""
        response = self.client.get(f"/deployments/{deployment_id}/api-spec")
        return self._handle_response(response)

    def execute_deployment(
        self,
        version_id: str,
        environment: str = "staging",
        gpu_enabled: bool | None = None,
        deployment_type: str = "local",
    ) -> dict:
        """Execute a local deployment."""
        self._check_write_allowed()
        payload: dict[str, Any] = {
            "release_id": version_id,
            "environment": environment,
            "deployment_type": deployment_type,
        }
        if gpu_enabled is not None:
            payload["gpu_enabled"] = gpu_enabled

        response = self.client.post("/deployments/execute", json=payload)
        return self._handle_response(response)

    def stop_deployment(self, deployment_id: str) -> dict:
        """Stop a local deployment."""
        self._check_write_allowed()
        response = self.client.post(f"/deployments/{deployment_id}/stop")
        return self._handle_response(response)

    def start_deployment(self, deployment_id: str) -> dict:
        """Start a stopped deployment."""
        self._check_write_allowed()
        response = self.client.post(f"/deployments/{deployment_id}/start")
        return self._handle_response(response)

    def restart_deployment(self, deployment_id: str) -> dict:
        """Restart a deployment."""
        self._check_write_allowed()
        response = self.client.post(f"/deployments/{deployment_id}/restart")
        return self._handle_response(response)

    # =========================================================================
    # Docker Builds
    # =========================================================================

    def list_docker_builds(
        self,
        version_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List Docker builds."""
        params: dict[str, Any] = {"size": min(limit, self._config.max_results)}
        if version_id:
            params["release_id"] = version_id
        # Note: status filter may need client-side filtering

        response = self.client.get("/docker/builds", params=params)
        data = self._handle_response(response)
        items = data.get("items", data) if isinstance(data, dict) else data

        if status and items:
            items = [b for b in items if b.get("status") == status]

        return items

    def get_docker_build(self, build_id: str) -> dict:
        """Get Docker build details."""
        response = self.client.get(f"/docker/builds/{build_id}")
        return self._handle_response(response)

    def get_docker_build_logs(self, build_id: str) -> str:
        """Get Docker build logs."""
        response = self.client.get(f"/docker/builds/{build_id}/logs")
        data = self._handle_response(response)
        return data.get("logs", "")

    def trigger_docker_build(
        self,
        version_id: str,
        build_type: str = "organic",
        image_tag: str | None = None,
    ) -> dict:
        """Trigger a Docker build."""
        self._check_write_allowed()
        payload: dict[str, Any] = {
            "release_id": version_id,
            "build_type": build_type,
        }
        if image_tag:
            payload["image_tag"] = image_tag

        response = self.client.post("/docker/builds", json=payload)
        return self._handle_response(response)

    def get_docker_disk_usage(self) -> dict:
        """Get Docker disk usage."""
        response = self.client.get("/docker/disk-usage")
        return self._handle_response(response)

    # =========================================================================
    # Benchmarks
    # =========================================================================

    def list_benchmarks(
        self,
        deployment_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List benchmarks."""
        if deployment_id:
            response = self.client.get(
                f"/benchmarks/deployment/{deployment_id}",
                params={"limit": min(limit, self._config.max_results)},
            )
        else:
            # No general list endpoint, return empty
            return []
        return self._handle_response(response)

    def get_benchmark(self, benchmark_id: str) -> dict:
        """Get benchmark details."""
        response = self.client.get(f"/benchmarks/{benchmark_id}")
        return self._handle_response(response)

    def run_benchmark(
        self,
        deployment_id: str,
        endpoint_path: str = "/v1/chat/completions",
        concurrent_requests: int = 10,
        total_requests: int = 100,
        request_body: dict | None = None,
    ) -> dict:
        """Run a benchmark."""
        self._check_write_allowed()
        payload: dict[str, Any] = {
            "deployment_id": deployment_id,
            "endpoint_path": endpoint_path,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
        }
        if request_body:
            payload["request_body"] = request_body

        response = self.client.post("/benchmarks/async", json=payload)
        return self._handle_response(response)

    def get_benchmark_summary(self, deployment_id: str) -> dict:
        """Get benchmark summary for a deployment."""
        response = self.client.get(f"/benchmarks/deployment/{deployment_id}/summary")
        return self._handle_response(response)

    # =========================================================================
    # Release Configs (Production Topology)
    # =========================================================================

    def list_release_configs(
        self,
        machine: str | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        """List release configs from release configs."""
        params: dict[str, Any] = {}
        if machine:
            params["machine"] = machine
        if model_name:
            params["model_name"] = model_name

        response = self.client.get("/release-configs", params=params)
        return self._handle_response(response)

    def get_release_config(self, machine: str, port: int) -> dict:
        """Get specific release config."""
        response = self.client.get(f"/release-configs/{machine}/{port}")
        return self._handle_response(response)

    def get_machine_topology(self) -> dict:
        """Get GPU allocation topology."""
        response = self.client.get("/release-configs/machines")
        return self._handle_response(response)

    def get_models_by_deployment(self) -> dict:
        """Get deployments grouped by model."""
        response = self.client.get("/release-configs/models")
        return self._handle_response(response)

    def list_docker_templates(self) -> list[str]:
        """List available docker-compose templates."""
        response = self.client.get("/release-configs/templates")
        return self._handle_response(response)

    def propose_release_config(
        self,
        machine: str,
        port: int,
        template_name: str,
        model_name: str,
        model_path: str,
        gpu_ids: str,
        tensor_parallel: int = 1,
        description: str = "",
    ) -> dict:
        """Create PR for production deployment."""
        self._check_production_write_allowed()
        payload = {
            "machine": machine,
            "port": port,
            "template_name": template_name,
            "model_name": model_name,
            "model_path": model_path,
            "gpu_ids": gpu_ids,
            "tensor_parallel": tensor_parallel,
        }
        params = {"description": description} if description else {}

        response = self.client.post("/release-configs/propose", json=payload, params=params)
        return self._handle_response(response)

    # =========================================================================
    # Artifacts
    # =========================================================================

    def list_artifacts(
        self,
        version_id: str | None = None,
        model_id: str | None = None,
        artifact_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List artifacts."""
        params: dict[str, Any] = {"limit": min(limit, self._config.max_results)}
        if version_id:
            params["release_id"] = version_id
        if model_id:
            params["model_id"] = model_id

        response = self.client.get("/artifacts", params=params)
        data = self._handle_response(response)
        items = data.get("items", data) if isinstance(data, dict) else data

        if artifact_type and items:
            items = [a for a in items if a.get("artifact_type") == artifact_type]

        return items

    def get_artifact(self, artifact_id: str) -> dict:
        """Get artifact details."""
        response = self.client.get(f"/artifacts/{artifact_id}")
        return self._handle_response(response)

    # =========================================================================
    # System
    # =========================================================================

    def health_check(self) -> dict:
        """Check registry health."""
        # Health endpoint is at /api/health (not under /api/v1)
        base_url = self._config.url.rstrip("/")
        response = httpx.get(
            f"{base_url}/health",
            headers={"X-API-Key": self._config.api_key},
            timeout=self._config.request_timeout_seconds,
        )
        return self._handle_response(response)

    def get_audit_logs(
        self,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get audit logs."""
        params: dict[str, Any] = {"limit": min(limit, self._config.max_results)}
        if action:
            params["action"] = action
        if resource_type:
            params["resource_type"] = resource_type

        response = self.client.get("/audit-logs", params=params)
        return self._handle_response(response)

    def get_storage_usage(self) -> dict:
        """Get storage usage statistics."""
        response = self.client.get("/system/storage")
        return self._handle_response(response)

    def browse_model_storage(self, relative_path: str = "") -> list[dict]:
        """Browse model storage filesystem."""
        # Strip leading slash for safety
        path = relative_path.lstrip("/")
        response = self.client.get("/system/files", params={"path": path})
        return self._handle_response(response)

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query: str,
        types: list[str] | None = None,
        limit: int = 20,
    ) -> dict:
        """Search across models, versions, and deployments."""
        results: dict[str, list] = {}
        search_types = types or ["models", "versions", "deployments"]
        per_type_limit = min(limit, self._config.max_results)

        if "models" in search_types:
            results["models"] = self.list_models(search=query, limit=per_type_limit)

        if "versions" in search_types:
            # Search versions by iterating models that match
            matching_models = self.list_models(search=query, limit=per_type_limit)
            versions = []
            for model in matching_models[:5]:  # Limit to avoid too many calls
                model_versions = self.list_versions(model_name=model.get("name"), limit=5)
                versions.extend(model_versions)
            results["versions"] = versions[:per_type_limit]

        if "deployments" in search_types:
            # Get deployments and filter client-side
            all_deployments = self.list_deployments(limit=per_type_limit * 2)
            query_lower = query.lower()
            results["deployments"] = [
                d
                for d in all_deployments
                if query_lower in d.get("image_name", "").lower()
                or query_lower in d.get("environment", "").lower()
            ][:per_type_limit]

        return results

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


# Global client instance
_client: RegistryClient | None = None


def get_client() -> RegistryClient:
    """Get or create the client singleton."""
    global _client
    if _client is None:
        _client = RegistryClient()
    return _client
