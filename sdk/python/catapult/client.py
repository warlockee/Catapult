"""Main Registry client."""
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx

from .models import Model, Version, Release, Deployment, ApiKey, Artifact, DockerBuild, AuditLog


class RegistryError(Exception):
    """Base exception for registry errors."""
    pass


class Registry:
    """Docker Release Registry client."""

    def __init__(self, base_url: str = None, api_key: str = None):
        """
        Initialize Registry client.

        Args:
            base_url: Base URL of the registry API
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv("REGISTRY_URL", "http://localhost/api")
        self.api_key = api_key or os.getenv("REGISTRY_API_KEY")

        if not self.api_key:
            raise RegistryError("API key is required. Set REGISTRY_API_KEY environment variable or pass api_key parameter.")

        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key},
            timeout=30.0,
        )

    @classmethod
    def from_env(cls) -> "Registry":
        """Create Registry client from environment variables."""
        return cls()

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise errors if needed."""
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = response.text

            raise RegistryError(f"API error ({response.status_code}): {error_detail}")

        return response.json()

    # ============================================================================
    # Models
    # ============================================================================

    def create_model(
        self,
        name: str,
        storage_path: str,
        repository: str = None,
        description: str = None,
        company: str = None,
        base_model: str = None,
        parameter_count: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Model:
        """
        Create a new model.

        Args:
            name: Model name
            storage_path: Storage path (e.g. s3://bucket/path)
            repository: Optional repository URL
            description: Optional description
            company: Optional company name
            base_model: Optional base model name
            parameter_count: Optional parameter count
            tags: Optional list of tags
            metadata: Optional metadata dict

        Returns:
            Created Model object
        """
        payload = {
            "name": name,
            "storage_path": storage_path,
            "description": description,
        }
        if repository:
            payload["repository"] = repository
        if company:
            payload["company"] = company
        if base_model:
            payload["base_model"] = base_model
        if parameter_count:
            payload["parameter_count"] = parameter_count
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        response = self.client.post(
            "/v1/models",
            json=payload,
        )
        data = self._handle_response(response)
        return Model(**data)

    def list_models(self, search: str = None, limit: int = 100, offset: int = 0) -> List[Model]:
        """
        List models.

        Args:
            search: Search term for model name
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Model objects
        """
        params = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search

        response = self.client.get("/v1/models", params=params)
        data = self._handle_response(response)
        if isinstance(data, dict):
            data = data.get("items", [])
        return [Model(**item) for item in data]

    def get_model(self, model_id: str) -> Model:
        """Get model by ID."""
        response = self.client.get(f"/v1/models/{model_id}")
        data = self._handle_response(response)
        return Model(**data)

    def delete_model(self, model_id: str) -> None:
        """Delete model by ID."""
        response = self.client.delete(f"/v1/models/{model_id}")
        self._handle_response(response)

    # ============================================================================
    # Versions (formerly Releases)
    # ============================================================================

    def create_version(
        self,
        model_name: str,
        version: str,
        tag: str,
        digest: str,
        size_bytes: int = None,
        platform: str = "linux/amd64",
        quantization: str = None,
        release_notes: str = None,
        metadata: Dict[str, Any] = None,
        ceph_path: str = None,
        auto_build: bool = False,
        build_config: Dict[str, Any] = None,
        is_release: bool = False,
    ) -> Version:
        """
        Create a new version.

        Args:
            model_name: Name of the model
            version: Version string
            tag: Docker tag
            digest: Image digest (sha256:...)
            size_bytes: Size in bytes
            platform: Platform (default: linux/amd64)
            quantization: Quantization level (e.g. fp16, int8)
            release_notes: Release notes
            metadata: Additional metadata
            ceph_path: Path on Ceph filesystem
            auto_build: Trigger Docker build automatically
            build_config: Configuration for auto build
            is_release: Whether this is a formal/promoted release

        Returns:
            Created Version object
        """
        # Get or create model
        models = self.list_models(search=model_name)
        model = next((m for m in models if m.name == model_name), None)

        if not model:
            raise RegistryError(f"Model '{model_name}' not found. Create it first with create_model().")

        # Create version
        payload = {
            "model_id": model.id,
            "version": version,
            "tag": tag,
            "digest": digest,
            "size_bytes": size_bytes,
            "platform": platform,
            "metadata": metadata or {},
            "ceph_path": ceph_path,
            "auto_build": auto_build,
            "is_release": is_release,
        }
        if quantization:
            payload["quantization"] = quantization
        if release_notes:
            payload["release_notes"] = release_notes
        if build_config:
            payload["build_config"] = build_config

        response = self.client.post(
            "/v1/versions",
            json=payload,
        )
        data = self._handle_response(response)
        return Version(**data)

    # Backward compatibility alias
    def create_release(self, *args, **kwargs) -> Version:
        """Create a new version. Deprecated: use create_version() instead."""
        return self.create_version(*args, **kwargs)

    def list_versions(
        self,
        model_name: str = None,
        version: str = None,
        is_release: bool = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Version]:
        """
        List versions.

        Args:
            model_name: Filter by model name
            version: Filter by version
            is_release: Filter by release status (True = promoted releases only)
            status: Filter by status (active/archived/etc)
            limit: Max results
            offset: Pagination offset

        Returns:
            List of Version objects
        """
        params = {
            "page": offset // limit + 1 if limit else 1,
            "size": limit,
        }
        if model_name:
            params["model_name"] = model_name
        if version:
            params["version"] = version
        if is_release is not None:
            params["is_release"] = is_release
        if status:
            params["status"] = status

        response = self.client.get("/v1/versions", params=params)
        data = self._handle_response(response)
        if isinstance(data, dict):
            data = data.get("items", [])
        return [Version(**item) for item in data]

    # Backward compatibility alias
    def list_releases(self, *args, **kwargs) -> List[Version]:
        """List versions. Deprecated: use list_versions() instead."""
        return self.list_versions(*args, **kwargs)

    def get_version(self, version_id: str) -> Version:
        """Get version by ID."""
        response = self.client.get(f"/v1/versions/{version_id}")
        data = self._handle_response(response)
        return Version(**data)

    # Backward compatibility alias
    def get_release(self, release_id: str) -> Version:
        """Get version by ID. Deprecated: use get_version() instead."""
        return self.get_version(release_id)

    def get_latest_version(
        self,
        model_name: str,
        environment: str = None,
    ) -> Optional[Version]:
        """
        Get latest version for a model.

        Args:
            model_name: Model name
            environment: Optional environment filter

        Returns:
            Latest Version object or None
        """
        params = {"model_name": model_name}
        if environment:
            params["environment"] = environment

        response = self.client.get("/v1/versions/latest", params=params)
        data = self._handle_response(response)
        return Version(**data) if data else None

    # Backward compatibility alias
    def get_latest_release(self, image_name: str, environment: str = None) -> Optional[Version]:
        """Get latest version. Deprecated: use get_latest_version() instead."""
        return self.get_latest_version(image_name, environment)

    def delete_version(self, version_id: str) -> None:
        """Delete version by ID."""
        response = self.client.delete(f"/v1/versions/{version_id}")
        self._handle_response(response)

    # Backward compatibility alias
    def delete_release(self, release_id: str) -> None:
        """Delete version by ID. Deprecated: use delete_version() instead."""
        return self.delete_version(release_id)

    def promote_version(self, version_id: str, is_release: bool = True) -> Version:
        """
        Promote or demote a version.

        Args:
            version_id: Version ID
            is_release: True to promote, False to demote

        Returns:
            Updated Version object
        """
        response = self.client.put(
            f"/v1/versions/{version_id}",
            json={"is_release": is_release}
        )
        data = self._handle_response(response)
        return Version(**data)

    # ============================================================================
    # Deployments
    # ============================================================================

    def deploy(
        self,
        release_id: str,
        environment: str,
        metadata: Dict[str, Any] = None,
        status: str = "success",
    ) -> Deployment:
        """
        Record a deployment.

        Args:
            release_id: ID of the release being deployed
            environment: Environment name (e.g., 'production', 'staging')
            metadata: Additional deployment metadata
            status: Deployment status (default: 'success')

        Returns:
            Created Deployment object
        """
        response = self.client.post(
            "/v1/deployments",
            json={
                "release_id": release_id,
                "environment": environment,
                "metadata": metadata or {},
                "status": status,
            },
        )
        data = self._handle_response(response)
        return Deployment(**data)

    def list_deployments(
        self,
        environment: str = None,
        release_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Deployment]:
        """
        List deployments.

        Args:
            environment: Filter by environment
            release_id: Filter by release ID
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Deployment objects
        """
        params = {"limit": limit, "offset": offset}
        if environment:
            params["environment"] = environment
        if release_id:
            params["release_id"] = release_id

        response = self.client.get("/v1/deployments", params=params)
        data = self._handle_response(response)
        if isinstance(data, dict):
            data = data.get("items", [])
        return [Deployment(**item) for item in data]

    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get deployment by ID."""
        response = self.client.get(f"/v1/deployments/{deployment_id}")
        data = self._handle_response(response)
        return Deployment(**data)

    # ============================================================================
    # API Keys
    # ============================================================================

    def create_api_key(self, name: str, expires_at: datetime = None) -> ApiKey:
        """
        Create a new API key.

        Args:
            name: Name for the API key
            expires_at: Optional expiration date

        Returns:
            Created ApiKey object (includes plaintext key)
        """
        payload = {"name": name}
        if expires_at:
            payload["expires_at"] = expires_at.isoformat()

        response = self.client.post("/v1/api-keys", json=payload)
        data = self._handle_response(response)
        return ApiKey(**data)

    def list_api_keys(self) -> List[ApiKey]:
        """List all API keys."""
        response = self.client.get("/v1/api-keys")
        data = self._handle_response(response)
        return [ApiKey(**item) for item in data]

    def revoke_api_key(self, key_id: str) -> None:
        """Revoke an API key."""
        response = self.client.delete(f"/v1/api-keys/{key_id}")
        self._handle_response(response)

    # ============================================================================
    # Artifacts
    # ============================================================================

    def list_artifacts(
        self,
        release_id: str = None,
        model_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Artifact]:
        """List artifacts."""
        params = {"limit": limit, "offset": offset}
        if release_id:
            params["release_id"] = release_id
        if model_id:
            params["model_id"] = model_id

        response = self.client.get("/v1/artifacts", params=params)
        data = self._handle_response(response)
        if isinstance(data, dict):
            data = data.get("items", [])
        return [Artifact(**item) for item in data]

    def get_artifact(self, artifact_id: str) -> Artifact:
        """Get artifact by ID."""
        response = self.client.get(f"/v1/artifacts/{artifact_id}")
        data = self._handle_response(response)
        return Artifact(**data)

    # ============================================================================
    # Docker Builds
    # ============================================================================

    def create_build(
        self,
        release_id: str,
        image_tag: str,
        build_type: str,
        artifact_id: str = None,
        artifact_ids: List[str] = None,
        dockerfile_content: str = None,
    ) -> DockerBuild:
        """Trigger a new Docker build."""
        payload = {
            "release_id": release_id,
            "image_tag": image_tag,
            "build_type": build_type,
        }
        if artifact_id:
            payload["artifact_id"] = artifact_id
        if artifact_ids:
            payload["artifact_ids"] = artifact_ids
        if dockerfile_content:
            payload["dockerfile_content"] = dockerfile_content

        response = self.client.post("/v1/docker/builds", json=payload)
        data = self._handle_response(response)
        return DockerBuild(**data)

    def list_builds(self, release_id: str = None) -> List[DockerBuild]:
        """List Docker builds."""
        params = {}
        if release_id:
            params["release_id"] = release_id

        response = self.client.get("/v1/docker/builds", params=params)
        data = self._handle_response(response)
        if isinstance(data, dict):
            data = data.get("items", [])
        return [DockerBuild(**item) for item in data]

    def get_build(self, build_id: str) -> DockerBuild:
        """Get build status."""
        response = self.client.get(f"/v1/docker/builds/{build_id}")
        data = self._handle_response(response)
        return DockerBuild(**data)

    def get_build_logs(self, build_id: str) -> str:
        """Get build logs."""
        response = self.client.get(f"/v1/docker/builds/{build_id}/logs")
        data = self._handle_response(response)
        return data.get("logs", "")

    def stream_build_logs(self, build_id: str):
        """Stream build logs (generator)."""
        with self.client.stream("GET", f"/v1/docker/builds/{build_id}/logs/stream") as response:
            for line in response.iter_lines():
                yield line

    # ============================================================================
    # Audit Logs
    # ============================================================================

    def list_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        resource_type: str = None,
        action: str = None,
    ) -> List[AuditLog]:
        """List audit logs."""
        params = {"limit": limit, "offset": offset}
        if resource_type:
            params["resource_type"] = resource_type
        if action:
            params["action"] = action

        response = self.client.get("/v1/audit-logs", params=params)
        data = self._handle_response(response)
        return [AuditLog(**item) for item in data]

    # ============================================================================
    # System
    # ============================================================================

    def get_storage_usage(self) -> Dict[str, int]:
        """Get storage usage statistics."""
        response = self.client.get("/v1/system/storage")
        return self._handle_response(response)

    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """List files in storage."""
        response = self.client.get("/v1/system/files", params={"path": path})
        return self._handle_response(response)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
