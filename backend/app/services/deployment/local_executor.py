"""
Local deployment executor using Docker.

Runs containers locally via the Docker daemon for model serving.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional, AsyncIterator
from uuid import UUID

import httpx

from app.core.config import settings
from app.core.exceptions import ContainerNotFoundError
from app.services.deployment.executor_base import (
    DeploymentExecutor,
    DeploymentConfig,
    DeploymentResult,
    ContainerStatus,
)
from app.services.deployment.gpu_service import GpuService

logger = logging.getLogger(__name__)


class LocalDeploymentExecutor(DeploymentExecutor):
    """
    Executor for local Docker deployments.

    Responsibilities:
    - Run Docker containers from pre-built images
    - Manage container lifecycle (start/stop/restart)
    - Health checks via HTTP endpoint
    - Log retrieval and streaming
    - Auto-detect GPU availability (delegated to GpuService)
    """

    def __init__(
        self,
        container_prefix: str = None,
        container_port: int = None,
        health_check_endpoint: str = None,
        health_check_timeout: int = None,
    ):
        """
        Initialize the local executor.

        Args:
            container_prefix: Prefix for container names
            container_port: Port inside the container
            health_check_endpoint: Health check path (e.g., /health)
            health_check_timeout: Health check timeout in seconds
        """
        self.container_prefix = container_prefix or settings.DEPLOYMENT_CONTAINER_PREFIX
        self.container_port = container_port or settings.DEPLOYMENT_CONTAINER_PORT
        self.health_check_endpoint = health_check_endpoint or settings.DEPLOYMENT_HEALTH_CHECK_ENDPOINT
        self.health_check_timeout = health_check_timeout or settings.DEPLOYMENT_HEALTH_CHECK_TIMEOUT

        # Initialize GPU service with the same container prefix
        self._gpu_service = GpuService(container_prefix=self.container_prefix)

    async def detect_gpu(self) -> tuple[bool, int]:
        """
        Detect if GPUs are available on the Docker host.

        Delegates to GpuService for actual detection.

        Returns:
            Tuple of (gpu_available, gpu_count)
        """
        return await self._gpu_service.detect_gpu()

    async def find_available_gpu(self) -> Optional[int]:
        """
        Find the GPU with lowest utilization/memory usage.

        Delegates to GpuService for GPU selection.

        Returns:
            GPU device ID (0, 1, 2, etc.) or 0 as default
        """
        return await self._gpu_service.find_available_gpu()

    async def find_available_gpus(self, count: int) -> List[int]:
        """
        Find the N GPUs with lowest utilization/memory usage.

        Excludes GPUs already assigned to other running deployment containers.
        Delegates to GpuService for GPU selection.

        Args:
            count: Number of GPUs to select

        Returns:
            List of GPU device IDs (e.g., [4, 5] for 2 GPUs)
        """
        return await self._gpu_service.find_available_gpus(count, exclude_deployed=True)

    def _get_container_name(self, deployment_id: UUID) -> str:
        """Generate container name from deployment ID."""
        return f"{self.container_prefix}-{str(deployment_id)[:8]}"

    def _build_docker_run_command(
        self,
        deployment_id: UUID,
        config: DeploymentConfig,
        host_port: Optional[int] = None,
        selected_gpus: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Build the docker run command with all flags.

        Args:
            deployment_id: UUID of the deployment
            config: Deployment configuration
            host_port: Port to expose on the host. If None, Docker will auto-assign.
            selected_gpus: List of GPU device IDs to use (e.g., [4, 5] for GPUs 4 and 5)

        Returns:
            List of command arguments
        """
        container_name = self._get_container_name(deployment_id)

        # If no host_port specified, use 0 to let Docker auto-assign
        port_mapping = f"{host_port}:{self.container_port}" if host_port else f"0:{self.container_port}"

        cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--name", container_name,
            "-p", port_mapping,
            "--restart", "unless-stopped",
            "--network", "catapult_registry-network",  # Connect to registry network for health checks
            "--shm-size", "16g",  # Increase shared memory for multi-GPU NCCL communication
        ]

        # GPU support - use smart GPU selection
        if config.gpu_enabled:
            if selected_gpus and len(selected_gpus) > 0:
                # Use specifically selected GPU(s) from smart selection
                # Note: Quotes ARE needed inside the string for Docker CLI to parse device list correctly
                device_ids = ",".join(str(g) for g in selected_gpus)
                cmd.extend(["--gpus", f'"device={device_ids}"'])
            else:
                # Fallback: use all GPUs
                cmd.extend(["--gpus", "all"])

        # Environment variables
        for key, value in config.environment_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Volume mounts
        for host_path, container_path in config.volume_mounts.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Resource limits
        if config.memory_limit:
            cmd.extend(["--memory", config.memory_limit])
        if config.cpu_limit:
            cmd.extend(["--cpus", str(config.cpu_limit)])

        # Image tag
        cmd.append(config.image_tag)

        return cmd

    async def _run_docker_command(
        self,
        cmd: List[str],
        timeout: int = 30,
    ) -> tuple[int, str, str]:
        """
        Run a docker command via subprocess.

        Args:
            cmd: Command arguments
            timeout: Timeout in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        logger.debug(f"Running Docker command: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return (
                process.returncode,
                stdout.decode().strip() if stdout else "",
                stderr.decode().strip() if stderr else "",
            )
        except asyncio.TimeoutError:
            logger.error(f"Docker command timed out: {' '.join(cmd)}")
            return -1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Docker command failed: {e}")
            return -1, "", str(e)

    async def _get_container_ip(self, container_id: str) -> Optional[str]:
        """
        Get the IP address of a running container.

        Args:
            container_id: Docker container ID

        Returns:
            Container IP address or None if not found
        """
        cmd = [
            "docker", "inspect",
            "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container_id,
        ]
        return_code, stdout, _ = await self._run_docker_command(cmd, timeout=10)
        if return_code == 0 and stdout:
            return stdout.strip()
        return None

    async def _get_container_host_port(self, container_id: str) -> Optional[int]:
        """
        Get the host port assigned to a container.

        Used when Docker auto-assigns a port (ephemeral port allocation).

        Args:
            container_id: Docker container ID

        Returns:
            Host port number or None if not found
        """
        cmd = ["docker", "port", container_id, str(self.container_port)]
        return_code, stdout, _ = await self._run_docker_command(cmd, timeout=10)
        if return_code == 0 and stdout:
            # Output format: "0.0.0.0:49153" or "[::]:49153"
            # Parse the port from the end
            for line in stdout.strip().split('\n'):
                if ':' in line:
                    try:
                        port_str = line.rsplit(':', 1)[1]
                        return int(port_str)
                    except (ValueError, IndexError):
                        continue
        return None

    async def deploy(
        self,
        deployment_id: UUID,
        config: DeploymentConfig,
        host_port: int = None,
    ) -> DeploymentResult:
        """
        Deploy a container with the given configuration.

        Automatically detects GPU availability on the host. If gpu_enabled is True
        but no GPUs are available, the deployment will fail with a clear error.

        Port allocation strategy:
        - If host_port is provided, use that specific port
        - If host_port is None, let Docker auto-assign an ephemeral port
          (this is the most reliable method as Docker handles atomically)

        Args:
            deployment_id: UUID of the deployment record
            config: Configuration for the deployment
            host_port: Port to expose. If None, Docker will auto-assign.

        Returns:
            DeploymentResult with container details or error
        """
        # Auto-detect GPU availability
        gpu_available, gpu_count = await self.detect_gpu()

        # Check if GPU is required but not available
        if config.gpu_enabled and not gpu_available:
            error_msg = "GPU requested but no GPUs detected on host. Run 'nvidia-smi' to verify GPU availability."
            logger.error(f"Deployment {deployment_id} failed: {error_msg}")
            return DeploymentResult(
                success=False,
                error_message=error_msg,
            )

        # Smart GPU selection - find the least loaded GPU(s)
        selected_gpus = None
        if config.gpu_enabled:
            if config.gpu_count and config.gpu_count > 1:
                # Multi-GPU: select the N least loaded GPUs
                selected_gpus = await self.find_available_gpus(config.gpu_count)
                logger.info(
                    f"Deployment {deployment_id}: GPU enabled, selected GPUs {selected_gpus} "
                    f"(out of {gpu_count} available)"
                )
            else:
                # Single GPU: use smart selection to find the least loaded GPU
                selected_gpu = await self.find_available_gpu()
                selected_gpus = [selected_gpu]
                logger.info(
                    f"Deployment {deployment_id}: GPU enabled, selected GPU {selected_gpu} "
                    f"(out of {gpu_count} available)"
                )
        else:
            logger.info(f"Deployment {deployment_id}: GPU disabled (available: {gpu_available}, count: {gpu_count})")

        # Log environment variables being passed
        logger.info(f"Deployment {deployment_id}: Environment vars: {config.environment_vars}")

        # Build and run the command
        cmd = self._build_docker_run_command(deployment_id, config, host_port, selected_gpus)

        return_code, stdout, stderr = await self._run_docker_command(cmd, timeout=60)

        if return_code != 0:
            error_msg = stderr or "Unknown error"
            logger.error(f"Failed to start container for {deployment_id}: {error_msg}")

            # Clean up any partially created container
            container_name = f"{self.container_prefix}-{str(deployment_id)[:8]}"
            await self._cleanup_failed_container(container_name)

            return DeploymentResult(
                success=False,
                error_message=error_msg,
            )

        # stdout contains the container ID
        container_id = stdout[:64] if stdout else None

        # Get actual port - either the one we specified or the one Docker assigned
        actual_port = host_port
        if container_id and not host_port:
            # Docker auto-assigned a port, retrieve it
            actual_port = await self._get_container_host_port(container_id)
            if not actual_port:
                logger.error(f"Failed to get auto-assigned port for container {container_id}")
                # Clean up the container since we can't track it
                await self._cleanup_failed_container(self._get_container_name(deployment_id))
                return DeploymentResult(
                    success=False,
                    error_message="Failed to retrieve auto-assigned port from Docker",
                )

        # Always use localhost with the dynamically allocated host port for external access
        endpoint_url = f"http://localhost:{actual_port}"

        logger.info(f"Started container {container_id} for deployment {deployment_id} on port {actual_port}")

        return DeploymentResult(
            success=True,
            container_id=container_id,
            endpoint_url=endpoint_url,
            port=actual_port,
        )

    async def stop(
        self,
        deployment_id: UUID,
        container_id: str,
    ) -> bool:
        """
        Stop and remove a running container.

        Args:
            deployment_id: UUID of the deployment record
            container_id: Docker container ID

        Returns:
            True if stopped successfully, False otherwise
        """
        # Stop the container
        stop_cmd = ["docker", "stop", container_id]
        return_code, _, stderr = await self._run_docker_command(stop_cmd, timeout=30)

        if return_code != 0:
            # Check if container doesn't exist
            if "No such container" in stderr:
                logger.warning(f"Container {container_id} not found, considering it stopped")
                return True
            logger.error(f"Failed to stop container {container_id}: {stderr}")
            return False

        # Remove the container
        rm_cmd = ["docker", "rm", container_id]
        return_code, _, stderr = await self._run_docker_command(rm_cmd, timeout=30)

        if return_code != 0 and "No such container" not in stderr:
            logger.warning(f"Failed to remove container {container_id}: {stderr}")
            # Still return True as container is stopped

        logger.info(f"Stopped container {container_id} for deployment {deployment_id}")
        return True

    async def _cleanup_failed_container(self, container_name: str) -> None:
        """
        Clean up a container that failed to start properly.

        When docker run fails (e.g., port conflict), the container may be left
        in 'Created' state. This method removes such orphaned containers.

        Args:
            container_name: Name of the container to clean up
        """
        try:
            # Check if container exists
            check_cmd = ["docker", "ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.ID}}"]
            return_code, stdout, _ = await self._run_docker_command(check_cmd, timeout=10)

            if return_code == 0 and stdout:
                container_id = stdout.strip()
                logger.info(f"Cleaning up failed container {container_name} ({container_id})")

                # Force remove the container
                rm_cmd = ["docker", "rm", "-f", container_id]
                await self._run_docker_command(rm_cmd, timeout=30)
                logger.info(f"Removed failed container {container_name}")
        except Exception as e:
            logger.warning(f"Error cleaning up failed container {container_name}: {e}")

    async def restart(
        self,
        deployment_id: UUID,
        container_id: str,
        config: DeploymentConfig,
        host_port: int = None,
    ) -> DeploymentResult:
        """
        Restart a container (stop and start with same config).

        Args:
            deployment_id: UUID of the deployment record
            container_id: Docker container ID to restart
            config: Configuration for the restart
            host_port: Port to use (should be same as before)

        Returns:
            DeploymentResult with new container details or error
        """
        # Stop the existing container
        stopped = await self.stop(deployment_id, container_id)
        if not stopped:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to stop container {container_id}",
            )

        # Start a new container
        return await self.deploy(deployment_id, config, host_port)

    async def get_status(
        self,
        container_id: str,
    ) -> ContainerStatus:
        """
        Get the status of a container.

        Args:
            container_id: Docker container ID

        Returns:
            ContainerStatus with container state
        """
        cmd = [
            "docker", "inspect", container_id,
            "--format", '{"running": {{.State.Running}}, "exit_code": {{.State.ExitCode}}, "started_at": "{{.State.StartedAt}}"}'
        ]

        return_code, stdout, stderr = await self._run_docker_command(cmd, timeout=10)

        if return_code != 0:
            if "No such object" in stderr or "No such container" in stderr:
                raise ContainerNotFoundError(container_id)
            return ContainerStatus(
                running=False,
                healthy=False,
                error=stderr,
            )

        try:
            data = json.loads(stdout)
            started_at = None
            if data.get("started_at"):
                try:
                    # Docker uses RFC3339 format
                    started_at_str = data["started_at"].split(".")[0]
                    started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
                except (ValueError, IndexError):
                    pass

            return ContainerStatus(
                running=data.get("running", False),
                healthy=data.get("running", False),  # Basic health = running
                exit_code=data.get("exit_code"),
                started_at=started_at,
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse container status: {e}")
            return ContainerStatus(
                running=False,
                healthy=False,
                error=f"Failed to parse status: {e}",
            )

    async def get_logs(
        self,
        container_id: str,
        tail: int = 100,
    ) -> str:
        """
        Get logs from a container.

        Args:
            container_id: Docker container ID
            tail: Number of lines to retrieve

        Returns:
            Log content as string
        """
        cmd = ["docker", "logs", "--tail", str(tail), container_id]

        return_code, stdout, stderr = await self._run_docker_command(cmd, timeout=30)

        if return_code != 0:
            if "No such container" in stderr:
                raise ContainerNotFoundError(container_id)
            return f"Error getting logs: {stderr}"

        # Combine stdout and stderr as logs can go to either
        return stdout + stderr if stderr else stdout

    async def stream_logs(
        self,
        container_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream logs from a container.

        Args:
            container_id: Docker container ID

        Yields:
            Log lines as they arrive
        """
        cmd = ["docker", "logs", "-f", "--tail", "100", container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield line.decode().rstrip("\n")

        except Exception as e:
            logger.error(f"Error streaming logs: {e}")
            yield f"Error streaming logs: {e}"

    async def health_check(
        self,
        endpoint_url: str,
        timeout: int = None,
    ) -> bool:
        """
        Check if a deployment is healthy via HTTP endpoint.

        Args:
            endpoint_url: Base URL of the deployment
            timeout: Timeout in seconds

        Returns:
            True if healthy, False otherwise
        """
        timeout = timeout or self.health_check_timeout
        health_url = f"{endpoint_url.rstrip('/')}{self.health_check_endpoint}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=timeout)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {health_url}: {e}")
            return False

    def _get_sample_request_body(self, endpoint_path: str, model_name: str = "model") -> dict:
        """Get a sample request body for probing an endpoint."""
        path_lower = endpoint_path.lower()

        if "/chat/completions" in path_lower:
            return {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "stream": False,
            }
        elif "/completions" in path_lower and "/chat" not in path_lower:
            return {
                "model": model_name,
                "prompt": "Hi",
                "max_tokens": 1,
                "stream": False,
            }
        elif "/embeddings" in path_lower:
            return {
                "model": model_name,
                "input": "test",
            }
        elif "/audio/speech" in path_lower:
            return {
                "model": model_name,
                "input": "Hello",
                "voice": "alloy",
            }
        elif "/audio/transcriptions" in path_lower:
            # Transcription requires file upload - can't probe properly
            return {}
        elif "/inference" in path_lower:
            return {
                "input": "test",
                "parameters": {},
            }
        elif "/generate" in path_lower:
            return {
                "model": model_name,
                "prompt": "Hi",
                "max_tokens": 1,
            }
        elif "/predict" in path_lower:
            return {
                "input": "test",
            }
        else:
            return {}

    async def discover_api_spec(
        self,
        endpoint_url: str,
        timeout: int = 5,
    ) -> dict:
        """
        Discover API endpoints from a running container.

        Probes the container for OpenAPI spec and common endpoints to determine
        what APIs are available.

        Args:
            endpoint_url: Base URL of the deployment
            timeout: Timeout in seconds for each probe

        Returns:
            Dictionary with discovered API info:
            {
                "api_type": "openai" | "fastapi" | "generic",
                "openapi_spec": {...} | None,
                "endpoints": [{"method": "GET", "path": "/health", "description": "..."}],
                "detected_endpoints": ["/health", "/v1/models", ...]
            }
        """
        base_url = endpoint_url.rstrip('/')
        result = {
            "api_type": "generic",
            "openapi_spec": None,
            "endpoints": [],
            "detected_endpoints": [],
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            # First, try to get model name from /v1/models
            model_name = "model"
            try:
                models_response = await client.get(f"{base_url}/v1/models")
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    if models_data.get("data") and len(models_data["data"]) > 0:
                        model_name = models_data["data"][0].get("id", "model")
            except Exception:
                pass

            # Try to fetch OpenAPI spec (FastAPI/Flask-RESTX/etc)
            openapi_paths = ["/openapi.json", "/swagger.json", "/api/openapi.json"]
            for path in openapi_paths:
                try:
                    response = await client.get(f"{base_url}{path}")
                    if response.status_code == 200:
                        spec = response.json()
                        result["openapi_spec"] = spec
                        result["api_type"] = "fastapi"

                        # Helper to detect if endpoint requires file upload
                        def requires_file_upload(details: dict) -> bool:
                            """Check if endpoint requires file upload based on OpenAPI spec."""
                            request_body = details.get("requestBody", {})
                            content = request_body.get("content", {})
                            # Check for multipart/form-data content type
                            if "multipart/form-data" in content:
                                return True
                            # Check for application/octet-stream
                            if "application/octet-stream" in content:
                                return True
                            # Check schema for file/binary types
                            for content_type, content_spec in content.items():
                                schema = content_spec.get("schema", {})
                                # Check if schema has file-related format
                                if schema.get("format") in ["binary", "byte"]:
                                    return True
                                # Check properties for file fields
                                properties = schema.get("properties", {})
                                for prop_name, prop_spec in properties.items():
                                    if prop_spec.get("type") == "string" and prop_spec.get("format") in ["binary", "byte"]:
                                        return True
                                    if prop_name.lower() in ["file", "files", "audio", "image", "video"]:
                                        return True
                            return False

                        # Extract endpoints from OpenAPI spec and probe each one
                        paths = spec.get("paths", {})
                        for endpoint_path, methods in paths.items():
                            for method, details in methods.items():
                                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                                    # Check if endpoint requires file upload
                                    needs_file_upload = requires_file_upload(details)
                                    # Probe the endpoint to get actual response
                                    try:
                                        if method.upper() == "GET":
                                            probe_response = await client.get(f"{base_url}{endpoint_path}")
                                        else:
                                            # Use appropriate request body for the endpoint
                                            request_body = self._get_sample_request_body(endpoint_path, model_name)
                                            probe_response = await client.request(
                                                method.upper(),
                                                f"{base_url}{endpoint_path}",
                                                json=request_body
                                            )

                                        # Skip endpoints that don't work (404 or 400)
                                        if probe_response.status_code in [404, 400]:
                                            logger.debug(f"Endpoint {endpoint_path} returned {probe_response.status_code}, skipping")
                                            continue

                                        # Capture actual response data
                                        response_data = None
                                        try:
                                            response_data = probe_response.json()
                                        except Exception:
                                            if probe_response.text:
                                                response_data = {"raw": probe_response.text[:500]}

                                        result["endpoints"].append({
                                            "method": method.upper(),
                                            "path": endpoint_path,
                                            "summary": details.get("summary", ""),
                                            "description": details.get("description", ""),
                                            "tags": details.get("tags", []),
                                            "status": probe_response.status_code,
                                            "response": response_data,
                                            "requires_file_upload": needs_file_upload,
                                        })
                                        result["detected_endpoints"].append(endpoint_path)

                                    except Exception as e:
                                        logger.debug(f"Failed to probe {endpoint_path}: {e}")
                                        # Still include endpoint but without response data
                                        result["endpoints"].append({
                                            "method": method.upper(),
                                            "path": endpoint_path,
                                            "summary": details.get("summary", ""),
                                            "description": details.get("description", ""),
                                            "tags": details.get("tags", []),
                                            "status": None,
                                            "response": None,
                                            "requires_file_upload": needs_file_upload,
                                        })
                                        result["detected_endpoints"].append(endpoint_path)

                        # Sort endpoints by priority: important APIs first
                        def endpoint_priority(ep):
                            p = ep["path"].lower()
                            # Priority 1: Core OpenAI-compatible endpoints
                            if p == "/v1/chat/completions":
                                return (0, p)
                            if p == "/v1/completions":
                                return (1, p)
                            if p == "/v1/models":
                                return (2, p)
                            if p == "/v1/embeddings":
                                return (3, p)
                            # Priority 2: Other v1 endpoints
                            if p.startswith("/v1/"):
                                return (10, p)
                            # Priority 3: Health/status
                            if p in ["/health", "/healthz", "/ready"]:
                                return (20, p)
                            # Priority 4: Common inference endpoints
                            if any(k in p for k in ["/generate", "/inference", "/predict"]):
                                return (30, p)
                            # Priority 5: Info endpoints
                            if p in ["/version", "/metrics", "/info"]:
                                return (40, p)
                            # Everything else
                            return (100, p)

                        result["endpoints"].sort(key=endpoint_priority)

                        # Also detect if this is OpenAI-compatible
                        openai_paths = {"/v1/chat/completions", "/v1/completions", "/v1/models"}
                        if openai_paths.intersection(set(result["detected_endpoints"])):
                            result["api_type"] = "openai"

                        # Set recommended benchmark endpoint (file uploads now supported by benchmarker)
                        # Exclude batch endpoints due to compatibility issues
                        for ep in result["endpoints"]:
                            if ep["method"] == "POST":
                                if not ep["path"].startswith("/health") and ep["path"] != "/metrics" and "/batch" not in ep["path"]:
                                    result["recommended_benchmark_endpoint"] = ep["path"]
                                    break

                        logger.info(f"Discovered OpenAPI spec from {path} with {len(result['endpoints'])} endpoints")
                        return result
                except Exception as e:
                    logger.debug(f"Failed to fetch {path}: {e}")

            # Probe common endpoints to detect API type
            # Format: (path, method, api_type, description, requires_file_upload)
            probe_endpoints = [
                # OpenAI/vLLM compatible
                ("/v1/models", "GET", "openai", "List available models", False),
                ("/v1/chat/completions", "POST", "openai", "Chat completions", False),
                ("/v1/completions", "POST", "openai", "Text completions", False),
                ("/v1/embeddings", "POST", "openai", "Generate embeddings", False),
                # Health/status
                ("/health", "GET", None, "Health check", False),
                ("/healthz", "GET", None, "Health check", False),
                ("/ready", "GET", None, "Readiness check", False),
                ("/", "GET", None, "Root endpoint", False),
                # Common API patterns
                ("/predict", "POST", None, "Model prediction", False),
                ("/inference", "POST", None, "Model inference", False),
                ("/generate", "POST", None, "Text generation", False),
                # Audio APIs (typically require file uploads)
                ("/synthesize", "POST", "audio", "Text-to-speech", False),
                ("/transcribe", "POST", "audio", "Speech-to-text", True),
                ("/transcribe/batch", "POST", "audio", "Batch speech-to-text", True),
                ("/tts", "POST", "audio", "Text-to-speech", False),
                ("/stt", "POST", "audio", "Speech-to-text", True),
                # Info endpoints
                ("/info", "GET", None, "Model info", False),
                ("/version", "GET", None, "Version info", False),
                ("/metrics", "GET", None, "Prometheus metrics", False),
            ]

            detected_api_type = None
            for path, method, api_type, description, needs_file_upload in probe_endpoints:
                try:
                    if method == "GET":
                        response = await client.get(f"{base_url}{path}")
                    else:
                        # Use appropriate request body for the endpoint
                        request_body = self._get_sample_request_body(path, model_name)
                        response = await client.post(f"{base_url}{path}", json=request_body)

                    # Consider endpoint works if we get a valid response
                    # Skip 404 (not found) and 400 (bad request - endpoint doesn't support this)
                    if response.status_code in [200, 201, 405, 422, 500]:
                        # Try to capture actual response data
                        response_data = None
                        try:
                            response_data = response.json()
                        except Exception:
                            # Response might not be JSON
                            if response.text:
                                response_data = {"raw": response.text[:500]}

                        result["detected_endpoints"].append(path)
                        result["endpoints"].append({
                            "method": method,
                            "path": path,
                            "summary": description,
                            "description": description,
                            "tags": [],
                            "status": response.status_code,
                            "response": response_data,
                            "requires_file_upload": needs_file_upload,
                        })
                        if api_type and not detected_api_type:
                            detected_api_type = api_type

                except Exception as e:
                    logger.debug(f"Probe failed for {method} {path}: {e}")

            if detected_api_type:
                result["api_type"] = detected_api_type

            # Set recommended benchmark endpoint (first POST endpoint, file uploads now supported)
            # Exclude batch endpoints due to compatibility issues
            for ep in result["endpoints"]:
                if ep["method"] == "POST":
                    if not ep["path"].startswith("/health") and ep["path"] != "/metrics" and "/batch" not in ep["path"]:
                        result["recommended_benchmark_endpoint"] = ep["path"]
                        break

            logger.info(f"Probed container, detected {len(result['endpoints'])} endpoints, type: {result['api_type']}")

        return result

    async def check_image_exists(self, image_tag: str) -> bool:
        """
        Check if a Docker image exists locally.

        Args:
            image_tag: Docker image tag

        Returns:
            True if image exists, False otherwise
        """
        cmd = ["docker", "image", "inspect", image_tag]
        return_code, _, _ = await self._run_docker_command(cmd, timeout=10)
        return return_code == 0


# Singleton instance
local_executor = LocalDeploymentExecutor()
