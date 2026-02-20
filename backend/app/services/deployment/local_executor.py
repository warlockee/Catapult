"""
Local deployment executor using Docker.

Runs containers locally via the Docker daemon for model serving.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, List, Optional
from uuid import UUID

import httpx

from app.core.config import settings
from app.services.api_discovery import (
    build_sample_body,
    detect_api_type_and_recommend,
    extract_request_schema,
    get_sample_request_body,
    requires_file_upload as check_file_upload,
    sort_endpoints_by_priority,
    sort_paths_by_priority,
)
from app.core.exceptions import ContainerNotFoundError
from app.services.deployment.executor_base import (
    ContainerStatus,
    DeploymentConfig,
    DeploymentExecutor,
    DeploymentResult,
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
            Dictionary with discovered API info including request_schema and
            sample_body for each endpoint.
        """
        base_url = endpoint_url.rstrip('/')
        result = {
            "api_type": "generic",
            "openapi_spec": None,
            "endpoints": [],
            "detected_endpoints": [],
            "recommended_benchmark_endpoint": None,
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

                        # Extract endpoints from OpenAPI spec and probe each one
                        paths = spec.get("paths", {})
                        for endpoint_path, methods in paths.items():
                            for method, details in methods.items():
                                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                                    needs_file_upload = check_file_upload(endpoint_path, details)
                                    # Extract schema and generate sample body
                                    request_schema = extract_request_schema(details, spec) if method.upper() != "GET" else None
                                    sample_body = build_sample_body(endpoint_path, model_name, details, spec) if method.upper() != "GET" else None

                                    # Probe the endpoint
                                    try:
                                        if method.upper() == "GET":
                                            probe_response = await client.get(f"{base_url}{endpoint_path}")
                                        else:
                                            probe_response = await client.request(
                                                method.upper(),
                                                f"{base_url}{endpoint_path}",
                                                json=sample_body or {}
                                            )

                                        if probe_response.status_code in [404, 400]:
                                            logger.debug(f"Endpoint {endpoint_path} returned {probe_response.status_code}, skipping")
                                            continue

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
                                            "request_schema": request_schema,
                                            "sample_body": sample_body,
                                        })
                                        result["detected_endpoints"].append(endpoint_path)

                                    except Exception as e:
                                        logger.debug(f"Failed to probe {endpoint_path}: {e}")
                                        result["endpoints"].append({
                                            "method": method.upper(),
                                            "path": endpoint_path,
                                            "summary": details.get("summary", ""),
                                            "description": details.get("description", ""),
                                            "tags": details.get("tags", []),
                                            "status": None,
                                            "response": None,
                                            "requires_file_upload": needs_file_upload,
                                            "request_schema": request_schema,
                                            "sample_body": sample_body,
                                        })
                                        result["detected_endpoints"].append(endpoint_path)

                        # Sort, detect type, recommend
                        result["endpoints"] = sort_endpoints_by_priority(result["endpoints"])
                        result["detected_endpoints"] = sort_paths_by_priority(result["detected_endpoints"])
                        result["api_type"], result["recommended_benchmark_endpoint"] = \
                            detect_api_type_and_recommend(result["detected_endpoints"], result["endpoints"])

                        logger.info(f"Discovered OpenAPI spec from {path} with {len(result['endpoints'])} endpoints")
                        return result
                except Exception as e:
                    logger.debug(f"Failed to fetch {path}: {e}")

            # Fallback: Probe common endpoints
            probe_endpoints = [
                ("/v1/models", "GET", "List available models", False),
                ("/v1/chat/completions", "POST", "Chat completions", False),
                ("/v1/completions", "POST", "Text completions", False),
                ("/v1/embeddings", "POST", "Generate embeddings", False),
                ("/health", "GET", "Health check", False),
                ("/healthz", "GET", "Health check", False),
                ("/ready", "GET", "Readiness check", False),
                ("/", "GET", "Root endpoint", False),
                ("/predict", "POST", "Model prediction", False),
                ("/inference", "POST", "Model inference", False),
                ("/generate", "POST", "Text generation", False),
                ("/synthesize", "POST", "Text-to-speech", False),
                ("/transcribe", "POST", "Speech-to-text", True),
                ("/transcribe/batch", "POST", "Batch speech-to-text", True),
                ("/tts", "POST", "Text-to-speech", False),
                ("/stt", "POST", "Speech-to-text", True),
                ("/info", "GET", "Model info", False),
                ("/version", "GET", "Version info", False),
                ("/metrics", "GET", "Prometheus metrics", False),
            ]

            for path, method, description, needs_file_upload in probe_endpoints:
                try:
                    sample_body = get_sample_request_body(path, model_name) if method == "POST" else None
                    if method == "GET":
                        response = await client.get(f"{base_url}{path}")
                    else:
                        response = await client.post(f"{base_url}{path}", json=sample_body or {})

                    if response.status_code not in [404, 400]:
                        response_data = None
                        try:
                            response_data = response.json()
                        except Exception:
                            if response.text:
                                response_data = {"raw": response.text[:500]}

                        result["detected_endpoints"].append(path)
                        result["endpoints"].append({
                            "method": method,
                            "path": path,
                            "summary": description,
                            "description": "",
                            "tags": [],
                            "status": response.status_code,
                            "response": response_data,
                            "requires_file_upload": needs_file_upload,
                            "request_schema": None,
                            "sample_body": sample_body,
                        })

                except Exception as e:
                    logger.debug(f"Probe failed for {method} {path}: {e}")

            # Sort, detect type, recommend
            result["endpoints"] = sort_endpoints_by_priority(result["endpoints"])
            result["detected_endpoints"] = sort_paths_by_priority(result["detected_endpoints"])
            result["api_type"], result["recommended_benchmark_endpoint"] = \
                detect_api_type_and_recommend(result["detected_endpoints"], result["endpoints"])

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
