"""
Service for executing benchmarks via Docker container.

Handles:
- Docker command construction for benchmarker container
- Container execution with log capture
- Progress updates from HTTP callbacks
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.benchmark import Benchmark
from app.models.deployment import Deployment
from app.models.release import Release
from app.models.model import Model

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkExecutionConfig:
    """Configuration for benchmark container execution."""
    endpoint_url: str
    endpoint_path: str = "/v1/chat/completions"
    method: str = "POST"
    concurrent_requests: int = 5
    total_requests: int = 20
    timeout_seconds: float = 60.0
    request_body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    server_type: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of benchmark container execution."""
    success: bool
    return_code: int
    log_path: Optional[str] = None
    error_message: Optional[str] = None


class BenchmarkExecutorService:
    """
    Service for executing benchmarks via Docker container.

    Responsibilities:
    - Build Docker run command for benchmarker container
    - Execute container with proper configuration
    - Handle container lifecycle (start, monitor, cleanup)
    """

    def __init__(
        self,
        benchmarker_image: Optional[str] = None,
        logs_dir: Optional[str] = None,
        callback_base_url: Optional[str] = None,
    ):
        """
        Initialize BenchmarkExecutorService.

        Args:
            benchmarker_image: Docker image for benchmarker (default: from settings)
            logs_dir: Directory for benchmark logs
            callback_base_url: Base URL for progress callbacks
        """
        self.benchmarker_image = benchmarker_image or getattr(
            settings, "BENCHMARKER_IMAGE", "benchmarker:latest"
        )
        self.logs_dir = logs_dir or getattr(
            settings, "BENCHMARK_LOGS_DIR", "/tmp/benchmark_logs"
        )
        self.callback_base_url = callback_base_url or self._get_default_callback_url()

        os.makedirs(self.logs_dir, exist_ok=True)

    def _get_default_callback_url(self) -> str:
        """Get the default callback URL from settings or construct it."""
        callback_url = getattr(settings, "BENCHMARK_CALLBACK_URL", None)
        if callback_url:
            return callback_url

        # Construct from backend host/port
        host = getattr(settings, "HOST", "localhost")
        port = getattr(settings, "PORT", 8000)
        return f"http://{host}:{port}/api/v1/internal/benchmarks"

    def get_log_path(self, benchmark_id: str) -> str:
        """Get log file path for a benchmark."""
        return os.path.join(self.logs_dir, f"benchmark_{benchmark_id}.log")

    def _build_docker_command(
        self,
        benchmark_id: UUID,
        config: BenchmarkExecutionConfig,
    ) -> List[str]:
        """
        Build docker run command for benchmarker container.

        Args:
            benchmark_id: ID of the benchmark
            config: Benchmark configuration

        Returns:
            List of command arguments
        """
        cmd = [
            "docker", "run",
            "--rm",  # Remove container after exit
            "--network", "host",  # Use host network for access to endpoints
            "--name", f"benchmarker-{str(benchmark_id)[:8]}",
        ]

        # Add the benchmarker image
        cmd.append(self.benchmarker_image)

        # Add benchmarker CLI arguments
        cmd.extend(["--endpoint-url", config.endpoint_url])
        cmd.extend(["--callback-url", self.callback_base_url])
        cmd.extend(["--benchmark-id", str(benchmark_id)])
        cmd.extend(["--endpoint-path", config.endpoint_path])
        cmd.extend(["--method", config.method])
        cmd.extend(["--concurrent-requests", str(config.concurrent_requests)])
        cmd.extend(["--total-requests", str(config.total_requests)])
        cmd.extend(["--timeout-seconds", str(config.timeout_seconds)])

        if config.server_type:
            cmd.extend(["--server-type", config.server_type])

        if config.request_body:
            cmd.extend(["--request-body", json.dumps(config.request_body)])

        return cmd

    async def execute(
        self,
        db: AsyncSession,
        benchmark_id: UUID,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute benchmark via Docker container.

        Args:
            db: Database session
            benchmark_id: ID of the benchmark to execute
            timeout: Optional timeout in seconds (default: from settings)

        Returns:
            ExecutionResult with success status and log path
        """
        # Get benchmark record
        result = await db.execute(
            select(Benchmark).where(Benchmark.id == benchmark_id)
        )
        benchmark = result.scalar_one_or_none()

        if not benchmark:
            return ExecutionResult(
                success=False,
                return_code=-1,
                error_message=f"Benchmark not found: {benchmark_id}",
            )

        # Determine endpoint URL and server_type
        endpoint_url = None
        server_type = None

        if benchmark.endpoint_url:
            endpoint_url = benchmark.endpoint_url
            # Try to get server_type from production endpoint
            if benchmark.production_endpoint_id:
                try:
                    from app.services.production_deployment_service import production_deployment_service
                    prod_endpoint = await production_deployment_service.get_endpoint_by_eid(
                        benchmark.production_endpoint_id
                    )
                    if prod_endpoint:
                        server_type = prod_endpoint.get("backend_type")
                except Exception as e:
                    logger.warning(f"Failed to get production endpoint backend_type: {e}")
        elif benchmark.deployment_id:
            result = await db.execute(
                select(Deployment).where(Deployment.id == benchmark.deployment_id)
            )
            deployment = result.scalar_one_or_none()
            if deployment and deployment.endpoint_url:
                endpoint_url = deployment.endpoint_url
                # Get server_type from deployment -> release -> model
                if deployment.release_id:
                    result = await db.execute(
                        select(Release).where(Release.id == deployment.release_id)
                    )
                    release = result.scalar_one_or_none()
                    if release and release.image_id:
                        result = await db.execute(
                            select(Model).where(Model.id == release.image_id)
                        )
                        model = result.scalar_one_or_none()
                        if model:
                            server_type = model.server_type
                            logger.info(f"Got server_type '{server_type}' from deployment model")

        if not endpoint_url:
            benchmark.status = "failed"
            benchmark.error_message = "No endpoint URL available"
            await db.commit()
            return ExecutionResult(
                success=False,
                return_code=-1,
                error_message="No endpoint URL available",
            )

        # Build execution config
        config = BenchmarkExecutionConfig(
            endpoint_url=endpoint_url,
            endpoint_path=benchmark.endpoint_path or "/v1/chat/completions",
            method=benchmark.method or "POST",
            concurrent_requests=benchmark.concurrent_requests or 5,
            total_requests=benchmark.total_requests or 20,
            timeout_seconds=benchmark.timeout_seconds or 60.0,
            request_body=benchmark.meta_data.get("request_body") if benchmark.meta_data else None,
            headers=benchmark.meta_data.get("headers", {}) if benchmark.meta_data else {},
            server_type=server_type,
        )

        # Build docker command
        cmd = self._build_docker_command(benchmark_id, config)

        # Update benchmark status
        benchmark.status = "running"
        benchmark.execution_mode = "docker"
        benchmark.started_at = datetime.utcnow()
        benchmark.stages_completed = []

        log_path = self.get_log_path(str(benchmark_id))
        benchmark.log_path = log_path
        await db.commit()

        logger.info(f"Starting benchmarker container: {' '.join(cmd)}")

        # Execute docker command
        timeout = timeout or getattr(settings, "BENCHMARKER_TIMEOUT", 600)

        try:
            with open(log_path, "w") as log_file:
                log_file.write(f"[{datetime.utcnow().isoformat()}] Starting benchmarker container\n")
                log_file.write(f"Command: {' '.join(cmd)}\n\n")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=log_file,
                )

                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    benchmark.status = "failed"
                    benchmark.error_message = f"Benchmark timed out after {timeout} seconds"
                    benchmark.completed_at = datetime.utcnow()
                    await db.commit()
                    return ExecutionResult(
                        success=False,
                        return_code=-1,
                        log_path=log_path,
                        error_message=f"Benchmark timed out after {timeout} seconds",
                    )

            # Note: The container reports completion via HTTP callback,
            # which updates the benchmark status directly.
            # Here we just check if the container exited cleanly.

            if process.returncode != 0:
                # Container failed - check if status was already updated by callback
                await db.refresh(benchmark)
                if benchmark.status != "completed":
                    benchmark.status = "failed"
                    if not benchmark.error_message:
                        benchmark.error_message = f"Container exited with code {process.returncode}"
                    benchmark.completed_at = datetime.utcnow()
                    await db.commit()

                return ExecutionResult(
                    success=False,
                    return_code=process.returncode,
                    log_path=log_path,
                    error_message=benchmark.error_message,
                )

            return ExecutionResult(
                success=True,
                return_code=0,
                log_path=log_path,
            )

        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            benchmark.status = "failed"
            benchmark.error_message = str(e)
            benchmark.completed_at = datetime.utcnow()
            await db.commit()

            return ExecutionResult(
                success=False,
                return_code=-1,
                log_path=log_path,
                error_message=str(e),
            )

    async def stop(self, benchmark_id: UUID) -> bool:
        """
        Stop a running benchmark container.

        Args:
            benchmark_id: ID of the benchmark

        Returns:
            True if container was stopped successfully
        """
        container_name = f"benchmarker-{str(benchmark_id)[:8]}"

        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "stop", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            return process.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to stop container {container_name}: {e}")
            return False


# Singleton instance
benchmark_executor = BenchmarkExecutorService()
