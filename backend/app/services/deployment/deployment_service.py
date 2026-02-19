"""
Deployment orchestration service.

Coordinates between:
- DeploymentRepository for database operations
- LocalDeploymentExecutor / K8sDeploymentExecutor for execution
- Celery tasks for async operations
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.exceptions import (
    ContainerNotFoundError,
    DeploymentAlreadyRunningError,
    DeploymentExecutionError,
    DeploymentNotFoundError,
    DeploymentNotRunningError,
    DockerImageNotFoundError,
    VersionNotFoundError,
)
from app.models.deployment import Deployment
from app.models.docker_build import DockerBuild
from app.models.version import Version
from app.services.deployment.executor_base import ContainerStatus, DeploymentConfig
from app.services.deployment.k8s_executor import K8sDeploymentExecutor
from app.services.deployment.local_executor import LocalDeploymentExecutor

logger = logging.getLogger(__name__)


class DeploymentService:
    """
    Orchestration service for deployments.

    Provides high-level methods for managing deployment lifecycle.
    """

    def __init__(self):
        """Initialize with executor instances."""
        self.local_executor = LocalDeploymentExecutor()
        self.k8s_executor = K8sDeploymentExecutor()

    def _mark_failed(self, deployment: Deployment, error: str) -> None:
        """Mark deployment as failed with error details (DRY helper)."""
        deployment.status = "failed"
        deployment.health_status = "unhealthy"
        deployment.meta_data = {
            **deployment.meta_data,
            "error": error,
        }

    def get_executor(self, deployment_type: str):
        """Get the appropriate executor for deployment type."""
        if deployment_type == "local":
            return self.local_executor
        elif deployment_type == "k8s":
            return self.k8s_executor
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")

    async def get_deployment_by_id(
        self,
        db: AsyncSession,
        deployment_id: UUID,
    ) -> Deployment:
        """Get deployment by ID with related data."""
        stmt = (
            select(Deployment)
            .options(joinedload(Deployment.version))
            .where(Deployment.id == deployment_id)
        )
        result = await db.execute(stmt)
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise DeploymentNotFoundError(str(deployment_id))

        return deployment

    async def get_docker_image_tag(
        self,
        db: AsyncSession,
        release_id: UUID,
    ) -> str:
        """Get the Docker image tag for a version."""
        # First check version metadata
        stmt = select(Version).where(Version.id == release_id)
        result = await db.execute(stmt)
        version = result.scalar_one_or_none()

        if not version:
            raise VersionNotFoundError(str(release_id))

        # Check if version has docker_image in metadata
        if version.meta_data and version.meta_data.get("docker_image"):
            return version.meta_data["docker_image"]

        # Otherwise, look for the latest successful build
        stmt = (
            select(DockerBuild)
            .where(
                DockerBuild.release_id == release_id,
                DockerBuild.status == "success",
            )
            .order_by(DockerBuild.completed_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        build = result.scalar_one_or_none()

        if not build:
            raise DockerImageNotFoundError(f"No Docker image found for version {release_id}")

        return build.image_tag

    async def create_deployment(
        self,
        db: AsyncSession,
        release_id: UUID,
        environment: str,
        deployment_type: str,
        api_key_id: Optional[UUID] = None,
        deployed_by: Optional[str] = None,
        gpu_enabled: bool = False,
        metadata: Optional[dict] = None,
        image_tag: Optional[str] = None,
    ) -> Deployment:
        """
        Create a new deployment record.

        Args:
            db: Database session
            release_id: Version to deploy
            environment: Target environment
            deployment_type: 'metadata', 'local', or 'k8s'
            api_key_id: API key that triggered deployment
            deployed_by: Name of deployer
            gpu_enabled: Whether GPU is enabled
            metadata: Additional metadata
            image_tag: Docker image tag used for deployment

        Returns:
            Created deployment record
        """
        deployment = Deployment(
            release_id=release_id,
            environment=environment,
            deployment_type=deployment_type,
            status="pending" if deployment_type != "metadata" else "success",
            api_key_id=api_key_id,
            deployed_by=deployed_by,
            gpu_enabled=gpu_enabled,
            meta_data=metadata or {},
            image_tag=image_tag,
        )

        db.add(deployment)
        await db.commit()
        await db.refresh(deployment)

        return deployment

    async def execute_deployment(
        self,
        db: AsyncSession,
        deployment_id: UUID,
        config: DeploymentConfig,
    ) -> Deployment:
        """
        Execute a pending deployment.

        This is called by the Celery worker.

        Port allocation strategy:
        - Let Docker handle port allocation atomically (no race conditions)
        - Docker assigns an ephemeral port when we pass host_port=None
        - We store the actual assigned port after container starts

        Args:
            db: Database session
            deployment_id: ID of the deployment to execute
            config: Deployment configuration

        Returns:
            Updated deployment record
        """
        deployment = await self.get_deployment_by_id(db, deployment_id)

        if deployment.status == "running":
            raise DeploymentAlreadyRunningError(str(deployment_id))

        # Update status to deploying
        deployment.status = "deploying"
        await db.commit()

        try:
            executor = self.get_executor(deployment.deployment_type)

            # Execute deployment - let Docker handle port allocation
            # By passing host_port=None, Docker atomically assigns an available port
            # This eliminates all TOCTOU race conditions
            result = await executor.deploy(deployment_id, config, host_port=None)

            if result.success:
                deployment.status = "running"
                deployment.container_id = result.container_id
                deployment.host_port = result.port  # Port assigned by Docker
                deployment.endpoint_url = result.endpoint_url
                deployment.started_at = datetime.utcnow()
                deployment.health_status = "unknown"
                logger.info(f"Deployment {deployment_id} started successfully on port {result.port}")
                await db.commit()
                await db.refresh(deployment)
                return deployment

            # Deployment failed
            self._mark_failed(deployment, result.error_message)
            await db.commit()
            await db.refresh(deployment)
            logger.error(f"Deployment {deployment_id} failed: {result.error_message}")
            return deployment

        except Exception as e:
            self._mark_failed(deployment, str(e))
            await db.commit()
            logger.error(f"Deployment {deployment_id} failed with exception: {e}")
            raise DeploymentExecutionError(str(deployment_id), str(e))

    async def stop_deployment(
        self,
        db: AsyncSession,
        deployment_id: UUID,
    ) -> Deployment:
        """
        Stop a running deployment.

        Args:
            db: Database session
            deployment_id: ID of the deployment to stop

        Returns:
            Updated deployment record
        """
        deployment = await self.get_deployment_by_id(db, deployment_id)

        # Accept both "running" and "stopping" status
        # "stopping" is set by API before dispatching task
        if deployment.status not in ("running", "stopping"):
            raise DeploymentNotRunningError(str(deployment_id), deployment.status)

        if not deployment.container_id:
            self._mark_failed(deployment, "No container ID found")
            await db.commit()
            raise DeploymentExecutionError(str(deployment_id), "No container ID found")

        try:
            executor = self.get_executor(deployment.deployment_type)
            stopped = await executor.stop(deployment_id, deployment.container_id)

            if stopped:
                deployment.status = "stopped"
                deployment.stopped_at = datetime.utcnow()
                deployment.health_status = "unknown"
                logger.info(f"Deployment {deployment_id} stopped successfully")
            else:
                self._mark_failed(deployment, "Failed to stop container")
                logger.error(f"Failed to stop deployment {deployment_id}")

            await db.commit()
            await db.refresh(deployment)
            return deployment

        except Exception as e:
            self._mark_failed(deployment, str(e))
            await db.commit()
            raise DeploymentExecutionError(str(deployment_id), str(e))

    async def restart_deployment(
        self,
        db: AsyncSession,
        deployment_id: UUID,
        config: DeploymentConfig,
    ) -> Deployment:
        """
        Restart a deployment.

        Stops the existing container and starts a new one.
        Let Docker handle port allocation atomically (no race conditions).

        Args:
            db: Database session
            deployment_id: ID of the deployment to restart
            config: Deployment configuration

        Returns:
            Updated deployment record
        """
        deployment = await self.get_deployment_by_id(db, deployment_id)

        if not deployment.container_id:
            self._mark_failed(deployment, "No container ID found")
            await db.commit()
            raise DeploymentExecutionError(str(deployment_id), "No container ID found")

        try:
            executor = self.get_executor(deployment.deployment_type)

            # Restart with Docker-assigned port (pass None to let Docker allocate)
            result = await executor.restart(
                deployment_id,
                deployment.container_id,
                config,
                host_port=None,  # Let Docker assign atomically
            )

            if result.success:
                deployment.status = "running"
                deployment.container_id = result.container_id
                deployment.host_port = result.port  # Port assigned by Docker
                deployment.endpoint_url = result.endpoint_url
                deployment.started_at = datetime.utcnow()
                deployment.stopped_at = None
                deployment.health_status = "unknown"
                logger.info(f"Deployment {deployment_id} restarted successfully on port {result.port}")
            else:
                self._mark_failed(deployment, result.error_message)

            await db.commit()
            await db.refresh(deployment)
            return deployment

        except Exception as e:
            self._mark_failed(deployment, str(e))
            await db.commit()
            raise DeploymentExecutionError(str(deployment_id), str(e))

    async def get_deployment_status(
        self,
        db: AsyncSession,
        deployment_id: UUID,
    ) -> ContainerStatus:
        """
        Get the status of a deployment's container.

        Args:
            db: Database session
            deployment_id: ID of the deployment

        Returns:
            ContainerStatus
        """
        deployment = await self.get_deployment_by_id(db, deployment_id)

        if not deployment.container_id:
            return ContainerStatus(
                running=False,
                healthy=False,
                error="No container ID",
            )

        executor = self.get_executor(deployment.deployment_type)
        return await executor.get_status(deployment.container_id)

    async def get_deployment_logs(
        self,
        db: AsyncSession,
        deployment_id: UUID,
        tail: int = 100,
    ) -> str:
        """
        Get logs from a deployment's container.

        Args:
            db: Database session
            deployment_id: ID of the deployment
            tail: Number of lines to retrieve

        Returns:
            Log content
        """
        deployment = await self.get_deployment_by_id(db, deployment_id)

        if not deployment.container_id:
            return "No container ID - deployment may not have started"

        executor = self.get_executor(deployment.deployment_type)
        try:
            return await executor.get_logs(deployment.container_id, tail)
        except ContainerNotFoundError:
            status_msg = f"[status: {deployment.status}]"
            return f"Container no longer exists {status_msg} - logs unavailable"

    async def cleanup_stuck_deployments(self, db: AsyncSession) -> int:
        """
        Clean up deployments stuck in transitional states.

        Marks as failed any deployments that have been stuck too long in:
        - pending (>10 min): task never started
        - deploying (>15 min): execution crashed
        - stopping (>10 min): stop task crashed

        Args:
            db: Database session

        Returns:
            Number of deployments cleaned up
        """
        now = datetime.utcnow()
        cleaned = 0

        # Stuck in "pending" > 10 minutes
        stmt = select(Deployment).where(
            Deployment.status == "pending",
            Deployment.deployment_type == "local",
            Deployment.deployed_at < now - timedelta(minutes=10),
        )
        result = await db.execute(stmt)
        for deployment in result.scalars().all():
            logger.warning(f"Deployment {deployment.id} stuck in pending, marking as failed")
            self._mark_failed(deployment, "Deployment task never started (timeout)")
            cleaned += 1

        # Stuck in "deploying" > 15 minutes
        stmt = select(Deployment).where(
            Deployment.status == "deploying",
            Deployment.deployment_type == "local",
            Deployment.deployed_at < now - timedelta(minutes=15),
        )
        result = await db.execute(stmt)
        for deployment in result.scalars().all():
            logger.warning(f"Deployment {deployment.id} stuck in deploying, marking as failed")
            self._mark_failed(deployment, "Deployment execution crashed (timeout)")
            cleaned += 1

        # Stuck in "stopping" > 10 minutes (use started_at as reference)
        stmt = select(Deployment).where(
            Deployment.status == "stopping",
            Deployment.deployment_type == "local",
            Deployment.started_at < now - timedelta(minutes=10),
        )
        result = await db.execute(stmt)
        for deployment in result.scalars().all():
            logger.warning(f"Deployment {deployment.id} stuck in stopping, marking as failed")
            self._mark_failed(deployment, "Stop task crashed (timeout)")
            deployment.stopped_at = now
            cleaned += 1

        if cleaned > 0:
            await db.commit()
            logger.info(f"Cleaned up {cleaned} stuck deployments")

        return cleaned

    async def run_health_checks(self, db: AsyncSession) -> int:
        """
        Run health checks on all running deployments.

        Also verifies containers actually exist and marks deployments as failed
        if their containers are gone.

        When a deployment becomes healthy for the first time, automatically
        triggers a benchmark to measure real performance metrics.

        Args:
            db: Database session

        Returns:
            Number of deployments checked
        """
        # First clean up any stuck transitional states
        await self.cleanup_stuck_deployments(db)

        stmt = select(Deployment).where(
            Deployment.status == "running",
            Deployment.deployment_type == "local",
            Deployment.endpoint_url.isnot(None),
        )
        result = await db.execute(stmt)
        deployments = result.scalars().all()

        checked = 0
        deployments_to_benchmark = []

        for deployment in deployments:
            try:
                executor = self.local_executor

                # First check if container actually exists
                if deployment.container_id:
                    container_status = await executor.get_status(deployment.container_id)
                    if not container_status.running:
                        # Container is dead - mark deployment as failed
                        logger.warning(
                            f"Deployment {deployment.id} container {deployment.container_id} "
                            f"is not running (exit_code={container_status.exit_code}), marking as failed"
                        )
                        deployment.status = "failed"
                        deployment.health_status = "unhealthy"
                        deployment.stopped_at = datetime.utcnow()
                        checked += 1
                        continue

                # Container exists and is running, check health endpoint
                healthy = await executor.health_check(deployment.endpoint_url)

                new_status = "healthy" if healthy else "unhealthy"
                old_status = deployment.health_status

                if old_status != new_status:
                    deployment.health_status = new_status
                    logger.info(f"Deployment {deployment.id} health status: {new_status}")

                    # Auto-trigger benchmark when deployment becomes healthy for the first time
                    if new_status == "healthy" and old_status == "unknown":
                        # Check if auto-benchmark already triggered
                        if not deployment.meta_data.get("auto_benchmark_triggered"):
                            deployments_to_benchmark.append(deployment)

                checked += 1
            except ContainerNotFoundError:
                # Container was removed externally - mark deployment as failed
                logger.warning(
                    f"Deployment {deployment.id} container {deployment.container_id} "
                    f"not found, marking as failed"
                )
                deployment.status = "failed"
                deployment.health_status = "unhealthy"
                deployment.stopped_at = datetime.utcnow()
                checked += 1
            except Exception as e:
                logger.warning(f"Health check failed for {deployment.id}: {e}")
                deployment.health_status = "unhealthy"

        await db.commit()

        # Trigger auto-benchmarks for newly healthy deployments
        for deployment in deployments_to_benchmark:
            await self._trigger_auto_benchmark(db, deployment)

        return checked

    async def _trigger_auto_benchmark(self, db: AsyncSession, deployment: Deployment) -> None:
        """
        Trigger an automatic benchmark for a newly healthy deployment.

        This runs inference benchmarks to measure real performance metrics
        like TTFT (Time To First Token) and TPS (Tokens Per Second).

        Args:
            db: Database session
            deployment: The deployment to benchmark
        """
        try:
            from app.services.benchmark_service import benchmark_service

            logger.info(f"Triggering auto-benchmark for deployment {deployment.id}")

            # Mark as triggered to prevent duplicate benchmarks
            deployment.meta_data = {
                **(deployment.meta_data or {}),
                "auto_benchmark_triggered": True,
                "auto_benchmark_triggered_at": datetime.utcnow().isoformat(),
            }
            await db.commit()

            # Determine appropriate endpoint based on model's server_type
            from app.models.model import Model
            from app.models.release import Release
            from app.services.benchmark_service import get_inference_endpoint

            endpoint_path = "/v1/chat/completions"  # Default
            request_body = None

            # Try to get server_type from deployment -> release -> model
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
                    if model and model.server_type:
                        endpoint_path, _, request_body = get_inference_endpoint(model.server_type)
                        logger.info(f"Auto-benchmark using endpoint {endpoint_path} for server_type {model.server_type}")

            # Create benchmark record for full verification suite
            # - Health check, inference test
            # - TTFT/TPS benchmark (for streaming endpoints)
            # - Stress test (5 concurrent, 20 total requests)
            benchmark = await benchmark_service.create_benchmark(
                db,
                deployment_id=deployment.id,
                endpoint_path=endpoint_path,
                method="POST",
                concurrent_requests=5,  # Stress test concurrency
                total_requests=20,  # Stress test total requests
                timeout_seconds=60.0,
                request_body=request_body,
            )

            # Dispatch benchmark task to worker via dispatcher
            from app.services.task_dispatcher import task_dispatcher
            task_dispatcher.dispatch_benchmark(benchmark.id)

            logger.info(f"Auto-benchmark {benchmark.id} queued for deployment {deployment.id}")

        except Exception as e:
            logger.error(f"Failed to trigger auto-benchmark for deployment {deployment.id}: {e}")

    async def get_running_deployments(
        self,
        db: AsyncSession,
        deployment_type: Optional[str] = None,
    ) -> List[Deployment]:
        """
        Get all running deployments.

        Args:
            db: Database session
            deployment_type: Optional filter by type

        Returns:
            List of running deployments
        """
        stmt = select(Deployment).where(Deployment.status == "running")

        if deployment_type:
            stmt = stmt.where(Deployment.deployment_type == deployment_type)

        result = await db.execute(stmt)
        return list(result.scalars().all())


# Singleton instance
deployment_service = DeploymentService()
