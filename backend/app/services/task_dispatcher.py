"""
Task dispatcher service for decoupling endpoints from Celery.

This module provides an abstraction layer between API endpoints and Celery tasks,
making the code testable without requiring a running Celery worker.

Usage:
    from app.services.task_dispatcher import task_dispatcher

    # In endpoint:
    task_dispatcher.dispatch_deployment(deployment_id, config_json)

    # In tests, replace with mock:
    with patch('app.services.task_dispatcher.task_dispatcher') as mock:
        mock.dispatch_deployment.return_value = "task-id"
        # ... test code
"""
import logging
from typing import Optional, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)


class TaskDispatcherProtocol(Protocol):
    """Protocol defining the task dispatcher interface."""

    def dispatch_deployment(
        self, deployment_id: UUID, config_json: str
    ) -> Optional[str]:
        """Dispatch a deployment task."""
        ...

    def dispatch_stop(self, deployment_id: UUID) -> Optional[str]:
        """Dispatch a stop deployment task."""
        ...

    def dispatch_restart(
        self, deployment_id: UUID, config_json: str
    ) -> Optional[str]:
        """Dispatch a restart deployment task."""
        ...

    def dispatch_docker_build(self, build_id: UUID) -> Optional[str]:
        """Dispatch a Docker build task."""
        ...

    def dispatch_mlflow_sync(self, version_id: UUID) -> Optional[str]:
        """Dispatch an MLflow metadata sync task."""
        ...

    def dispatch_benchmark(self, benchmark_id: UUID) -> Optional[str]:
        """Dispatch a benchmark execution task."""
        ...

    def dispatch_disk_usage_check(self) -> Optional[str]:
        """Dispatch a Docker disk usage check task."""
        ...

    def dispatch_docker_cleanup(self) -> Optional[str]:
        """Dispatch a Docker image cleanup task."""
        ...

    def dispatch_health_check(self) -> Optional[str]:
        """Dispatch a deployment health check task."""
        ...


class CeleryTaskDispatcher:
    """
    Task dispatcher implementation using Celery.

    This is the production implementation that dispatches tasks to Celery workers.
    All Celery imports are deferred to method calls to avoid circular imports.
    """

    def dispatch_deployment(
        self, deployment_id: UUID, config_json: str
    ) -> Optional[str]:
        """
        Dispatch a container deployment task.

        Args:
            deployment_id: UUID of the deployment
            config_json: JSON-serialized DeploymentConfig

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import deploy_container_task

            result = deploy_container_task.delay(str(deployment_id), config_json)
            logger.info(f"Dispatched deployment task for {deployment_id}: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch deployment task: {e}")
            return None

    def dispatch_stop(self, deployment_id: UUID) -> Optional[str]:
        """
        Dispatch a container stop task.

        Args:
            deployment_id: UUID of the deployment to stop

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import stop_container_task

            result = stop_container_task.delay(str(deployment_id))
            logger.info(f"Dispatched stop task for {deployment_id}: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch stop task: {e}")
            return None

    def dispatch_restart(
        self, deployment_id: UUID, config_json: str
    ) -> Optional[str]:
        """
        Dispatch a container restart task.

        Args:
            deployment_id: UUID of the deployment to restart
            config_json: JSON-serialized DeploymentConfig

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import restart_container_task

            result = restart_container_task.delay(str(deployment_id), config_json)
            logger.info(f"Dispatched restart task for {deployment_id}: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch restart task: {e}")
            return None

    def dispatch_docker_build(self, build_id: UUID) -> Optional[str]:
        """
        Dispatch a Docker image build task.

        Args:
            build_id: UUID of the docker build

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import build_docker_image

            result = build_docker_image.delay(str(build_id))
            logger.info(f"Dispatched docker build task for {build_id}: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch docker build task: {e}")
            return None

    def dispatch_mlflow_sync(self, version_id: UUID) -> Optional[str]:
        """
        Dispatch an MLflow metadata sync task.

        Args:
            version_id: UUID of the version to sync

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import sync_mlflow_metadata_task

            result = sync_mlflow_metadata_task.delay(str(version_id))
            logger.info(f"Dispatched MLflow sync task for {version_id}: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch MLflow sync task: {e}")
            return None

    def dispatch_benchmark(self, benchmark_id: UUID) -> Optional[str]:
        """
        Dispatch a benchmark execution task.

        Args:
            benchmark_id: UUID of the benchmark to run

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import run_benchmark_task

            result = run_benchmark_task.delay(str(benchmark_id))
            logger.info(f"Dispatched benchmark task for {benchmark_id}: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch benchmark task: {e}")
            return None

    def dispatch_disk_usage_check(self) -> Optional[str]:
        """
        Dispatch a Docker disk usage check task.

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import get_docker_disk_usage_task

            result = get_docker_disk_usage_task.delay()
            logger.info(f"Dispatched disk usage check task: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch disk usage check task: {e}")
            return None

    def dispatch_docker_cleanup(self) -> Optional[str]:
        """
        Dispatch a Docker image cleanup task.

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import cleanup_docker_images_task

            result = cleanup_docker_images_task.delay()
            logger.info(f"Dispatched Docker cleanup task: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch Docker cleanup task: {e}")
            return None

    def dispatch_health_check(self) -> Optional[str]:
        """
        Dispatch a deployment health check task.

        Returns:
            Task ID if dispatched successfully, None otherwise
        """
        try:
            from app.worker import health_check_deployments_task

            result = health_check_deployments_task.delay()
            logger.info(f"Dispatched health check task: {result.id}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to dispatch health check task: {e}")
            return None


class NoOpTaskDispatcher:
    """
    No-op task dispatcher for testing.

    This implementation does nothing, allowing tests to run without Celery.
    """

    def dispatch_deployment(
        self, deployment_id: UUID, config_json: str
    ) -> Optional[str]:
        logger.debug(f"NoOp: dispatch_deployment({deployment_id})")
        return f"noop-deployment-{deployment_id}"

    def dispatch_stop(self, deployment_id: UUID) -> Optional[str]:
        logger.debug(f"NoOp: dispatch_stop({deployment_id})")
        return f"noop-stop-{deployment_id}"

    def dispatch_restart(
        self, deployment_id: UUID, config_json: str
    ) -> Optional[str]:
        logger.debug(f"NoOp: dispatch_restart({deployment_id})")
        return f"noop-restart-{deployment_id}"

    def dispatch_docker_build(self, build_id: UUID) -> Optional[str]:
        logger.debug(f"NoOp: dispatch_docker_build({build_id})")
        return f"noop-build-{build_id}"

    def dispatch_mlflow_sync(self, version_id: UUID) -> Optional[str]:
        logger.debug(f"NoOp: dispatch_mlflow_sync({version_id})")
        return f"noop-mlflow-{version_id}"

    def dispatch_benchmark(self, benchmark_id: UUID) -> Optional[str]:
        logger.debug(f"NoOp: dispatch_benchmark({benchmark_id})")
        return f"noop-benchmark-{benchmark_id}"

    def dispatch_disk_usage_check(self) -> Optional[str]:
        logger.debug("NoOp: dispatch_disk_usage_check()")
        return "noop-disk-usage"

    def dispatch_docker_cleanup(self) -> Optional[str]:
        logger.debug("NoOp: dispatch_docker_cleanup()")
        return "noop-docker-cleanup"

    def dispatch_health_check(self) -> Optional[str]:
        logger.debug("NoOp: dispatch_health_check()")
        return "noop-health-check"


def _create_dispatcher() -> TaskDispatcherProtocol:
    """
    Create the appropriate task dispatcher based on environment.

    Returns CeleryTaskDispatcher for production, NoOpTaskDispatcher for tests.
    """
    import os

    environment = os.getenv("ENVIRONMENT", "production")

    if environment == "test":
        logger.info("Using NoOpTaskDispatcher for test environment")
        return NoOpTaskDispatcher()

    return CeleryTaskDispatcher()


# Singleton instance for shared use
task_dispatcher: TaskDispatcherProtocol = _create_dispatcher()
