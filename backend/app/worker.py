"""
Celery tasks for background job execution.

This module contains all Celery tasks that run in the worker process.
Tasks handle Docker operations, deployment execution, builds, and more.

Uses run_async from async_helpers for clean async/await execution.
Deployment tasks use DeploymentTask base class for automatic failure handling.
"""
import json
import logging
from uuid import UUID
from app.core.celery_app import celery_app, DeploymentTask
from app.core.async_helpers import run_async, run_async_with_db
from app.services.docker_service import docker_service

logger = logging.getLogger(__name__)


# =============================================================================
# Deployment Execution Tasks
#
# These tasks use the DeploymentTask base class which automatically marks
# deployments as failed via the on_failure hook when a task raises an exception.
# =============================================================================

@celery_app.task(base=DeploymentTask, bind=True, acks_late=True)
def deploy_container_task(self, deployment_id: str, config_json: str):
    """
    Celery task to deploy a Docker container.
    Runs in worker which has Docker socket access.

    Uses DeploymentTask base class for automatic failure handling.

    Args:
        deployment_id: UUID of the deployment
        config_json: JSON-serialized DeploymentConfig
    """
    logger.info(f"Starting deployment task for {deployment_id}")

    from app.services.deployment.deployment_service import deployment_service
    from app.services.deployment.executor_base import DeploymentConfig

    config_data = json.loads(config_json)
    config = DeploymentConfig(**config_data)

    async def run_deployment(db):
        await deployment_service.execute_deployment(
            db,
            UUID(deployment_id),
            config,
        )

    run_async_with_db(run_deployment)

    logger.info(f"Deployment task for {deployment_id} completed")
    return f"Deployment {deployment_id} completed"


@celery_app.task(base=DeploymentTask, bind=True, acks_late=True)
def stop_container_task(self, deployment_id: str):
    """
    Celery task to stop a Docker container.
    Runs in worker which has Docker socket access.

    Uses DeploymentTask base class for automatic failure handling.

    Args:
        deployment_id: UUID of the deployment to stop
    """
    logger.info(f"Starting stop task for deployment {deployment_id}")

    from app.services.deployment.deployment_service import deployment_service

    async def run_stop(db):
        await deployment_service.stop_deployment(db, UUID(deployment_id))

    run_async_with_db(run_stop)

    logger.info(f"Stop task for {deployment_id} completed")
    return f"Deployment {deployment_id} stopped"


@celery_app.task(base=DeploymentTask, bind=True, acks_late=True)
def restart_container_task(self, deployment_id: str, config_json: str):
    """
    Celery task to restart a Docker container.
    Runs in worker which has Docker socket access.

    Uses DeploymentTask base class for automatic failure handling.

    Args:
        deployment_id: UUID of the deployment to restart
        config_json: JSON-serialized DeploymentConfig
    """
    logger.info(f"Starting restart task for deployment {deployment_id}")

    from app.services.deployment.deployment_service import deployment_service
    from app.services.deployment.executor_base import DeploymentConfig

    config_data = json.loads(config_json)
    config = DeploymentConfig(**config_data)

    async def run_restart(db):
        await deployment_service.restart_deployment(
            db,
            UUID(deployment_id),
            config,
        )

    run_async_with_db(run_restart)

    logger.info(f"Restart task for {deployment_id} completed")
    return f"Deployment {deployment_id} restarted"


@celery_app.task
def get_container_status_task(container_id: str) -> dict:
    """
    Celery task to get container status.
    Runs in worker which has Docker socket access.

    Args:
        container_id: Docker container ID

    Returns:
        Dict with container status
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "inspect", container_id, "--format",
             '{"running": {{.State.Running}}, "exit_code": {{.State.ExitCode}}, "started_at": "{{.State.StartedAt}}"}'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"running": False, "error": result.stderr}
    except Exception as e:
        return {"running": False, "error": str(e)}


@celery_app.task
def get_container_logs_task(container_id: str, tail: int = 100) -> dict:
    """
    Celery task to get container logs.
    Runs in worker which has Docker socket access.

    Args:
        container_id: Docker container ID
        tail: Number of lines to retrieve

    Returns:
        Dict with logs content
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_id],
            capture_output=True, text=True, timeout=30
        )
        return {
            "logs": result.stdout + result.stderr,
            "truncated": False,
        }
    except Exception as e:
        return {"logs": f"Error getting logs: {e}", "truncated": False}


@celery_app.task(acks_late=True)
def health_check_deployments_task():
    """
    Periodic task to check health of all running deployments.
    Updates health_status in database.
    """
    logger.info("Running deployment health checks")
    try:
        from app.services.deployment.deployment_service import deployment_service

        async def run_health_checks(db):
            return await deployment_service.run_health_checks(db)

        count = run_async_with_db(run_health_checks)

        logger.info(f"Health checks completed for {count} deployments")
        return f"Health checks completed for {count} deployments"
    except Exception as e:
        logger.error(f"Health check task failed: {e}")
        return f"Health check failed: {e}"


# =============================================================================
# Docker Build Tasks
# =============================================================================

@celery_app.task(acks_late=True, expires=4*3600)  # Task expires after 4 hours
def build_docker_image(build_id: str):
    """
    Celery task to run docker build.

    Includes validation to skip stale/completed builds.
    """
    logger.info(f"Starting build task for {build_id}")
    try:
        async def validate_and_run():
            from app.models.docker_build import DockerBuild
            from app.core.database import async_session_maker
            from sqlalchemy import select
            from datetime import datetime

            async with async_session_maker() as db:
                # Fetch the build record
                result = await db.execute(
                    select(DockerBuild).where(DockerBuild.id == UUID(build_id))
                )
                build = result.scalar_one_or_none()

                if not build:
                    logger.warning(f"Build {build_id} not found in database, skipping")
                    return "Build not found"

                # Skip if already completed
                if build.status in ('success', 'failed', 'cancelled'):
                    logger.info(f"Build {build_id} already {build.status}, skipping")
                    return f"Build already {build.status}"

                # Skip if build is too old (created > 4 hours ago)
                created_at = build.created_at
                if created_at.tzinfo:
                    created_at = created_at.replace(tzinfo=None)
                age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
                if age_hours > 4:
                    logger.warning(f"Build {build_id} is {age_hours:.1f}h old, marking as failed")
                    build.status = 'failed'
                    build.error_message = f'Build expired (queued {age_hours:.1f}h ago)'
                    build.completed_at = datetime.utcnow()
                    await db.commit()
                    return "Build expired"

            # Build is valid, proceed with actual build
            await docker_service.run_build(UUID(build_id))
            return "Build completed"

        result = run_async(validate_and_run())

        logger.info(f"Build task for {build_id}: {result}")
        return f"Build {build_id}: {result}"
    except Exception as e:
        logger.error(f"Build task for {build_id} failed: {e}")
        return f"Build {build_id} failed: {e}"


@celery_app.task(acks_late=True)
def recover_orphaned_builds_task():
    """
    Periodic task to detect and recover orphaned builds.

    This runs periodically to catch builds that got stuck while the worker
    was running (not just on startup).
    """
    logger.info("Running periodic orphan build recovery")
    try:
        async def run_recovery():
            from app.services.docker.build_recovery import run_orphan_recovery
            return await run_orphan_recovery()

        found, recovered = run_async(run_recovery())

        if found > 0:
            logger.info(f"Periodic orphan recovery: {recovered}/{found} builds recovered")
        return f"Orphan recovery: {recovered}/{found} builds recovered"
    except Exception as e:
        logger.error(f"Periodic orphan recovery failed: {e}")
        return f"Orphan recovery failed: {e}"


@celery_app.task(acks_late=True)
def cleanup_docker_images_task():
    """
    Celery task to cleanup Docker images.
    Runs in worker which has Docker socket access.
    """
    logger.info("Starting Docker cleanup task")
    try:
        async def run_cleanup():
            from app.services.garbage_collector import cleanup_docker_images, cleanup_superseded_images
            await cleanup_docker_images()
            await cleanup_superseded_images()

        run_async(run_cleanup())

        logger.info("Docker cleanup task completed")
        return "Docker cleanup completed"
    except Exception as e:
        logger.error(f"Docker cleanup task failed: {e}")
        return f"Docker cleanup failed: {e}"


@celery_app.task
def get_docker_disk_usage_task():
    """
    Celery task to get Docker disk usage.
    Runs in worker which has Docker socket access.
    Returns dict with Docker usage stats.
    """
    import subprocess
    import shutil

    def parse_docker_size(size_str: str) -> int:
        """Parse Docker size string (e.g., '1.5GB', '500MB') to bytes."""
        if not size_str or size_str == "0B":
            return 0
        size_str = size_str.strip().upper()
        # Check longest units first to avoid "GB" matching "B"
        units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]
        for unit, multiplier in units:
            if size_str.endswith(unit):
                try:
                    value = float(size_str[:-len(unit)])
                    return int(value * multiplier)
                except ValueError:
                    return 0
        return 0

    def format_bytes(size_bytes: int) -> str:
        """Format bytes to human readable string."""
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{size:.1f} {units[i]}"

    try:
        # Run docker system df
        result = subprocess.run(
            ["docker", "system", "df", "--format", "{{.Type}}\t{{.Size}}\t{{.TotalCount}}"],
            capture_output=True, text=True, timeout=30
        )

        docker_usage = {
            "Images": {"size": 0, "count": 0},
            "Containers": {"size": 0, "count": 0},
            "Local Volumes": {"size": 0, "count": 0},
            "Build Cache": {"size": 0, "count": 0},
        }

        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    type_name = parts[0]
                    size_str = parts[1]
                    count = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                    if type_name in docker_usage:
                        docker_usage[type_name] = {
                            "size": parse_docker_size(size_str),
                            "count": count
                        }

        # Get disk space
        try:
            disk = shutil.disk_usage("/var/lib/docker")
        except (FileNotFoundError, PermissionError):
            disk = shutil.disk_usage("/")

        total_docker = sum(d["size"] for d in docker_usage.values())

        return {
            "images": {
                "count": docker_usage["Images"]["count"],
                "size_bytes": docker_usage["Images"]["size"],
                "size_human": format_bytes(docker_usage["Images"]["size"]),
            },
            "build_cache": {
                "count": docker_usage["Build Cache"]["count"],
                "size_bytes": docker_usage["Build Cache"]["size"],
                "size_human": format_bytes(docker_usage["Build Cache"]["size"]),
            },
            "containers": {
                "count": docker_usage["Containers"]["count"],
                "size_bytes": docker_usage["Containers"]["size"],
                "size_human": format_bytes(docker_usage["Containers"]["size"]),
            },
            "volumes": {
                "count": docker_usage["Local Volumes"]["count"],
                "size_bytes": docker_usage["Local Volumes"]["size"],
                "size_human": format_bytes(docker_usage["Local Volumes"]["size"]),
            },
            "total_docker_bytes": total_docker,
            "total_docker_human": format_bytes(total_docker),
            "disk_available_bytes": disk.free,
            "disk_available_human": format_bytes(disk.free),
            "disk_total_bytes": disk.total,
            "disk_total_human": format_bytes(disk.total),
        }
    except Exception as e:
        logger.error(f"Failed to get Docker disk usage: {e}")
        # Return fallback with just disk info
        try:
            disk = shutil.disk_usage("/")
        except Exception:
            disk = type('obj', (object,), {'free': 0, 'total': 0})()

        return {
            "images": {"count": 0, "size_bytes": 0, "size_human": "N/A"},
            "build_cache": {"count": 0, "size_bytes": 0, "size_human": "N/A"},
            "containers": {"count": 0, "size_bytes": 0, "size_human": "N/A"},
            "volumes": {"count": 0, "size_bytes": 0, "size_human": "N/A"},
            "total_docker_bytes": 0,
            "total_docker_human": "N/A",
            "disk_available_bytes": disk.free,
            "disk_available_human": format_bytes(disk.free),
            "disk_total_bytes": disk.total,
            "disk_total_human": format_bytes(disk.total),
        }


# =============================================================================
# MLflow Metadata Tasks
# =============================================================================

@celery_app.task
def sync_mlflow_metadata_task(version_id: str):
    """
    Celery task to fetch MLflow metadata for a version and store it.
    Fire-and-forget: logs errors but doesn't raise.

    Args:
        version_id: UUID of the version with an mlflow_url
    """
    logger.info(f"Starting MLflow metadata sync for version {version_id}")
    try:
        async def do_sync(db):
            from datetime import datetime, timezone
            from sqlalchemy import select
            from app.models.version import Version
            from app.services.mlflow_service import mlflow_service

            result = await db.execute(
                select(Version).where(Version.id == UUID(version_id))
            )
            version = result.scalar_one_or_none()

            if not version or not version.mlflow_url:
                logger.warning(f"Version {version_id} not found or has no mlflow_url")
                return

            mlflow_data = await mlflow_service.fetch_metadata(version.mlflow_url)
            mlflow_data["fetched_at"] = datetime.now(timezone.utc).isoformat()

            current_metadata = dict(version.meta_data or {})
            current_metadata["mlflow"] = mlflow_data
            version.meta_data = current_metadata

            await db.commit()
            logger.info(f"MLflow metadata synced for version {version_id}")

        run_async_with_db(do_sync)

        return f"MLflow sync for {version_id} completed"
    except Exception as e:
        logger.error(f"MLflow sync for version {version_id} failed: {e}")
        return f"MLflow sync for {version_id} failed: {e}"


# =============================================================================
# Benchmark Tasks
# =============================================================================

@celery_app.task(acks_late=True)
def run_benchmark_task(benchmark_id: str):
    """
    Celery task to run a benchmark.
    Runs in worker for better isolation and to not block the API.

    Routes to either:
    - Docker execution (execution_mode='docker'): Runs benchmarker container
    - Inline execution (execution_mode='inline'): Runs directly in worker

    Args:
        benchmark_id: UUID of the benchmark to execute
    """
    logger.info(f"Starting benchmark task for {benchmark_id}")
    try:
        from sqlalchemy import select
        from app.models.benchmark import Benchmark

        async def run_benchmark(db):
            # Get benchmark to check execution_mode
            result = await db.execute(
                select(Benchmark).where(Benchmark.id == UUID(benchmark_id))
            )
            benchmark = result.scalar_one_or_none()

            if not benchmark:
                logger.error(f"Benchmark {benchmark_id} not found")
                return

            execution_mode = getattr(benchmark, 'execution_mode', 'inline') or 'inline'

            if execution_mode == 'docker':
                # Execute via Docker container
                from app.services.benchmark.executor_service import benchmark_executor
                logger.info(f"Running benchmark {benchmark_id} via Docker container")
                await benchmark_executor.execute(db, UUID(benchmark_id))
            else:
                # Execute inline (legacy mode)
                from app.services.benchmark_service import benchmark_service
                logger.info(f"Running benchmark {benchmark_id} inline")
                await benchmark_service.execute_benchmark(db, UUID(benchmark_id))

        run_async_with_db(run_benchmark)

        logger.info(f"Benchmark task for {benchmark_id} completed")
        return f"Benchmark {benchmark_id} completed"
    except Exception as e:
        logger.error(f"Benchmark task for {benchmark_id} failed: {e}")
        return f"Benchmark {benchmark_id} failed: {e}"
