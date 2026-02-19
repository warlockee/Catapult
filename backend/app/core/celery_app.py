import asyncio
import logging
import os
from uuid import UUID

from celery import Celery, Task
from celery.signals import worker_ready

logger = logging.getLogger(__name__)


class DeploymentTask(Task):
    """
    Base Celery task class for deployment-related tasks.

    Provides automatic failure handling to mark deployments as failed
    when a task errors out, preventing stuck "deploying"/"stopping" states.

    Usage:
        @celery_app.task(base=DeploymentTask, bind=True, acks_late=True)
        def deploy_container_task(self, deployment_id: str, config_json: str):
            # Task implementation
            pass
    """

    # Subclasses can override which statuses should be marked as failed
    fail_on_statuses = ("pending", "deploying", "stopping")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Called when the task raises an exception.

        Automatically marks the deployment as failed to prevent
        the deployment from being stuck in an intermediate state.

        Args:
            exc: The exception raised by the task
            task_id: The unique task ID
            args: The positional arguments passed to the task
            kwargs: The keyword arguments passed to the task
            einfo: The exception info (traceback)
        """
        # The first argument is always deployment_id for deployment tasks
        deployment_id = args[0] if args else None

        if deployment_id:
            logger.error(
                f"Deployment task failed for {deployment_id}: {exc}",
                exc_info=einfo.exc_info if einfo else None
            )
            self._mark_deployment_failed(deployment_id, str(exc))
        else:
            logger.error(f"Task failed but no deployment_id found: {exc}")

        super().on_failure(exc, task_id, args, kwargs, einfo)

    def _mark_deployment_failed(self, deployment_id: str, error: str) -> bool:
        """
        Mark a deployment as failed in the database.

        Uses a separate event loop since this runs in the Celery worker context.

        Args:
            deployment_id: UUID of the deployment
            error: Error message to store

        Returns:
            True if marked successfully, False otherwise
        """
        try:
            from sqlalchemy import select

            from app.core.async_helpers import run_async_with_db
            from app.models.deployment import Deployment

            async def mark_failed(db):
                stmt = select(Deployment).where(Deployment.id == UUID(deployment_id))
                result = await db.execute(stmt)
                deployment = result.scalar_one_or_none()

                if deployment and deployment.status in self.fail_on_statuses:
                    deployment.status = "failed"
                    deployment.health_status = "unhealthy"
                    deployment.host_port = None  # Release any allocated port
                    deployment.meta_data = {
                        **(deployment.meta_data or {}),
                        "error": error,
                    }
                    await db.commit()
                    logger.info(f"Marked deployment {deployment_id} as failed due to task error")
                    return True
                return False

            return run_async_with_db(mark_failed)
        except Exception as e:
            logger.warning(f"Could not mark deployment {deployment_id} as failed: {e}")
            return False

# Get Redis URL from environment or default
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Build task expiration - tasks older than this are discarded
BUILD_TASK_EXPIRY_HOURS = int(os.getenv("BUILD_TASK_EXPIRY_HOURS", "4"))

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Default task expiration (4 hours) - prevents stale tasks from running
    task_default_expires=BUILD_TASK_EXPIRY_HOURS * 3600,
    # Result expiration (1 day)
    result_expires=86400,
    # Beat schedule for periodic tasks
    beat_schedule={
        'recover-orphaned-builds': {
            'task': 'app.worker.recover_orphaned_builds_task',
            'schedule': 3600.0,  # Every hour
        },
        'health-check-deployments': {
            'task': 'app.worker.health_check_deployments_task',
            'schedule': 30.0,  # Every 30 seconds
        },
    },
)


def purge_stale_build_tasks():
    """
    Purge stale build tasks from the queue on worker startup.

    This prevents old tasks from being processed after worker restarts.
    """
    try:
        # Get the number of messages purged
        purged = celery_app.control.purge()
        if purged:
            logger.warning(f"Purged {purged} stale tasks from queue on startup")
        return purged
    except Exception as e:
        logger.error(f"Failed to purge stale tasks: {e}")
        return 0


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Handle worker ready signal.

    1. Purges stale tasks from the queue (prevents processing old builds after restart)
    2. Runs orphan build recovery to mark stuck builds as failed
    """
    logger.info("Worker ready - running startup tasks...")

    # Step 1: Purge stale tasks from queue
    # This prevents old build tasks from being picked up after restart
    purged = purge_stale_build_tasks()
    if purged:
        logger.info(f"Purged {purged} stale tasks from queue")

    # Step 2: Run orphan build recovery
    try:
        from app.services.docker.build_recovery import run_orphan_recovery

        # Create a new event loop for this sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Dispose any existing connections first
            from app.core.database import engine
            loop.run_until_complete(engine.dispose())

            # Run orphan build recovery
            found, recovered = loop.run_until_complete(run_orphan_recovery())
            if found > 0:
                logger.info(f"Orphan recovery: {recovered}/{found} builds marked as failed")
            else:
                logger.info("Orphan recovery: No orphaned builds found")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    except Exception as e:
        logger.error(f"Error during worker startup tasks: {e}")
