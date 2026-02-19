"""
Docker build recovery service.

Handles detection and recovery of orphaned builds that were interrupted
by worker restarts or crashes.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.config import settings
from app.models.docker_build import DockerBuild

logger = logging.getLogger(__name__)


async def find_orphaned_builds(db: AsyncSession) -> List[DockerBuild]:
    """
    Find builds that are stuck in 'building' or 'pending' status.

    A build is considered orphaned if:
    - Status is 'building' or 'pending'
    - AND either:
      - It has a celery_task_id but the task is no longer active
      - It has been in this status longer than the timeout threshold

    Args:
        db: Database session

    Returns:
        List of orphaned DockerBuild records
    """
    timeout_hours = getattr(settings, 'DOCKER_BUILD_TIMEOUT_HOURS', 4)
    timeout_threshold = datetime.utcnow() - timedelta(hours=timeout_hours)

    # Find all builds stuck in building/pending status
    stmt = select(DockerBuild).where(
        DockerBuild.status.in_(['building', 'pending'])
    )
    result = await db.execute(stmt)
    stuck_builds = list(result.scalars().all())

    if not stuck_builds:
        return []

    orphaned = []

    # Get active Celery tasks
    active_task_ids = set()
    try:
        inspect = celery_app.control.inspect()

        # Get active tasks
        active = inspect.active() or {}
        for worker_tasks in active.values():
            for task in worker_tasks:
                active_task_ids.add(task.get('id'))

        # Get reserved tasks (queued but not started)
        reserved = inspect.reserved() or {}
        for worker_tasks in reserved.values():
            for task in worker_tasks:
                active_task_ids.add(task.get('id'))

        # Get scheduled tasks
        scheduled = inspect.scheduled() or {}
        for worker_tasks in scheduled.values():
            for task in worker_tasks:
                active_task_ids.add(task.get('request', {}).get('id'))

    except Exception as e:
        logger.warning(f"Could not inspect Celery tasks: {e}")
        # If we can't inspect, fall back to timeout-based detection only
        active_task_ids = None

    for build in stuck_builds:
        is_orphaned = False
        reason = ""

        # Check if task is no longer active
        if build.celery_task_id and active_task_ids is not None:
            if build.celery_task_id not in active_task_ids:
                is_orphaned = True
                reason = f"Celery task {build.celery_task_id} no longer active"

        # Check if build has exceeded timeout
        if build.created_at:
            # Handle timezone-aware vs naive datetimes
            created_at = build.created_at
            if created_at.tzinfo:
                created_at = created_at.replace(tzinfo=None)

            if created_at < timeout_threshold:
                is_orphaned = True
                if not reason:
                    reason = f"Build exceeded {timeout_hours}h timeout"

        # If no celery_task_id and pending for too long, also consider orphaned
        if not build.celery_task_id and build.status == 'pending':
            if build.created_at:
                created_at = build.created_at
                if created_at.tzinfo:
                    created_at = created_at.replace(tzinfo=None)
                # Pending without task_id for more than 5 minutes is suspicious
                if created_at < datetime.utcnow() - timedelta(minutes=5):
                    is_orphaned = True
                    reason = "Pending build without Celery task ID"

        if is_orphaned:
            logger.info(f"Found orphaned build {build.id}: {reason}")
            orphaned.append(build)

    return orphaned


async def recover_orphaned_builds(db: AsyncSession) -> Tuple[int, int]:
    """
    Recover orphaned builds by marking them as failed.

    Args:
        db: Database session

    Returns:
        Tuple of (total_found, total_recovered)
    """
    if not getattr(settings, 'DOCKER_BUILD_ORPHAN_RECOVERY', True):
        logger.info("Orphan build recovery is disabled")
        return 0, 0

    orphaned = await find_orphaned_builds(db)

    if not orphaned:
        logger.info("No orphaned builds found")
        return 0, 0

    recovered = 0
    for build in orphaned:
        try:
            build.status = 'failed'
            build.error_message = 'Build interrupted by worker restart or timeout'
            build.completed_at = datetime.utcnow()
            recovered += 1
            logger.info(f"Marked orphaned build {build.id} ({build.image_tag}) as failed")
        except Exception as e:
            logger.error(f"Failed to recover build {build.id}: {e}")

    if recovered > 0:
        await db.commit()

    logger.info(f"Orphan recovery complete: {recovered}/{len(orphaned)} builds recovered")
    return len(orphaned), recovered


async def run_orphan_recovery():
    """
    Run orphan recovery using a fresh database session.

    This function is designed to be called on worker startup.
    """
    from app.core.database import async_session_maker

    logger.info("Starting orphan build recovery check...")

    try:
        async with async_session_maker() as db:
            found, recovered = await recover_orphaned_builds(db)
            return found, recovered
    except Exception as e:
        logger.error(f"Orphan recovery failed: {e}")
        return 0, 0
