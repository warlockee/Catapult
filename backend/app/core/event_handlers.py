"""
Event handlers for domain events.

This module registers handlers that react to domain events,
enabling loose coupling between services.
"""
import logging
from uuid import UUID

from app.core.events import (
    DockerBuildCompletedEvent,
    DockerBuildFailedEvent,
    DockerBuildRequestedEvent,
    ReleaseCreatedEvent,
    event_dispatcher,
    handles,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Release Event Handlers
# =============================================================================

@handles(ReleaseCreatedEvent)
async def on_release_created(event: ReleaseCreatedEvent):
    """
    Handle release creation - trigger auto-build if requested.

    Instead of the release endpoint directly calling docker_service,
    it emits a ReleaseCreatedEvent. This handler then decides
    whether to trigger a Docker build.
    """
    if not event.auto_build:
        return

    logger.info(f"Auto-build requested for release {event.release_id}")

    # Emit a Docker build request event
    build_config = event.build_config or {}

    build_event = DockerBuildRequestedEvent(
        release_id=event.release_id,
        image_tag=build_config.get("image_tag"),
        build_type=build_config.get("build_type", "standard"),
        artifact_ids=build_config.get("artifact_ids", []),
        dockerfile_content=build_config.get("dockerfile_content"),
    )

    await event_dispatcher.dispatch_async(build_event)


@handles(DockerBuildRequestedEvent)
async def on_docker_build_requested(event: DockerBuildRequestedEvent):
    """
    Handle Docker build request - create build and enqueue task.

    This handler creates the build record and enqueues a Celery task
    to execute the actual build.
    """
    from app.core.database import async_session_maker
    from app.services.docker_service import docker_service

    if not event.release_id or not event.image_tag:
        logger.warning("DockerBuildRequestedEvent missing required fields")
        return

    try:
        async with async_session_maker() as db:
            # Convert artifact_ids from list to proper format
            artifact_ids = [UUID(str(aid)) for aid in event.artifact_ids] if event.artifact_ids else None

            build = await docker_service.create_build(
                db=db,
                release_id=event.release_id,
                image_tag=event.image_tag,
                build_type=event.build_type or "standard",
                artifact_ids=artifact_ids,
                dockerfile_content=event.dockerfile_content,
            )

            logger.info(f"Created Docker build {build.id} for release {event.release_id}")

            # Enqueue Celery task via dispatcher
            from app.services.task_dispatcher import task_dispatcher
            task_dispatcher.dispatch_docker_build(build.id)

    except Exception as e:
        logger.error(f"Failed to create Docker build: {e}")


# =============================================================================
# Docker Build Event Handlers
# =============================================================================

@handles(DockerBuildCompletedEvent)
async def on_docker_build_completed(event: DockerBuildCompletedEvent):
    """
    Handle Docker build completion - could trigger notifications, etc.
    """
    logger.info(
        f"Docker build {event.build_id} completed successfully: {event.image_tag}"
    )
    # Future: Send notifications, update dashboards, etc.


@handles(DockerBuildFailedEvent)
async def on_docker_build_failed(event: DockerBuildFailedEvent):
    """
    Handle Docker build failure - could trigger alerts, etc.
    """
    logger.warning(
        f"Docker build {event.build_id} failed: {event.error_message}"
    )
    # Future: Send alerts, create incident tickets, etc.


# =============================================================================
# Registration
# =============================================================================

def register_all_handlers():
    """
    Explicitly register all handlers.

    Call this during application startup to ensure all handlers
    are registered before events are dispatched.

    Note: The @handles decorator auto-registers handlers when
    this module is imported, but this function can be called
    for explicit registration if needed.
    """
    # Handlers are already registered via @handles decorator
    # This function exists for explicit startup registration if needed
    logger.info("Event handlers registered")


# Auto-register when module is imported
# (handlers are registered via @handles decorator)
