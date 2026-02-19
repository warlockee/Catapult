import os
import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from app.core.database import async_session_maker
from app.core.config import settings
from app.models.artifact import Artifact
from app.models.docker_build import DockerBuild

logger = logging.getLogger(__name__)

async def cleanup_artifacts():
    """
    Permanently delete artifacts that were soft-deleted more than 30 days ago.
    """
    logger.info("Starting artifact cleanup job...")
    
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    try:
        async with async_session_maker() as db:
            # Find artifacts to delete
            query = select(Artifact).where(Artifact.deleted_at <= cutoff_date)
            result = await db.execute(query)
            artifacts_to_delete = result.scalars().all()
            
            if not artifacts_to_delete:
                logger.info("No artifacts found for cleanup.")
                return

            logger.info(f"Found {len(artifacts_to_delete)} artifacts to permanently delete.")
            
            deleted_count = 0
            errors_count = 0
            
            for artifact in artifacts_to_delete:
                try:
                    # Delete file from disk
                    if artifact.file_path and os.path.exists(artifact.file_path):
                        os.remove(artifact.file_path)
                        logger.info(f"Deleted file: {artifact.file_path}")
                    elif artifact.file_path:
                        logger.warning(f"File not found (already deleted?): {artifact.file_path}")
                    
                    # Delete from database
                    await db.delete(artifact)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting artifact {artifact.id}: {e}")
                    errors_count += 1
            
            await db.commit()
            logger.info(f"Cleanup complete. Deleted: {deleted_count}, Errors: {errors_count}")
            
    except Exception as e:
        logger.error(f"Error during artifact cleanup job: {e}")

async def cleanup_docker_images():
    """
    Prune Docker build cache and dangling images.
    This runs daily and handles general Docker hygiene.
    """
    logger.info("Starting Docker cleanup job...")

    try:
        # Prune build cache
        logger.info("Pruning Docker build cache...")
        proc = await asyncio.create_subprocess_exec(
            "docker", "builder", "prune", "-f", "--filter", "until=24h",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()

        # Prune dangling images
        logger.info("Pruning dangling Docker images...")
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "prune", "-f",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()

        logger.info("Docker cleanup complete.")
    except Exception as e:
        logger.error(f"Error during Docker cleanup: {e}")


async def cleanup_superseded_images():
    """
    Remove Docker images for builds that have passed their retention period.

    This finds builds where:
    - superseded_at is set (not the current build)
    - superseded_at + retention_days has passed
    - cleaned_at is NULL (not already cleaned)

    For each qualifying build, removes the Docker image and sets cleaned_at.
    """
    logger.info("Starting superseded Docker image cleanup...")

    retention_days = settings.DOCKER_IMAGE_RETENTION_DAYS
    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    cleaned_count = 0
    error_count = 0
    space_reclaimed = 0

    try:
        async with async_session_maker() as db:
            # Find superseded successful builds past retention
            stmt = select(DockerBuild).where(
                and_(
                    DockerBuild.superseded_at.isnot(None),
                    DockerBuild.superseded_at <= cutoff,
                    DockerBuild.cleaned_at.is_(None),
                    DockerBuild.status == "success",
                )
            )
            result = await db.execute(stmt)
            builds_to_clean = result.scalars().all()

            if not builds_to_clean:
                logger.info("No superseded images to clean up.")
            else:
                logger.info(f"Found {len(builds_to_clean)} superseded images to clean up.")

                for build in builds_to_clean:
                    try:
                        # Get image size before removal
                        size_bytes = await _get_image_size(build.image_tag)

                        # Remove Docker image
                        success = await _remove_docker_image(build.image_tag)

                        if success:
                            build.cleaned_at = datetime.utcnow()
                            cleaned_count += 1
                            space_reclaimed += size_bytes
                            logger.info(
                                f"Cleaned image {build.image_tag} (build {build.id}), "
                                f"reclaimed {_format_bytes(size_bytes)}"
                            )
                        else:
                            # Image might already be gone - mark as cleaned anyway
                            build.cleaned_at = datetime.utcnow()
                            cleaned_count += 1
                            logger.warning(
                                f"Image {build.image_tag} not found or already removed, "
                                f"marking build {build.id} as cleaned"
                            )

                    except Exception as e:
                        logger.error(f"Error cleaning build {build.id} ({build.image_tag}): {e}")
                        error_count += 1

                await db.commit()

            # Also clean failed builds past retention
            await _cleanup_failed_builds(db)

        logger.info(
            f"Superseded image cleanup complete. "
            f"Cleaned: {cleaned_count}, Errors: {error_count}, "
            f"Space reclaimed: {_format_bytes(space_reclaimed)}"
        )

    except Exception as e:
        logger.error(f"Error during superseded image cleanup: {e}")


async def _cleanup_failed_builds(db):
    """Clean up images from failed builds past retention."""
    retention_days = settings.DOCKER_FAILED_BUILD_RETENTION_DAYS
    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    stmt = select(DockerBuild).where(
        and_(
            DockerBuild.status == "failed",
            DockerBuild.completed_at.isnot(None),
            DockerBuild.completed_at <= cutoff,
            DockerBuild.cleaned_at.is_(None),
        )
    )
    result = await db.execute(stmt)
    failed_builds = result.scalars().all()

    if failed_builds:
        logger.info(f"Cleaning up {len(failed_builds)} failed builds past retention...")
        for build in failed_builds:
            try:
                await _remove_docker_image(build.image_tag)
                build.cleaned_at = datetime.utcnow()
                logger.info(f"Cleaned failed build {build.id} ({build.image_tag})")
            except Exception as e:
                logger.error(f"Error cleaning failed build {build.id}: {e}")

        await db.commit()


async def _get_image_size(image_tag: str) -> int:
    """Get Docker image size in bytes."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", image_tag,
            "--format", "{{.Size}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()

        if proc.returncode == 0 and stdout:
            return int(stdout.decode().strip())
    except Exception:
        pass
    return 0


async def _remove_docker_image(image_tag: str) -> bool:
    """Remove a Docker image. Returns True if successful or image doesn't exist."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "rmi", image_tag,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()

        if proc.returncode == 0:
            return True

        # Image not found is also success (already removed)
        stderr_text = stderr.decode() if stderr else ""
        if "No such image" in stderr_text or "not found" in stderr_text.lower():
            return True

        logger.warning(f"Failed to remove image {image_tag}: {stderr_text}")
        return False

    except Exception as e:
        logger.error(f"Error removing Docker image {image_tag}: {e}")
        return False


def _format_bytes(size_bytes: int) -> str:
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
