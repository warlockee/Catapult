"""
Service for Docker build job archiving.

Handles:
- Archiving completed build metadata
- Syncing archived jobs back to database
"""
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import yaml

from app.core.config import settings

logger = logging.getLogger(__name__)


class BuildArchiveService:
    """
    Service for archiving Docker build jobs.

    Responsibilities:
    - Archive build metadata for disaster recovery
    - Restore archived jobs to database on startup
    """

    def __init__(self, archive_dir: Optional[str] = None):
        """
        Initialize BuildArchiveService.

        Args:
            archive_dir: Directory for archived jobs
        """
        self.archive_dir = archive_dir or settings.DOCKER_JOBS_ARCHIVE_DIR
        os.makedirs(self.archive_dir, exist_ok=True)

    def archive_build(
        self,
        build_id: str,
        release_id: str,
        model_id: Optional[str],
        image_tag: str,
        build_type: str,
        dockerfile_path: str,
        started_at: Optional[datetime],
        completed_at: datetime,
        build_args: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Archive a completed build job.

        Args:
            build_id: Build identifier
            release_id: Associated release ID
            model_id: Associated model ID
            image_tag: Docker image tag
            build_type: Type of build
            dockerfile_path: Path to Dockerfile used
            started_at: Build start time
            completed_at: Build completion time
            build_args: Optional build arguments used

        Returns:
            Path to archived job directory
        """
        job_dir = os.path.join(self.archive_dir, build_id)
        os.makedirs(job_dir, exist_ok=True)

        # Copy Dockerfile
        if os.path.exists(dockerfile_path):
            shutil.copy(dockerfile_path, os.path.join(job_dir, "Dockerfile"))

        # Create metadata
        metadata = {
            "job_id": build_id,
            "release_id": release_id,
            "model_id": model_id,
            "image_tag": image_tag,
            "build_type": build_type,
            "started_at": started_at.isoformat() if started_at else None,
            "completed_at": completed_at.isoformat(),
            "build_args": [
                {"name": k, "value": v}
                for k, v in (build_args or {}).items()
            ],
        }

        with open(os.path.join(job_dir, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)

        logger.info(f"Archived build job to {job_dir}")
        return job_dir

    def list_archived_jobs(self) -> list:
        """
        List all archived job IDs.

        Returns:
            List of archived job ID strings
        """
        if not os.path.exists(self.archive_dir):
            return []

        return [
            d for d in os.listdir(self.archive_dir)
            if os.path.isdir(os.path.join(self.archive_dir, d))
        ]

    def load_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for an archived job.

        Args:
            job_id: Job identifier

        Returns:
            Metadata dict or None if not found
        """
        metadata_path = os.path.join(self.archive_dir, job_id, "metadata.yaml")

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, "r") as f:
            return yaml.safe_load(f)

    def get_dockerfile(self, job_id: str) -> Optional[str]:
        """
        Get Dockerfile content for an archived job.

        Args:
            job_id: Job identifier

        Returns:
            Dockerfile content or None if not found
        """
        dockerfile_path = os.path.join(self.archive_dir, job_id, "Dockerfile")

        if not os.path.exists(dockerfile_path):
            return None

        with open(dockerfile_path, "r") as f:
            return f.read()


# Singleton instance
archive_service = BuildArchiveService()
