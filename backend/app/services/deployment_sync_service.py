"""
Deployment filesystem materialization service.

Handles persisting deployment records to the filesystem for recovery
and syncing them back to the database on startup.

File structure:
  /fsx/deployments/
    {model_name}/
      {environment}.json

Each JSON file contains the deployment data needed to restore the record.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.deployment import Deployment
from app.models.model import Model
from app.models.release import Release

logger = logging.getLogger(__name__)


class DeploymentSyncService:
    """Service for materializing and syncing deployments to/from filesystem."""

    def __init__(self):
        self.deployments_path = Path(settings.SYNC_DEPLOYMENTS_PATH)

    def _get_deployment_file_path(self, model_name: str, environment: str) -> Path:
        """Get the file path for a deployment."""
        # Sanitize names to be filesystem-safe
        safe_model = model_name.replace("/", "_").replace("\\", "_")
        safe_env = environment.replace("/", "_").replace("\\", "_")
        return self.deployments_path / safe_model / f"{safe_env}.json"

    async def materialize_deployment(
        self,
        deployment: Deployment,
        model_name: str,
        release_version: str,
    ) -> bool:
        """
        Write deployment to filesystem for recovery.

        Args:
            deployment: The deployment object to materialize
            model_name: Name of the model (used for directory structure)
            release_version: Version of the release

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_deployment_file_path(model_name, deployment.environment)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Build deployment data (including execution fields for recovery)
            deployment_data = {
                "release_version": release_version,
                "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
                "terminated_at": deployment.terminated_at.isoformat() if deployment.terminated_at else None,
                "status": deployment.status,
                "deployed_by": deployment.deployed_by,
                "cluster": deployment.cluster,
                "k8s_namespace": deployment.k8s_namespace,
                "endpoint_url": deployment.endpoint_url,
                "replicas": deployment.replicas,
                "metadata": deployment.meta_data or {},
                # Execution fields
                "container_id": deployment.container_id,
                "host_port": deployment.host_port,
                "deployment_type": deployment.deployment_type,
                "health_status": deployment.health_status,
                "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
                "stopped_at": deployment.stopped_at.isoformat() if deployment.stopped_at else None,
                "gpu_enabled": deployment.gpu_enabled,
            }

            # Write atomically (write to temp, then rename)
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(deployment_data, f, indent=2, default=str)
            temp_path.rename(file_path)

            logger.info(f"Materialized deployment: {model_name}/{deployment.environment}")
            return True

        except Exception as e:
            logger.error(f"Failed to materialize deployment {model_name}/{deployment.environment}: {e}")
            return False

    async def remove_deployment(self, model_name: str, environment: str) -> bool:
        """
        Remove deployment file from filesystem.

        Args:
            model_name: Name of the model
            environment: Deployment environment

        Returns:
            True if successful or file didn't exist, False on error
        """
        try:
            file_path = self._get_deployment_file_path(model_name, environment)

            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed deployment file: {model_name}/{environment}")

                # Clean up empty parent directory
                if file_path.parent.exists() and not any(file_path.parent.iterdir()):
                    file_path.parent.rmdir()

            return True

        except Exception as e:
            logger.error(f"Failed to remove deployment file {model_name}/{environment}: {e}")
            return False

    async def sync_deployments(self, db: AsyncSession) -> int:
        """
        Sync deployments from filesystem to database.

        Scans the deployments directory and creates/updates deployment records
        in the database for any found deployment files.

        Args:
            db: Database session

        Returns:
            Number of deployments synced
        """
        if not self.deployments_path.exists():
            logger.info(f"Deployments path {self.deployments_path} does not exist, creating it.")
            self.deployments_path.mkdir(parents=True, exist_ok=True)
            return 0

        logger.info(f"Scanning deployments in {self.deployments_path}")

        deployments_synced = 0
        deployments_skipped = 0

        try:
            # Iterate over model directories
            for model_dir in self.deployments_path.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name

                # Find the model in the database
                stmt = select(Model).where(Model.name == model_name)
                result = await db.execute(stmt)
                model = result.scalar_one_or_none()

                if not model:
                    logger.warning(f"Model '{model_name}' not found, skipping its deployments")
                    continue

                # Iterate over deployment files in this model directory
                for deployment_file in model_dir.glob("*.json"):
                    environment = deployment_file.stem  # filename without .json

                    try:
                        with open(deployment_file) as f:
                            data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to read deployment file {deployment_file}: {e}")
                        continue

                    release_version = data.get("release_version")
                    if not release_version:
                        logger.warning(f"No release_version in {deployment_file}, skipping")
                        continue

                    # Find the release
                    stmt = select(Release).where(
                        Release.image_id == model.id,
                        Release.version == release_version
                    )
                    result = await db.execute(stmt)
                    release = result.scalars().first()

                    if not release:
                        logger.warning(
                            f"Release '{release_version}' for model '{model_name}' not found, skipping"
                        )
                        continue

                    # Check if deployment already exists for this release and environment
                    stmt = select(Deployment).where(
                        Deployment.release_id == release.id,
                        Deployment.environment == environment
                    )
                    result = await db.execute(stmt)
                    existing = result.scalars().first()

                    if existing:
                        deployments_skipped += 1
                        continue

                    # Parse dates helper
                    def parse_iso_date(date_str):
                        if not date_str:
                            return None
                        try:
                            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        except ValueError:
                            return None

                    deployed_at = parse_iso_date(data.get("deployed_at")) or datetime.utcnow()
                    terminated_at = parse_iso_date(data.get("terminated_at"))
                    started_at = parse_iso_date(data.get("started_at"))
                    stopped_at = parse_iso_date(data.get("stopped_at"))

                    # Create the deployment (including execution fields for recovery)
                    deployment = Deployment(
                        release_id=release.id,
                        environment=environment,
                        deployed_at=deployed_at,
                        terminated_at=terminated_at,
                        status=data.get("status", "success"),
                        deployed_by=data.get("deployed_by"),
                        cluster=data.get("cluster"),
                        k8s_namespace=data.get("k8s_namespace"),
                        endpoint_url=data.get("endpoint_url"),
                        replicas=data.get("replicas"),
                        meta_data=data.get("metadata", {}),
                        # Execution fields
                        container_id=data.get("container_id"),
                        host_port=data.get("host_port"),
                        deployment_type=data.get("deployment_type", "metadata"),
                        health_status=data.get("health_status", "unknown"),
                        started_at=started_at,
                        stopped_at=stopped_at,
                        gpu_enabled=data.get("gpu_enabled", False),
                    )
                    db.add(deployment)
                    deployments_synced += 1

                    logger.info(f"Synced deployment: {model_name}/{environment}")

            await db.commit()
            logger.info(
                f"Deployment sync complete: {deployments_synced} synced, {deployments_skipped} already existed"
            )
            return deployments_synced

        except Exception as e:
            logger.error(f"Error syncing deployments: {e}")
            await db.rollback()
            return 0


# Singleton instance
deployment_sync_service = DeploymentSyncService()
