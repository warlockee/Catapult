import hashlib
import logging
import os
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.artifact import Artifact
from app.models.model import Model
from app.models.release import Release
from app.services.deployment_sync_service import deployment_sync_service

logger = logging.getLogger(__name__)

# Advisory lock ID for filesystem sync (arbitrary unique number)
FS_SYNC_LOCK_ID = 54321

class FileSystemSyncService:
    def __init__(self):
        self.models_path = Path(settings.SYNC_MODELS_PATH)
        self.artifacts_path = Path(settings.SYNC_ARTIFACTS_PATH)

    async def sync_storage(self):
        """
        Main entry point to sync filesystem to database.

        Uses advisory locking to prevent race conditions when multiple workers
        start simultaneously. Only one process will run the sync at a time.
        """
        logger.info("Starting File System Sync...")

        async with async_session_maker() as lock_session:
            # Try to acquire advisory lock (non-blocking)
            # Returns true if lock acquired, false if another process holds it
            result = await lock_session.execute(
                text(f"SELECT pg_try_advisory_lock({FS_SYNC_LOCK_ID})")
            )
            lock_acquired = result.scalar()

            if not lock_acquired:
                logger.info("Another process is running filesystem sync, skipping.")
                return

            try:
                # Sync models with dedicated session
                async with async_session_maker() as db:
                    await self._sync_models(db)

                # Sync artifacts with dedicated session (isolated from model sync)
                async with async_session_maker() as db:
                    await self._sync_artifacts(db)

                # Sync deployments (must be after models/releases exist)
                async with async_session_maker() as db:
                    await deployment_sync_service.sync_deployments(db)

                logger.info("File System Sync Completed.")
            finally:
                # Release advisory lock
                await lock_session.execute(
                    text(f"SELECT pg_advisory_unlock({FS_SYNC_LOCK_ID})")
                )

    async def _sync_models(self, db: AsyncSession):
        """
        Sync models from filesystem storage.

        Scans the configured models directory for model directories
        and creates database entries for any new models found.
        """
        if not self.models_path.exists():
            logger.warning(f"Models sync path {self.models_path} does not exist.")
            return

        logger.info(f"Scanning models in {self.models_path} (filesystem mode)")

        models_synced = 0
        releases_created = 0

        try:
            # Iterate over directories (local/dev models)
            for entry in self.models_path.iterdir():
                if entry.is_dir():
                    model_name = entry.name

                    # Check if model exists
                    stmt = select(Model).where(Model.name == model_name)
                    result = await db.execute(stmt)
                    existing_model = result.scalar_one_or_none()

                    if not existing_model:
                        logger.info(f"Found new model on disk: {model_name}")
                        new_model = Model(
                            name=model_name,
                            storage_path=str(entry.absolute()),
                            source='filesystem',
                            description=f"Automatically imported from {entry}"
                        )
                        db.add(new_model)
                        # Flush to get ID
                        await db.flush()
                        model_id = new_model.id
                        model_name_for_release = new_model.name
                        models_synced += 1
                    else:
                        model_id = existing_model.id
                        model_name_for_release = existing_model.name

                    # Check for releases
                    stmt = select(Release).where(Release.image_id == model_id)
                    result = await db.execute(stmt)
                    if not result.scalar_one_or_none():
                        logger.info(f"Creating default release for model: {model_name_for_release}")
                        new_release = Release(
                            image_id=model_id,
                            version="v1.0.0",
                            tag="latest",
                            status="active",
                            digest=f"sha256:{model_name_for_release}",  # Placeholder
                            meta_data={"source": "fs_sync", "auto_generated": True}
                        )
                        db.add(new_release)
                        releases_created += 1

            await db.commit()
            logger.info(f"Model sync complete: {models_synced} new models, {releases_created} new releases")

        except Exception as e:
            logger.error(f"Error syncing models: {e}")
            await db.rollback()

    def _exists_on_filesystem(self, storage_path: str) -> bool:
        """Check if model exists on local filesystem."""
        return Path(storage_path).exists()

    async def _sync_artifacts(self, db: AsyncSession):
        """
        Scan {SYNC_ARTIFACTS_PATH} and create Artifact records.
        """
        if not self.artifacts_path.exists():
            logger.warning(f"Artifacts sync path {self.artifacts_path} does not exist.")
            return

        logger.info(f"Scanning artifacts in {self.artifacts_path}")

        artifacts_synced = 0
        artifacts_skipped = 0
        batch_size = 50  # Commit in batches to avoid memory issues

        try:
            # Recursive scan
            for root, dirs, files in os.walk(self.artifacts_path):
                for filename in files:
                    filepath = Path(root) / filename

                    # Skip if already exists (check by path)
                    abs_path = str(filepath.absolute())

                    stmt = select(Artifact).where(Artifact.file_path == abs_path)
                    result = await db.execute(stmt)
                    if result.scalar_one_or_none():
                        artifacts_skipped += 1
                        continue

                    logger.info(f"Found new artifact: {filename}")

                    try:
                        # Calculate checksum
                        checksum = self._calculate_checksum(filepath)
                        size = filepath.stat().st_size
                    except Exception as e:
                        logger.warning(f"Could not read artifact {filepath}: {e}")
                        continue

                    # Try to link to model from parent directory name
                    parent_dir = filepath.parent.name
                    model_id = None

                    if parent_dir != self.artifacts_path.name:
                        stmt = select(Model).where(Model.name == parent_dir)
                        result = await db.execute(stmt)
                        model = result.scalar_one_or_none()
                        if model:
                            model_id = model.id

                    artifact = Artifact(
                        name=filename,
                        artifact_type=self._infer_type(filename),
                        file_path=abs_path,
                        size_bytes=size,
                        checksum=checksum,
                        model_id=model_id,
                        meta_data={"source": "fs_sync"}
                    )
                    db.add(artifact)
                    artifacts_synced += 1

                    # Commit in batches
                    if artifacts_synced % batch_size == 0:
                        await db.commit()
                        logger.info(f"Artifact sync progress: {artifacts_synced} synced")

            # Final commit for remaining artifacts
            await db.commit()
            logger.info(f"Artifact sync complete: {artifacts_synced} new artifacts, {artifacts_skipped} already existed")

        except Exception as e:
            logger.error(f"Error syncing artifacts: {e}")
            await db.rollback()

    def _calculate_checksum(self, filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _infer_type(self, filename: str) -> str:
        if filename.endswith(".whl"):
            return "wheel"
        if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            return "tarball"
        if filename.endswith(".sif"):
            return "singularity"
        return "file"

fs_sync_service = FileSystemSyncService()
