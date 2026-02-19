"""
Repository for Artifact entity database operations.
"""
from datetime import datetime
from typing import Optional, List, Tuple
from uuid import UUID
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.repositories.base import BaseRepository
from app.models.artifact import Artifact
from app.models.version import Version
from app.models.model import Model
from app.core.exceptions import ArtifactNotFoundError, ArtifactAlreadyExistsError, VersionNotFoundError


class ArtifactRepository(BaseRepository[Artifact]):
    """Repository for Artifact database operations."""

    model = Artifact

    async def get_by_id_with_version(self, id: UUID) -> Optional[Artifact]:
        """Get an artifact by ID with version and model eagerly loaded."""
        result = await self.db.execute(
            select(Artifact)
            .options(joinedload(Artifact.version).joinedload(Version.model))
            .where(Artifact.id == id, Artifact.deleted_at.is_(None))
        )
        return result.unique().scalar_one_or_none()

    async def get_by_id_or_raise(self, id: UUID) -> Artifact:
        """Get an artifact by ID, raising exception if not found."""
        artifact = await self.get_by_id_with_version(id)
        if not artifact:
            raise ArtifactNotFoundError(str(id))
        return artifact

    async def list_artifacts(
        self,
        release_id: Optional[UUID] = None,
        model_id: Optional[UUID] = None,
        artifact_type: Optional[str] = None,
        platform: Optional[str] = None,
        page: int = 1,
        size: int = 100,
    ) -> Tuple[List[Artifact], int]:
        """List artifacts with optional filters."""
        conditions = [Artifact.deleted_at.is_(None)]

        if release_id:
            conditions.append(Artifact.release_id == release_id)
        if model_id:
            conditions.append(Artifact.model_id == model_id)
        if artifact_type:
            conditions.append(Artifact.artifact_type == artifact_type)
        if platform:
            conditions.append(Artifact.platform == platform)

        count_stmt = select(func.count(Artifact.id)).where(and_(*conditions))
        total = (await self.db.execute(count_stmt)).scalar_one()

        skip = (page - 1) * size
        query = (
            select(Artifact)
            .options(joinedload(Artifact.version).joinedload(Version.model))
            .where(and_(*conditions))
            .order_by(desc(Artifact.created_at), Artifact.id)
            .offset(skip)
            .limit(size)
        )

        result = await self.db.execute(query)
        return list(result.unique().scalars().all()), total

    async def check_name_exists(
        self,
        name: str,
        model_id: Optional[UUID] = None,
        release_id: Optional[UUID] = None,
        exclude_id: Optional[UUID] = None,
    ) -> bool:
        """Check if an artifact with the given name exists."""
        conditions = [Artifact.name == name, Artifact.deleted_at.is_(None)]

        if model_id:
            conditions.append(Artifact.model_id == model_id)
        elif release_id:
            conditions.append(Artifact.release_id == release_id)

        if exclude_id:
            conditions.append(Artifact.id != exclude_id)

        result = await self.db.execute(
            select(func.count(Artifact.id)).where(and_(*conditions))
        )
        return result.scalar_one() > 0

    async def create_artifact(
        self,
        name: str,
        artifact_type: str,
        file_path: str,
        size_bytes: int,
        checksum: str,
        uploaded_by: str,
        release_id: Optional[UUID] = None,
        model_id: Optional[UUID] = None,
        checksum_type: str = "sha256",
        platform: str = "any",
        python_version: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Artifact:
        """Create a new artifact."""
        if release_id:
            version_result = await self.db.execute(
                select(Version).where(Version.id == release_id)
            )
            version = version_result.scalar_one_or_none()
            if not version:
                raise VersionNotFoundError(str(release_id))
            if not model_id:
                model_id = version.image_id

        if await self.check_name_exists(name, model_id, release_id):
            raise ArtifactAlreadyExistsError(name)

        artifact = Artifact(
            release_id=release_id,
            model_id=model_id,
            name=name,
            artifact_type=artifact_type,
            file_path=file_path,
            size_bytes=size_bytes,
            checksum=checksum,
            checksum_type=checksum_type,
            platform=platform,
            python_version=python_version,
            meta_data=metadata or {},
            uploaded_by=uploaded_by,
        )

        self.db.add(artifact)
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact

    async def update_artifact(
        self,
        artifact: Artifact,
        file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Artifact:
        """Update an artifact."""
        if file_path is not None:
            artifact.file_path = file_path
        if metadata is not None:
            artifact.meta_data = metadata

        return await self.update(artifact)

    async def soft_delete(self, artifact: Artifact) -> Artifact:
        """Soft delete an artifact by setting deleted_at."""
        artifact.deleted_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(artifact)
        return artifact
