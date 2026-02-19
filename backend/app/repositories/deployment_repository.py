"""
Repository for Deployment entity database operations.
"""
from typing import Optional, List, Tuple
from uuid import UUID
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.repositories.base import BaseRepository
from app.models.deployment import Deployment
from app.models.version import Version
from app.models.model import Model
from app.core.exceptions import DeploymentNotFoundError, VersionNotFoundError


class DeploymentRepository(BaseRepository[Deployment]):
    """Repository for Deployment database operations."""

    model = Deployment

    async def get_by_id_with_version(self, id: UUID) -> Optional[Deployment]:
        """Get a deployment by ID with version and model eagerly loaded."""
        result = await self.db.execute(
            select(Deployment)
            .options(joinedload(Deployment.version).joinedload(Version.model))
            .where(Deployment.id == id)
        )
        return result.unique().scalar_one_or_none()

    async def get_by_id_or_raise(self, id: UUID) -> Deployment:
        """Get a deployment by ID, raising exception if not found."""
        deployment = await self.get_by_id_with_version(id)
        if not deployment:
            raise DeploymentNotFoundError(str(id))
        return deployment

    async def list_deployments(
        self,
        environment: Optional[str] = None,
        status: Optional[str] = None,
        release_id: Optional[UUID] = None,
        page: int = 1,
        size: int = 100,
    ) -> Tuple[List[Deployment], int]:
        """List deployments with optional filters."""
        conditions = []

        if environment:
            conditions.append(Deployment.environment == environment)
        if status:
            conditions.append(Deployment.status == status)
        if release_id:
            conditions.append(Deployment.release_id == release_id)

        count_stmt = select(func.count(Deployment.id))
        if conditions:
            count_stmt = count_stmt.where(and_(*conditions))
        total = (await self.db.execute(count_stmt)).scalar_one()

        skip = (page - 1) * size
        query = (
            select(Deployment)
            .options(joinedload(Deployment.version).joinedload(Version.model))
            .order_by(desc(Deployment.deployed_at), Deployment.id)
            .offset(skip)
            .limit(size)
        )
        if conditions:
            query = query.where(and_(*conditions))

        result = await self.db.execute(query)
        return list(result.unique().scalars().all()), total

    async def create_deployment(
        self,
        release_id: UUID,
        environment: str,
        deployed_by: str,
        api_key_id: UUID,
        status: str = "success",
        metadata: Optional[dict] = None,
    ) -> Deployment:
        """Create a new deployment."""
        version_result = await self.db.execute(
            select(Version).where(Version.id == release_id)
        )
        version = version_result.scalar_one_or_none()
        if not version:
            raise VersionNotFoundError(str(release_id))

        deployment = Deployment(
            release_id=release_id,
            environment=environment,
            deployed_by=deployed_by,
            api_key_id=api_key_id,
            status=status,
            meta_data=metadata or {},
        )

        self.db.add(deployment)
        await self.db.commit()
        await self.db.refresh(deployment)

        # Eagerly load version with model for materialization
        result = await self.db.execute(
            select(Deployment)
            .options(joinedload(Deployment.version).joinedload(Version.model))
            .where(Deployment.id == deployment.id)
        )
        return result.unique().scalar_one()

    async def update_status(
        self,
        deployment_id: UUID,
        status: str,
    ) -> Deployment:
        """
        Update deployment status.

        Args:
            deployment_id: Deployment UUID
            status: New status value

        Returns:
            Updated deployment

        Raises:
            DeploymentNotFoundError: If deployment not found
        """
        deployment = await self.get_by_id_or_raise(deployment_id)
        deployment.status = status
        await self.db.commit()
        return deployment

    async def update_metadata(
        self,
        deployment_id: UUID,
        metadata: dict,
        merge: bool = True,
    ) -> Deployment:
        """
        Update deployment metadata.

        Args:
            deployment_id: Deployment UUID
            metadata: Metadata to set or merge
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            Updated deployment

        Raises:
            DeploymentNotFoundError: If deployment not found
        """
        deployment = await self.get_by_id_or_raise(deployment_id)

        if merge:
            current = deployment.meta_data or {}
            deployment.meta_data = {**current, **metadata}
        else:
            deployment.meta_data = metadata

        await self.db.commit()
        return deployment

    async def update_health_status(
        self,
        deployment_id: UUID,
        healthy: bool,
    ) -> Deployment:
        """
        Update deployment health status.

        Args:
            deployment_id: Deployment UUID
            healthy: Whether the deployment is healthy

        Returns:
            Updated deployment

        Raises:
            DeploymentNotFoundError: If deployment not found
        """
        deployment = await self.get_by_id_or_raise(deployment_id)
        deployment.health_status = "healthy" if healthy else "unhealthy"
        await self.db.commit()
        await self.db.refresh(deployment)
        return deployment
