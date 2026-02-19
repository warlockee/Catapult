"""
Repository for DockerBuild entity database operations.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, desc, func, select

from app.core.exceptions import DockerBuildNotFoundError
from app.models.docker_build import DockerBuild
from app.repositories.base import BaseRepository


class DockerBuildRepository(BaseRepository[DockerBuild]):
    """Repository for DockerBuild database operations."""

    model = DockerBuild

    async def get_by_id_or_raise(self, id: UUID) -> DockerBuild:
        """Get a docker build by ID, raising exception if not found."""
        build = await self.get_by_id(id)
        if not build:
            raise DockerBuildNotFoundError(str(id))
        return build

    async def list_builds(
        self,
        release_id: Optional[UUID] = None,
        status: Optional[str] = None,
        build_type: Optional[str] = None,
        page: int = 1,
        size: int = 100,
    ) -> Tuple[List[DockerBuild], int]:
        """List docker builds with optional filters."""
        conditions = []

        if release_id:
            conditions.append(DockerBuild.release_id == release_id)
        if status:
            conditions.append(DockerBuild.status == status)
        if build_type:
            conditions.append(DockerBuild.build_type == build_type)

        count_stmt = select(func.count(DockerBuild.id))
        if conditions:
            count_stmt = count_stmt.where(and_(*conditions))
        total = (await self.db.execute(count_stmt)).scalar_one()

        skip = (page - 1) * size
        query = (
            select(DockerBuild)
            .order_by(desc(DockerBuild.created_at), DockerBuild.id)
            .offset(skip)
            .limit(size)
        )
        if conditions:
            query = query.where(and_(*conditions))

        result = await self.db.execute(query)
        return list(result.scalars().all()), total
