"""
Repository for ApiKey entity database operations.
"""
from typing import Optional, List, Tuple
from uuid import UUID
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository
from app.models.api_key import ApiKey
from app.core.exceptions import ApiKeyNotFoundError, ApiKeyAlreadyExistsError


class ApiKeyRepository(BaseRepository[ApiKey]):
    """Repository for ApiKey database operations."""

    model = ApiKey

    async def get_by_id_or_raise(self, id: UUID) -> ApiKey:
        """Get an API key by ID, raising exception if not found."""
        api_key = await self.get_by_id(id)
        if not api_key:
            raise ApiKeyNotFoundError(str(id))
        return api_key

    async def get_by_name(self, name: str) -> Optional[ApiKey]:
        """Get an API key by name."""
        result = await self.db.execute(
            select(ApiKey).where(ApiKey.name == name)
        )
        return result.scalar_one_or_none()

    async def check_name_exists(self, name: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if an API key with the given name exists."""
        query = select(func.count(ApiKey.id)).where(ApiKey.name == name)
        if exclude_id:
            query = query.where(ApiKey.id != exclude_id)
        result = await self.db.execute(query)
        return result.scalar_one() > 0

    async def list_api_keys(
        self,
        page: int = 1,
        size: int = 100,
    ) -> Tuple[List[ApiKey], int]:
        """List API keys with pagination."""
        count_stmt = select(func.count(ApiKey.id))
        total = (await self.db.execute(count_stmt)).scalar_one()

        skip = (page - 1) * size
        query = (
            select(ApiKey)
            .order_by(desc(ApiKey.created_at), ApiKey.id)
            .offset(skip)
            .limit(size)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all()), total

    async def create_api_key(
        self,
        name: str,
        key_hash: str,
        role: str,
        expires_at=None,
    ) -> ApiKey:
        """Create a new API key."""
        if await self.check_name_exists(name):
            raise ApiKeyAlreadyExistsError(name)

        api_key = ApiKey(
            name=name,
            key_hash=key_hash,
            role=role,
            expires_at=expires_at,
        )

        self.db.add(api_key)
        await self.db.commit()
        await self.db.refresh(api_key)
        return api_key
