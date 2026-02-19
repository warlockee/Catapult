"""
Base repository class with common CRUD operations.

Provides a foundation for domain-specific repositories with:
- Type-safe generic operations
- Consistent error handling
- Pagination support
- Query building helpers
"""
from typing import Any, Generic, List, Optional, Tuple, Type, TypeVar
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

T = TypeVar("T", bound=DeclarativeBase)


class BaseRepository(Generic[T]):
    """
    Generic base repository for CRUD operations.

    Subclass this and set the `model` class attribute to your SQLAlchemy model.
    Override methods as needed for domain-specific behavior.
    """

    model: Type[T]

    def __init__(self, db: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            db: SQLAlchemy async session
        """
        self.db = db

    async def get_by_id(self, id: UUID) -> Optional[T]:
        """
        Get a single record by ID.

        Args:
            id: Record UUID

        Returns:
            Record if found, None otherwise
        """
        result = await self.db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        page: int = 1,
        size: int = 100,
        order_by: Optional[Any] = None,
    ) -> Tuple[List[T], int]:
        """
        Get all records with pagination.

        Args:
            page: Page number (1-based)
            size: Page size
            order_by: Optional column to order by (default: created_at desc)

        Returns:
            Tuple of (list of records, total count)
        """
        # Count query
        count_stmt = select(func.count(self.model.id))
        total = (await self.db.execute(count_stmt)).scalar_one()

        # Data query
        query = select(self.model)

        if order_by is not None:
            query = query.order_by(order_by)
        elif hasattr(self.model, "created_at"):
            query = query.order_by(desc(self.model.created_at), self.model.id)
        else:
            query = query.order_by(self.model.id)

        skip = (page - 1) * size
        query = query.offset(skip).limit(size)

        result = await self.db.execute(query)
        return list(result.scalars().all()), total

    async def create(self, entity: T) -> T:
        """
        Create a new record.

        Args:
            entity: Entity to create

        Returns:
            Created entity with ID populated
        """
        self.db.add(entity)
        await self.db.commit()
        await self.db.refresh(entity)
        return entity

    async def update(self, entity: T) -> T:
        """
        Update an existing record.

        Args:
            entity: Entity with updated values

        Returns:
            Updated entity
        """
        await self.db.commit()
        await self.db.refresh(entity)
        return entity

    async def delete(self, entity: T) -> bool:
        """
        Delete a record.

        Args:
            entity: Entity to delete

        Returns:
            True if deleted successfully
        """
        await self.db.delete(entity)
        await self.db.commit()
        return True

    async def delete_by_id(self, id: UUID) -> bool:
        """
        Delete a record by ID.

        Args:
            id: Record UUID

        Returns:
            True if deleted, False if not found
        """
        entity = await self.get_by_id(id)
        if not entity:
            return False
        return await self.delete(entity)

    async def exists(self, id: UUID) -> bool:
        """
        Check if a record exists.

        Args:
            id: Record UUID

        Returns:
            True if exists, False otherwise
        """
        result = await self.db.execute(
            select(func.count(self.model.id)).where(self.model.id == id)
        )
        return result.scalar_one() > 0

    async def count(self) -> int:
        """
        Count total records.

        Returns:
            Total count
        """
        result = await self.db.execute(
            select(func.count(self.model.id))
        )
        return result.scalar_one()
