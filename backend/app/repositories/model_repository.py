"""
Repository for Model entity database operations.

Encapsulates all database queries related to models, providing a clean
interface for services and endpoints to interact with model data.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import desc, func, select

from app.core.exceptions import ModelAlreadyExistsError, ModelNotFoundError
from app.models.model import Model
from app.models.version import Version
from app.repositories.base import BaseRepository


class ModelRepository(BaseRepository[Model]):
    """Repository for Model database operations."""

    model = Model

    async def get_by_name(self, name: str) -> Optional[Model]:
        """
        Get a model by name.

        Args:
            name: Model name

        Returns:
            Model if found, None otherwise
        """
        result = await self.db.execute(
            select(Model).where(Model.name == name)
        )
        return result.scalar_one_or_none()

    async def get_by_name_or_raise(self, name: str) -> Model:
        """
        Get a model by name, raising exception if not found.

        Args:
            name: Model name

        Returns:
            Model

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.get_by_name(name)
        if not model:
            raise ModelNotFoundError(name)
        return model

    async def get_by_id_or_raise(self, id: UUID) -> Model:
        """
        Get a model by ID, raising exception if not found.

        Args:
            id: Model UUID

        Returns:
            Model

        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        model = await self.get_by_id(id)
        if not model:
            raise ModelNotFoundError(str(id))
        return model

    async def list_with_version_counts(
        self,
        page: int = 1,
        size: int = 10,
        search: Optional[str] = None,
        source: Optional[str] = None,
        exclude_orphaned: bool = True,
    ) -> Tuple[List[Tuple[Model, int]], int]:
        """
        List models with their version counts.

        Args:
            page: Page number (1-based)
            size: Page size
            search: Optional search string for name/repository
            source: Optional filter by source (filesystem, manual, orphaned)
            exclude_orphaned: Exclude orphaned models (default True)

        Returns:
            Tuple of (list of (model, version_count) tuples, total count)
        """
        # Build filter conditions
        conditions = []
        if search:
            conditions.append(
                (Model.name.ilike(f"%{search}%")) | (Model.repository.ilike(f"%{search}%"))
            )
        if source:
            conditions.append(Model.source == source)
        elif exclude_orphaned:
            # No explicit source filter, but exclude orphaned by default
            conditions.append(Model.source != 'orphaned')

        # Count query
        count_query = select(func.count(Model.id))
        if conditions:
            count_query = count_query.where(*conditions)
        total = (await self.db.execute(count_query)).scalar_one()

        # Data query with version count
        query = (
            select(Model, func.count(Version.id).label("version_count"))
            .outerjoin(Version, Model.id == Version.image_id)
            .group_by(Model.id)
        )

        if conditions:
            query = query.where(*conditions)

        # Pagination with deterministic ordering
        skip = (page - 1) * size
        query = query.order_by(desc(Model.created_at), Model.id).offset(skip).limit(size)

        result = await self.db.execute(query)
        return list(result.all()), total

    async def list_options(self) -> List[Tuple[UUID, str]]:
        """
        List all models with minimal data for dropdowns.

        Returns:
            List of (id, name) tuples
        """
        result = await self.db.execute(
            select(Model.id, Model.name).order_by(Model.name)
        )
        return list(result.all())

    async def create_model(
        self,
        name: str,
        repository: Optional[str] = None,
        company: Optional[str] = None,
        base_model: Optional[str] = None,
        parameter_count: Optional[str] = None,
        storage_path: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Model:
        """
        Create a new model.

        Args:
            name: Model name (must be unique)
            repository: Optional repository URL
            company: Optional company name
            base_model: Optional base model name
            parameter_count: Optional parameter count
            storage_path: Optional storage path
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata dict

        Returns:
            Created model

        Raises:
            ModelAlreadyExistsError: If model with name already exists
        """
        # Check for existing model with same name
        existing = await self.get_by_name(name)
        if existing:
            raise ModelAlreadyExistsError(name)

        model = Model(
            name=name,
            repository=repository,
            company=company,
            base_model=base_model,
            parameter_count=parameter_count,
            storage_path=storage_path,
            description=description,
            tags=tags or [],
            meta_data=metadata or {},
        )

        return await self.create(model)

    async def update_model(
        self,
        model: Model,
        repository: Optional[str] = None,
        company: Optional[str] = None,
        base_model: Optional[str] = None,
        parameter_count: Optional[str] = None,
        storage_path: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        requires_gpu: Optional[bool] = None,
        server_type: Optional[str] = None,
    ) -> Model:
        """
        Update an existing model.

        Args:
            model: Model to update
            repository: Optional new repository URL
            company: Optional new company name
            base_model: Optional new base model name
            parameter_count: Optional new parameter count
            storage_path: Optional new storage path
            description: Optional new description
            tags: Optional new tags list
            metadata: Optional new metadata dict
            requires_gpu: Optional GPU requirement flag
            server_type: Optional server type (vllm, audio, onnx, etc.)

        Returns:
            Updated model
        """
        if repository is not None:
            model.repository = repository
        if company is not None:
            model.company = company
        if base_model is not None:
            model.base_model = base_model
        if parameter_count is not None:
            model.parameter_count = parameter_count
        if storage_path is not None:
            model.storage_path = storage_path
        if description is not None:
            model.description = description
        if tags is not None:
            model.tags = tags
        if metadata is not None:
            model.meta_data = metadata
        if requires_gpu is not None:
            model.requires_gpu = requires_gpu
        if server_type is not None:
            model.server_type = server_type

        return await self.update(model)

    async def name_exists(self, name: str, exclude_id: Optional[UUID] = None) -> bool:
        """
        Check if a model name already exists.

        Args:
            name: Model name to check
            exclude_id: Optional ID to exclude from check (for updates)

        Returns:
            True if name exists, False otherwise
        """
        query = select(func.count(Model.id)).where(Model.name == name)
        if exclude_id:
            query = query.where(Model.id != exclude_id)
        result = await self.db.execute(query)
        return result.scalar_one() > 0
