"""
Repository for Version entity database operations.

Encapsulates all database queries related to versions, providing a clean
interface for services and endpoints to interact with version data.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, desc, exists, func, select
from sqlalchemy.orm import contains_eager, joinedload

from app.core.exceptions import ModelNotFoundError, VersionAlreadyExistsError, VersionNotFoundError
from app.models.deployment import Deployment
from app.models.model import Model
from app.models.version import Version
from app.repositories.base import BaseRepository


class VersionRepository(BaseRepository[Version]):
    """Repository for Version database operations."""

    model = Version

    async def get_by_id_with_model(self, id: UUID) -> Optional[Version]:
        """
        Get a version by ID with model eagerly loaded.

        Args:
            id: Version UUID

        Returns:
            Version with model loaded, or None if not found
        """
        result = await self.db.execute(
            select(Version)
            .options(joinedload(Version.model))
            .where(Version.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_or_raise(self, id: UUID) -> Version:
        """
        Get a version by ID, raising exception if not found.

        Args:
            id: Version UUID

        Returns:
            Version

        Raises:
            VersionNotFoundError: If version doesn't exist
        """
        version = await self.get_by_id_with_model(id)
        if not version:
            raise VersionNotFoundError(str(id))
        return version

    async def list_versions(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        environment: Optional[str] = None,
        is_release: Optional[bool] = None,
        status: Optional[str] = None,
        page: int = 1,
        size: int = 100,
    ) -> Tuple[List[Version], int]:
        """
        List versions with optional filters.

        Args:
            model_name: Filter by model name
            version: Filter by version (exact match)
            environment: Filter by environment (versions deployed to this environment)
            is_release: Filter by is_release flag
            status: Filter by status
            page: Page number (1-based)
            size: Page size

        Returns:
            Tuple of (list of versions, total count)
        """
        query = select(Version)

        # Build filter conditions
        conditions = []
        if version:
            conditions.append(Version.version == version)
        if is_release is not None:
            conditions.append(Version.is_release == is_release)
        if status:
            conditions.append(Version.status == status)

        # Model filter requires join
        model_join_needed = bool(model_name)
        if model_name:
            conditions.append(Model.name == model_name)

        # --- Count Query ---
        count_stmt = select(func.count(Version.id.distinct()))
        if model_join_needed:
            count_stmt = count_stmt.join(Model, Version.image_id == Model.id)

        # Use correlated EXISTS for environment filter
        if environment:
            env_exists = exists().where(
                Deployment.release_id == Version.id,
                Deployment.environment == environment
            )
            count_stmt = count_stmt.where(env_exists)

        if conditions:
            count_stmt = count_stmt.where(and_(*conditions))

        total = (await self.db.execute(count_stmt)).scalar_one()

        # --- Data Query ---
        if model_join_needed:
            query = query.join(Model, Version.image_id == Model.id).options(
                contains_eager(Version.model)
            )
        else:
            query = query.options(joinedload(Version.model))

        # Use correlated EXISTS for environment filter
        if environment:
            env_exists = exists().where(
                Deployment.release_id == Version.id,
                Deployment.environment == environment
            )
            query = query.where(env_exists)

        if conditions:
            query = query.where(and_(*conditions))

        # Pagination with deterministic ordering
        skip = (page - 1) * size
        query = query.order_by(desc(Version.created_at), Version.id).offset(skip).limit(size)

        result = await self.db.execute(query)
        return list(result.unique().scalars().all()), total

    async def get_latest_for_model(
        self,
        model_name: str,
        environment: Optional[str] = None,
    ) -> Optional[Version]:
        """
        Get the latest version for a model.

        Args:
            model_name: Model name
            environment: Optional environment filter

        Returns:
            Latest version if found, None otherwise
        """
        query = (
            select(Version)
            .join(Model)
            .options(contains_eager(Version.model))
            .where(Model.name == model_name)
        )

        if environment:
            env_exists = exists().where(
                Deployment.release_id == Version.id,
                Deployment.environment == environment
            )
            query = query.where(env_exists)

        query = query.order_by(desc(Version.created_at)).limit(1)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def check_version_exists(
        self,
        model_id: UUID,
        version: str,
        quantization: Optional[str] = None,
    ) -> bool:
        """
        Check if a version with given version/quantization exists for a model.

        Args:
            model_id: Model UUID
            version: Version string
            quantization: Optional quantization string

        Returns:
            True if exists, False otherwise
        """
        query = select(func.count(Version.id)).where(
            and_(
                Version.image_id == model_id,
                Version.version == version,
            )
        )

        if quantization is None:
            query = query.where(Version.quantization.is_(None))
        else:
            query = query.where(Version.quantization == quantization)

        result = await self.db.execute(query)
        return result.scalar_one() > 0

    async def create_version(
        self,
        model_id: UUID,
        version: str,
        tag: str,
        digest: str,
        quantization: Optional[str] = None,
        size_bytes: Optional[int] = None,
        platform: str = "linux/amd64",
        architecture: str = "amd64",
        os: str = "linux",
        release_notes: Optional[str] = None,
        metadata: Optional[dict] = None,
        ceph_path: Optional[str] = None,
        mlflow_url: Optional[str] = None,
        is_release: bool = False,
    ) -> Version:
        """
        Create a new version.

        Args:
            model_id: Model UUID
            version: Version string
            tag: Docker tag
            digest: Docker digest
            quantization: Optional quantization
            size_bytes: Optional size in bytes
            platform: Platform (default: linux/amd64)
            architecture: Architecture (default: amd64)
            os: Operating system (default: linux)
            release_notes: Optional release notes
            metadata: Optional metadata dict
            ceph_path: Optional ceph storage path
            is_release: Whether this is a production release

        Returns:
            Created version

        Raises:
            ModelNotFoundError: If model doesn't exist
            VersionAlreadyExistsError: If version with same version already exists
        """
        # Verify model exists
        model_result = await self.db.execute(
            select(Model).where(Model.id == model_id)
        )
        if not model_result.scalar_one_or_none():
            raise ModelNotFoundError(str(model_id))

        # Check for existing version
        if await self.check_version_exists(model_id, version, quantization):
            raise VersionAlreadyExistsError(version)

        new_version = Version(
            image_id=model_id,
            version=version,
            tag=tag,
            digest=digest,
            quantization=quantization,
            size_bytes=size_bytes,
            platform=platform,
            architecture=architecture,
            os=os,
            release_notes=release_notes,
            meta_data=metadata or {},
            ceph_path=ceph_path,
            mlflow_url=mlflow_url,
            is_release=is_release,
        )

        self.db.add(new_version)

        # Handle race condition
        try:
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise VersionAlreadyExistsError(version)
            raise

        await self.db.refresh(new_version)
        return new_version

    async def update_metadata(
        self,
        version_id: UUID,
        metadata: dict,
        merge: bool = True,
    ) -> Version:
        """
        Update version metadata.

        Args:
            version_id: Version UUID
            metadata: Metadata to set or merge
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            Updated version

        Raises:
            VersionNotFoundError: If version not found
        """
        version = await self.get_by_id_or_raise(version_id)

        if merge:
            current = version.meta_data or {}
            version.meta_data = {**current, **metadata}
        else:
            version.meta_data = metadata

        await self.db.commit()
        await self.db.refresh(version)
        return version

    async def update_version(
        self,
        version: Version,
        quantization: Optional[str] = None,
        release_notes: Optional[str] = None,
        metadata: Optional[dict] = None,
        status: Optional[str] = None,
        ceph_path: Optional[str] = None,
        mlflow_url: Optional[str] = None,
        is_release: Optional[bool] = None,
    ) -> Version:
        """
        Update a version.

        Args:
            version: Version to update
            quantization: Optional new quantization
            release_notes: Optional new release notes
            metadata: Optional new metadata
            status: Optional new status
            ceph_path: Optional new ceph path
            mlflow_url: Optional new MLflow URL
            is_release: Optional new is_release flag

        Returns:
            Updated version
        """
        if quantization is not None:
            version.quantization = quantization
        if release_notes is not None:
            version.release_notes = release_notes
        if metadata is not None:
            version.meta_data = metadata
        if status is not None:
            version.status = status
        if ceph_path is not None:
            version.ceph_path = ceph_path
        if mlflow_url is not None:
            version.mlflow_url = mlflow_url
        if is_release is not None:
            version.is_release = is_release

        return await self.update(version)

    async def list_by_model_id(
        self,
        model_id: UUID,
        page: int = 1,
        size: int = 100,
    ) -> Tuple[List[Version], int]:
        """
        List versions for a specific model.

        Args:
            model_id: Model UUID
            page: Page number (1-based)
            size: Page size

        Returns:
            Tuple of (list of versions, total count)
        """
        # Count
        count_stmt = select(func.count(Version.id)).where(Version.image_id == model_id)
        total = (await self.db.execute(count_stmt)).scalar_one()

        # Data
        skip = (page - 1) * size
        query = (
            select(Version)
            .where(Version.image_id == model_id)
            .order_by(desc(Version.created_at), Version.id)
            .offset(skip)
            .limit(size)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all()), total

    async def list_options_by_model(self, model_id: UUID) -> List[Tuple[UUID, str]]:
        """
        List versions for a model with minimal data for dropdowns.

        Args:
            model_id: Model UUID

        Returns:
            List of (id, version) tuples
        """
        result = await self.db.execute(
            select(Version.id, Version.version)
            .where(Version.image_id == model_id)
            .order_by(desc(Version.created_at))
        )
        return list(result.all())

    async def list_options(self) -> List[Version]:
        """
        List all versions with model data for dropdowns.

        Returns:
            List of versions with model eagerly loaded
        """
        result = await self.db.execute(
            select(Version)
            .join(Model, Version.image_id == Model.id)
            .options(contains_eager(Version.model))
            .order_by(desc(Version.created_at))
        )
        return list(result.unique().scalars().all())

    async def list_deployments_for_version(
        self,
        version_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Deployment]:
        """
        List deployments for a specific version.

        Args:
            version_id: Version UUID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of deployments
        """
        result = await self.db.execute(
            select(Deployment)
            .where(Deployment.release_id == version_id)
            .order_by(desc(Deployment.deployed_at))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# Backward compatibility alias (deprecated)
ReleaseRepository = VersionRepository
