"""
API endpoints for versions.

Uses VersionRepository for data access and domain exceptions for error handling.

TERMINOLOGY:
    - Version: Any version of a model (is_release=false by default)
    - Release: A promoted/verified version (is_release=true)
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import InvalidConfigurationError
from app.core.security import require_operator, verify_api_key
from app.models.api_key import ApiKey
from app.repositories.model_repository import ModelRepository
from app.repositories.version_repository import VersionRepository
from app.schemas.deployment import DeploymentResponse
from app.schemas.docker_build import DockerBuildCreate
from app.schemas.mlflow import MlflowMetadataResponse
from app.schemas.pagination import PaginatedResponse
from app.schemas.version import VersionCreate, VersionOption, VersionResponse, VersionUpdate, VersionWithModel
from app.services.audit_service import create_audit_log
from app.services.docker_service import docker_service
from app.services.task_dispatcher import task_dispatcher

router = APIRouter()


@router.post("", response_model=VersionResponse, status_code=status.HTTP_201_CREATED)
async def create_version(
    version_data: VersionCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> VersionResponse:
    """
    Create a new version. Requires operator role.

    Raises:
        ModelNotFoundError: If model doesn't exist (404)
        VersionAlreadyExistsError: If version with same version already exists (409)
    """
    repo = VersionRepository(db)

    # Create version - raises domain exceptions on validation failures
    version = await repo.create_version(
        model_id=version_data.get_model_id(),
        version=version_data.version,
        tag=version_data.tag,
        digest=version_data.digest,
        quantization=version_data.quantization,
        size_bytes=version_data.size_bytes,
        platform=version_data.platform,
        architecture=version_data.architecture,
        os=version_data.os,
        release_notes=version_data.release_notes,
        metadata=version_data.metadata,
        ceph_path=version_data.ceph_path,
        mlflow_url=version_data.mlflow_url,
        is_release=version_data.is_release,
    )

    await create_audit_log(
        db=db,
        action="create_version",
        resource_type="version",
        resource_id=version.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={
            "version": version.version,
            "tag": version.tag,
            "digest": version.digest,
        },
        ip_address=request.client.host if request.client else None,
    )

    # Trigger MLflow metadata sync if mlflow_url is set
    if version.mlflow_url:
        task_dispatcher.dispatch_mlflow_sync(version.id)

    # Trigger auto-build if requested
    if version_data.auto_build:
        build_config = version_data.build_config or {}

        # Fetch model name for the image tag
        model_repo = ModelRepository(db)
        model = await model_repo.get_by_id(version.image_id)
        model_name = model.name if model else "unknown"

        build_data = DockerBuildCreate(
            release_id=version.id,
            image_tag=f"{model_name}:{version.version}",
            build_type=build_config.get("build_type", "organic"),
            nocache=build_config.get("nocache", False),
            dockerfile_content=build_config.get("dockerfile_content"),
        )

        build = await docker_service.create_build(
            db=db,
            release_id=build_data.release_id,
            image_tag=build_data.image_tag,
            build_type=build_data.build_type,
            artifact_id=build_data.artifact_id,
            dockerfile_content=build_data.dockerfile_content,
        )

        # Trigger async build
        task_dispatcher.dispatch_docker_build(build.id)

    return VersionResponse.from_version(version)


@router.get("", response_model=PaginatedResponse[VersionWithModel])
async def list_versions(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    version: Optional[str] = Query(None, description="Filter by version"),
    environment: Optional[str] = Query(None, description="Filter by deployment environment"),
    is_release: Optional[bool] = Query(None, description="Filter by is_release flag"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> PaginatedResponse[VersionWithModel]:
    """List versions with optional filters (paginated)."""
    repo = VersionRepository(db)
    versions, total = await repo.list_versions(
        model_name=model_name,
        version=version,
        environment=environment,
        is_release=is_release,
        status=status,
        page=page,
        size=size,
    )

    items = [VersionWithModel.from_version(v) for v in versions]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get("/options", response_model=list[VersionOption])
async def list_version_options(
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> list[VersionOption]:
    """List all versions with minimal data for dropdowns/selects."""
    repo = VersionRepository(db)
    rows = await repo.list_options()

    return [
        VersionOption(
            id=row.id,
            version=row.version,
            tag=row.tag,
            image_name=row.model.name,
            model_name=row.model.name,
        )
        for row in rows
    ]


@router.get("/latest", response_model=VersionWithModel)
async def get_latest_version(
    model_name: str = Query(..., description="Model name"),
    environment: Optional[str] = Query(None, description="Filter by deployment environment"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> VersionWithModel:
    """
    Get the latest version for a model.

    Raises:
        VersionNotFoundError: If no version found (404)
    """
    from app.core.exceptions import VersionNotFoundError

    repo = VersionRepository(db)
    version = await repo.get_latest_for_model(model_name, environment)

    if not version:
        raise VersionNotFoundError(f"No version found for model '{model_name}'")

    return VersionWithModel.from_version(version)


@router.get("/{version_id}", response_model=VersionWithModel)
async def get_version(
    version_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> VersionWithModel:
    """
    Get a version by ID.

    Raises:
        VersionNotFoundError: If version not found (404)
    """
    repo = VersionRepository(db)
    version = await repo.get_by_id_or_raise(version_id)

    return VersionWithModel.from_version(version)


@router.put("/{version_id}", response_model=VersionResponse)
async def update_version(
    version_id: UUID,
    version_data: VersionUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> VersionResponse:
    """
    Update a version. Requires operator role.

    Raises:
        VersionNotFoundError: If version not found (404)
    """
    repo = VersionRepository(db)
    version = await repo.get_by_id_or_raise(version_id)

    old_mlflow_url = version.mlflow_url

    version = await repo.update_version(
        version,
        quantization=version_data.quantization,
        release_notes=version_data.release_notes,
        metadata=version_data.metadata,
        status=version_data.status,
        ceph_path=version_data.ceph_path,
        mlflow_url=version_data.mlflow_url,
        is_release=version_data.is_release,
    )

    await create_audit_log(
        db=db,
        action="update_version",
        resource_type="version",
        resource_id=version.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"version": version.version},
        ip_address=request.client.host if request.client else None,
    )

    # Trigger MLflow metadata sync if mlflow_url was set or changed
    if version.mlflow_url and version.mlflow_url != old_mlflow_url:
        task_dispatcher.dispatch_mlflow_sync(version.id)

    return VersionResponse.from_version(version)


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_version(
    version_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> None:
    """
    Delete a version. Requires operator role.

    Raises:
        VersionNotFoundError: If version not found (404)
    """
    repo = VersionRepository(db)
    version = await repo.get_by_id_or_raise(version_id)
    version_str = version.version

    await repo.delete(version)

    await create_audit_log(
        db=db,
        action="delete_version",
        resource_type="version",
        resource_id=version_id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"version": version_str},
        ip_address=request.client.host if request.client else None,
    )


@router.get("/{version_id}/deployments", response_model=List[DeploymentResponse])
async def list_version_deployments(
    version_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> List[DeploymentResponse]:
    """
    Get deployment history for a version.

    Raises:
        VersionNotFoundError: If version not found (404)
    """
    repo = VersionRepository(db)

    # Verify version exists
    await repo.get_by_id_or_raise(version_id)

    # Get deployments
    deployments = await repo.list_deployments_for_version(version_id, skip, limit)

    return deployments


@router.get("/{version_id}/mlflow-metadata", response_model=MlflowMetadataResponse)
async def get_mlflow_metadata(
    version_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> MlflowMetadataResponse:
    """
    Get cached MLflow metadata for a version.

    Returns the metadata stored under the version's metadata.mlflow key.
    Use POST .../mlflow-metadata/sync to fetch/refresh from MLflow.

    Raises:
        VersionNotFoundError: If version not found (404)
        InvalidConfigurationError: If no mlflow_url or no cached metadata (400)
    """
    repo = VersionRepository(db)
    version = await repo.get_by_id_or_raise(version_id)

    if not version.mlflow_url:
        raise InvalidConfigurationError(
            "mlflow_url", "No MLflow URL configured for this version"
        )

    mlflow_data = (version.meta_data or {}).get("mlflow")
    if not mlflow_data:
        raise InvalidConfigurationError(
            "mlflow_metadata",
            "No cached MLflow metadata. Use POST .../mlflow-metadata/sync to fetch.",
        )

    return MlflowMetadataResponse(**mlflow_data)


@router.post("/{version_id}/mlflow-metadata/sync", response_model=MlflowMetadataResponse)
async def sync_mlflow_metadata(
    version_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> MlflowMetadataResponse:
    """
    Fetch/refresh MLflow metadata from the MLflow server and store it.

    Parses the version's mlflow_url, calls the MLflow REST API,
    and stores the normalized metadata under metadata.mlflow.

    Raises:
        VersionNotFoundError: If version not found (404)
        InvalidConfigurationError: If no mlflow_url or URL is invalid (400)
        ServiceUnavailableError: If MLflow server is unreachable (503)
    """
    from datetime import datetime, timezone

    from app.services.mlflow_service import mlflow_service

    repo = VersionRepository(db)
    version = await repo.get_by_id_or_raise(version_id)

    if not version.mlflow_url:
        raise InvalidConfigurationError(
            "mlflow_url", "No MLflow URL configured for this version"
        )

    try:
        mlflow_data = await mlflow_service.fetch_metadata(version.mlflow_url)
    except ValueError as e:
        raise InvalidConfigurationError("mlflow_url", str(e))

    mlflow_data["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # Merge into existing metadata (preserve other keys)
    await repo.update_metadata(version_id, {"mlflow": mlflow_data})

    return MlflowMetadataResponse(**mlflow_data)
