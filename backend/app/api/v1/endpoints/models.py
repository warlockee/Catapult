"""
API endpoints for models.

Uses ModelRepository for data access and domain exceptions for error handling.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_admin, require_operator, verify_api_key
from app.models.api_key import ApiKey
from app.repositories.model_repository import ModelRepository
from app.repositories.version_repository import VersionRepository
from app.schemas.model import ModelCreate, ModelOption, ModelResponse, ModelUpdate, ModelWithVersions
from app.schemas.pagination import PaginatedResponse
from app.schemas.version import VersionResponse
from app.services.audit_service import create_audit_log

router = APIRouter()


@router.get("", response_model=PaginatedResponse[ModelWithVersions])
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    search: str = None,
    source: str = Query(None, description="Filter by source: filesystem, manual, orphaned. Default excludes orphaned."),
    include_orphaned: bool = Query(False, description="Include orphaned models (excluded by default)"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> PaginatedResponse[ModelWithVersions]:
    """List all models with version counts (paginated)."""
    repo = ModelRepository(db)
    rows, total = await repo.list_with_version_counts(
        page=page, size=size, search=search, source=source, exclude_orphaned=not include_orphaned
    )

    items = [
        ModelWithVersions.from_model_with_count(model, version_count)
        for model, version_count in rows
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get("/options", response_model=list[ModelOption])
async def list_model_options(
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> list[ModelOption]:
    """List all models with minimal data for dropdowns/selects."""
    repo = ModelRepository(db)
    rows = await repo.list_options()
    return [ModelOption(id=row[0], name=row[1]) for row in rows]


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    model_data: ModelCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> ModelResponse:
    """
    Create a new model. Requires operator role.

    Raises:
        ModelAlreadyExistsError: If model with the same name already exists (409)
    """
    repo = ModelRepository(db)

    # This raises ModelAlreadyExistsError if name exists (handled by exception middleware)
    model = await repo.create_model(
        name=model_data.name,
        repository=model_data.repository,
        company=model_data.company,
        base_model=model_data.base_model,
        parameter_count=model_data.parameter_count,
        storage_path=model_data.storage_path,
        description=model_data.description,
        tags=model_data.tags,
        metadata=model_data.metadata,
    )

    await create_audit_log(
        db=db,
        action="create_model",
        resource_type="model",
        resource_id=model.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"name": model.name, "storage_path": model.storage_path},
        ip_address=request.client.host if request.client else None,
    )

    return ModelResponse.from_model(model)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> ModelResponse:
    """
    Get a model by ID.

    Raises:
        ModelNotFoundError: If model not found (404)
    """
    repo = ModelRepository(db)
    model = await repo.get_by_id_or_raise(model_id)

    return ModelResponse.from_model(model)


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: UUID,
    model_data: ModelUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> ModelResponse:
    """
    Update a model. Requires operator role.

    Raises:
        ModelNotFoundError: If model not found (404)
    """
    repo = ModelRepository(db)
    model = await repo.get_by_id_or_raise(model_id)

    model = await repo.update_model(
        model,
        repository=model_data.repository,
        company=model_data.company,
        base_model=model_data.base_model,
        parameter_count=model_data.parameter_count,
        storage_path=model_data.storage_path,
        description=model_data.description,
        tags=model_data.tags,
        metadata=model_data.metadata,
        requires_gpu=model_data.requires_gpu,
        server_type=model_data.server_type,
    )

    await create_audit_log(
        db=db,
        action="update_model",
        resource_type="model",
        resource_id=model.id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"name": model.name},
        ip_address=request.client.host if request.client else None,
    )

    return ModelResponse.from_model(model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_admin),
) -> None:
    """
    Delete a model. Requires admin role.

    Raises:
        ModelNotFoundError: If model not found (404)
    """
    repo = ModelRepository(db)
    model = await repo.get_by_id_or_raise(model_id)
    model_name = model.name

    await repo.delete(model)

    await create_audit_log(
        db=db,
        action="delete_model",
        resource_type="model",
        resource_id=model_id,
        api_key_name=api_key.name,
        api_key_id=api_key.id,
        details={"name": model_name},
        ip_address=request.client.host if request.client else None,
    )


@router.get("/{model_id}/versions", response_model=PaginatedResponse[VersionResponse])
async def list_model_versions(
    model_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> PaginatedResponse[VersionResponse]:
    """
    List all versions for a model (paginated).

    Raises:
        ModelNotFoundError: If model not found (404)
    """
    # Verify model exists
    model_repo = ModelRepository(db)
    await model_repo.get_by_id_or_raise(model_id)

    # Get versions
    version_repo = VersionRepository(db)
    versions, total = await version_repo.list_by_model_id(model_id, page=page, size=size)

    return PaginatedResponse.create(items=versions, total=total, page=page, size=size)
