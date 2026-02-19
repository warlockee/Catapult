"""
API endpoints for API key management.

Uses ApiKeyRepository for data access and domain exceptions for error handling.
"""
import secrets
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import hash_api_key, invalidate_api_key_cache, require_admin
from app.core.config import settings
from app.core.exceptions import ApiKeyNotFoundError
from app.models.api_key import ApiKey
from app.repositories.api_key_repository import ApiKeyRepository
from app.schemas.api_key import ApiKeyCreate, ApiKeyResponse, ApiKeyCreated
from app.services.audit_service import create_audit_log

router = APIRouter()


def generate_api_key() -> str:
    """Generate a secure random API key (64 characters)."""
    return secrets.token_urlsafe(48)


class CannotDeleteOwnKeyError(Exception):
    """Raised when attempting to delete own API key."""

    def __init__(self):
        super().__init__("Cannot delete your own API key")


@router.post("", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    api_key_data: ApiKeyCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_api_key: ApiKey = Depends(require_admin),
) -> ApiKeyCreated:
    """
    Create a new API key. Requires admin role.

    Returns:
        Created API key with plaintext key (only returned once)

    Raises:
        ApiKeyAlreadyExistsError: If API key with the same name already exists (409)
    """
    repo = ApiKeyRepository(db)

    # Generate new API key
    plaintext_key = generate_api_key()
    key_hash = hash_api_key(plaintext_key, settings.API_KEY_SALT)

    api_key = await repo.create_api_key(
        name=api_key_data.name,
        key_hash=key_hash,
        role=api_key_data.role.value,
        expires_at=api_key_data.expires_at,
    )

    await create_audit_log(
        db=db,
        action="create_api_key",
        resource_type="api_key",
        resource_id=api_key.id,
        api_key_name=current_api_key.name,
        api_key_id=current_api_key.id,
        details={"created_key_name": api_key.name, "role": api_key.role},
        ip_address=request.client.host if request.client else None,
    )

    return ApiKeyCreated(
        id=api_key.id,
        name=api_key.name,
        role=api_key.role,
        is_active=api_key.is_active,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        key=plaintext_key,
    )


@router.get("", response_model=List[ApiKeyResponse])
async def list_api_keys(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_admin),
) -> List[ApiKeyResponse]:
    """
    List all API keys. Requires admin role.

    Returns:
        List of API keys (without plaintext keys)
    """
    repo = ApiKeyRepository(db)
    # Convert skip/limit to page/size for consistency
    page = (skip // limit) + 1 if limit > 0 else 1
    api_keys, _ = await repo.list_api_keys(page=page, size=limit)
    return api_keys


@router.delete("/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    api_key_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_api_key: ApiKey = Depends(require_admin),
) -> None:
    """
    Delete (revoke) an API key. Requires admin role.

    Raises:
        ApiKeyNotFoundError: If API key not found (404)
        ValidationError: If trying to delete own key (400)
    """
    from app.core.exceptions import ValidationError

    repo = ApiKeyRepository(db)
    api_key = await repo.get_by_id_or_raise(api_key_id)

    # Prevent deleting own API key
    if api_key.id == current_api_key.id:
        raise ValidationError("self_delete", "Cannot delete your own API key")

    api_key_name = api_key.name

    await repo.delete(api_key)

    # Invalidate cache for the deleted key
    invalidate_api_key_cache(str(api_key_id))

    await create_audit_log(
        db=db,
        action="delete_api_key",
        resource_type="api_key",
        resource_id=api_key_id,
        api_key_name=current_api_key.name,
        api_key_id=current_api_key.id,
        details={"deleted_key_name": api_key_name},
        ip_address=request.client.host if request.client else None,
    )
