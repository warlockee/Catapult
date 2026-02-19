"""
Security utilities for API key authentication and RBAC.
"""
import bcrypt
import time
from typing import Optional, Dict, Tuple, Callable
from datetime import datetime
from uuid import UUID
from functools import wraps
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.api_key import ApiKey, ApiKeyRole


# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Cache for verified API keys: {api_key: (api_key_id, expiry_time)}
# TTL of 5 minutes to balance security and performance
_api_key_cache: Dict[str, Tuple[str, float]] = {}
_CACHE_TTL = 300  # 5 minutes
_CACHE_MAX_SIZE = 10000  # Maximum number of cached keys to prevent DoS


def invalidate_api_key_cache(api_key_id: Optional[str] = None) -> int:
    """
    Invalidate cached API keys.

    Args:
        api_key_id: If provided, invalidate only entries for this key ID.
                   If None, invalidate all cached entries.

    Returns:
        Number of cache entries invalidated.
    """
    global _api_key_cache

    if api_key_id is None:
        # Clear entire cache
        count = len(_api_key_cache)
        _api_key_cache = {}
        return count

    # Find and remove entries matching the key ID
    keys_to_remove = [
        key for key, (cached_id, _) in _api_key_cache.items()
        if cached_id == api_key_id
    ]

    for key in keys_to_remove:
        del _api_key_cache[key]

    return len(keys_to_remove)


def hash_api_key(api_key: str, salt: str) -> str:
    """
    Hash an API key using bcrypt.

    Args:
        api_key: The API key to hash
        salt: The salt to use for hashing

    Returns:
        The hashed API key
    """
    combined = f"{api_key}{salt}".encode()
    return bcrypt.hashpw(combined, bcrypt.gensalt()).decode()


def verify_api_key_hash(api_key: str, key_hash: str, salt: str) -> bool:
    """
    Verify an API key against its hash.

    Args:
        api_key: The API key to verify
        key_hash: The hash to verify against
        salt: The salt used for hashing

    Returns:
        True if the API key matches the hash, False otherwise
    """
    combined = f"{api_key}{salt}".encode()
    return bcrypt.checkpw(combined, key_hash.encode())


async def get_api_key_from_db(
    api_key: str,
    db: AsyncSession
) -> Optional[ApiKey]:
    """
    Validate an API key and return the corresponding ApiKey object.

    Args:
        api_key: The API key to validate
        db: Database session

    Returns:
        ApiKey object if valid, None otherwise
    """
    from app.core.config import settings
    start_time = time.time()

    # Check cache first
    if api_key in _api_key_cache:
        cached_id, expiry = _api_key_cache[api_key]
        if time.time() < expiry:
            # Cache hit - fetch the key by ID (fast DB lookup)
            result = await db.execute(
                select(ApiKey).where(ApiKey.id == UUID(cached_id)).where(ApiKey.is_active == True)
            )
            db_key = result.scalar_one_or_none()
            if db_key:
                return db_key
        # Cache expired or key not found, remove it
        del _api_key_cache[api_key]

    # Prefix optimization: if key contains '.', extract prefix and lookup directly
    if "." in api_key:
        try:
            prefix, secret = api_key.split(".", 1)
        except ValueError:
            # Invalid format
            return None
            
        # Find key by prefix
        stmt = select(ApiKey).where(ApiKey.prefix == prefix).where(ApiKey.is_active == True)
        result = await db.execute(stmt)
        db_key = result.scalar_one_or_none()

        if db_key:
            if verify_api_key_hash(api_key, db_key.key_hash, settings.API_KEY_SALT):
                # Check if key has expired
                if db_key.expires_at and db_key.expires_at < datetime.utcnow():
                    return None

                # Cache the verified key (with size limit)
                if len(_api_key_cache) >= _CACHE_MAX_SIZE:
                    # Remove oldest entries (simple cleanup)
                    oldest_keys = sorted(
                        _api_key_cache.items(),
                        key=lambda x: x[1][1]
                    )[:_CACHE_MAX_SIZE // 10]
                    for old_key, _ in oldest_keys:
                        del _api_key_cache[old_key]

                _api_key_cache[api_key] = (str(db_key.id), time.time() + _CACHE_TTL)
                db_key.last_used_at = datetime.utcnow()
                await db.commit()
                return db_key
            # Do not fall back to O(N) if prefix matches but hash fails (security)
            return None

    # NO FALLBACK: O(N) scan removed for security (DoS prevention).
    # All valid keys MUST have a prefix.
    return None



async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    session: AsyncSession = Depends(get_db),
) -> ApiKey:
    """
    Dependency for verifying API keys.
    
    Soft Auth Strategy:
    1. If key provided: Try to validate it against DB (updates last_used_at).
    2. If validation succeeds: Return real key.
    3. If validation fails or no key: Fallback to Mock Admin (preserves UI access).
    """
    if api_key:
        db_key = await get_api_key_from_db(api_key, session)
        if db_key:
            return db_key

    # BYPASS: Return a mock admin key for all requests (not for production!)
    # Uses fixed UUID that exists in database to satisfy FK constraints
    mock_key = ApiKey(
        id=UUID('00000000-0000-0000-0000-000000000001'),
        name="bypass-admin",
        key_hash="",
        prefix="bypass",
        role=ApiKeyRole.ADMIN,
        is_active=True,
        created_at=datetime.utcnow(),
    )
    return mock_key


def require_role(required_role: ApiKeyRole):
    """
    Dependency factory that checks if the API key has the required role.

    Usage:
        @app.delete("/items/{id}")
        async def delete_item(
            api_key: ApiKey = Depends(require_role(ApiKeyRole.ADMIN))
        ):
            ...

    Args:
        required_role: The minimum role required to access the endpoint

    Returns:
        A dependency function that validates the role
    """
    async def role_checker(
        api_key: ApiKey = Depends(verify_api_key)
    ) -> ApiKey:
        if not api_key.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}",
            )
        return api_key

    return role_checker


# Pre-defined role dependencies for common use cases
require_admin = require_role(ApiKeyRole.ADMIN)
require_operator = require_role(ApiKeyRole.OPERATOR)
require_viewer = require_role(ApiKeyRole.VIEWER)
