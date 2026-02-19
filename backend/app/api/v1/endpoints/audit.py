"""
API endpoints for audit logs.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_api_key
from app.models.api_key import ApiKey
from app.repositories.audit_log_repository import AuditLogRepository
from app.schemas.audit_log import AuditLogResponse

router = APIRouter()


@router.get("", response_model=List[AuditLogResponse])
async def list_audit_logs(
    action: Optional[str] = Query(None, description="Filter by action"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    api_key_name: Optional[str] = Query(None, description="Filter by API key name"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> List[AuditLogResponse]:
    """
    List audit logs with optional filters.

    Args:
        action: Filter by action
        resource_type: Filter by resource type
        api_key_name: Filter by API key name
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        api_key: Verified API key

    Returns:
        List of audit logs
    """
    repo = AuditLogRepository(db)
    audit_logs = await repo.list_audit_logs(
        action=action,
        resource_type=resource_type,
        api_key_name=api_key_name,
        skip=skip,
        limit=limit,
    )

    return audit_logs
