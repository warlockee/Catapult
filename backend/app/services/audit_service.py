"""
Service for creating audit logs.
"""
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog


async def create_audit_log(
    db: AsyncSession,
    action: str,
    resource_type: str,
    resource_id: Optional[UUID] = None,
    api_key_name: Optional[str] = None,
    api_key_id: Optional[UUID] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """
    Create an audit log entry.

    Args:
        db: Database session
        action: Action performed (e.g., 'create_image', 'delete_release')
        resource_type: Type of resource (e.g., 'image', 'release', 'deployment')
        resource_id: ID of the resource affected
        api_key_name: Name of the API key used (deprecated, kept for backward compatibility)
        api_key_id: UUID of the API key used (preferred)
        details: Additional details about the action
        ip_address: IP address of the requester

    Returns:
        Created audit log entry
    """
    audit_log = AuditLog(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        api_key_name=api_key_name,
        api_key_id=api_key_id,
        details_=details or {},
        ip_address=ip_address,
    )

    db.add(audit_log)
    await db.commit()
    await db.refresh(audit_log)

    return audit_log
