"""
Repository for AuditLog entity database operations.
"""
from typing import List, Optional

from sqlalchemy import and_, desc, select

from app.models.audit_log import AuditLog
from app.repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for AuditLog database operations."""

    model = AuditLog

    async def list_audit_logs(
        self,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        api_key_name: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AuditLog]:
        """
        List audit logs with optional filters.

        Args:
            action: Filter by action
            resource_type: Filter by resource type
            api_key_name: Filter by API key name
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of audit logs
        """
        conditions = []

        if action:
            conditions.append(AuditLog.action == action)
        if resource_type:
            conditions.append(AuditLog.resource_type == resource_type)
        if api_key_name:
            conditions.append(AuditLog.api_key_name == api_key_name)

        query = select(AuditLog).order_by(desc(AuditLog.created_at)).offset(skip).limit(limit)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.db.execute(query)
        return list(result.scalars().all())
