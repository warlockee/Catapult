"""
Pydantic schemas for Audit Log.
"""
from datetime import datetime
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Dict, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class AuditLogResponse(BaseModel):
    """Schema for Audit Log response."""
    id: UUID
    api_key_name: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[UUID] = None
    details: Dict[str, Any] = Field(..., validation_alias="details_")
    ip_address: Optional[Union[str, IPv4Address, IPv6Address]] = None
    created_at: datetime

    class Config:
        from_attributes = True
