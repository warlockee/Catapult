"""
Pydantic schemas for API Key.
"""
from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID


class ApiKeyRoleEnum(str, Enum):
    """Roles for API key authentication."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class ApiKeyCreate(BaseModel):
    """Schema for creating an API Key."""
    name: str = Field(..., max_length=255)
    role: ApiKeyRoleEnum = Field(default=ApiKeyRoleEnum.VIEWER)
    expires_at: Optional[datetime] = None


class ApiKeyResponse(BaseModel):
    """Schema for API Key response."""
    id: UUID
    name: str
    role: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ApiKeyCreated(ApiKeyResponse):
    """Schema for API Key response after creation (includes plaintext key)."""
    key: str  # Only returned on creation
