"""
API Key model for authentication.
"""
import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ApiKeyRole(str, Enum):
    """Roles for API key authentication."""
    ADMIN = "admin"      # Full access to all operations
    OPERATOR = "operator"  # Read + write operations (deployments, releases, etc.)
    VIEWER = "viewer"    # Read-only access


class ApiKey(Base):
    """API Key for authentication."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    prefix = Column(String(10), nullable=True, index=True)
    key_hash = Column(String(255), nullable=False, unique=True)
    # Role-based access control
    role = Column(String(20), nullable=False, default=ApiKeyRole.VIEWER.value, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    def has_role(self, required_role: ApiKeyRole) -> bool:
        """Check if this API key has the required role or higher."""
        role_hierarchy = {
            ApiKeyRole.VIEWER: 0,
            ApiKeyRole.OPERATOR: 1,
            ApiKeyRole.ADMIN: 2,
        }
        current_role = ApiKeyRole(self.role) if self.role else ApiKeyRole.VIEWER
        return role_hierarchy.get(current_role, 0) >= role_hierarchy.get(required_role, 0)
