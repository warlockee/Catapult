"""
Audit Log model for tracking all operations.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.orm import relationship

from app.core.database import Base


class AuditLog(Base):
    """Audit log for tracking operations."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Legacy string column (deprecated, kept for backward compatibility)
    api_key_name = Column(String(255), nullable=True, index=True)
    # Proper FK reference to api_keys table
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False)  # 'register_release', 'deploy', etc.
    resource_type = Column(String(50), nullable=False)  # 'image', 'release', 'deployment'
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    details_ = Column("details", JSONB, default=dict, nullable=False)
    ip_address = Column(INET, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    api_key = relationship("ApiKey", backref="audit_logs", foreign_keys=[api_key_id])
