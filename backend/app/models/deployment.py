"""
Deployment model for tracking deployments.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


class Deployment(Base):
    """Deployment record."""

    __tablename__ = "deployments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    release_id = Column(UUID(as_uuid=True), ForeignKey("versions.id", ondelete="CASCADE"), nullable=False, index=True)
    environment = Column(String(100), nullable=False, index=True)
    cluster = Column(String(255), nullable=True, index=True)
    k8s_namespace = Column(String(255), nullable=True, index=True)
    endpoint_url = Column(String(500), nullable=True)
    replicas = Column(Integer, nullable=True)
    # Legacy string column (deprecated, kept for backward compatibility)
    deployed_by = Column(String(255), nullable=True)
    # Proper FK reference to api_keys table
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True, index=True)
    deployed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    terminated_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="success", index=True)
    meta_data = Column("metadata", JSONB, default=dict, nullable=False)

    # Deployment execution fields
    container_id = Column(String(64), nullable=True, index=True)
    host_port = Column(Integer, nullable=True, index=True)
    deployment_type = Column(String(20), nullable=False, default="metadata")  # 'metadata', 'local', 'k8s'
    health_status = Column(String(20), nullable=False, default="unknown")  # 'unknown', 'healthy', 'unhealthy'
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)
    gpu_enabled = Column(Boolean, nullable=False, default=False)
    image_tag = Column(String(500), nullable=True)  # Docker image tag used for deployment

    # Relationships
    version = relationship("Version", back_populates="deployments")
    api_key = relationship("ApiKey", backref="deployments", foreign_keys=[api_key_id])

    @property
    def is_running(self) -> bool:
        """Check if the deployment is currently running."""
        return self.status == "running" and self.container_id is not None

    @property
    def is_local(self) -> bool:
        """Check if this is a local deployment."""
        return self.deployment_type == "local"
