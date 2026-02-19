"""
Model metadata for ML models.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Model(Base):
    """ML Model metadata."""

    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    storage_path = Column(String(1000), nullable=False)
    repository = Column(String(500), nullable=True)
    company = Column(String(255), nullable=True, index=True)
    base_model = Column(String(100), nullable=True, index=True)
    parameter_count = Column(String(50), nullable=True, index=True)
    description = Column(Text, nullable=True)
    tags = Column(JSONB, server_default='[]', nullable=False)
    meta_data = Column("metadata", JSONB, server_default='{}', nullable=False)
    requires_gpu = Column(Boolean, nullable=False, default=True, server_default='true')
    # Server type for deployment: vllm, audio, onnx, triton, generic, custom
    # If not set, auto-detection will be attempted during Docker build
    server_type = Column(String(50), nullable=True, index=True)
    # Source of model discovery: filesystem, manual, orphaned
    source = Column(String(50), nullable=False, default='filesystem', server_default='filesystem', index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    versions = relationship("Version", back_populates="model", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="model", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_models_tags", "tags", postgresql_using="gin"),
    )
