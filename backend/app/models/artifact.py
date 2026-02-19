"""
Artifact model for prebuilt wheels and other build artifacts.
"""
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Artifact(Base):
    """Build Artifact (wheels, tarballs, etc.)."""

    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"), nullable=True, index=True)
    release_id = Column(UUID(as_uuid=True), ForeignKey("versions.id", ondelete="CASCADE"), nullable=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    artifact_type = Column(String(50), nullable=False, index=True)  # wheel, sdist, tarball, binary, etc.
    file_path = Column(String(1000), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    checksum = Column(String(128), nullable=False)  # SHA256 hash
    checksum_type = Column(String(20), default="sha256")
    platform = Column(String(100), nullable=True)  # linux, darwin, win32, any
    python_version = Column(String(50), nullable=True)  # 3.11, 3.12, etc.
    meta_data = Column("metadata", JSONB, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    uploaded_by = Column(String(100), nullable=True)
    deleted_at = Column(DateTime, nullable=True, index=True)

    # Relationships
    version = relationship("Version", back_populates="artifacts")
    model = relationship("Model", back_populates="artifacts")
    docker_builds = relationship("DockerBuild", back_populates="artifact", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_artifacts_type", "artifact_type"),
    )
