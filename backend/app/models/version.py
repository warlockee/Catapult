"""
Version model for model versions.

NAMING CONVENTION:
    The database column is named 'image_id' for historical reasons (the table
    was originally called 'images' before being renamed to 'models').

    For semantic clarity in code, use the 'model_id' property instead of
    accessing 'image_id' directly. The property provides a getter/setter
    that maps to the underlying column.

    Example:
        # Preferred - use property
        version.model_id = some_uuid
        model_uuid = version.model_id

        # Avoid - direct column access
        version.image_id = some_uuid  # Works but less clear

    The API response schemas automatically serialize 'image_id' as 'model_id'
    for external consistency.

TERMINOLOGY:
    - Version: Any version of a model (is_release=false by default)
    - Release: A promoted/verified version (is_release=true)
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, BigInteger, DateTime, ForeignKey, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


class Version(Base):
    """
    Model Version.

    Represents a specific version of a model that can be deployed.
    Each version belongs to exactly one Model (via model_id/image_id).
    A version can be promoted to a "release" by setting is_release=True.
    """

    # Table was renamed from "releases" to "versions" in migration i9d0e1f2g3h4
    __tablename__ = "versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign key to models table
    # Note: Column named 'image_id' for backward compatibility with migrations.
    # Use the 'model_id' property in application code for clarity.
    image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to models table. Use model_id property in code."
    )

    @property
    def model_id(self):
        """Get the associated model's UUID. Preferred over accessing image_id directly."""
        return self.image_id

    @model_id.setter
    def model_id(self, value):
        """Set the associated model's UUID."""
        self.image_id = value
    version = Column(String(500), nullable=False, index=True)
    tag = Column(String(500), nullable=False)
    digest = Column(String(255), nullable=False)
    quantization = Column(String(50), nullable=True, index=True)
    size_bytes = Column(BigInteger, nullable=True)
    platform = Column(String(50), default="linux/amd64")
    architecture = Column(String(50), default="amd64")
    os = Column(String(50), default="linux")
    status = Column(String(50), default="active", index=True)
    release_notes = Column(String, nullable=True)
    meta_data = Column("metadata", JSONB, default=dict, nullable=False)
    ceph_path = Column(String(1000), nullable=True)
    mlflow_url = Column(String(1000), nullable=True)
    is_release = Column("is_release", Boolean, default=False, nullable=False, server_default='false', index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    model = relationship("Model", back_populates="versions")
    artifacts = relationship("Artifact", back_populates="version", cascade="all, delete-orphan")
    deployments = relationship("Deployment", back_populates="version", cascade="all, delete-orphan")
    docker_builds = relationship("DockerBuild", back_populates="version", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_versions_image_version_quant", "image_id", "version", "quantization", unique=True),
    )
