"""
Junction table for Docker Build to Artifact many-to-many relationship.
This replaces the denormalized artifact_ids ARRAY column.
"""
import uuid

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class DockerBuildArtifact(Base):
    """Junction table linking DockerBuilds to Artifacts."""

    __tablename__ = "docker_build_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    docker_build_id = Column(
        UUID(as_uuid=True),
        ForeignKey("docker_builds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    artifact_id = Column(
        UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
