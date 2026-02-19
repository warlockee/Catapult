import uuid
from datetime import datetime, timedelta

from sqlalchemy import ARRAY, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.config import settings
from app.core.database import Base


class DockerBuild(Base):
    __tablename__ = "docker_builds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    release_id = Column(UUID(as_uuid=True), ForeignKey("versions.id"))
    # Legacy single artifact reference (deprecated, kept for backward compatibility)
    artifact_id = Column(UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=True)
    # Legacy array column (deprecated, data migrated to docker_build_artifacts junction table)
    artifact_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    status = Column(String, nullable=False)  # pending, building, success, failed
    image_tag = Column(String, nullable=False)
    build_type = Column(String, nullable=False)  # organic, azure
    log_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    dockerfile_content = Column(Text, nullable=True)
    server_type = Column(String(50), nullable=True)  # Detected or explicit server type
    celery_task_id = Column(String(50), nullable=True)
    # GC tracking columns
    superseded_at = Column(DateTime(timezone=True), nullable=True)  # NULL = current, Set = cleanup scheduled
    cleaned_at = Column(DateTime(timezone=True), nullable=True)  # NULL = image exists, Set = removed by GC

    # Relationships
    version = relationship("Version", back_populates="docker_builds")
    # Legacy single artifact relationship
    artifact = relationship("Artifact", back_populates="docker_builds", foreign_keys=[artifact_id])
    # Proper many-to-many relationship via junction table
    build_artifacts = relationship(
        "DockerBuildArtifact",
        backref="docker_build",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def get_all_artifact_ids(self):
        """
        Get all artifact IDs associated with this build.
        Handles both legacy columns and new junction table.
        """
        artifact_ids_set = set()

        # From junction table (preferred)
        if self.build_artifacts:
            for ba in self.build_artifacts:
                artifact_ids_set.add(ba.artifact_id)

        # From legacy artifact_ids array
        if self.artifact_ids:
            for aid in self.artifact_ids:
                artifact_ids_set.add(aid)

        # From legacy single artifact_id
        if self.artifact_id:
            artifact_ids_set.add(self.artifact_id)

        return list(artifact_ids_set)

    @property
    def is_current(self) -> bool:
        """True if this is the current/active build (not superseded)."""
        return self.superseded_at is None and self.status == "success"

    @property
    def is_cleaned(self) -> bool:
        """True if the Docker image has been removed by GC."""
        return self.cleaned_at is not None

    @property
    def cleanup_scheduled_at(self) -> datetime | None:
        """When this build's image is scheduled for cleanup, or None if current."""
        if self.superseded_at is None:
            return None
        retention_days = getattr(settings, 'DOCKER_IMAGE_RETENTION_DAYS', 7)
        return self.superseded_at + timedelta(days=retention_days)

    @property
    def days_until_cleanup(self) -> int | None:
        """
        Days until cleanup, or None if current.
        Negative if overdue for cleanup.
        """
        scheduled = self.cleanup_scheduled_at
        if scheduled is None:
            return None
        now = datetime.now(scheduled.tzinfo) if scheduled.tzinfo else datetime.utcnow()
        delta = scheduled - now
        return delta.days
