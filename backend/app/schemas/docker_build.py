import re
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, field_validator

# Docker image tag regex pattern
# Repository: lowercase letters, digits, separators (., _, -, /)
# Tag: alphanumeric, underscores, periods, hyphens
DOCKER_TAG_PATTERN = re.compile(
    r'^[a-z0-9]([a-z0-9._/-]*[a-z0-9])?:[a-zA-Z0-9_][a-zA-Z0-9._-]*$'
)


def validate_docker_image_tag(v: str) -> str:
    """Validate Docker image tag format.

    Docker requires:
    - Repository name must be lowercase
    - Valid characters: a-z, 0-9, ., _, -, /
    - Tag (after :) can have mixed case
    """
    if not v or ':' not in v:
        raise ValueError('Image tag must include repository and tag (e.g., "my-image:v1.0")')

    repo, tag = v.rsplit(':', 1)

    # Check repository is lowercase
    if repo != repo.lower():
        raise ValueError(
            f'Repository name must be lowercase. Got "{repo}", use "{repo.lower()}" instead'
        )

    # Validate full pattern
    if not DOCKER_TAG_PATTERN.match(v):
        raise ValueError(
            f'Invalid Docker image tag format: "{v}". '
            'Repository must contain only lowercase letters, digits, and separators (._-/)'
        )

    return v


class DockerBuildBase(BaseModel):
    release_id: UUID
    artifact_id: Optional[UUID] = None
    artifact_ids: Optional[List[UUID]] = None
    image_tag: str
    build_type: str  # organic, azure, test, optimized
    dockerfile_content: Optional[str] = None


class DockerBuildCreate(DockerBuildBase):
    """Schema for creating a new Docker build - validates image_tag format."""

    @field_validator('image_tag')
    @classmethod
    def validate_image_tag(cls, v: str) -> str:
        return validate_docker_image_tag(v)

class DockerBuildUpdate(BaseModel):
    status: Optional[str] = None
    log_path: Optional[str] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class DockerDiskUsageComponent(BaseModel):
    """Component disk usage info."""
    count: Optional[int] = None
    size_bytes: int
    size_human: str


class DockerDiskUsageResponse(BaseModel):
    """Docker disk usage summary."""
    images: DockerDiskUsageComponent
    build_cache: DockerDiskUsageComponent
    containers: DockerDiskUsageComponent
    volumes: DockerDiskUsageComponent
    total_docker_bytes: int
    total_docker_human: str
    disk_available_bytes: int
    disk_available_human: str
    disk_total_bytes: int
    disk_total_human: str


class DockerBuildResponse(DockerBuildBase):
    id: UUID
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    log_path: Optional[str] = None
    server_type: Optional[str] = None  # Detected or explicit server type
    # GC tracking fields
    superseded_at: Optional[datetime] = None
    cleaned_at: Optional[datetime] = None
    cleanup_scheduled_at: Optional[datetime] = None
    days_until_cleanup: Optional[int] = None
    is_current: bool = False
    is_cleaned: bool = False

    class Config:
        from_attributes = True
