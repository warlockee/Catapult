"""
Pydantic schemas for Artifact.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class ArtifactBase(BaseModel):
    """Base schema for Artifact."""
    name: str = Field(..., max_length=255)
    artifact_type: str = Field(..., max_length=50, description="Type: wheel, sdist, tarball, binary, etc.")
    file_path: str = Field(..., max_length=1000)
    size_bytes: int = Field(..., ge=0)
    checksum: str = Field(..., max_length=128, description="SHA256 hash")
    checksum_type: str = Field(default="sha256", max_length=20)
    platform: Optional[str] = Field(None, max_length=100, description="Platform: linux, darwin, win32, any")
    python_version: Optional[str] = Field(None, max_length=50, description="Python version: 3.11, 3.12, etc.")
    metadata: Optional[Dict[str, Any]] = None


class ArtifactCreate(ArtifactBase):
    """Schema for creating an Artifact."""
    release_id: Optional[UUID] = None
    model_id: Optional[UUID] = None


class ArtifactRegister(BaseModel):
    """Schema for registering an existing Artifact."""
    release_id: Optional[UUID] = None
    model_id: Optional[UUID] = None
    name: Optional[str] = Field(None, max_length=255)
    artifact_type: Optional[str] = Field(None, max_length=50, description="Type: wheel, sdist, tarball, binary, etc.")
    file_path: str = Field(..., max_length=1000)
    platform: Optional[str] = Field(None, max_length=100, description="Platform: linux, darwin, win32, any")
    python_version: Optional[str] = Field(None, max_length=50, description="Python version: 3.11, 3.12, etc.")
    metadata: Optional[Dict[str, Any]] = None


class ArtifactUpdate(BaseModel):
    """Schema for updating an Artifact."""
    metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = Field(None, max_length=1000)


class ArtifactResponse(ArtifactBase):
    """Schema for Artifact response."""
    id: UUID
    release_id: Optional[UUID] = None
    model_id: Optional[UUID] = None
    created_at: datetime
    uploaded_by: Optional[str] = None

    class Config:
        from_attributes = True

    @classmethod
    def from_artifact(cls, artifact) -> "ArtifactResponse":
        """Convert an Artifact ORM instance to response schema."""
        return cls(
            id=artifact.id,
            release_id=artifact.release_id,
            model_id=artifact.model_id,
            name=artifact.name,
            artifact_type=artifact.artifact_type,
            file_path=artifact.file_path,
            size_bytes=artifact.size_bytes,
            checksum=artifact.checksum,
            checksum_type=artifact.checksum_type,
            platform=artifact.platform,
            python_version=artifact.python_version,
            metadata=artifact.meta_data or {},
            created_at=artifact.created_at,
            uploaded_by=artifact.uploaded_by,
        )


class ArtifactWithRelease(ArtifactResponse):
    """Schema for Artifact with Release details."""
    release_version: Optional[str] = None
    image_name: Optional[str] = None

    @classmethod
    def from_artifact(cls, artifact) -> "ArtifactWithRelease":
        """Convert an Artifact ORM instance with release info to response schema."""
        return cls(
            id=artifact.id,
            release_id=artifact.release_id,
            name=artifact.name,
            artifact_type=artifact.artifact_type,
            file_path=artifact.file_path,
            size_bytes=artifact.size_bytes,
            checksum=artifact.checksum,
            checksum_type=artifact.checksum_type,
            platform=artifact.platform,
            python_version=artifact.python_version,
            metadata=artifact.meta_data or {},
            created_at=artifact.created_at,
            uploaded_by=artifact.uploaded_by,
            release_version=artifact.version.version if artifact.version else None,
            image_name=artifact.version.model.name if artifact.version and artifact.version.model else None,
        )
