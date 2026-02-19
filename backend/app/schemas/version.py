"""
Pydantic schemas for Version.

TERMINOLOGY:
    - Version: Any version of a model (is_release=false by default)
    - Release: A promoted/verified version (is_release=true)
"""
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class VersionBase(BaseModel):
    """Base schema for Version."""
    version: str = Field(..., max_length=500)
    tag: str = Field(..., max_length=500)
    digest: str = Field(..., max_length=255)
    quantization: Optional[str] = Field(None, max_length=50)
    size_bytes: Optional[int] = None
    platform: str = "linux/amd64"
    architecture: str = "amd64"
    os: str = "linux"
    release_notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ceph_path: Optional[str] = Field(None, max_length=1000)
    mlflow_url: Optional[str] = Field(None, max_length=1000)
    is_release: bool = False


class VersionCreate(VersionBase):
    """Schema for creating a Version."""
    # Accept both model_id (preferred) and image_id (backward compat)
    model_id: Optional[UUID] = Field(None)
    image_id: Optional[UUID] = Field(None)
    auto_build: bool = False
    build_config: Optional[Dict[str, Any]] = None

    model_config = {"populate_by_name": True, "protected_namespaces": ()}

    def get_model_id(self) -> UUID:
        """Return model_id, falling back to image_id for backward compat."""
        mid = self.model_id or self.image_id
        if not mid:
            raise ValueError("Either model_id or image_id is required")
        return mid


class VersionUpdate(BaseModel):
    """Schema for updating a Version."""
    quantization: Optional[str] = Field(None, max_length=50)
    release_notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, max_length=50)
    ceph_path: Optional[str] = Field(None, max_length=1000)
    mlflow_url: Optional[str] = Field(None, max_length=1000)
    is_release: Optional[bool] = None


class VersionResponse(VersionBase):
    """Schema for Version response."""
    id: UUID
    # Use image_id as the primary field name for backward compatibility
    # The database column is 'image_id', and existing clients expect this field name
    image_id: UUID
    status: str
    created_at: datetime
    is_release: bool
    metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias="meta_data")

    model_config = {"from_attributes": True, "populate_by_name": True, "protected_namespaces": ()}

    @classmethod
    def from_version(cls, version) -> "VersionResponse":
        """Convert a Version ORM instance to response schema."""
        return cls(
            id=version.id,
            image_id=version.image_id,
            version=version.version,
            tag=version.tag,
            digest=version.digest,
            quantization=version.quantization,
            size_bytes=version.size_bytes,
            platform=version.platform,
            architecture=version.architecture,
            os=version.os,
            release_notes=version.release_notes,
            metadata=version.meta_data or {},
            ceph_path=version.ceph_path,
            mlflow_url=version.mlflow_url,
            status=version.status,
            created_at=version.created_at,
            is_release=version.is_release,
        )


class VersionWithModel(VersionResponse):
    """Schema for Version with Model details."""
    # Note: Keep 'image_name' for API backward compatibility, but prefer 'model_name' in new code
    image_name: Optional[str] = Field(None, deprecated=True, description="Use model_name instead")
    model_name: Optional[str] = None  # Preferred field
    image_repository: Optional[str] = Field(None, deprecated=True, description="Use model_repository instead")
    model_repository: Optional[str] = None  # Preferred field

    @classmethod
    def from_version(cls, version) -> "VersionWithModel":
        """Convert a Version ORM instance with its model to response schema."""
        return cls(
            id=version.id,
            image_id=version.image_id,
            version=version.version,
            tag=version.tag,
            digest=version.digest,
            quantization=version.quantization,
            size_bytes=version.size_bytes,
            platform=version.platform,
            architecture=version.architecture,
            os=version.os,
            release_notes=version.release_notes,
            metadata=version.meta_data or {},
            ceph_path=version.ceph_path,
            mlflow_url=version.mlflow_url,
            status=version.status,
            created_at=version.created_at,
            is_release=version.is_release,
            image_name=version.model.name if version.model else None,
            model_name=version.model.name if version.model else None,
            image_repository=version.model.storage_path if version.model else None,
            model_repository=version.model.storage_path if version.model else None,
        )


class VersionOption(BaseModel):
    """Slim schema for dropdown/select options - reduces data transfer."""
    id: UUID
    version: str
    tag: str
    # Note: Keep 'image_name' for API backward compatibility
    image_name: Optional[str] = Field(None, deprecated=True, description="Use model_name instead")
    model_name: Optional[str] = None  # Preferred field

    model_config = {"from_attributes": True}


# Backward compatibility aliases (deprecated)
ReleaseBase = VersionBase
ReleaseCreate = VersionCreate
ReleaseUpdate = VersionUpdate
ReleaseResponse = VersionResponse
ReleaseWithImage = VersionWithModel
ReleaseOption = VersionOption
