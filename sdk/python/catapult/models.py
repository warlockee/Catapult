"""Data models for SDK."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class Model(BaseModel):
    """Model metadata."""
    id: str
    name: str
    storage_path: str
    repository: Optional[str] = None
    company: Optional[str] = None
    base_model: Optional[str] = None
    parameter_count: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime


class Version(BaseModel):
    """Version model (a version of a model, can be promoted to a release)."""
    id: str
    image_id: str
    version: str
    tag: str
    digest: str
    size_bytes: Optional[int] = None
    platform: str = "linux/amd64"
    architecture: str = "amd64"
    os: str = "linux"
    quantization: Optional[str] = None
    release_notes: Optional[str] = None
    metadata: Dict[str, Any] = {}
    ceph_path: Optional[str] = None
    status: str = "active"
    is_release: bool = False
    created_at: datetime
    # Backward compatibility alias
    model_id: Optional[str] = None

    def __init__(self, **data):
        # Handle model_id -> image_id mapping for backward compatibility
        if 'model_id' in data and 'image_id' not in data:
            data['image_id'] = data['model_id']
        elif 'image_id' in data and 'model_id' not in data:
            data['model_id'] = data['image_id']
        super().__init__(**data)


# Backward compatibility alias
Release = Version


class Artifact(BaseModel):
    """Artifact model."""
    id: str
    name: str
    artifact_type: str
    file_path: str
    size_bytes: int
    checksum: str
    checksum_type: str = "sha256"
    platform: Optional[str] = None
    python_version: Optional[str] = None
    release_id: Optional[str] = None
    model_id: Optional[str] = None
    created_at: datetime
    uploaded_by: Optional[str] = None
    metadata: Dict[str, Any] = {}


class DockerBuild(BaseModel):
    """Docker Build model."""
    id: str
    release_id: str
    artifact_id: Optional[str] = None
    artifact_ids: Optional[List[str]] = None
    image_tag: str
    build_type: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    log_path: Optional[str] = None
    dockerfile_content: Optional[str] = None


class Deployment(BaseModel):
    """Deployment model."""
    id: str
    release_id: str
    environment: str
    deployed_by: Optional[str] = None
    deployed_at: datetime
    status: str = "success"
    metadata: Dict[str, Any] = {}


class ApiKey(BaseModel):
    """API Key model."""
    id: str
    name: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    key: Optional[str] = None  # Only present when creating


class AuditLog(BaseModel):
    """Audit Log model."""
    id: str
    api_key_name: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    created_at: datetime
