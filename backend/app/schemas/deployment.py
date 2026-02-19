"""
Pydantic schemas for Deployment.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class DeploymentType(str, Enum):
    """Type of deployment."""
    METADATA = "metadata"  # Just records metadata (legacy behavior)
    LOCAL = "local"        # Run locally via Docker
    K8S = "k8s"            # Deploy to Kubernetes (placeholder)


class DeploymentStatus(str, Enum):
    """Status of a deployment."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    SUCCESS = "success"  # Legacy status for metadata-only deployments


class HealthStatus(str, Enum):
    """Health status of a deployment."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class DeploymentBase(BaseModel):
    """Base schema for Deployment."""
    environment: str = Field(..., max_length=100)
    cluster: Optional[str] = Field(None, max_length=255)
    k8s_namespace: Optional[str] = Field(None, max_length=255)
    endpoint_url: Optional[str] = Field(None, max_length=500)
    replicas: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeploymentCreate(DeploymentBase):
    """Schema for creating a Deployment (metadata only, legacy)."""
    release_id: UUID
    status: str = "success"


class DeploymentExecuteCreate(BaseModel):
    """Schema for creating and executing a deployment."""
    release_id: UUID
    environment: str = Field(..., max_length=100)
    deployment_type: DeploymentType = DeploymentType.LOCAL

    # Execution configuration
    gpu_enabled: bool = False
    gpu_count: int = Field(default=1, ge=1)
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    volume_mounts: Dict[str, str] = Field(default_factory=dict)  # host_path: container_path
    memory_limit: Optional[str] = None  # e.g., "8g", "16g"
    cpu_limit: Optional[float] = None   # e.g., 4.0 for 4 CPUs

    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeploymentResponse(BaseModel):
    """Schema for Deployment response."""
    id: UUID
    release_id: UUID
    environment: str
    cluster: Optional[str] = None
    k8s_namespace: Optional[str] = None
    endpoint_url: Optional[str] = None
    replicas: Optional[int] = None
    deployed_by: Optional[str] = None
    deployed_at: datetime
    terminated_at: Optional[datetime] = None
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True
        populate_by_name = True

    @classmethod
    def from_deployment(cls, deployment) -> "DeploymentResponse":
        """Convert a Deployment ORM instance to response schema."""
        return cls(
            id=deployment.id,
            release_id=deployment.release_id,
            environment=deployment.environment,
            deployed_by=deployment.deployed_by,
            deployed_at=deployment.deployed_at,
            status=deployment.status,
            metadata=deployment.meta_data or {},
        )


class DeploymentExecuteResponse(DeploymentResponse):
    """Extended response for executed deployments."""
    container_id: Optional[str] = None
    host_port: Optional[int] = None
    deployment_type: DeploymentType = DeploymentType.METADATA
    health_status: HealthStatus = HealthStatus.UNKNOWN
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    gpu_enabled: bool = False
    image_tag: Optional[str] = None

    @classmethod
    def from_deployment(cls, deployment) -> "DeploymentExecuteResponse":
        """Convert a Deployment ORM instance to execute response schema."""
        return cls(
            id=deployment.id,
            release_id=deployment.release_id,
            environment=deployment.environment,
            deployed_by=deployment.deployed_by,
            deployed_at=deployment.deployed_at,
            status=deployment.status,
            metadata=deployment.meta_data or {},
            container_id=deployment.container_id,
            host_port=deployment.host_port,
            deployment_type=deployment.deployment_type,
            health_status=deployment.health_status,
            started_at=deployment.started_at,
            stopped_at=deployment.stopped_at,
            gpu_enabled=deployment.gpu_enabled,
            image_tag=deployment.image_tag,
        )


class DeploymentWithRelease(DeploymentExecuteResponse):
    """Schema for Deployment with Release details."""
    release_version: Optional[str] = None
    image_name: Optional[str] = None

    @classmethod
    def from_deployment(cls, deployment) -> "DeploymentWithRelease":
        """Convert a Deployment ORM instance with release info to response schema."""
        return cls(
            id=deployment.id,
            release_id=deployment.release_id,
            environment=deployment.environment,
            deployed_by=deployment.deployed_by,
            deployed_at=deployment.deployed_at,
            status=deployment.status,
            metadata=deployment.meta_data or {},
            release_version=deployment.version.version if deployment.version else None,
            image_name=deployment.version.model.name if deployment.version and deployment.version.model else None,
            container_id=deployment.container_id,
            host_port=deployment.host_port,
            deployment_type=deployment.deployment_type,
            health_status=deployment.health_status,
            started_at=deployment.started_at,
            stopped_at=deployment.stopped_at,
            gpu_enabled=deployment.gpu_enabled,
            endpoint_url=deployment.endpoint_url,
            image_tag=deployment.image_tag,
        )


class ContainerStatusResponse(BaseModel):
    """Container status response."""
    running: bool
    healthy: bool
    exit_code: Optional[int] = None
    started_at: Optional[datetime] = None
    error: Optional[str] = None


class DeploymentLogsResponse(BaseModel):
    """Container logs response."""
    deployment_id: UUID
    logs: str
    truncated: bool = False
