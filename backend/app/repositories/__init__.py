"""
Repository layer for database access.
"""
from app.repositories.api_key_repository import ApiKeyRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.base import BaseRepository
from app.repositories.deployment_repository import DeploymentRepository
from app.repositories.docker_build_repository import DockerBuildRepository
from app.repositories.model_repository import ModelRepository
from app.repositories.release_repository import ReleaseRepository  # Backward compat alias
from app.repositories.version_repository import VersionRepository

__all__ = [
    "BaseRepository",
    "ModelRepository",
    "VersionRepository",
    "ReleaseRepository",  # Backward compatibility alias
    "ArtifactRepository",
    "DeploymentRepository",
    "DockerBuildRepository",
    "ApiKeyRepository",
]
