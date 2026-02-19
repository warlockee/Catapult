"""
Docker build services.

This package contains decomposed services for Docker build operations:
- DockerfileService: Template resolution and Dockerfile generation
- BuildWorkspaceService: Build directory setup and file staging
- BuildExecutorService: Docker command execution and logging
- BuildArchiveService: Build job archiving for disaster recovery
"""
from app.services.docker.dockerfile_service import DockerfileService, dockerfile_service
from app.services.docker.workspace_service import BuildWorkspaceService, workspace_service
from app.services.docker.executor_service import BuildExecutorService, executor_service
from app.services.docker.archive_service import BuildArchiveService, archive_service

__all__ = [
    "DockerfileService",
    "BuildWorkspaceService",
    "BuildExecutorService",
    "BuildArchiveService",
    # Singleton instances
    "dockerfile_service",
    "workspace_service",
    "executor_service",
    "archive_service",
]
