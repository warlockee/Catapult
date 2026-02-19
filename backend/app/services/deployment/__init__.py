"""
Deployment execution services.

This package provides services for executing deployments locally via Docker
or on Kubernetes (placeholder).
"""
from app.services.deployment.executor_base import (
    DeploymentExecutor,
    DeploymentConfig,
    DeploymentResult,
    ContainerStatus,
)
from app.services.deployment.local_executor import LocalDeploymentExecutor
from app.services.deployment.k8s_executor import K8sDeploymentExecutor
from app.services.deployment.deployment_service import deployment_service
from app.services.deployment.port_allocator import port_allocator

__all__ = [
    "DeploymentExecutor",
    "DeploymentConfig",
    "DeploymentResult",
    "ContainerStatus",
    "LocalDeploymentExecutor",
    "K8sDeploymentExecutor",
    "deployment_service",
    "port_allocator",
]
