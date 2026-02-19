"""
Kubernetes deployment executor (placeholder).

This executor will be implemented in a future iteration to support:
- Kubernetes deployment creation
- Service and ingress management
- Scaling operations
- Health monitoring via k8s probes
"""
from typing import AsyncIterator
from uuid import UUID

from app.services.deployment.executor_base import (
    DeploymentExecutor,
    DeploymentConfig,
    DeploymentResult,
    ContainerStatus,
)


class K8sDeploymentExecutor(DeploymentExecutor):
    """
    Placeholder executor for Kubernetes deployments.

    All methods raise NotImplementedError with helpful messages
    directing users to use local deployment instead.
    """

    async def deploy(
        self,
        deployment_id: UUID,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        """Deploy to Kubernetes (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes deployment is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments. "
            "K8s support is planned for a future release."
        )

    async def stop(
        self,
        deployment_id: UUID,
        container_id: str,
    ) -> bool:
        """Stop Kubernetes deployment (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes deployment stop is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments."
        )

    async def restart(
        self,
        deployment_id: UUID,
        container_id: str,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        """Restart Kubernetes deployment (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes deployment restart is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments."
        )

    async def get_status(
        self,
        container_id: str,
    ) -> ContainerStatus:
        """Get Kubernetes pod status (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes status check is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments."
        )

    async def get_logs(
        self,
        container_id: str,
        tail: int = 100,
    ) -> str:
        """Get Kubernetes pod logs (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes log retrieval is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments."
        )

    async def stream_logs(
        self,
        container_id: str,
    ) -> AsyncIterator[str]:
        """Stream Kubernetes pod logs (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes log streaming is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments."
        )
        # This yield is never reached but satisfies the type checker
        yield ""  # pragma: no cover

    async def health_check(
        self,
        endpoint_url: str,
        timeout: int = 5,
    ) -> bool:
        """Check Kubernetes deployment health (not yet implemented)."""
        raise NotImplementedError(
            "Kubernetes health check is not yet implemented. "
            "Use deployment_type='local' for local Docker deployments."
        )


# Singleton instance
k8s_executor = K8sDeploymentExecutor()
