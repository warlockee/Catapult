"""
Abstract base class for deployment executors.

Defines the interface that all deployment executors must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Dict, Optional
from uuid import UUID


@dataclass
class DeploymentConfig:
    """Configuration for a deployment execution."""

    image_tag: str
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volume_mounts: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    gpu_enabled: bool = False
    gpu_count: int = 1
    memory_limit: Optional[str] = None  # e.g., "8g", "16g"
    cpu_limit: Optional[float] = None   # e.g., 4.0 for 4 CPUs


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""

    success: bool
    container_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    port: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class ContainerStatus:
    """Status of a running container."""

    running: bool
    healthy: bool
    exit_code: Optional[int] = None
    started_at: Optional[datetime] = None
    error: Optional[str] = None


class DeploymentExecutor(ABC):
    """
    Abstract base class for deployment executors.

    Implementations must provide methods for:
    - Deploying containers
    - Stopping containers
    - Restarting containers
    - Getting container status
    - Getting container logs
    - Health checking deployments
    """

    @abstractmethod
    async def deploy(
        self,
        deployment_id: UUID,
        config: DeploymentConfig,
    ) -> DeploymentResult:
        """
        Deploy a container with the given configuration.

        Args:
            deployment_id: UUID of the deployment record
            config: Configuration for the deployment

        Returns:
            DeploymentResult with container details or error
        """
        pass

    @abstractmethod
    async def stop(
        self,
        deployment_id: UUID,
        container_id: str,
    ) -> bool:
        """
        Stop a running container.

        Args:
            deployment_id: UUID of the deployment record
            container_id: Docker container ID

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    async def restart(
        self,
        deployment_id: UUID,
        container_id: str,
        config: DeploymentConfig,
        host_port: int = None,
    ) -> DeploymentResult:
        """
        Restart a container.

        Args:
            deployment_id: UUID of the deployment record
            container_id: Docker container ID to restart
            config: Configuration for the restart
            host_port: Optional port to use (for local deployments)

        Returns:
            DeploymentResult with new container details or error
        """
        pass

    @abstractmethod
    async def get_status(
        self,
        container_id: str,
    ) -> ContainerStatus:
        """
        Get the status of a container.

        Args:
            container_id: Docker container ID

        Returns:
            ContainerStatus with container state
        """
        pass

    @abstractmethod
    async def get_logs(
        self,
        container_id: str,
        tail: int = 100,
    ) -> str:
        """
        Get logs from a container.

        Args:
            container_id: Docker container ID
            tail: Number of lines to retrieve

        Returns:
            Log content as string
        """
        pass

    @abstractmethod
    async def stream_logs(
        self,
        container_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream logs from a container.

        Args:
            container_id: Docker container ID

        Yields:
            Log lines as they arrive
        """
        pass

    @abstractmethod
    async def health_check(
        self,
        endpoint_url: str,
        timeout: int = 5,
    ) -> bool:
        """
        Check if a deployment is healthy via HTTP endpoint.

        Args:
            endpoint_url: URL to check (e.g., http://localhost:9001/health)
            timeout: Timeout in seconds

        Returns:
            True if healthy, False otherwise
        """
        pass
