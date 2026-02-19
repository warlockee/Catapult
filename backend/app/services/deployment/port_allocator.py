"""
Port allocation service for local deployments.

Manages port allocation in a configurable range (default 9000-9999)
to avoid conflicts between running deployments.
"""
import asyncio
import logging
from typing import Optional, Set
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import PortAllocationError
from app.models.deployment import Deployment

logger = logging.getLogger(__name__)


class PortAllocator:
    """
    Thread-safe port allocation with database persistence.

    Allocates ports from a configurable range, checking both
    database records and actual system port availability.
    """

    def __init__(
        self,
        port_range_start: int = None,
        port_range_end: int = None,
    ):
        """
        Initialize the port allocator.

        Args:
            port_range_start: Start of port range (default from settings)
            port_range_end: End of port range (default from settings)
        """
        self.port_range_start = port_range_start or settings.DEPLOYMENT_PORT_RANGE_START
        self.port_range_end = port_range_end or settings.DEPLOYMENT_PORT_RANGE_END

    async def get_used_ports(
        self, db: AsyncSession, exclude_deployment_id: Optional[UUID] = None
    ) -> Set[int]:
        """
        Get all ports currently in use by deployments.

        Includes running, deploying, and stopping statuses since containers
        may still be using ports during state transitions.

        Args:
            db: Database session
            exclude_deployment_id: Optional deployment ID to exclude from check
                                   (used during restart to not count own port)

        Returns:
            Set of port numbers in use
        """
        stmt = select(Deployment.host_port).where(
            Deployment.status.in_(["running", "deploying", "stopping"]),
            Deployment.host_port.isnot(None),
        )
        if exclude_deployment_id:
            stmt = stmt.where(Deployment.id != exclude_deployment_id)
        result = await db.execute(stmt)
        return {row[0] for row in result.fetchall()}

    async def get_host_listening_ports(self) -> Set[int]:
        """
        Get all TCP ports currently listening on the HOST.

        This runs a container with host networking to check the actual host
        network namespace, not the container's namespace. This is critical
        because the worker runs in a container with its own network namespace,
        but Docker port publishing happens on the HOST.

        Returns:
            Set of port numbers listening on the host
        """
        ports = set()
        try:
            # Use ss via Docker with host networking to see actual host ports
            # This is necessary because we're running inside a container
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm", "--net=host",
                "busybox", "sh", "-c",
                "netstat -tln 2>/dev/null | tail -n +3",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0 and stdout:
                # Parse netstat output: "tcp  0  0  0.0.0.0:9000  0.0.0.0:*  LISTEN"
                for line in stdout.decode().strip().split("\n"):
                    if not line or "LISTEN" not in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        # Local address is in format "0.0.0.0:port" or "127.0.0.1:port"
                        local_addr = parts[3]
                        if ":" in local_addr:
                            try:
                                port = int(local_addr.rsplit(":", 1)[1])
                                if self.port_range_start <= port <= self.port_range_end:
                                    ports.add(port)
                            except (ValueError, IndexError):
                                continue
            elif stderr:
                logger.warning(f"Host port check error: {stderr.decode()}")
        except Exception as e:
            logger.warning(f"Error checking host ports: {e}")

        return ports

    async def get_docker_ports_in_use(self) -> Set[int]:
        """
        Get all ports currently bound by RUNNING Docker containers.

        This catches containers that may be running but not tracked in DB
        (orphaned containers, containers restarted by Docker, etc.)

        NOTE: Only checks running containers because stopped containers
        don't actually hold ports (their port config is just metadata).

        Returns:
            Set of port numbers bound by Docker containers
        """
        ports = set()
        try:
            # Only check RUNNING containers - stopped containers don't hold ports
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps", "--format", "{{.Ports}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0 and stdout:
                # Parse port mappings like "0.0.0.0:9000->8000/tcp"
                for line in stdout.decode().strip().split("\n"):
                    if not line:
                        continue
                    # Handle multiple port mappings separated by comma
                    for mapping in line.split(", "):
                        if "->" in mapping and ":" in mapping:
                            try:
                                # Extract host port from "0.0.0.0:9000->8000/tcp"
                                host_part = mapping.split("->")[0]
                                host_port = int(host_part.split(":")[-1])
                                if self.port_range_start <= host_port <= self.port_range_end:
                                    ports.add(host_port)
                            except (ValueError, IndexError):
                                continue
            elif stderr:
                logger.warning(f"Docker ps error: {stderr.decode()}")
        except Exception as e:
            logger.warning(f"Error checking Docker ports: {e}")

        return ports

    async def cleanup_orphaned_container_on_port(self, port: int) -> bool:
        """
        Remove orphaned RUNNING container blocking a port.

        An orphaned container is one using a port but not tracked as
        'running' in the database.

        Args:
            port: Port number to free up

        Returns:
            True if a container was removed, False otherwise
        """
        try:
            # Find RUNNING container using this port (stopped containers don't hold ports)
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps", "--filter", f"publish={port}",
                "--format", "{{.ID}} {{.Names}} {{.Status}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0 and stdout:
                for line in stdout.decode().strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        container_id = parts[0]
                        container_name = parts[1]
                        logger.warning(
                            f"Found orphaned container {container_name} ({container_id}) "
                            f"using port {port}, removing..."
                        )
                        # Stop and remove the container
                        stop_proc = await asyncio.create_subprocess_exec(
                            "docker", "rm", "-f", container_id,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await stop_proc.communicate()
                        if stop_proc.returncode == 0:
                            logger.info(f"Removed orphaned container {container_name}")
                            return True
                        else:
                            logger.error(f"Failed to remove container {container_name}")
        except Exception as e:
            logger.error(f"Error cleaning up orphaned container on port {port}: {e}")

        return False

    async def allocate(self, db: AsyncSession) -> int:
        """
        Allocate an available port from the range.

        Checks database records, Docker container ports, and HOST system availability.
        Automatically cleans up orphaned containers blocking ports.

        Args:
            db: Database session

        Returns:
            Available port number

        Raises:
            PortAllocationError: If no ports are available
        """
        db_used_ports = await self.get_used_ports(db)
        docker_used_ports = await self.get_docker_ports_in_use()
        host_used_ports = await self.get_host_listening_ports()

        # Find orphaned ports (in Docker but not in DB as running)
        orphaned_ports = docker_used_ports - db_used_ports
        if orphaned_ports:
            logger.warning(f"Found orphaned Docker ports not in DB: {orphaned_ports}")

        # Combine all used ports
        all_used_ports = db_used_ports | docker_used_ports | host_used_ports

        if host_used_ports:
            logger.info(f"Host ports in use within range: {sorted(host_used_ports)}")

        for port in range(self.port_range_start, self.port_range_end + 1):
            # Skip if port is used by anything
            if port in all_used_ports:
                # If it's an orphaned Docker container, try cleanup
                if port in orphaned_ports:
                    logger.info(f"Port {port} in use by orphaned Docker container, attempting cleanup...")
                    if await self.cleanup_orphaned_container_on_port(port):
                        logger.info(f"Allocated port {port} after cleanup")
                        return port
                continue

            logger.info(f"Allocated port {port}")
            return port

        raise PortAllocationError(self.port_range_start, self.port_range_end)

    async def is_available(
        self, db: AsyncSession, port: int, exclude_deployment_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if a specific port is available.

        Args:
            db: Database session
            port: Port number to check
            exclude_deployment_id: Optional deployment ID to exclude from DB check
                                   (used during restart to not count own port)

        Returns:
            True if available, False otherwise
        """
        if port < self.port_range_start or port > self.port_range_end:
            return False

        # Check database (exclude self if specified)
        db_used_ports = await self.get_used_ports(db, exclude_deployment_id)
        if port in db_used_ports:
            return False

        # Check Docker containers
        docker_used_ports = await self.get_docker_ports_in_use()
        if port in docker_used_ports:
            return False

        # Check HOST listening ports (not container namespace)
        host_used_ports = await self.get_host_listening_ports()
        if port in host_used_ports:
            return False

        return True

    async def release(self, db: AsyncSession, port: int) -> None:
        """
        Release a port (for documentation - actual release happens via deployment status update).

        Note: Port release is handled by updating the deployment status to 'stopped'.
        This method is provided for completeness but doesn't need to do anything
        since the database query in get_used_ports only looks at 'running' deployments.

        Args:
            db: Database session
            port: Port number to release
        """
        logger.info(f"Port {port} released (deployment stopped)")


# Singleton instance
port_allocator = PortAllocator()
