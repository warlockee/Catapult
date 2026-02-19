"""Docker build tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_docker_build_tools(mcp: FastMCP) -> None:
    """Register Docker build-related tools."""

    @mcp.tool()
    def list_docker_builds(
        version_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List Docker builds.

        Args:
            version_id: Filter by version UUID
            status: Filter by: pending, building, success, failed
            limit: Maximum results (default 20)

        Returns:
            List of builds with: id, version_id, image_tag, build_type,
            status, created_at, completed_at
        """
        client = get_client()
        return client.list_docker_builds(
            version_id=version_id,
            status=status,
            limit=limit,
        )

    @mcp.tool()
    def get_docker_build(build_id: str) -> dict:
        """Get Docker build details.

        Args:
            build_id: Build UUID

        Returns:
            Full build with: id, release_id, image_tag, build_type, status,
            error_message, created_at, completed_at, is_current
        """
        client = get_client()
        return client.get_docker_build(build_id)

    @mcp.tool()
    def get_docker_build_logs(build_id: str) -> str:
        """Get Docker build logs.

        Args:
            build_id: Build UUID

        Returns:
            Build log text
        """
        client = get_client()
        return client.get_docker_build_logs(build_id)

    @mcp.tool()
    def trigger_docker_build(
        version_id: str,
        build_type: str = "organic",
        image_tag: str | None = None,
    ) -> dict:
        """Trigger a Docker image build.

        This is a WRITE operation.

        Queues an async build job via Celery.

        Args:
            version_id: Version UUID to build
            build_type: Build type - organic, azure, optimized, test (default "organic")
            image_tag: Custom tag (auto-generated if None)

        Returns:
            {id, status: "pending", image_tag}
        """
        client = get_client()
        return client.trigger_docker_build(
            version_id=version_id,
            build_type=build_type,
            image_tag=image_tag,
        )

    @mcp.tool()
    def get_docker_disk_usage() -> dict:
        """Get Docker disk usage statistics.

        Useful for checking if cleanup is needed before builds.

        Returns:
            {images: {count, size_human}, build_cache: {count, size_human},
            containers: {count, size_human}, volumes: {count, size_human},
            disk_available_human, disk_total_human}
        """
        client = get_client()
        return client.get_docker_disk_usage()
