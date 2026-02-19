"""System tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_system_tools(mcp: FastMCP) -> None:
    """Register system-related tools."""

    @mcp.tool()
    def health_check() -> dict:
        """Check registry system health.

        Verifies database, cache, and service connectivity.

        Returns:
            {status: "healthy"|"unhealthy", database: bool, cache: bool,
            celery: bool, version: str}
        """
        client = get_client()
        return client.health_check()

    @mcp.tool()
    def get_audit_logs(
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get system audit logs.

        Track who did what and when.

        Args:
            action: Filter by action: create, update, delete, deploy, etc.
            resource_type: Filter by type: model, version, deployment
            limit: Maximum results (default 50)

        Returns:
            List of logs with: id, action, resource_type, resource_id,
            user, details, created_at
        """
        client = get_client()
        return client.get_audit_logs(action=action, resource_type=resource_type, limit=limit)

    @mcp.tool()
    def get_storage_usage() -> dict:
        """Get storage usage statistics.

        Shows model storage consumption on Ceph.

        Returns:
            {total_bytes, used_bytes, free_bytes, models_count,
            versions_count, largest_models: [{name, size}]}
        """
        client = get_client()
        return client.get_storage_usage()

    @mcp.tool()
    def browse_model_storage(relative_path: str = "") -> list[dict]:
        """Browse model storage filesystem.

        SECURITY: Sandboxed to STORAGE_ROOT. Cannot access paths outside
        the model storage directory.

        Args:
            relative_path: Path relative to storage root (default: root)

        Returns:
            List of entries: [{name, type: "file"|"dir", size, modified}]
        """
        client = get_client()
        return client.browse_model_storage(relative_path)
