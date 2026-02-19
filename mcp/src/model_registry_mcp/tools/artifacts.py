"""Artifact tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_artifact_tools(mcp: FastMCP) -> None:
    """Register artifact-related tools."""

    @mcp.tool()
    def list_artifacts(
        version_id: str | None = None,
        model_id: str | None = None,
        artifact_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List artifacts (evaluation results, configs, etc.).

        Artifacts are files attached to models or versions.

        Args:
            version_id: Filter by version UUID
            model_id: Filter by model UUID
            artifact_type: Filter by type: evaluation, config, checkpoint, log
            limit: Maximum results (default 50)

        Returns:
            List of artifacts with: id, model_id, version_id, artifact_type,
            filename, size_bytes, storage_path, created_at
        """
        client = get_client()
        return client.list_artifacts(
            version_id=version_id,
            model_id=model_id,
            artifact_type=artifact_type,
            limit=limit,
        )

    @mcp.tool()
    def get_artifact(artifact_id: str) -> dict:
        """Get artifact details.

        Args:
            artifact_id: Artifact UUID

        Returns:
            Full artifact with: id, model_id, version_id, artifact_type,
            filename, size_bytes, storage_path, metadata, created_at
        """
        client = get_client()
        return client.get_artifact(artifact_id)
