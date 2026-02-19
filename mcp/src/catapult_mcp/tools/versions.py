"""Version tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_version_tools(mcp: FastMCP) -> None:
    """Register version-related tools."""

    @mcp.tool()
    def list_versions(
        model_name: str | None = None,
        is_release: bool | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List model versions.

        A version is a specific checkpoint. Versions can be promoted to
        "releases" to mark them as production-ready.

        Args:
            model_name: Filter by model name
            is_release: True=only releases, False=only non-releases, None=all
            limit: Maximum results (default 50)

        Returns:
            List of versions with: id, model_name, version, tag, is_release,
            quantization, ceph_path, created_at
        """
        client = get_client()
        return client.list_versions(
            model_name=model_name,
            is_release=is_release,
            limit=limit,
        )

    @mcp.tool()
    def get_version(version_id: str) -> dict:
        """Get detailed version information.

        Args:
            version_id: Version UUID

        Returns:
            Full version with: id, model_id, model_name, version, tag, digest,
            quantization, is_release, ceph_path, metadata, created_at
        """
        client = get_client()
        return client.get_version(version_id)

    @mcp.tool()
    def get_latest_version(model_name: str) -> dict:
        """Get the most recent version of a model.

        Args:
            model_name: Model name

        Returns:
            Latest version or error if none exists
        """
        client = get_client()
        return client.get_latest_version(model_name)

    @mcp.tool()
    def create_version(
        model_id: str,
        version: str,
        tag: str,
        digest: str,
        ceph_path: str | None = None,
        quantization: str | None = None,
        is_release: bool = False,
        auto_build: bool = False,
    ) -> dict:
        """Create a new model version.

        This is a WRITE operation.

        Args:
            model_id: Parent model UUID
            version: Version string (e.g., "1.0.0")
            tag: Docker tag
            digest: Content digest (sha256:...)
            ceph_path: Path to checkpoint on Ceph
            quantization: Quantization type (fp16, int8, etc.)
            is_release: Mark as production release (default False)
            auto_build: Trigger Docker build automatically (default False)

        Returns:
            Created version
        """
        client = get_client()
        return client.create_version(
            model_id=model_id,
            version=version,
            tag=tag,
            digest=digest,
            ceph_path=ceph_path,
            quantization=quantization,
            is_release=is_release,
            auto_build=auto_build,
        )

    @mcp.tool()
    def promote_version(version_id: str, is_release: bool = True) -> dict:
        """Promote or demote a version to/from release status.

        This is a WRITE operation.

        Args:
            version_id: Version UUID
            is_release: True to promote, False to demote (default True)

        Returns:
            Updated version
        """
        client = get_client()
        return client.promote_version(version_id, is_release)
