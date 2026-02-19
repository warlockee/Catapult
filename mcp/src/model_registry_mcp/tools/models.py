"""Model tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client, MCPError


def register_model_tools(mcp: FastMCP) -> None:
    """Register model-related tools."""

    @mcp.tool()
    def list_models(
        search: str | None = None,
        server_type: str | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List models in the registry.

        Models are ML model definitions with metadata. Each model can have
        multiple versions (checkpoints).

        Args:
            search: Filter by name (partial match)
            server_type: Filter by type: vllm, audio, whisper, tts, embedding
            source: Filter by source: filesystem, manual
            limit: Maximum results (default 50)

        Returns:
            List of models with: id, name, server_type, storage_path,
            description, tags, requires_gpu, created_at
        """
        client = get_client()
        return client.list_models(
            search=search,
            server_type=server_type,
            source=source,
            limit=limit,
        )

    @mcp.tool()
    def get_model(model_id: str) -> dict:
        """Get detailed model information.

        Args:
            model_id: Model UUID or exact name

        Returns:
            Full model with: id, name, storage_path, repository, company,
            base_model, parameter_count, description, tags, metadata,
            server_type, source, requires_gpu, created_at, updated_at
        """
        client = get_client()
        return client.get_model(model_id)

    @mcp.tool()
    def create_model(
        name: str,
        storage_path: str,
        server_type: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        requires_gpu: bool = True,
    ) -> dict:
        """Register a new model in the registry.

        This is a WRITE operation.

        Args:
            name: Unique model name
            storage_path: Path to model files (Ceph path)
            server_type: Server type for deployment (vllm, audio, etc.)
            description: Model description
            tags: Searchable tags
            requires_gpu: Whether model needs GPU (default True)

        Returns:
            Created model
        """
        client = get_client()
        return client.create_model(
            name=name,
            storage_path=storage_path,
            server_type=server_type,
            description=description,
            tags=tags,
            requires_gpu=requires_gpu,
        )

    @mcp.tool()
    def update_model(
        model_id: str,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Update model metadata.

        This is a WRITE operation.

        Args:
            model_id: Model UUID or name
            description: New description (None to keep current)
            tags: New tags (None to keep current)
            metadata: New metadata (None to keep current)

        Returns:
            Updated model
        """
        client = get_client()
        return client.update_model(
            model_id=model_id,
            description=description,
            tags=tags,
            metadata=metadata,
        )

    @mcp.tool()
    def delete_model(model_id: str) -> dict:
        """Delete a model and all its versions.

        This is a DESTRUCTIVE WRITE operation.

        Args:
            model_id: Model UUID or name

        Returns:
            Confirmation with deleted_id
        """
        client = get_client()
        return client.delete_model(model_id)
