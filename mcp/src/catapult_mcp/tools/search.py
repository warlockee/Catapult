"""Search tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related tools."""

    @mcp.tool()
    def search(
        query: str,
        types: list[str] | None = None,
        limit: int = 20,
    ) -> dict:
        """Search across models, versions, and deployments.

        Unified search for finding resources by name or keyword.

        Args:
            query: Search term (partial match on names)
            types: Resource types to search: ["models", "versions", "deployments"]
                   (default: all)
            limit: Max results per type (default 20)

        Returns:
            {models: [...], versions: [...], deployments: [...]}
        """
        client = get_client()
        return client.search(query=query, types=types, limit=limit)
