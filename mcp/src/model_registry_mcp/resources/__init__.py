"""MCP Resources for Model Registry."""

from mcp.server.fastmcp import FastMCP


def register_all_resources(mcp: FastMCP) -> None:
    """Register all resources with the MCP server."""
    from .resources import register_resources

    register_resources(mcp)
