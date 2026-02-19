"""MCP Prompts for Catapult."""

from mcp.server.fastmcp import FastMCP


def register_all_prompts(mcp: FastMCP) -> None:
    """Register all prompts with the MCP server."""
    from .prompts import register_prompts

    register_prompts(mcp)
