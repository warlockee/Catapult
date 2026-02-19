"""MCP Server for Model Registry."""

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    "Model Registry",
    instructions="MCP server for Catapult - manage models, versions, and deployments",
)

# Import and register tools
from .tools import register_all_tools
from .resources import register_all_resources
from .prompts import register_all_prompts

register_all_tools(mcp)
register_all_resources(mcp)
register_all_prompts(mcp)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
