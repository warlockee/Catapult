"""MCP Tools for Model Registry."""

from mcp.server.fastmcp import FastMCP


def register_all_tools(mcp: FastMCP) -> None:
    """Register all tools with the MCP server."""
    from .models import register_model_tools
    from .versions import register_version_tools
    from .deployments import register_deployment_tools
    from .docker_builds import register_docker_build_tools
    from .benchmarks import register_benchmark_tools
    from .release_configs import register_release_config_tools
    from .artifacts import register_artifact_tools
    from .system import register_system_tools
    from .search import register_search_tools

    register_model_tools(mcp)
    register_version_tools(mcp)
    register_deployment_tools(mcp)
    register_docker_build_tools(mcp)
    register_benchmark_tools(mcp)
    register_release_config_tools(mcp)
    register_artifact_tools(mcp)
    register_system_tools(mcp)
    register_search_tools(mcp)
