"""Configuration for MCP Server."""

from pydantic_settings import BaseSettings


class MCPConfig(BaseSettings):
    """MCP Server Configuration.

    Environment variables:
        REGISTRY_URL: Registry API URL (default: http://localhost/api)
        REGISTRY_API_KEY: API key for authentication (default: admin-key)
        REGISTRY_READ_ONLY: Disable all write operations (default: false)
        REGISTRY_ALLOW_PRODUCTION_CHANGES: Allow propose_release_config (default: true)
        REGISTRY_CACHE_TTL_SECONDS: Cache TTL for list operations (default: 60)
        REGISTRY_REQUEST_TIMEOUT_SECONDS: API request timeout (default: 30)
        REGISTRY_MAX_LOG_LINES: Maximum log lines to return (default: 5000)
        REGISTRY_MAX_RESULTS: Maximum results for list operations (default: 100)
    """

    # Registry connection
    url: str = "http://localhost/api"
    api_key: str = "admin-key"

    # Operation modes
    read_only: bool = False
    allow_production_changes: bool = True

    # Performance
    cache_ttl_seconds: int = 60
    request_timeout_seconds: int = 30

    # Limits
    max_log_lines: int = 5000
    max_results: int = 100

    model_config = {
        "env_prefix": "REGISTRY_",
        "case_sensitive": False,
    }


# Global config instance - initialized lazily
_config: MCPConfig | None = None


def get_config() -> MCPConfig:
    """Get or create the config singleton."""
    global _config
    if _config is None:
        _config = MCPConfig()
    return _config
