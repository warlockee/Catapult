"""MCP Resources for Catapult.

Resources provide context that can be attached to conversations.
"""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_resources(mcp: FastMCP) -> None:
    """Register resource handlers."""

    @mcp.resource("registry://models")
    def get_models_resource() -> str:
        """Get all models as a resource."""
        client = get_client()
        models = client.list_models(limit=100)
        lines = ["# Models in Registry\n"]
        for m in models:
            lines.append(f"- **{m.get('name')}** ({m.get('server_type', 'unknown')})")
            if m.get("description"):
                lines.append(f"  {m.get('description')}")
        return "\n".join(lines)

    @mcp.resource("registry://models/{model_id}")
    def get_model_resource(model_id: str) -> str:
        """Get a specific model as a resource."""
        client = get_client()
        model = client.get_model(model_id)
        lines = [
            f"# Model: {model.get('name')}\n",
            f"- **ID**: {model.get('id')}",
            f"- **Server Type**: {model.get('server_type', 'N/A')}",
            f"- **Storage Path**: {model.get('storage_path', 'N/A')}",
            f"- **Requires GPU**: {model.get('requires_gpu', True)}",
            f"- **Created**: {model.get('created_at', 'N/A')}",
        ]
        if model.get("description"):
            lines.append(f"\n## Description\n{model.get('description')}")
        if model.get("tags"):
            lines.append(f"\n## Tags\n{', '.join(model.get('tags'))}")
        return "\n".join(lines)

    @mcp.resource("registry://versions/{version_id}")
    def get_version_resource(version_id: str) -> str:
        """Get a specific version as a resource."""
        client = get_client()
        version = client.get_version(version_id)
        lines = [
            f"# Version: {version.get('version')}\n",
            f"- **ID**: {version.get('id')}",
            f"- **Model**: {version.get('model_name', 'N/A')}",
            f"- **Tag**: {version.get('tag', 'N/A')}",
            f"- **Is Release**: {version.get('is_release', False)}",
            f"- **Quantization**: {version.get('quantization', 'N/A')}",
            f"- **Ceph Path**: {version.get('ceph_path', 'N/A')}",
            f"- **Created**: {version.get('created_at', 'N/A')}",
        ]
        return "\n".join(lines)

    @mcp.resource("registry://deployments")
    def get_deployments_resource() -> str:
        """Get all local deployments as a resource."""
        client = get_client()
        deployments = client.list_deployments(limit=50)
        lines = ["# Local Deployments\n"]
        for d in deployments:
            status = d.get("status", "unknown")
            emoji = {"running": "🟢", "stopped": "🔴", "failed": "❌"}.get(status, "⚪")
            lines.append(
                f"- {emoji} **{d.get('image_name', 'unknown')}** "
                f"({d.get('environment', 'N/A')}) - {d.get('endpoint_url', 'N/A')}"
            )
        return "\n".join(lines)

    @mcp.resource("registry://deployments/{deployment_id}")
    def get_deployment_resource(deployment_id: str) -> str:
        """Get a specific deployment as a resource."""
        client = get_client()
        deployment = client.get_deployment(deployment_id)
        lines = [
            f"# Deployment: {deployment.get('image_name', 'unknown')}\n",
            f"- **ID**: {deployment.get('id')}",
            f"- **Status**: {deployment.get('status', 'N/A')}",
            f"- **Health**: {deployment.get('health_status', 'N/A')}",
            f"- **Environment**: {deployment.get('environment', 'N/A')}",
            f"- **Endpoint**: {deployment.get('endpoint_url', 'N/A')}",
            f"- **Port**: {deployment.get('host_port', 'N/A')}",
            f"- **GPU Enabled**: {deployment.get('gpu_enabled', False)}",
            f"- **Container ID**: {deployment.get('container_id', 'N/A')}",
            f"- **Deployed At**: {deployment.get('started_at', 'N/A')}",
        ]
        return "\n".join(lines)

    @mcp.resource("registry://production")
    def get_production_resource() -> str:
        """Get production topology as a resource."""
        client = get_client()
        configs = client.list_release_configs()
        lines = ["# Production Deployments (Release Configs)\n"]

        # Group by machine
        by_machine: dict[str, list] = {}
        for c in configs:
            machine = c.get("machine", "unknown")
            if machine not in by_machine:
                by_machine[machine] = []
            by_machine[machine].append(c)

        for machine, machine_configs in sorted(by_machine.items()):
            lines.append(f"\n## {machine}\n")
            for c in machine_configs:
                lines.append(
                    f"- Port {c.get('port')}: **{c.get('model_name')}** "
                    f"(GPUs: {c.get('gpu_ids', 'N/A')})"
                )

        return "\n".join(lines)

    @mcp.resource("registry://production/{machine}")
    def get_machine_resource(machine: str) -> str:
        """Get production config for a specific machine."""
        client = get_client()
        configs = client.list_release_configs(machine=machine)
        lines = [f"# Machine: {machine}\n"]

        for c in configs:
            lines.append(f"\n## Port {c.get('port')}\n")
            lines.append(f"- **Model**: {c.get('model_name')}")
            lines.append(f"- **Path**: {c.get('model_path', 'N/A')}")
            lines.append(f"- **Template**: {c.get('template', 'N/A')}")
            lines.append(f"- **GPUs**: {c.get('gpu_ids', 'N/A')}")
            lines.append(f"- **Tensor Parallel**: {c.get('tensor_parallel', 1)}")

        return "\n".join(lines)

    @mcp.resource("registry://health")
    def get_health_resource() -> str:
        """Get system health as a resource."""
        client = get_client()
        health = client.health_check()
        lines = [
            "# Registry Health\n",
            f"- **Status**: {health.get('status', 'unknown')}",
            f"- **Database**: {'✓' if health.get('database') else '✗'}",
            f"- **Cache**: {'✓' if health.get('cache') else '✗'}",
            f"- **Celery**: {'✓' if health.get('celery') else '✗'}",
            f"- **Version**: {health.get('version', 'N/A')}",
        ]
        return "\n".join(lines)
