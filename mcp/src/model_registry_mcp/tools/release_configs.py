"""Release config tools for MCP.

These tools provide READ-ONLY visibility into production topology (release configs).
Production changes require PR workflow with human approval.
"""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_release_config_tools(mcp: FastMCP) -> None:
    """Register release config tools."""

    @mcp.tool()
    def list_release_configs(
        machine: str | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        """List production deployments from release configs.

        Shows what's actually deployed in production (GitOps source of truth).
        These are NOT local deployments - use list_deployments for local.

        Args:
            machine: Filter by machine hostname
            model_name: Filter by model name

        Returns:
            List of configs with: machine, port, model_name, model_path,
            template, gpu_ids, tensor_parallel
        """
        client = get_client()
        return client.list_release_configs(machine=machine, model_name=model_name)

    @mcp.tool()
    def get_release_config(machine: str, port: int) -> dict:
        """Get specific production deployment config.

        Args:
            machine: Machine hostname
            port: Port number

        Returns:
            Full config with: machine, port, model_name, model_path,
            template, gpu_ids, tensor_parallel, extra_env
        """
        client = get_client()
        return client.get_release_config(machine, port)

    @mcp.tool()
    def get_machine_topology() -> dict:
        """Get GPU allocation topology across all machines.

        Shows which GPUs are allocated to which models on each machine.
        Useful for capacity planning and identifying free GPUs.

        Returns:
            {machine: {gpus: [gpu_info], deployments: [...]}}
        """
        client = get_client()
        return client.get_machine_topology()

    @mcp.tool()
    def get_models_by_deployment() -> dict:
        """Get production deployments grouped by model.

        Shows all machines/ports where each model is deployed.
        Useful for understanding model distribution.

        Returns:
            {model_name: [{machine, port, gpu_ids}]}
        """
        client = get_client()
        return client.get_models_by_deployment()

    @mcp.tool()
    def list_docker_templates() -> list[str]:
        """List available docker-compose templates for production.

        Templates define deployment configurations (vllm, audio, etc.).

        Returns:
            List of template names (e.g., ["vllm.yml", "audio.yml"])
        """
        client = get_client()
        return client.list_docker_templates()

    @mcp.tool()
    def propose_release_config(
        machine: str,
        port: int,
        template_name: str,
        model_name: str,
        model_path: str,
        gpu_ids: str,
        tensor_parallel: int = 1,
        description: str = "",
    ) -> dict:
        """Propose a production deployment change via PR.

        This is a PRODUCTION WRITE operation requiring human approval.

        Creates a pull request in release configs that must be reviewed
        and merged by a human. The change takes effect when merged.

        Args:
            machine: Target machine hostname
            port: Target port number
            template_name: Docker template (from list_docker_templates)
            model_name: Model name for the deployment
            model_path: Path to model files
            gpu_ids: Comma-separated GPU IDs (e.g., "0,1,2,3")
            tensor_parallel: Tensor parallelism degree (default 1)
            description: PR description (recommended)

        Returns:
            {pr_url, pr_number, branch_name} - review and merge the PR
        """
        client = get_client()
        return client.propose_release_config(
            machine=machine,
            port=port,
            template_name=template_name,
            model_name=model_name,
            model_path=model_path,
            gpu_ids=gpu_ids,
            tensor_parallel=tensor_parallel,
            description=description,
        )
