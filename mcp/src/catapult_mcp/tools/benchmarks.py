"""Benchmark tools for MCP."""

from mcp.server.fastmcp import FastMCP

from ..client import get_client


def register_benchmark_tools(mcp: FastMCP) -> None:
    """Register benchmark-related tools."""

    @mcp.tool()
    def list_benchmarks(
        deployment_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List benchmark results.

        Args:
            deployment_id: Filter by deployment UUID (required for now)
            limit: Maximum results (default 20)

        Returns:
            List of benchmarks with: id, deployment_id, status,
            latency_p50_ms, latency_p99_ms, requests_per_second,
            error_rate, completed_at
        """
        client = get_client()
        return client.list_benchmarks(deployment_id=deployment_id, limit=limit)

    @mcp.tool()
    def get_benchmark(benchmark_id: str) -> dict:
        """Get detailed benchmark results.

        Args:
            benchmark_id: Benchmark UUID

        Returns:
            Full metrics: latency percentiles (p50, p90, p95, p99),
            throughput (requests_per_second), error analysis,
            request/response details, duration
        """
        client = get_client()
        return client.get_benchmark(benchmark_id)

    @mcp.tool()
    def run_benchmark(
        deployment_id: str,
        endpoint_path: str = "/v1/chat/completions",
        concurrent_requests: int = 10,
        total_requests: int = 100,
        request_body: dict | None = None,
    ) -> dict:
        """Run performance benchmark on a deployment.

        This is a WRITE operation that sends real traffic.

        Args:
            deployment_id: Deployment UUID to benchmark
            endpoint_path: API endpoint to test (default "/v1/chat/completions")
            concurrent_requests: Parallel request count (default 10)
            total_requests: Total requests to send (default 100)
            request_body: Custom request payload (uses default if None)

        Returns:
            {benchmark_id, status: "pending"} - poll get_benchmark for results
        """
        client = get_client()
        return client.run_benchmark(
            deployment_id=deployment_id,
            endpoint_path=endpoint_path,
            concurrent_requests=concurrent_requests,
            total_requests=total_requests,
            request_body=request_body,
        )

    @mcp.tool()
    def get_benchmark_summary(deployment_id: str) -> dict:
        """Get latest benchmark summary for a deployment.

        Args:
            deployment_id: Deployment UUID

        Returns:
            {has_data: bool, latency_p50_ms, latency_p99_ms,
            requests_per_second, error_rate, last_run_at}
        """
        client = get_client()
        return client.get_benchmark_summary(deployment_id)
