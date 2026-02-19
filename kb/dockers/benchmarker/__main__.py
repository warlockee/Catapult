#!/usr/bin/env python3
"""
CLI entry point for the benchmarker.

Usage:
    python -m benchmarker \
        --endpoint-url "http://10.0.0.1:8000" \
        --callback-url "http://backend:8000/internal/benchmarks" \
        --benchmark-id "abc123" \
        --config '{"concurrent_requests": 5, "total_requests": 20}'
"""
import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

from .metrics import BenchmarkConfig
from .progress import NoOpProgressReporter, ProgressReporter
from .runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmarks against ML model endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic benchmark with HTTP callbacks
    python -m benchmarker \\
        --endpoint-url "http://10.0.0.1:8000" \\
        --callback-url "http://backend:8000/internal/benchmarks" \\
        --benchmark-id "abc123"

    # Benchmark with custom configuration
    python -m benchmarker \\
        --endpoint-url "http://10.0.0.1:8000" \\
        --callback-url "http://backend:8000/internal/benchmarks" \\
        --benchmark-id "abc123" \\
        --endpoint-path "/v1/chat/completions" \\
        --concurrent-requests 5 \\
        --total-requests 20 \\
        --server-type "vllm"

    # Benchmark without callbacks (for testing)
    python -m benchmarker \\
        --endpoint-url "http://10.0.0.1:8000" \\
        --no-callback
        """,
    )

    parser.add_argument(
        "--endpoint-url",
        required=True,
        help="Base URL of the deployment to benchmark (e.g., http://10.0.0.1:8000)",
    )
    parser.add_argument(
        "--callback-url",
        help="Base URL for progress callbacks (e.g., http://backend:8000/internal/benchmarks)",
    )
    parser.add_argument(
        "--benchmark-id",
        help="ID of the benchmark for callback reporting",
    )
    parser.add_argument(
        "--no-callback",
        action="store_true",
        help="Disable HTTP callbacks (logs progress to stdout instead)",
    )
    parser.add_argument(
        "--endpoint-path",
        default="/v1/chat/completions",
        help="API endpoint path to benchmark (default: /v1/chat/completions)",
    )
    parser.add_argument(
        "--method",
        default="POST",
        help="HTTP method (default: POST)",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent requests for stress test (default: 5)",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=20,
        help="Total number of requests for stress test (default: 20)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout per request in seconds (default: 60)",
    )
    parser.add_argument(
        "--server-type",
        help="Server type hint (vllm, audio, asr, embedding, etc.)",
    )
    parser.add_argument(
        "--request-body",
        help="JSON request body for POST requests",
    )
    parser.add_argument(
        "--config",
        help="JSON configuration object (overrides individual options)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Build benchmark configuration from CLI arguments."""
    # Start with defaults
    config_dict = {
        "endpoint_url": args.endpoint_url,
        "endpoint_path": args.endpoint_path,
        "method": args.method,
        "concurrent_requests": args.concurrent_requests,
        "total_requests": args.total_requests,
        "timeout_seconds": args.timeout_seconds,
        "server_type": args.server_type,
        "request_body": None,
        "headers": {},
    }

    # Parse request body if provided
    if args.request_body:
        try:
            config_dict["request_body"] = json.loads(args.request_body)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --request-body: {e}")
            sys.exit(1)

    # Override with config JSON if provided
    if args.config:
        try:
            config_override = json.loads(args.config)
            config_dict.update(config_override)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --config: {e}")
            sys.exit(1)

    return BenchmarkConfig(**config_dict)


async def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate callback arguments
    if not args.no_callback and (not args.callback_url or not args.benchmark_id):
        logger.error("--callback-url and --benchmark-id are required unless --no-callback is set")
        return 1

    # Build configuration
    config = build_config(args)

    logger.info(f"Starting benchmark for {config.endpoint_url}")
    logger.info(f"  Endpoint path: {config.endpoint_path}")
    logger.info(f"  Server type: {config.server_type or 'auto-detect'}")
    logger.info(f"  Concurrent requests: {config.concurrent_requests}")
    logger.info(f"  Total requests: {config.total_requests}")
    logger.info(f"  Timeout: {config.timeout_seconds}s")

    # Create progress reporter
    if args.no_callback:
        progress = NoOpProgressReporter()
    else:
        progress = ProgressReporter(
            callback_base_url=args.callback_url,
            benchmark_id=args.benchmark_id,
        )

    # Run benchmark
    try:
        async with progress:
            runner = BenchmarkRunner(progress_reporter=progress)
            result = await runner.run(config)

        if result.success:
            logger.info("Benchmark completed successfully")
            logger.info(f"  TTFT avg: {result.ttft_avg_ms:.2f}ms")
            logger.info(f"  TPS avg: {result.tokens_per_second_avg:.2f}")
            logger.info(f"  Latency avg: {result.latency_avg_ms:.2f}ms")
            logger.info(f"  RPS: {result.requests_per_second:.2f}")
            logger.info(f"  Duration: {result.duration_seconds:.2f}s")
            return 0
        else:
            logger.error(f"Benchmark failed: {result.error_message}")
            return 1

    except Exception as e:
        logger.exception(f"Benchmark crashed: {e}")
        if not args.no_callback:
            async with ProgressReporter(args.callback_url, args.benchmark_id) as p:
                await p.report_error(str(e))
        return 2


def run():
    """Entry point for the CLI."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
