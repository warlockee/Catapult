"""
API endpoints for deployment benchmarks.
"""
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import InvalidBenchmarkConfigError
from app.core.security import require_operator, verify_api_key
from app.models.api_key import ApiKey
from app.repositories.benchmark_repository import BenchmarkRepository
from app.schemas.benchmark import (
    BenchmarkCreate,
    BenchmarkResponse,
    BenchmarkSummary,
)
from app.services.benchmark_service import benchmark_service
from app.services.task_dispatcher import task_dispatcher

router = APIRouter()


@router.post("", response_model=BenchmarkResponse, status_code=status.HTTP_201_CREATED)
async def create_and_run_benchmark(
    benchmark_data: BenchmarkCreate,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> BenchmarkResponse:
    """
    Create and run a benchmark for a deployment or production endpoint.

    This endpoint creates a benchmark record and immediately executes it.
    For long-running benchmarks, consider using the async endpoint instead.

    Provide either deployment_id (for local deployments) or endpoint_url (for production endpoints).
    """
    headers = benchmark_data.headers or {}

    # Determine which type of benchmark to create
    if benchmark_data.endpoint_url:
        # Production endpoint benchmark
        benchmark = await benchmark_service.create_production_benchmark(
            db=db,
            endpoint_url=benchmark_data.endpoint_url,
            production_endpoint_id=benchmark_data.production_endpoint_id,
            endpoint_path=benchmark_data.endpoint_path,
            method=benchmark_data.method,
            concurrent_requests=benchmark_data.concurrent_requests,
            total_requests=benchmark_data.total_requests,
            timeout_seconds=benchmark_data.timeout_seconds,
            request_body=benchmark_data.request_body,
            headers=headers,
            execution_mode=benchmark_data.execution_mode,
        )
    elif benchmark_data.deployment_id:
        # Local deployment benchmark
        benchmark = await benchmark_service.create_benchmark(
            db=db,
            deployment_id=benchmark_data.deployment_id,
            endpoint_path=benchmark_data.endpoint_path,
            method=benchmark_data.method,
            concurrent_requests=benchmark_data.concurrent_requests,
            total_requests=benchmark_data.total_requests,
            timeout_seconds=benchmark_data.timeout_seconds,
            request_body=benchmark_data.request_body,
            headers=headers,
            execution_mode=benchmark_data.execution_mode,
        )
    else:
        raise InvalidBenchmarkConfigError(
            "Either deployment_id or endpoint_url must be provided"
        )

    # Execute benchmark synchronously (inline mode for sync endpoint)
    benchmark = await benchmark_service.execute_benchmark(db, benchmark.id)

    return BenchmarkResponse.from_benchmark(benchmark)


@router.post("/async", response_model=BenchmarkResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_benchmark_async(
    benchmark_data: BenchmarkCreate,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> BenchmarkResponse:
    """
    Create a benchmark and queue it for async execution.

    Returns immediately with the pending benchmark. Poll the GET endpoint
    to check for completion.

    Provide either deployment_id (for local deployments) or endpoint_url (for production endpoints).
    """
    # Build headers
    headers = benchmark_data.headers or {}

    # Determine which type of benchmark to create
    if benchmark_data.endpoint_url:
        # Production endpoint benchmark
        benchmark = await benchmark_service.create_production_benchmark(
            db=db,
            endpoint_url=benchmark_data.endpoint_url,
            production_endpoint_id=benchmark_data.production_endpoint_id,
            endpoint_path=benchmark_data.endpoint_path,
            method=benchmark_data.method,
            concurrent_requests=benchmark_data.concurrent_requests,
            total_requests=benchmark_data.total_requests,
            timeout_seconds=benchmark_data.timeout_seconds,
            request_body=benchmark_data.request_body,
            headers=headers,
            execution_mode=benchmark_data.execution_mode,
        )
    elif benchmark_data.deployment_id:
        # Local deployment benchmark
        benchmark = await benchmark_service.create_benchmark(
            db=db,
            deployment_id=benchmark_data.deployment_id,
            endpoint_path=benchmark_data.endpoint_path,
            method=benchmark_data.method,
            concurrent_requests=benchmark_data.concurrent_requests,
            total_requests=benchmark_data.total_requests,
            timeout_seconds=benchmark_data.timeout_seconds,
            request_body=benchmark_data.request_body,
            headers=headers,
            execution_mode=benchmark_data.execution_mode,
        )
    else:
        raise InvalidBenchmarkConfigError(
            "Either deployment_id or endpoint_url must be provided"
        )

    # Queue for async execution
    task_dispatcher.dispatch_benchmark(benchmark.id)

    return BenchmarkResponse.from_benchmark(benchmark)


@router.get("/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> BenchmarkResponse:
    """Get a benchmark by ID."""
    repo = BenchmarkRepository(db)
    benchmark = await repo.get_by_id_or_raise(benchmark_id)
    return BenchmarkResponse.from_benchmark(benchmark)


@router.post("/{benchmark_id}/cancel", response_model=BenchmarkResponse)
async def cancel_benchmark(
    benchmark_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> BenchmarkResponse:
    """Cancel a running benchmark."""
    repo = BenchmarkRepository(db)
    benchmark = await repo.mark_cancelled(benchmark_id)
    return BenchmarkResponse.from_benchmark(benchmark)


@router.get("/deployment/{deployment_id}", response_model=List[BenchmarkResponse])
async def list_deployment_benchmarks(
    deployment_id: UUID,
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> List[BenchmarkResponse]:
    """List benchmarks for a deployment."""
    benchmarks = await benchmark_service.list_benchmarks(db, deployment_id, limit)
    return [BenchmarkResponse.from_benchmark(b) for b in benchmarks]


@router.get("/deployment/{deployment_id}/summary", response_model=BenchmarkSummary)
async def get_deployment_benchmark_summary(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> BenchmarkSummary:
    """Get benchmark summary for a deployment (latest completed benchmark)."""
    benchmark = await benchmark_service.get_latest_benchmark(db, deployment_id)

    if not benchmark:
        return BenchmarkSummary(has_data=False)

    return BenchmarkSummary(
        has_data=True,
        last_run_at=benchmark.completed_at,
        status=benchmark.status,
        model_id=benchmark.model_id,
        latency_avg_ms=benchmark.latency_avg_ms,
        latency_p50_ms=benchmark.latency_p50_ms,
        latency_p95_ms=benchmark.latency_p95_ms,
        latency_p99_ms=benchmark.latency_p99_ms,
        requests_per_second=benchmark.requests_per_second,
        total_requests=benchmark.total_requests_sent,
        error_rate=benchmark.error_rate,
        # Inference metrics (TTFT/TPS)
        ttft_avg_ms=benchmark.ttft_avg_ms,
        ttft_p50_ms=benchmark.ttft_p50_ms,
        ttft_p95_ms=benchmark.ttft_p95_ms,
        tokens_per_second_avg=benchmark.tokens_per_second_avg,
        total_tokens_generated=benchmark.total_tokens_generated,
    )


@router.get("/production-endpoint/{endpoint_id}/summary", response_model=BenchmarkSummary)
async def get_production_endpoint_benchmark_summary(
    endpoint_id: int,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> BenchmarkSummary:
    """Get benchmark summary for a production endpoint (latest completed benchmark)."""
    return await benchmark_service.get_production_benchmark_summary(db, endpoint_id)


@router.get("/production-endpoint/{endpoint_id}", response_model=List[BenchmarkResponse])
async def list_production_endpoint_benchmarks(
    endpoint_id: int,
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> List[BenchmarkResponse]:
    """List benchmarks for a production endpoint."""
    repo = BenchmarkRepository(db)
    benchmarks = await repo.list_by_production_endpoint(endpoint_id, limit)
    return [BenchmarkResponse.from_benchmark(b) for b in benchmarks]
