"""
Repository for Benchmark entity database operations.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository
from app.models.benchmark import Benchmark
from app.core.exceptions import (
    BenchmarkNotFoundError,
    BenchmarkNotCancellableError,
    BenchmarkNotUpdatableError,
)


class BenchmarkRepository(BaseRepository[Benchmark]):
    """Repository for Benchmark database operations."""

    model = Benchmark

    async def get_by_id_or_raise(self, id: UUID) -> Benchmark:
        """
        Get a benchmark by ID, raising exception if not found.

        Args:
            id: Benchmark UUID

        Returns:
            Benchmark entity

        Raises:
            BenchmarkNotFoundError: If benchmark not found
        """
        benchmark = await self.get_by_id(id)
        if not benchmark:
            raise BenchmarkNotFoundError(str(id))
        return benchmark

    async def list_by_deployment(
        self,
        deployment_id: UUID,
        limit: int = 10,
    ) -> List[Benchmark]:
        """
        List benchmarks for a deployment, ordered by creation date desc.

        Args:
            deployment_id: Deployment UUID
            limit: Maximum number of results

        Returns:
            List of benchmarks
        """
        result = await self.db.execute(
            select(Benchmark)
            .where(Benchmark.deployment_id == deployment_id)
            .order_by(desc(Benchmark.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_production_endpoint(
        self,
        endpoint_id: int,
        limit: int = 10,
    ) -> List[Benchmark]:
        """
        List benchmarks for a production endpoint, ordered by creation date desc.

        Args:
            endpoint_id: Production endpoint ID
            limit: Maximum number of results

        Returns:
            List of benchmarks
        """
        result = await self.db.execute(
            select(Benchmark)
            .where(Benchmark.production_endpoint_id == endpoint_id)
            .order_by(desc(Benchmark.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_latest_by_deployment(
        self,
        deployment_id: UUID,
        status: Optional[str] = "completed",
    ) -> Optional[Benchmark]:
        """
        Get the latest benchmark for a deployment.

        Args:
            deployment_id: Deployment UUID
            status: Filter by status (default: "completed")

        Returns:
            Latest benchmark or None
        """
        query = (
            select(Benchmark)
            .where(Benchmark.deployment_id == deployment_id)
            .order_by(desc(Benchmark.created_at))
            .limit(1)
        )
        if status:
            query = query.where(Benchmark.status == status)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_latest_by_production_endpoint(
        self,
        endpoint_id: int,
        status: Optional[str] = "completed",
    ) -> Optional[Benchmark]:
        """
        Get the latest benchmark for a production endpoint.

        Args:
            endpoint_id: Production endpoint ID
            status: Filter by status (default: "completed")

        Returns:
            Latest benchmark or None
        """
        query = (
            select(Benchmark)
            .where(Benchmark.production_endpoint_id == endpoint_id)
            .order_by(desc(Benchmark.created_at))
            .limit(1)
        )
        if status:
            query = query.where(Benchmark.status == status)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_progress(
        self,
        benchmark_id: UUID,
        stage: str,
        progress: Optional[str] = None,
    ) -> Benchmark:
        """
        Update benchmark progress.

        Args:
            benchmark_id: Benchmark UUID
            stage: Current stage name
            progress: Progress within stage (e.g., "3/5")

        Returns:
            Updated benchmark

        Raises:
            BenchmarkNotFoundError: If benchmark not found
            BenchmarkNotUpdatableError: If benchmark is not in updatable state
        """
        benchmark = await self.get_by_id_or_raise(benchmark_id)

        if benchmark.status not in ("pending", "running"):
            raise BenchmarkNotUpdatableError(str(benchmark_id), benchmark.status)

        benchmark.current_stage = stage
        benchmark.stage_progress = progress

        if benchmark.status == "pending":
            benchmark.status = "running"
            benchmark.started_at = datetime.utcnow()

        await self.db.commit()
        return benchmark

    async def update_stage_complete(
        self,
        benchmark_id: UUID,
        stage: str,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
    ) -> Benchmark:
        """
        Mark a benchmark stage as complete.

        Args:
            benchmark_id: Benchmark UUID
            stage: Stage name that completed
            success: Whether the stage succeeded
            data: Additional stage data

        Returns:
            Updated benchmark

        Raises:
            BenchmarkNotFoundError: If benchmark not found
            BenchmarkNotUpdatableError: If benchmark is not in updatable state
        """
        benchmark = await self.get_by_id_or_raise(benchmark_id)

        if benchmark.status not in ("pending", "running"):
            raise BenchmarkNotUpdatableError(str(benchmark_id), benchmark.status)

        stages = list(benchmark.stages_completed or [])
        stage_record = {
            "stage": stage,
            "success": success,
            **(data or {}),
        }
        stages.append(stage_record)
        benchmark.stages_completed = stages
        benchmark.stage_progress = None

        await self.db.commit()
        return benchmark

    async def mark_completed(
        self,
        benchmark_id: UUID,
        results: Dict[str, Any],
        success: bool = True,
    ) -> Benchmark:
        """
        Mark benchmark as completed with results.

        Args:
            benchmark_id: Benchmark UUID
            results: Final benchmark results
            success: Whether the benchmark succeeded

        Returns:
            Updated benchmark

        Raises:
            BenchmarkNotFoundError: If benchmark not found
            BenchmarkNotUpdatableError: If benchmark is not in updatable state
        """
        benchmark = await self.get_by_id_or_raise(benchmark_id)

        if benchmark.status not in ("pending", "running"):
            raise BenchmarkNotUpdatableError(str(benchmark_id), benchmark.status)

        benchmark.status = "completed" if success else "failed"
        benchmark.current_stage = "completed" if success else "failed"
        benchmark.stage_progress = None
        benchmark.completed_at = datetime.utcnow()

        if not success and results.get("error_message"):
            benchmark.error_message = results["error_message"]

        # Update latency metrics
        benchmark.latency_avg_ms = results.get("latency_avg_ms")
        benchmark.latency_min_ms = results.get("latency_min_ms")
        benchmark.latency_max_ms = results.get("latency_max_ms")
        benchmark.latency_p50_ms = results.get("latency_p50_ms")
        benchmark.latency_p90_ms = results.get("latency_p90_ms")
        benchmark.latency_p95_ms = results.get("latency_p95_ms")
        benchmark.latency_p99_ms = results.get("latency_p99_ms")

        # Update throughput metrics
        benchmark.requests_per_second = results.get("requests_per_second")
        benchmark.total_requests_sent = results.get("total_requests_sent")
        benchmark.successful_requests = results.get("successful_requests")
        benchmark.failed_requests = results.get("failed_requests")
        benchmark.error_rate = results.get("error_rate")

        # Update inference metrics (TTFT/TPS)
        benchmark.ttft_avg_ms = results.get("ttft_avg_ms")
        benchmark.ttft_min_ms = results.get("ttft_min_ms")
        benchmark.ttft_max_ms = results.get("ttft_max_ms")
        benchmark.ttft_p50_ms = results.get("ttft_p50_ms")
        benchmark.ttft_p90_ms = results.get("ttft_p90_ms")
        benchmark.ttft_p95_ms = results.get("ttft_p95_ms")
        benchmark.ttft_p99_ms = results.get("ttft_p99_ms")
        benchmark.tokens_per_second_avg = results.get("tokens_per_second_avg")
        benchmark.tokens_per_second_min = results.get("tokens_per_second_min")
        benchmark.tokens_per_second_max = results.get("tokens_per_second_max")
        benchmark.total_tokens_generated = results.get("total_tokens_generated")

        # Update model ID
        benchmark.model_id = results.get("model_id")

        # Update timing
        benchmark.duration_seconds = results.get("duration_seconds")
        if results.get("started_at"):
            try:
                benchmark.started_at = datetime.fromisoformat(
                    results["started_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        await self.db.commit()
        return benchmark

    async def mark_failed(
        self,
        benchmark_id: UUID,
        error_message: str,
        stage: Optional[str] = None,
    ) -> Benchmark:
        """
        Mark benchmark as failed with error.

        Args:
            benchmark_id: Benchmark UUID
            error_message: Error description
            stage: Stage where error occurred

        Returns:
            Updated benchmark

        Raises:
            BenchmarkNotFoundError: If benchmark not found
        """
        benchmark = await self.get_by_id_or_raise(benchmark_id)

        # Allow marking as failed even if already completed (for late errors)
        if benchmark.status in ("pending", "running"):
            benchmark.status = "failed"
            benchmark.current_stage = stage or "failed"
            benchmark.stage_progress = None
            benchmark.error_message = error_message
            benchmark.completed_at = datetime.utcnow()
            await self.db.commit()

        return benchmark

    async def mark_cancelled(
        self,
        benchmark_id: UUID,
    ) -> Benchmark:
        """
        Cancel a benchmark.

        Args:
            benchmark_id: Benchmark UUID

        Returns:
            Updated benchmark

        Raises:
            BenchmarkNotFoundError: If benchmark not found
            BenchmarkNotCancellableError: If benchmark cannot be cancelled
        """
        benchmark = await self.get_by_id_or_raise(benchmark_id)

        if benchmark.status not in ("pending", "running"):
            raise BenchmarkNotCancellableError(str(benchmark_id), benchmark.status)

        benchmark.status = "cancelled"
        benchmark.current_stage = "cancelled"
        benchmark.error_message = "Cancelled by user"
        benchmark.completed_at = datetime.utcnow()

        await self.db.commit()
        return benchmark
