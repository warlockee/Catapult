"""
Tests for the BenchmarkRepository.

Tests cover:
- Basic CRUD operations
- Query methods (list_by_deployment, list_by_production_endpoint)
- State transitions (update_progress, mark_completed, mark_failed, mark_cancelled)
- Exception handling

Run with: pytest backend/tests/test_benchmark_repository.py -v
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.core.exceptions import (
    BenchmarkNotFoundError,
    BenchmarkNotCancellableError,
    BenchmarkNotUpdatableError,
)


class TestBenchmarkRepositoryGetById:
    """Tests for get_by_id and get_by_id_or_raise methods."""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_benchmark(self):
        """Test get_by_id returns benchmark when found."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.id = benchmark_id

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.get_by_id(benchmark_id)

        assert result == mock_benchmark

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(self):
        """Test get_by_id returns None when not found."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.get_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_or_raise_returns_benchmark(self):
        """Test get_by_id_or_raise returns benchmark when found."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.id = benchmark_id

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.get_by_id_or_raise(benchmark_id)

        assert result == mock_benchmark

    @pytest.mark.asyncio
    async def test_get_by_id_or_raise_raises_when_not_found(self):
        """Test get_by_id_or_raise raises BenchmarkNotFoundError when not found."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)

        with pytest.raises(BenchmarkNotFoundError) as exc_info:
            await repo.get_by_id_or_raise(benchmark_id)

        assert str(benchmark_id) in str(exc_info.value)


class TestBenchmarkRepositoryListMethods:
    """Tests for list methods."""

    @pytest.mark.asyncio
    async def test_list_by_deployment(self):
        """Test list_by_deployment returns benchmarks for deployment."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        deployment_id = uuid4()
        mock_benchmarks = [MagicMock(), MagicMock()]

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_benchmarks
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.list_by_deployment(deployment_id, limit=10)

        assert result == mock_benchmarks
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_by_production_endpoint(self):
        """Test list_by_production_endpoint returns benchmarks for endpoint."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        endpoint_id = 123
        mock_benchmarks = [MagicMock(), MagicMock()]

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_benchmarks
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.list_by_production_endpoint(endpoint_id, limit=10)

        assert result == mock_benchmarks

    @pytest.mark.asyncio
    async def test_get_latest_by_deployment(self):
        """Test get_latest_by_deployment returns latest benchmark."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        deployment_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.get_latest_by_deployment(deployment_id)

        assert result == mock_benchmark

    @pytest.mark.asyncio
    async def test_get_latest_by_production_endpoint(self):
        """Test get_latest_by_production_endpoint returns latest benchmark."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        endpoint_id = 456
        mock_benchmark = MagicMock()
        mock_benchmark.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.get_latest_by_production_endpoint(endpoint_id)

        assert result == mock_benchmark


class TestBenchmarkRepositoryUpdateProgress:
    """Tests for update_progress method."""

    @pytest.mark.asyncio
    async def test_update_progress_success(self):
        """Test update_progress updates benchmark progress."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "running"
        mock_benchmark.current_stage = None
        mock_benchmark.stage_progress = None

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        result = await repo.update_progress(benchmark_id, "inference_test", "3/5")

        assert mock_benchmark.current_stage == "inference_test"
        assert mock_benchmark.stage_progress == "3/5"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_progress_starts_pending_benchmark(self):
        """Test update_progress starts a pending benchmark."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "pending"
        mock_benchmark.started_at = None

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        await repo.update_progress(benchmark_id, "starting", None)

        assert mock_benchmark.status == "running"
        assert mock_benchmark.started_at is not None

    @pytest.mark.asyncio
    async def test_update_progress_raises_when_not_updatable(self):
        """Test update_progress raises error for completed benchmark."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)

        with pytest.raises(BenchmarkNotUpdatableError):
            await repo.update_progress(benchmark_id, "stage", None)


class TestBenchmarkRepositoryStageComplete:
    """Tests for update_stage_complete method."""

    @pytest.mark.asyncio
    async def test_update_stage_complete_adds_stage(self):
        """Test update_stage_complete adds stage to completed list."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "running"
        mock_benchmark.stages_completed = []
        mock_benchmark.stage_progress = "3/5"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        await repo.update_stage_complete(
            benchmark_id,
            "health_check",
            success=True,
            data={"latency": 100},
        )

        assert len(mock_benchmark.stages_completed) == 1
        assert mock_benchmark.stages_completed[0]["stage"] == "health_check"
        assert mock_benchmark.stages_completed[0]["success"] is True
        assert mock_benchmark.stages_completed[0]["latency"] == 100
        assert mock_benchmark.stage_progress is None
        mock_db.commit.assert_called_once()


class TestBenchmarkRepositoryMarkCompleted:
    """Tests for mark_completed method."""

    @pytest.mark.asyncio
    async def test_mark_completed_success(self):
        """Test mark_completed updates all metrics."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "running"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        results = {
            "latency_avg_ms": 150.5,
            "latency_p99_ms": 300.0,
            "requests_per_second": 100.0,
            "error_rate": 0.01,
            "model_id": "test-model",
        }
        await repo.mark_completed(benchmark_id, results, success=True)

        assert mock_benchmark.status == "completed"
        assert mock_benchmark.current_stage == "completed"
        assert mock_benchmark.latency_avg_ms == 150.5
        assert mock_benchmark.latency_p99_ms == 300.0
        assert mock_benchmark.requests_per_second == 100.0
        assert mock_benchmark.error_rate == 0.01
        assert mock_benchmark.model_id == "test-model"
        assert mock_benchmark.completed_at is not None
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_completed_failure(self):
        """Test mark_completed with failure sets error message."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "running"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        results = {"error_message": "Connection timeout"}
        await repo.mark_completed(benchmark_id, results, success=False)

        assert mock_benchmark.status == "failed"
        assert mock_benchmark.error_message == "Connection timeout"


class TestBenchmarkRepositoryMarkFailed:
    """Tests for mark_failed method."""

    @pytest.mark.asyncio
    async def test_mark_failed_sets_error(self):
        """Test mark_failed sets error status and message."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "running"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        await repo.mark_failed(benchmark_id, "Network error", "inference_test")

        assert mock_benchmark.status == "failed"
        assert mock_benchmark.error_message == "Network error"
        assert mock_benchmark.current_stage == "inference_test"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_failed_ignores_completed_benchmark(self):
        """Test mark_failed doesn't change completed benchmarks."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        await repo.mark_failed(benchmark_id, "Late error")

        assert mock_benchmark.status == "completed"  # Not changed
        mock_db.commit.assert_not_called()


class TestBenchmarkRepositoryMarkCancelled:
    """Tests for mark_cancelled method."""

    @pytest.mark.asyncio
    async def test_mark_cancelled_success(self):
        """Test mark_cancelled cancels benchmark."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "running"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)
        await repo.mark_cancelled(benchmark_id)

        assert mock_benchmark.status == "cancelled"
        assert mock_benchmark.current_stage == "cancelled"
        assert mock_benchmark.error_message == "Cancelled by user"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_cancelled_raises_for_completed(self):
        """Test mark_cancelled raises for already completed benchmark."""
        from app.repositories.benchmark_repository import BenchmarkRepository

        benchmark_id = uuid4()
        mock_benchmark = MagicMock()
        mock_benchmark.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_benchmark
        mock_db.execute.return_value = mock_result

        repo = BenchmarkRepository(mock_db)

        with pytest.raises(BenchmarkNotCancellableError):
            await repo.mark_cancelled(benchmark_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
