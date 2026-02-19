"""
Tests for the EvaluationRepository.

Tests cover:
- Basic CRUD operations
- Query methods (list_by_deployment, list_by_production_endpoint)
- State transitions (mark_cancelled, mark_failed)
- Exception handling

Run with: pytest backend/tests/test_evaluation_repository.py -v
"""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import (
    EvaluationNotCancellableError,
    EvaluationNotFoundError,
)


class TestEvaluationRepositoryGetById:
    """Tests for get_by_id and get_by_id_or_raise methods."""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_evaluation(self):
        """Test get_by_id returns evaluation when found."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.id = evaluation_id

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_by_id(evaluation_id)

        assert result == mock_evaluation

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(self):
        """Test get_by_id returns None when not found."""
        from app.repositories.evaluation_repository import EvaluationRepository

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_or_raise_returns_evaluation(self):
        """Test get_by_id_or_raise returns evaluation when found."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.id = evaluation_id

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_by_id_or_raise(evaluation_id)

        assert result == mock_evaluation

    @pytest.mark.asyncio
    async def test_get_by_id_or_raise_raises_when_not_found(self):
        """Test get_by_id_or_raise raises EvaluationNotFoundError when not found."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)

        with pytest.raises(EvaluationNotFoundError) as exc_info:
            await repo.get_by_id_or_raise(evaluation_id)

        assert str(evaluation_id) in str(exc_info.value)


class TestEvaluationRepositoryListMethods:
    """Tests for list methods."""

    @pytest.mark.asyncio
    async def test_list_by_deployment(self):
        """Test list_by_deployment returns evaluations for deployment."""
        from app.repositories.evaluation_repository import EvaluationRepository

        deployment_id = uuid4()
        mock_evaluations = [MagicMock(), MagicMock()]

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_evaluations
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.list_by_deployment(deployment_id, limit=10)

        assert result == mock_evaluations
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_by_production_endpoint(self):
        """Test list_by_production_endpoint returns evaluations for endpoint."""
        from app.repositories.evaluation_repository import EvaluationRepository

        endpoint_id = 123
        mock_evaluations = [MagicMock(), MagicMock()]

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_evaluations
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.list_by_production_endpoint(endpoint_id, limit=10)

        assert result == mock_evaluations


class TestEvaluationRepositoryGetLatest:
    """Tests for get_latest methods."""

    @pytest.mark.asyncio
    async def test_get_latest_completed_by_deployment(self):
        """Test get_latest_completed_by_deployment returns latest evaluation."""
        from app.repositories.evaluation_repository import EvaluationRepository

        deployment_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "completed"
        mock_evaluation.wer = 0.15

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_latest_completed_by_deployment(deployment_id)

        assert result == mock_evaluation

    @pytest.mark.asyncio
    async def test_get_latest_completed_by_deployment_returns_none(self):
        """Test get_latest_completed_by_deployment returns None when no evaluation."""
        from app.repositories.evaluation_repository import EvaluationRepository

        deployment_id = uuid4()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_latest_completed_by_deployment(deployment_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_completed_by_production_endpoint(self):
        """Test get_latest_completed_by_production_endpoint returns latest evaluation."""
        from app.repositories.evaluation_repository import EvaluationRepository

        endpoint_id = 456
        mock_evaluation = MagicMock()
        mock_evaluation.status = "completed"
        mock_evaluation.wer = 0.12

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_latest_completed_by_production_endpoint(endpoint_id)

        assert result == mock_evaluation

    @pytest.mark.asyncio
    async def test_get_latest_completed_without_wer_filter(self):
        """Test get_latest_completed can skip WER filter."""
        from app.repositories.evaluation_repository import EvaluationRepository

        deployment_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "completed"
        mock_evaluation.wer = None  # No WER data

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        result = await repo.get_latest_completed_by_deployment(
            deployment_id, with_wer=False
        )

        assert result == mock_evaluation


class TestEvaluationRepositoryMarkCancelled:
    """Tests for mark_cancelled method."""

    @pytest.mark.asyncio
    async def test_mark_cancelled_success(self):
        """Test mark_cancelled cancels pending evaluation."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "pending"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        await repo.mark_cancelled(evaluation_id)

        assert mock_evaluation.status == "cancelled"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_cancelled_running_evaluation(self):
        """Test mark_cancelled cancels running evaluation."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "running"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        await repo.mark_cancelled(evaluation_id)

        assert mock_evaluation.status == "cancelled"

    @pytest.mark.asyncio
    async def test_mark_cancelled_raises_for_completed(self):
        """Test mark_cancelled raises for completed evaluation."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)

        with pytest.raises(EvaluationNotCancellableError):
            await repo.mark_cancelled(evaluation_id)

    @pytest.mark.asyncio
    async def test_mark_cancelled_raises_when_not_found(self):
        """Test mark_cancelled raises when evaluation not found."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)

        with pytest.raises(EvaluationNotFoundError):
            await repo.mark_cancelled(evaluation_id)


class TestEvaluationRepositoryMarkFailed:
    """Tests for mark_failed method."""

    @pytest.mark.asyncio
    async def test_mark_failed_sets_error(self):
        """Test mark_failed sets error status and message."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "running"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        await repo.mark_failed(evaluation_id, "Dataset not found")

        assert mock_evaluation.status == "failed"
        assert mock_evaluation.error_message == "Dataset not found"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_failed_ignores_completed(self):
        """Test mark_failed doesn't change completed evaluations."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()
        mock_evaluation = MagicMock()
        mock_evaluation.status = "completed"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_evaluation
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)
        await repo.mark_failed(evaluation_id, "Late error")

        assert mock_evaluation.status == "completed"  # Not changed
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_mark_failed_raises_when_not_found(self):
        """Test mark_failed raises when evaluation not found."""
        from app.repositories.evaluation_repository import EvaluationRepository

        evaluation_id = uuid4()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        repo = EvaluationRepository(mock_db)

        with pytest.raises(EvaluationNotFoundError):
            await repo.mark_failed(evaluation_id, "Error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
