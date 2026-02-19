"""
Tests for the DeploymentTask Celery base class.

Tests cover:
- on_failure hook marking deployments as failed
- Status filtering (only certain statuses get marked as failed)
- Error handling and logging

Run with: pytest backend/tests/test_deployment_task.py -v
"""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Skip all tests if celery is not installed
celery = pytest.importorskip("celery")


class TestDeploymentTaskOnFailure:
    """Tests for DeploymentTask on_failure hook."""

    def test_on_failure_marks_deployment_failed(self):
        """Test that on_failure marks deployment as failed."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()
        deployment_id = str(uuid4())

        with patch.object(task, '_mark_deployment_failed', return_value=True) as mock_mark:
            task.on_failure(
                exc=Exception("Test error"),
                task_id="test-task-123",
                args=[deployment_id],
                kwargs={},
                einfo=None
            )

            mock_mark.assert_called_once_with(deployment_id, "Test error")

    def test_on_failure_handles_no_deployment_id(self):
        """Test that on_failure handles missing deployment_id gracefully."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()

        with patch.object(task, '_mark_deployment_failed') as mock_mark:
            # No args provided
            task.on_failure(
                exc=Exception("Test error"),
                task_id="test-task-123",
                args=[],
                kwargs={},
                einfo=None
            )

            # Should not call _mark_deployment_failed when no deployment_id
            mock_mark.assert_not_called()

    def test_on_failure_with_kwargs_deployment_id(self):
        """Test on_failure when deployment_id is in kwargs (not args)."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()
        deployment_id = str(uuid4())

        with patch.object(task, '_mark_deployment_failed') as mock_mark:
            # deployment_id only in kwargs, not in args
            task.on_failure(
                exc=Exception("Test error"),
                task_id="test-task-123",
                args=[],  # Empty args
                kwargs={"deployment_id": deployment_id},
                einfo=None
            )

            # Current implementation only checks args[0], not kwargs
            mock_mark.assert_not_called()


class TestDeploymentTaskMarkFailed:
    """Tests for DeploymentTask._mark_deployment_failed method."""

    @pytest.mark.asyncio
    async def test_mark_deployment_failed_updates_status(self):
        """Test that _mark_deployment_failed updates deployment status in DB."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()
        deployment_id = str(uuid4())

        # Create mock deployment
        mock_deployment = MagicMock()
        mock_deployment.status = "deploying"
        mock_deployment.health_status = "healthy"
        mock_deployment.host_port = 8080
        mock_deployment.meta_data = {}

        # Create mock DB session
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_deployment
        mock_db.execute.return_value = mock_result

        with patch('app.core.celery_app.run_async_with_db') as mock_run:
            # Capture the async function passed to run_async_with_db
            def capture_and_run(func):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(func(mock_db))
                finally:
                    loop.close()

            mock_run.side_effect = capture_and_run

            result = task._mark_deployment_failed(deployment_id, "Test error")

            # Verify deployment was updated
            assert mock_deployment.status == "failed"
            assert mock_deployment.health_status == "unhealthy"
            assert mock_deployment.host_port is None
            assert mock_deployment.meta_data["error"] == "Test error"
            mock_db.commit.assert_called_once()

    def test_mark_deployment_failed_only_updates_certain_statuses(self):
        """Test that _mark_deployment_failed only updates pending/deploying/stopping statuses."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()
        deployment_id = str(uuid4())

        # Create mock deployment that's already running (should not be marked failed)
        mock_deployment = MagicMock()
        mock_deployment.status = "running"  # Not in fail_on_statuses
        mock_deployment.health_status = "healthy"
        mock_deployment.host_port = 8080
        mock_deployment.meta_data = {}

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_deployment
        mock_db.execute.return_value = mock_result

        with patch('app.core.celery_app.run_async_with_db') as mock_run:
            def capture_and_run(func):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(func(mock_db))
                finally:
                    loop.close()

            mock_run.side_effect = capture_and_run

            result = task._mark_deployment_failed(deployment_id, "Test error")

            # Verify deployment was NOT updated (status should still be "running")
            assert mock_deployment.status == "running"
            assert mock_deployment.health_status == "healthy"
            mock_db.commit.assert_not_called()

    def test_mark_deployment_failed_handles_not_found(self):
        """Test that _mark_deployment_failed handles deployment not found."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()
        deployment_id = str(uuid4())

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # Not found
        mock_db.execute.return_value = mock_result

        with patch('app.core.celery_app.run_async_with_db') as mock_run:
            def capture_and_run(func):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(func(mock_db))
                finally:
                    loop.close()

            mock_run.side_effect = capture_and_run

            result = task._mark_deployment_failed(deployment_id, "Test error")

            # Should return False when deployment not found
            assert result is False
            mock_db.commit.assert_not_called()

    def test_mark_deployment_failed_handles_exception(self):
        """Test that _mark_deployment_failed handles exceptions gracefully."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()
        deployment_id = str(uuid4())

        with patch('app.core.celery_app.run_async_with_db') as mock_run:
            mock_run.side_effect = Exception("Database connection failed")

            # Should not raise, should return False
            result = task._mark_deployment_failed(deployment_id, "Test error")

            assert result is False


class TestDeploymentTaskFailOnStatuses:
    """Tests for DeploymentTask fail_on_statuses configuration."""

    def test_default_fail_on_statuses(self):
        """Test default fail_on_statuses values."""
        from app.core.celery_app import DeploymentTask

        task = DeploymentTask()

        assert "pending" in task.fail_on_statuses
        assert "deploying" in task.fail_on_statuses
        assert "stopping" in task.fail_on_statuses
        assert "running" not in task.fail_on_statuses
        assert "failed" not in task.fail_on_statuses

    def test_custom_fail_on_statuses(self):
        """Test that subclasses can override fail_on_statuses."""
        from app.core.celery_app import DeploymentTask

        class CustomTask(DeploymentTask):
            fail_on_statuses = ("custom_status",)

        task = CustomTask()

        assert task.fail_on_statuses == ("custom_status",)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
