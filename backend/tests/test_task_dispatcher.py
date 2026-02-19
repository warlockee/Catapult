"""
Tests for the TaskDispatcher service.

Tests cover:
- CeleryTaskDispatcher methods
- NoOpTaskDispatcher methods
- Dispatcher selection based on environment

Run with: pytest backend/tests/test_task_dispatcher.py -v
"""
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


class TestNoOpTaskDispatcher:
    """Tests for NoOpTaskDispatcher (always available, no dependencies)."""

    def test_dispatch_deployment_returns_task_id(self):
        """Test dispatch_deployment returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()
        deployment_id = uuid4()

        result = dispatcher.dispatch_deployment(deployment_id, '{"config": "test"}')

        assert result == f"noop-deployment-{deployment_id}"

    def test_dispatch_stop_returns_task_id(self):
        """Test dispatch_stop returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()
        deployment_id = uuid4()

        result = dispatcher.dispatch_stop(deployment_id)

        assert result == f"noop-stop-{deployment_id}"

    def test_dispatch_restart_returns_task_id(self):
        """Test dispatch_restart returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()
        deployment_id = uuid4()

        result = dispatcher.dispatch_restart(deployment_id, '{"config": "test"}')

        assert result == f"noop-restart-{deployment_id}"

    def test_dispatch_docker_build_returns_task_id(self):
        """Test dispatch_docker_build returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()
        build_id = uuid4()

        result = dispatcher.dispatch_docker_build(build_id)

        assert result == f"noop-build-{build_id}"

    def test_dispatch_mlflow_sync_returns_task_id(self):
        """Test dispatch_mlflow_sync returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()
        version_id = uuid4()

        result = dispatcher.dispatch_mlflow_sync(version_id)

        assert result == f"noop-mlflow-{version_id}"

    def test_dispatch_benchmark_returns_task_id(self):
        """Test dispatch_benchmark returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()
        benchmark_id = uuid4()

        result = dispatcher.dispatch_benchmark(benchmark_id)

        assert result == f"noop-benchmark-{benchmark_id}"

    def test_dispatch_disk_usage_check_returns_task_id(self):
        """Test dispatch_disk_usage_check returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()

        result = dispatcher.dispatch_disk_usage_check()

        assert result == "noop-disk-usage"

    def test_dispatch_docker_cleanup_returns_task_id(self):
        """Test dispatch_docker_cleanup returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()

        result = dispatcher.dispatch_docker_cleanup()

        assert result == "noop-docker-cleanup"

    def test_dispatch_health_check_returns_task_id(self):
        """Test dispatch_health_check returns a task ID."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()

        result = dispatcher.dispatch_health_check()

        assert result == "noop-health-check"


class TestCeleryTaskDispatcher:
    """Tests for CeleryTaskDispatcher (mocked Celery tasks)."""

    def test_dispatch_deployment_calls_celery_task(self):
        """Test dispatch_deployment calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        deployment_id = uuid4()
        config_json = '{"test": "config"}'

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-task-123"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(deploy_container_task=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                # Need to reimport to pick up the mock
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_deployment(deployment_id, config_json)

        assert result == "celery-task-123"
        mock_task.delay.assert_called_once_with(str(deployment_id), config_json)

    def test_dispatch_stop_calls_celery_task(self):
        """Test dispatch_stop calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        deployment_id = uuid4()

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-task-456"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(stop_container_task=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_stop(deployment_id)

        assert result == "celery-task-456"
        mock_task.delay.assert_called_once_with(str(deployment_id))

    def test_dispatch_restart_calls_celery_task(self):
        """Test dispatch_restart calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        deployment_id = uuid4()
        config_json = '{"restart": "config"}'

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-task-789"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(restart_container_task=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_restart(deployment_id, config_json)

        assert result == "celery-task-789"
        mock_task.delay.assert_called_once_with(str(deployment_id), config_json)

    def test_dispatch_docker_build_calls_celery_task(self):
        """Test dispatch_docker_build calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        build_id = uuid4()

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-build-123"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(build_docker_image=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_docker_build(build_id)

        assert result == "celery-build-123"
        mock_task.delay.assert_called_once_with(str(build_id))

    def test_dispatch_mlflow_sync_calls_celery_task(self):
        """Test dispatch_mlflow_sync calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        version_id = uuid4()

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-mlflow-123"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(sync_mlflow_metadata_task=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_mlflow_sync(version_id)

        assert result == "celery-mlflow-123"
        mock_task.delay.assert_called_once_with(str(version_id))

    def test_dispatch_benchmark_calls_celery_task(self):
        """Test dispatch_benchmark calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        benchmark_id = uuid4()

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-benchmark-123"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(run_benchmark_task=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_benchmark(benchmark_id)

        assert result == "celery-benchmark-123"
        mock_task.delay.assert_called_once_with(str(benchmark_id))

    def test_dispatch_disk_usage_check_calls_celery_task(self):
        """Test dispatch_disk_usage_check calls the Celery task."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()

        mock_task = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "celery-disk-123"
        mock_task.delay.return_value = mock_result

        with patch.dict('sys.modules', {'app.worker': MagicMock(get_docker_disk_usage_task=mock_task)}):
            with patch('app.services.task_dispatcher.logger'):
                from app.services.task_dispatcher import CeleryTaskDispatcher
                dispatcher = CeleryTaskDispatcher()
                result = dispatcher.dispatch_disk_usage_check()

        assert result == "celery-disk-123"
        mock_task.delay.assert_called_once()


class TestCeleryTaskDispatcherErrorHandling:
    """Tests for CeleryTaskDispatcher error handling."""

    def test_dispatch_deployment_returns_none_on_error(self):
        """Test dispatch_deployment returns None when task dispatch fails."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        deployment_id = uuid4()

        # Make the import fail
        with patch.dict('sys.modules', {'app.worker': None}):
            with patch('app.services.task_dispatcher.logger'):
                result = dispatcher.dispatch_deployment(deployment_id, '{}')

        assert result is None

    def test_dispatch_stop_returns_none_on_error(self):
        """Test dispatch_stop returns None when task dispatch fails."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()
        deployment_id = uuid4()

        with patch.dict('sys.modules', {'app.worker': None}):
            with patch('app.services.task_dispatcher.logger'):
                result = dispatcher.dispatch_stop(deployment_id)

        assert result is None


class TestDispatcherSelection:
    """Tests for dispatcher selection based on environment."""

    def test_creates_noop_dispatcher_for_test_environment(self):
        """Test that NoOpTaskDispatcher is created in test environment."""
        from app.services.task_dispatcher import NoOpTaskDispatcher, _create_dispatcher

        with patch.dict('os.environ', {'ENVIRONMENT': 'test'}):
            dispatcher = _create_dispatcher()

        assert isinstance(dispatcher, NoOpTaskDispatcher)

    def test_creates_celery_dispatcher_for_production(self):
        """Test that CeleryTaskDispatcher is created in production."""
        from app.services.task_dispatcher import CeleryTaskDispatcher, _create_dispatcher

        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            dispatcher = _create_dispatcher()

        assert isinstance(dispatcher, CeleryTaskDispatcher)

    def test_creates_celery_dispatcher_by_default(self):
        """Test that CeleryTaskDispatcher is created when ENVIRONMENT is not set."""
        from app.services.task_dispatcher import CeleryTaskDispatcher, _create_dispatcher

        with patch.dict('os.environ', {}, clear=True):
            # Remove ENVIRONMENT if it exists
            import os
            env_backup = os.environ.get('ENVIRONMENT')
            if 'ENVIRONMENT' in os.environ:
                del os.environ['ENVIRONMENT']

            try:
                dispatcher = _create_dispatcher()
                assert isinstance(dispatcher, CeleryTaskDispatcher)
            finally:
                # Restore environment
                if env_backup:
                    os.environ['ENVIRONMENT'] = env_backup


class TestTaskDispatcherProtocol:
    """Tests for TaskDispatcherProtocol compliance."""

    def test_noop_dispatcher_implements_protocol(self):
        """Test NoOpTaskDispatcher implements the protocol."""
        from app.services.task_dispatcher import NoOpTaskDispatcher

        dispatcher = NoOpTaskDispatcher()

        # Check all required methods exist
        assert hasattr(dispatcher, 'dispatch_deployment')
        assert hasattr(dispatcher, 'dispatch_stop')
        assert hasattr(dispatcher, 'dispatch_restart')
        assert hasattr(dispatcher, 'dispatch_docker_build')
        assert hasattr(dispatcher, 'dispatch_mlflow_sync')
        assert hasattr(dispatcher, 'dispatch_benchmark')
        assert hasattr(dispatcher, 'dispatch_disk_usage_check')

    def test_celery_dispatcher_implements_protocol(self):
        """Test CeleryTaskDispatcher implements the protocol."""
        from app.services.task_dispatcher import CeleryTaskDispatcher

        dispatcher = CeleryTaskDispatcher()

        # Check all required methods exist
        assert hasattr(dispatcher, 'dispatch_deployment')
        assert hasattr(dispatcher, 'dispatch_stop')
        assert hasattr(dispatcher, 'dispatch_restart')
        assert hasattr(dispatcher, 'dispatch_docker_build')
        assert hasattr(dispatcher, 'dispatch_mlflow_sync')
        assert hasattr(dispatcher, 'dispatch_benchmark')
        assert hasattr(dispatcher, 'dispatch_disk_usage_check')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
