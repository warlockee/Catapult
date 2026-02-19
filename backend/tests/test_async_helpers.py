"""
Tests for the async_helpers module.

Tests cover:
- run_async: Basic coroutine execution, exception propagation, cleanup
- run_async_with_db: Database session handling
- AsyncTaskRunner: Context manager for multiple async operations

Run with: pytest backend/tests/test_async_helpers.py -v
"""
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock


# Common patch targets - these are where the imports happen
ENGINE_PATCH = 'app.core.database.engine'
SESSION_MAKER_PATCH = 'app.core.database.async_session_maker'


class TestRunAsync:
    """Tests for run_async function."""

    def test_run_async_executes_coroutine(self):
        """Test that run_async executes a coroutine and returns its result."""
        from app.core.async_helpers import run_async

        async def simple_coro():
            return "hello"

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(simple_coro())

        assert result == "hello"

    def test_run_async_with_arguments(self):
        """Test run_async with coroutine that takes arguments."""
        from app.core.async_helpers import run_async

        async def add_numbers(a, b):
            return a + b

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(add_numbers(5, 3))

        assert result == 8

    def test_run_async_propagates_exceptions(self):
        """Test that exceptions from coroutines are propagated."""
        from app.core.async_helpers import run_async

        async def failing_coro():
            raise ValueError("Test error")

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with pytest.raises(ValueError) as exc_info:
                run_async(failing_coro())

        assert "Test error" in str(exc_info.value)

    def test_run_async_disposes_engine(self):
        """Test that run_async disposes the database engine."""
        from app.core.async_helpers import run_async

        async def simple_coro():
            return True

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            run_async(simple_coro())
            mock_engine.dispose.assert_called_once()

    def test_run_async_cleans_up_on_success(self):
        """Test that event loop is properly cleaned up on success."""
        from app.core.async_helpers import run_async

        async def simple_coro():
            return "done"

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(simple_coro())

        assert result == "done"
        # If cleanup failed, this test would hang or raise

    def test_run_async_cleans_up_on_exception(self):
        """Test that event loop is cleaned up even when coroutine raises."""
        from app.core.async_helpers import run_async

        async def failing_coro():
            raise RuntimeError("Test failure")

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with pytest.raises(RuntimeError):
                run_async(failing_coro())
        # If cleanup failed, this test would hang

    def test_run_async_with_async_generator_cleanup(self):
        """Test that async generators are properly cleaned up."""
        from app.core.async_helpers import run_async

        cleanup_called = []

        async def coro_with_generator():
            async def gen():
                try:
                    yield 1
                    yield 2
                finally:
                    cleanup_called.append(True)

            async for _ in gen():
                break  # Exit early to test cleanup
            return "done"

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(coro_with_generator())

        assert result == "done"
        assert len(cleanup_called) == 1

    def test_run_async_with_nested_coroutines(self):
        """Test run_async with nested coroutine calls."""
        from app.core.async_helpers import run_async

        async def inner():
            await asyncio.sleep(0)  # Yield control
            return 42

        async def outer():
            result = await inner()
            return result * 2

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(outer())

        assert result == 84

    def test_run_async_can_be_called_multiple_times(self):
        """Test that run_async can be called multiple times sequentially."""
        from app.core.async_helpers import run_async

        async def counter(n):
            return n

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            results = [run_async(counter(i)) for i in range(3)]

        assert results == [0, 1, 2]
        assert mock_engine.dispose.call_count == 3


class TestRunAsyncWithDb:
    """Tests for run_async_with_db function."""

    def test_run_async_with_db_provides_session(self):
        """Test that run_async_with_db provides a database session."""
        from app.core.async_helpers import run_async_with_db

        session_received = []

        async def db_operation(db):
            session_received.append(db)
            return "done"

        with patch(ENGINE_PATCH) as mock_engine, \
             patch(SESSION_MAKER_PATCH) as mock_session_maker:
            mock_engine.dispose = AsyncMock()
            mock_session = AsyncMock()
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

            result = run_async_with_db(db_operation)

        assert result == "done"
        assert len(session_received) == 1
        assert session_received[0] == mock_session

    def test_run_async_with_db_commits_when_requested(self):
        """Test that run_async_with_db commits session when commit=True."""
        from app.core.async_helpers import run_async_with_db

        async def db_operation(db):
            return "committed"

        with patch(ENGINE_PATCH) as mock_engine, \
             patch(SESSION_MAKER_PATCH) as mock_session_maker:
            mock_engine.dispose = AsyncMock()
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

            run_async_with_db(db_operation, commit=True)

        mock_session.commit.assert_called_once()

    def test_run_async_with_db_no_commit_by_default(self):
        """Test that run_async_with_db doesn't commit by default."""
        from app.core.async_helpers import run_async_with_db

        async def db_operation(db):
            return "not committed"

        with patch(ENGINE_PATCH) as mock_engine, \
             patch(SESSION_MAKER_PATCH) as mock_session_maker:
            mock_engine.dispose = AsyncMock()
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

            run_async_with_db(db_operation)

        mock_session.commit.assert_not_called()

    def test_run_async_with_db_propagates_exceptions(self):
        """Test that exceptions from db operations are propagated."""
        from app.core.async_helpers import run_async_with_db

        async def failing_db_operation(db):
            raise ValueError("DB operation failed")

        with patch(ENGINE_PATCH) as mock_engine, \
             patch(SESSION_MAKER_PATCH) as mock_session_maker:
            mock_engine.dispose = AsyncMock()
            mock_session = AsyncMock()
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(ValueError) as exc_info:
                run_async_with_db(failing_db_operation)

        assert "DB operation failed" in str(exc_info.value)

    def test_run_async_with_db_returns_result(self):
        """Test that run_async_with_db returns the function result."""
        from app.core.async_helpers import run_async_with_db

        async def db_query(db):
            return {"id": 1, "name": "test"}

        with patch(ENGINE_PATCH) as mock_engine, \
             patch(SESSION_MAKER_PATCH) as mock_session_maker:
            mock_engine.dispose = AsyncMock()
            mock_session = AsyncMock()
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

            result = run_async_with_db(db_query)

        assert result == {"id": 1, "name": "test"}


class TestAsyncTaskRunner:
    """Tests for AsyncTaskRunner context manager."""

    def test_async_task_runner_basic_usage(self):
        """Test basic AsyncTaskRunner usage."""
        from app.core.async_helpers import AsyncTaskRunner

        async def simple_task():
            return "task result"

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with AsyncTaskRunner() as runner:
                result = runner.run(simple_task())

        assert result == "task result"

    def test_async_task_runner_multiple_tasks(self):
        """Test running multiple tasks with AsyncTaskRunner."""
        from app.core.async_helpers import AsyncTaskRunner

        async def task1():
            return 1

        async def task2():
            return 2

        async def task3():
            return 3

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with AsyncTaskRunner() as runner:
                results = [
                    runner.run(task1()),
                    runner.run(task2()),
                    runner.run(task3()),
                ]

        assert results == [1, 2, 3]
        # Engine should only be disposed once
        mock_engine.dispose.assert_called_once()

    def test_async_task_runner_disposes_engine_once(self):
        """Test that engine is only disposed on first run."""
        from app.core.async_helpers import AsyncTaskRunner

        async def task():
            return True

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with AsyncTaskRunner() as runner:
                for _ in range(5):
                    runner.run(task())

        # Should only dispose once, not 5 times
        mock_engine.dispose.assert_called_once()

    def test_async_task_runner_raises_outside_context(self):
        """Test that run() raises when called outside context manager."""
        from app.core.async_helpers import AsyncTaskRunner

        async def task():
            return True

        runner = AsyncTaskRunner()
        with pytest.raises(RuntimeError) as exc_info:
            runner.run(task())

        assert "must be used as a context manager" in str(exc_info.value)

    def test_async_task_runner_propagates_exceptions(self):
        """Test that exceptions from tasks are propagated."""
        from app.core.async_helpers import AsyncTaskRunner

        async def failing_task():
            raise ValueError("Task failed")

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with pytest.raises(ValueError) as exc_info:
                with AsyncTaskRunner() as runner:
                    runner.run(failing_task())

        assert "Task failed" in str(exc_info.value)

    def test_async_task_runner_cleans_up_on_exception(self):
        """Test that runner cleans up even when task raises."""
        from app.core.async_helpers import AsyncTaskRunner

        async def failing_task():
            raise RuntimeError("Failure")

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            try:
                with AsyncTaskRunner() as runner:
                    runner.run(failing_task())
            except RuntimeError:
                pass

        # If cleanup failed, subsequent tests would fail or hang

    def test_async_task_runner_with_dependent_tasks(self):
        """Test running tasks that depend on each other's results."""
        from app.core.async_helpers import AsyncTaskRunner

        async def get_base():
            return 10

        async def multiply(value):
            return value * 2

        async def add(value):
            return value + 5

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            with AsyncTaskRunner() as runner:
                base = runner.run(get_base())
                doubled = runner.run(multiply(base))
                final = runner.run(add(doubled))

        assert base == 10
        assert doubled == 20
        assert final == 25


class TestIntegrationWithCeleryTasks:
    """Integration-style tests simulating Celery task patterns."""

    def test_typical_celery_task_pattern(self):
        """Test the pattern typically used in Celery tasks."""
        from app.core.async_helpers import run_async

        # Simulate a Celery task that does async DB work
        task_result = None

        def celery_task_simulation(item_id: str):
            nonlocal task_result

            async def do_work():
                # Simulate async operations
                await asyncio.sleep(0)
                return f"Processed {item_id}"

            with patch(ENGINE_PATCH) as mock_engine:
                mock_engine.dispose = AsyncMock()
                task_result = run_async(do_work())

            return task_result

        result = celery_task_simulation("item-123")
        assert result == "Processed item-123"
        assert task_result == "Processed item-123"

    def test_task_with_db_session_pattern(self):
        """Test the pattern for tasks that need database sessions."""
        from app.core.async_helpers import run_async_with_db

        def celery_task_with_db(deployment_id: str):
            async def update_deployment(db):
                # Simulate database operation
                return {"id": deployment_id, "status": "updated"}

            with patch(ENGINE_PATCH) as mock_engine, \
                 patch(SESSION_MAKER_PATCH) as mock_session_maker:
                mock_engine.dispose = AsyncMock()
                mock_session = AsyncMock()
                mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_maker.return_value.__aexit__ = AsyncMock(return_value=None)

                return run_async_with_db(update_deployment)

        result = celery_task_with_db("deploy-456")
        assert result["id"] == "deploy-456"
        assert result["status"] == "updated"

    def test_task_with_multiple_async_operations(self):
        """Test pattern for tasks with multiple sequential async operations."""
        from app.core.async_helpers import AsyncTaskRunner

        def celery_task_multi_op():
            async def step1():
                return "step1_result"

            async def step2(prev_result):
                return f"{prev_result} -> step2"

            async def step3(prev_result):
                return f"{prev_result} -> step3"

            with patch(ENGINE_PATCH) as mock_engine:
                mock_engine.dispose = AsyncMock()
                with AsyncTaskRunner() as runner:
                    r1 = runner.run(step1())
                    r2 = runner.run(step2(r1))
                    r3 = runner.run(step3(r2))
                    return r3

        result = celery_task_multi_op()
        assert result == "step1_result -> step2 -> step3"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_run_async_with_none_result(self):
        """Test run_async with coroutine that returns None."""
        from app.core.async_helpers import run_async

        async def returns_none():
            return None

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(returns_none())

        assert result is None

    def test_run_async_with_complex_return_type(self):
        """Test run_async with complex return types."""
        from app.core.async_helpers import run_async

        async def returns_complex():
            return {
                "list": [1, 2, 3],
                "nested": {"a": {"b": "c"}},
                "tuple": (1, 2),
            }

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(returns_complex())

        assert result["list"] == [1, 2, 3]
        assert result["nested"]["a"]["b"] == "c"

    def test_run_async_with_long_running_task(self):
        """Test run_async with a task that takes some time."""
        from app.core.async_helpers import run_async

        async def slow_task():
            await asyncio.sleep(0.01)  # 10ms
            return "completed"

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(slow_task())

        assert result == "completed"

    def test_run_async_with_concurrent_subtasks(self):
        """Test run_async with a coroutine that runs concurrent subtasks."""
        from app.core.async_helpers import run_async

        async def concurrent_work():
            async def subtask(n):
                await asyncio.sleep(0)
                return n * 2

            results = await asyncio.gather(
                subtask(1),
                subtask(2),
                subtask(3),
            )
            return results

        with patch(ENGINE_PATCH) as mock_engine:
            mock_engine.dispose = AsyncMock()
            result = run_async(concurrent_work())

        assert result == [2, 4, 6]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
