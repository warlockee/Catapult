"""
Async helpers for running async code in synchronous contexts.

This module provides utilities for running async code within Celery tasks
and other synchronous contexts that need to execute async database operations.

Usage:
    from app.core.async_helpers import run_async, run_async_with_db

    # Simple async execution
    result = run_async(my_async_function())

    # With database session
    async def my_db_operation(db: AsyncSession):
        # ... do database work
        return result

    result = run_async_with_db(my_db_operation)
"""
import asyncio
import logging
from typing import TypeVar, Callable, Awaitable, Optional, Any

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine in a new event loop.

    Handles the boilerplate of creating an event loop, disposing the database
    engine (to ensure fresh connections), running the coroutine, and cleanup.

    This is designed for use in Celery tasks where each task runs in its own
    thread/process and needs a fresh event loop.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine

    Example:
        @celery_app.task
        def my_task():
            async def do_work():
                async with async_session_maker() as db:
                    # ... database operations
                    return result
            return run_async(do_work())
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Dispose the global engine to ensure new connections use the current loop.
        # This is critical for SQLAlchemy async to work properly in worker processes.
        # The dispose() may trigger asyncpg connection.close() which can raise
        # "RuntimeError: Event loop is closed" if connections were created on a
        # previous (now-closed) event loop. This is harmless — the connections
        # are being discarded anyway — so we suppress it.
        from app.core.database import engine
        # Temporarily raise the log level of SQLAlchemy pool loggers during dispose()
        # to suppress harmless "Event loop is closed" ERROR tracebacks from asyncpg
        # connections that reference a previous (now-closed) event loop.
        _pool_loggers = [logging.getLogger(n) for n in ('sqlalchemy.pool', 'sqlalchemy.pool.impl')]
        _prev_levels = [pl.level for pl in _pool_loggers]
        for pl in _pool_loggers:
            pl.setLevel(logging.CRITICAL)
        try:
            loop.run_until_complete(engine.dispose())
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
        finally:
            for pl, lv in zip(_pool_loggers, _prev_levels):
                pl.setLevel(lv)

        # Run the actual coroutine
        result = loop.run_until_complete(coro)
        return result
    finally:
        # Clean up async generators and close the loop
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def run_async_with_db(
    func: Callable[[AsyncSession], Awaitable[T]],
    *,
    commit: bool = False,
) -> T:
    """
    Run an async function with a database session.

    This is a convenience wrapper that creates a database session and passes
    it to your function. Handles session lifecycle automatically.

    Args:
        func: Async function that takes a database session and returns a result
        commit: If True, commits the session after the function completes
                (useful for read-only operations that don't need explicit commit)

    Returns:
        The result of the function

    Raises:
        Any exception raised by the function

    Example:
        @celery_app.task
        def my_task(item_id: str):
            async def do_work(db: AsyncSession):
                item = await db.get(Item, UUID(item_id))
                item.status = "processed"
                await db.commit()
                return item.id

            return run_async_with_db(do_work)
    """
    async def wrapper():
        from app.core.database import async_session_maker
        async with async_session_maker() as db:
            result = await func(db)
            if commit:
                await db.commit()
            return result

    return run_async(wrapper())


class AsyncTaskRunner:
    """
    Context manager for running multiple async operations in sequence.

    Useful when you need to run several async operations but want to
    share the event loop setup/teardown.

    Example:
        with AsyncTaskRunner() as runner:
            result1 = runner.run(operation1())
            result2 = runner.run(operation2())
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._engine_disposed = False

    def __enter__(self) -> "AsyncTaskRunner":
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._loop:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            self._loop = None
        return False

    def run(self, coro: Awaitable[T]) -> T:
        """
        Run a coroutine in the managed event loop.

        Args:
            coro: The coroutine to execute

        Returns:
            The result of the coroutine
        """
        if not self._loop:
            raise RuntimeError("AsyncTaskRunner must be used as a context manager")

        # Dispose engine on first run
        if not self._engine_disposed:
            from app.core.database import engine
            _pool_loggers = [logging.getLogger(n) for n in ('sqlalchemy.pool', 'sqlalchemy.pool.impl')]
            _prev_levels = [pl.level for pl in _pool_loggers]
            for pl in _pool_loggers:
                pl.setLevel(logging.CRITICAL)
            try:
                self._loop.run_until_complete(engine.dispose())
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    raise
            finally:
                for pl, lv in zip(_pool_loggers, _prev_levels):
                    pl.setLevel(lv)
            self._engine_disposed = True

        return self._loop.run_until_complete(coro)

    async def get_db_session(self):
        """
        Get a database session for use within an async context.

        Returns:
            AsyncSession context manager
        """
        from app.core.database import async_session_maker
        return async_session_maker()
