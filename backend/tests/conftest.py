"""
Pytest configuration and fixtures for backend tests.

This file is automatically loaded by pytest before running tests.
It sets up necessary environment variables and common fixtures.
"""
import os
import pytest

# Set environment variables BEFORE any app imports
# These are required by Settings class
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("API_KEY_SALT", "test_salt_for_testing_only")
os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture
def mock_db_session():
    """Provide a mock database session for tests that don't need real DB."""
    from unittest.mock import AsyncMock
    return AsyncMock()


@pytest.fixture
def mock_engine():
    """Provide a mock database engine."""
    from unittest.mock import AsyncMock, MagicMock
    engine = MagicMock()
    engine.dispose = AsyncMock()
    return engine
