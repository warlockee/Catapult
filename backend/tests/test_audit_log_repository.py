"""
Tests for AuditLogRepository.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

from app.repositories.audit_log_repository import AuditLogRepository


class TestAuditLogRepositoryListAuditLogs:
    """Tests for list_audit_logs method."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def mock_audit_logs(self):
        """Create mock audit logs."""
        logs = []
        for i in range(3):
            log = MagicMock()
            log.id = uuid4()
            log.action = f"action_{i}"
            log.resource_type = "deployment"
            log.api_key_name = "test-key"
            log.created_at = datetime.utcnow()
            logs.append(log)
        return logs

    @pytest.mark.asyncio
    async def test_list_audit_logs_no_filters(self, mock_db, mock_audit_logs):
        """Test list_audit_logs returns all logs without filters."""
        repo = AuditLogRepository(mock_db)

        # Mock the execute result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_audit_logs
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs()

        assert len(result) == 3
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_audit_logs_with_action_filter(self, mock_db, mock_audit_logs):
        """Test list_audit_logs filters by action."""
        repo = AuditLogRepository(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_audit_logs[0]]
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs(action="action_0")

        assert len(result) == 1
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_audit_logs_with_resource_type_filter(self, mock_db, mock_audit_logs):
        """Test list_audit_logs filters by resource_type."""
        repo = AuditLogRepository(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_audit_logs
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs(resource_type="deployment")

        assert len(result) == 3
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_audit_logs_with_api_key_name_filter(self, mock_db, mock_audit_logs):
        """Test list_audit_logs filters by api_key_name."""
        repo = AuditLogRepository(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_audit_logs
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs(api_key_name="test-key")

        assert len(result) == 3
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_audit_logs_with_pagination(self, mock_db, mock_audit_logs):
        """Test list_audit_logs respects skip and limit."""
        repo = AuditLogRepository(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_audit_logs[1]]
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs(skip=1, limit=1)

        assert len(result) == 1
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_audit_logs_with_multiple_filters(self, mock_db, mock_audit_logs):
        """Test list_audit_logs with multiple filters combined."""
        repo = AuditLogRepository(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_audit_logs[0]]
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs(
            action="action_0",
            resource_type="deployment",
            api_key_name="test-key",
        )

        assert len(result) == 1
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_audit_logs_empty_result(self, mock_db):
        """Test list_audit_logs returns empty list when no matches."""
        repo = AuditLogRepository(mock_db)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await repo.list_audit_logs(action="nonexistent")

        assert result == []
        mock_db.execute.assert_called_once()
