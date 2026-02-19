"""
Tests for DeploymentRepository.

Tests the repository methods for deployment state management.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.repositories.deployment_repository import DeploymentRepository
from app.core.exceptions import DeploymentNotFoundError


class TestDeploymentRepositoryUpdateStatus:
    """Tests for update_status method."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def mock_deployment(self):
        """Create a mock deployment."""
        deployment = MagicMock()
        deployment.id = uuid4()
        deployment.status = "running"
        deployment.meta_data = {}
        deployment.health_status = None
        return deployment

    @pytest.mark.asyncio
    async def test_update_status_success(self, mock_db, mock_deployment):
        """Test update_status sets the new status."""
        repo = DeploymentRepository(mock_db)

        # Mock get_by_id_or_raise to return the deployment
        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_deployment

            result = await repo.update_status(mock_deployment.id, "stopping")

            assert mock_deployment.status == "stopping"
            mock_db.commit.assert_called_once()
            assert result == mock_deployment

    @pytest.mark.asyncio
    async def test_update_status_raises_when_not_found(self, mock_db):
        """Test update_status raises DeploymentNotFoundError when not found."""
        repo = DeploymentRepository(mock_db)
        deployment_id = uuid4()

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = DeploymentNotFoundError(str(deployment_id))

            with pytest.raises(DeploymentNotFoundError):
                await repo.update_status(deployment_id, "stopping")


class TestDeploymentRepositoryUpdateMetadata:
    """Tests for update_metadata method."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def mock_deployment(self):
        """Create a mock deployment."""
        deployment = MagicMock()
        deployment.id = uuid4()
        deployment.status = "running"
        deployment.meta_data = {"existing_key": "existing_value"}
        return deployment

    @pytest.mark.asyncio
    async def test_update_metadata_merges_by_default(self, mock_db, mock_deployment):
        """Test update_metadata merges with existing metadata by default."""
        repo = DeploymentRepository(mock_db)

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_deployment

            result = await repo.update_metadata(mock_deployment.id, {"new_key": "new_value"})

            assert mock_deployment.meta_data == {
                "existing_key": "existing_value",
                "new_key": "new_value",
            }
            mock_db.commit.assert_called_once()
            assert result == mock_deployment

    @pytest.mark.asyncio
    async def test_update_metadata_replaces_when_merge_false(self, mock_db, mock_deployment):
        """Test update_metadata replaces existing metadata when merge=False."""
        repo = DeploymentRepository(mock_db)

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_deployment

            result = await repo.update_metadata(
                mock_deployment.id,
                {"new_key": "new_value"},
                merge=False
            )

            assert mock_deployment.meta_data == {"new_key": "new_value"}
            mock_db.commit.assert_called_once()
            assert result == mock_deployment

    @pytest.mark.asyncio
    async def test_update_metadata_handles_none_existing(self, mock_db, mock_deployment):
        """Test update_metadata handles None existing metadata."""
        repo = DeploymentRepository(mock_db)
        mock_deployment.meta_data = None

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_deployment

            result = await repo.update_metadata(mock_deployment.id, {"new_key": "new_value"})

            assert mock_deployment.meta_data == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_update_metadata_raises_when_not_found(self, mock_db):
        """Test update_metadata raises DeploymentNotFoundError when not found."""
        repo = DeploymentRepository(mock_db)
        deployment_id = uuid4()

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = DeploymentNotFoundError(str(deployment_id))

            with pytest.raises(DeploymentNotFoundError):
                await repo.update_metadata(deployment_id, {"key": "value"})


class TestDeploymentRepositoryUpdateHealthStatus:
    """Tests for update_health_status method."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def mock_deployment(self):
        """Create a mock deployment."""
        deployment = MagicMock()
        deployment.id = uuid4()
        deployment.health_status = None
        return deployment

    @pytest.mark.asyncio
    async def test_update_health_status_sets_healthy(self, mock_db, mock_deployment):
        """Test update_health_status sets healthy status."""
        repo = DeploymentRepository(mock_db)

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_deployment

            result = await repo.update_health_status(mock_deployment.id, healthy=True)

            assert mock_deployment.health_status == "healthy"
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once_with(mock_deployment)
            assert result == mock_deployment

    @pytest.mark.asyncio
    async def test_update_health_status_sets_unhealthy(self, mock_db, mock_deployment):
        """Test update_health_status sets unhealthy status."""
        repo = DeploymentRepository(mock_db)

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_deployment

            result = await repo.update_health_status(mock_deployment.id, healthy=False)

            assert mock_deployment.health_status == "unhealthy"
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once_with(mock_deployment)
            assert result == mock_deployment

    @pytest.mark.asyncio
    async def test_update_health_status_raises_when_not_found(self, mock_db):
        """Test update_health_status raises DeploymentNotFoundError when not found."""
        repo = DeploymentRepository(mock_db)
        deployment_id = uuid4()

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = DeploymentNotFoundError(str(deployment_id))

            with pytest.raises(DeploymentNotFoundError):
                await repo.update_health_status(deployment_id, healthy=True)
