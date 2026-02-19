"""
Tests for VersionRepository update_metadata method.
"""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.exceptions import VersionNotFoundError
from app.repositories.version_repository import VersionRepository


class TestVersionRepositoryUpdateMetadata:
    """Tests for update_metadata method."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def mock_version(self):
        """Create a mock version."""
        version = MagicMock()
        version.id = uuid4()
        version.meta_data = {"existing_key": "existing_value"}
        return version

    @pytest.mark.asyncio
    async def test_update_metadata_merges_by_default(self, mock_db, mock_version):
        """Test update_metadata merges with existing metadata by default."""
        repo = VersionRepository(mock_db)

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_version

            result = await repo.update_metadata(mock_version.id, {"new_key": "new_value"})

            assert mock_version.meta_data == {
                "existing_key": "existing_value",
                "new_key": "new_value",
            }
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once_with(mock_version)
            assert result == mock_version

    @pytest.mark.asyncio
    async def test_update_metadata_replaces_when_merge_false(self, mock_db, mock_version):
        """Test update_metadata replaces existing metadata when merge=False."""
        repo = VersionRepository(mock_db)

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_version

            result = await repo.update_metadata(
                mock_version.id,
                {"new_key": "new_value"},
                merge=False
            )

            assert mock_version.meta_data == {"new_key": "new_value"}
            mock_db.commit.assert_called_once()
            assert result == mock_version

    @pytest.mark.asyncio
    async def test_update_metadata_handles_none_existing(self, mock_db, mock_version):
        """Test update_metadata handles None existing metadata."""
        repo = VersionRepository(mock_db)
        mock_version.meta_data = None

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_version

            result = await repo.update_metadata(mock_version.id, {"new_key": "new_value"})

            assert mock_version.meta_data == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_update_metadata_nested_merge(self, mock_db, mock_version):
        """Test update_metadata with nested data (like mlflow)."""
        repo = VersionRepository(mock_db)
        mock_version.meta_data = {"existing": "value"}

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_version

            mlflow_data = {
                "model_name": "test-model",
                "fetched_at": "2024-01-01T00:00:00Z"
            }
            result = await repo.update_metadata(mock_version.id, {"mlflow": mlflow_data})

            assert mock_version.meta_data == {
                "existing": "value",
                "mlflow": mlflow_data,
            }

    @pytest.mark.asyncio
    async def test_update_metadata_raises_when_not_found(self, mock_db):
        """Test update_metadata raises VersionNotFoundError when not found."""
        repo = VersionRepository(mock_db)
        version_id = uuid4()

        with patch.object(repo, 'get_by_id_or_raise', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = VersionNotFoundError(str(version_id))

            with pytest.raises(VersionNotFoundError):
                await repo.update_metadata(version_id, {"key": "value"})
