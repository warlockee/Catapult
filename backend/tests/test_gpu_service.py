"""
Tests for the GPU service.

Tests cover:
- GPU detection (detect_gpu)
- GPU info querying (query_gpu_info)
- GPU selection (find_available_gpu, find_available_gpus)
- Used GPU tracking (get_gpus_used_by_deployments)
- Caching behavior

Run with: pytest backend/tests/test_gpu_service.py -v
"""
from unittest.mock import AsyncMock, patch

import pytest


class TestGpuServiceDetection:
    """Tests for GPU detection functionality."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset GPU cache before each test."""
        from app.services.deployment.gpu_service import GpuService
        GpuService.reset_cache()
        yield
        GpuService.reset_cache()

    @pytest.mark.asyncio
    async def test_detect_gpu_with_nvidia_runtime(self):
        """Test GPU detection when nvidia runtime is available."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock docker info returning nvidia runtime
        mock_docker_info = AsyncMock()
        mock_docker_info.communicate = AsyncMock(return_value=(b"nvidia runc", b""))
        mock_docker_info.returncode = 0

        # Mock nvidia-smi returning GPU list
        mock_nvidia_smi = AsyncMock()
        mock_nvidia_smi.communicate = AsyncMock(return_value=(b"NVIDIA A100\nNVIDIA A100", b""))
        mock_nvidia_smi.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # First call is docker info, second is nvidia-smi
            mock_exec.side_effect = [mock_docker_info, mock_nvidia_smi]

            available, count = await gpu_service.detect_gpu()

        assert available is True
        assert count == 2

    @pytest.mark.asyncio
    async def test_detect_gpu_no_nvidia_runtime(self):
        """Test GPU detection when nvidia runtime is not available."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock docker info returning no nvidia runtime
        mock_docker_info = AsyncMock()
        mock_docker_info.communicate = AsyncMock(return_value=(b"runc", b""))
        mock_docker_info.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_docker_info):
            available, count = await gpu_service.detect_gpu()

        assert available is False
        assert count == 0

    @pytest.mark.asyncio
    async def test_detect_gpu_caching(self):
        """Test that GPU detection results are cached."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock docker info returning nvidia runtime
        mock_docker_info = AsyncMock()
        mock_docker_info.communicate = AsyncMock(return_value=(b"nvidia runc", b""))
        mock_docker_info.returncode = 0

        # Mock nvidia-smi
        mock_nvidia_smi = AsyncMock()
        mock_nvidia_smi.communicate = AsyncMock(return_value=(b"NVIDIA A100", b""))
        mock_nvidia_smi.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.side_effect = [mock_docker_info, mock_nvidia_smi]

            # First call
            available1, count1 = await gpu_service.detect_gpu()
            # Second call should use cache
            available2, count2 = await gpu_service.detect_gpu()

        assert available1 is True and available2 is True
        assert count1 == count2 == 1
        # Should only have been called twice (docker info + nvidia-smi), not four times
        assert mock_exec.call_count == 2

    @pytest.mark.asyncio
    async def test_reset_cache(self):
        """Test that reset_cache clears the GPU detection cache."""
        from app.services.deployment.gpu_service import GpuService

        # Set cache values
        GpuService._gpu_available = True
        GpuService._gpu_count = 4

        # Reset
        GpuService.reset_cache()

        assert GpuService._gpu_available is None
        assert GpuService._gpu_count is None


class TestGpuServiceQuery:
    """Tests for GPU info querying."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset GPU cache before each test."""
        from app.services.deployment.gpu_service import GpuService
        GpuService.reset_cache()
        yield

    @pytest.mark.asyncio
    async def test_query_gpu_info_success(self):
        """Test querying GPU info successfully."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock nvidia-smi output
        nvidia_output = b"""0, 1234, 24576, 45
1, 5678, 24576, 80
2, 100, 24576, 10"""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(nvidia_output, b""))
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            gpu_info = await gpu_service.query_gpu_info()

        assert len(gpu_info) == 3
        assert gpu_info[0].index == 0
        assert gpu_info[0].memory_used_mb == 1234
        assert gpu_info[0].memory_total_mb == 24576
        assert gpu_info[0].memory_free_mb == 24576 - 1234
        assert gpu_info[0].utilization_percent == 45

        assert gpu_info[1].index == 1
        assert gpu_info[1].utilization_percent == 80

        assert gpu_info[2].index == 2
        assert gpu_info[2].utilization_percent == 10

    @pytest.mark.asyncio
    async def test_query_gpu_info_failure_returns_empty(self):
        """Test that failed GPU query returns empty list."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock both direct and docker nvidia-smi failing
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_process.returncode = 1

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            gpu_info = await gpu_service.query_gpu_info()

        assert gpu_info == []

    @pytest.mark.asyncio
    async def test_query_gpu_info_fallback_to_docker(self):
        """Test that query falls back to Docker when direct nvidia-smi fails."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # First call (direct) fails, second (docker) succeeds
        mock_fail = AsyncMock()
        mock_fail.communicate = AsyncMock(return_value=(b"", b"not found"))
        mock_fail.returncode = 1

        mock_success = AsyncMock()
        mock_success.communicate = AsyncMock(return_value=(b"0, 1000, 8000, 50", b""))
        mock_success.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.side_effect = [mock_fail, mock_success]
            gpu_info = await gpu_service.query_gpu_info()

        assert len(gpu_info) == 1
        assert gpu_info[0].index == 0
        # Should have tried both direct and docker
        assert mock_exec.call_count == 2


class TestGpuServiceSelection:
    """Tests for GPU selection functionality."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset GPU cache before each test."""
        from app.services.deployment.gpu_service import GpuService
        GpuService.reset_cache()
        yield

    @pytest.mark.asyncio
    async def test_find_available_gpu_selects_least_utilized(self):
        """Test that find_available_gpu selects the least utilized GPU."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # GPU 2 has lowest utilization (10%)
        nvidia_output = b"""0, 5000, 24576, 80
1, 1234, 24576, 45
2, 100, 24576, 10"""

        mock_nvidia = AsyncMock()
        mock_nvidia.communicate = AsyncMock(return_value=(nvidia_output, b""))
        mock_nvidia.returncode = 0

        # Mock for docker ps (used GPUs check) - returns no containers
        mock_docker_ps = AsyncMock()
        mock_docker_ps.communicate = AsyncMock(return_value=(b"", b""))
        mock_docker_ps.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Order: docker ps (for used GPUs), then nvidia-smi (for GPU info)
            mock_exec.side_effect = [mock_docker_ps, mock_nvidia]
            selected = await gpu_service.find_available_gpu()

        assert selected == 2  # GPU 2 has lowest utilization

    @pytest.mark.asyncio
    async def test_find_available_gpus_excludes_deployed(self):
        """Test that find_available_gpus excludes GPUs used by deployments."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # All GPUs have similar utilization
        nvidia_output = b"""0, 1000, 24576, 50
1, 1000, 24576, 50
2, 1000, 24576, 50
3, 1000, 24576, 50"""

        mock_nvidia = AsyncMock()
        mock_nvidia.communicate = AsyncMock(return_value=(nvidia_output, b""))
        mock_nvidia.returncode = 0

        # Mock docker ps showing deployment containers
        mock_docker_ps = AsyncMock()
        mock_docker_ps.communicate = AsyncMock(return_value=(b"deployment-abc123\ndeployment-def456", b""))
        mock_docker_ps.returncode = 0

        # Mock docker inspect returning GPU assignments
        mock_inspect1 = AsyncMock()
        mock_inspect1.communicate = AsyncMock(return_value=(b"[0]", b""))  # Container 1 uses GPU 0
        mock_inspect1.returncode = 0

        mock_inspect2 = AsyncMock()
        mock_inspect2.communicate = AsyncMock(return_value=(b"[1]", b""))  # Container 2 uses GPU 1
        mock_inspect2.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Order: nvidia-smi, docker ps, inspect1, inspect2
            mock_exec.side_effect = [mock_docker_ps, mock_inspect1, mock_inspect2, mock_nvidia]

            selected = await gpu_service.find_available_gpus(count=2, exclude_deployed=True)

        # Should select GPUs 2 and 3 (excluding 0 and 1 which are in use)
        assert set(selected) == {2, 3}

    @pytest.mark.asyncio
    async def test_find_available_gpus_fallback_when_not_enough(self):
        """Test fallback when not enough GPUs available after exclusion."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Only 2 GPUs total
        nvidia_output = b"""0, 1000, 24576, 50
1, 1000, 24576, 60"""

        mock_nvidia = AsyncMock()
        mock_nvidia.communicate = AsyncMock(return_value=(nvidia_output, b""))
        mock_nvidia.returncode = 0

        # GPU 0 is in use
        mock_docker_ps = AsyncMock()
        mock_docker_ps.communicate = AsyncMock(return_value=(b"deployment-abc123", b""))
        mock_docker_ps.returncode = 0

        mock_inspect = AsyncMock()
        mock_inspect.communicate = AsyncMock(return_value=(b"[0]", b""))
        mock_inspect.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.side_effect = [mock_docker_ps, mock_inspect, mock_nvidia]

            # Request 2 GPUs but only 1 available after exclusion
            selected = await gpu_service.find_available_gpus(count=2, exclude_deployed=True)

        # Should fall back to all GPUs since not enough available
        assert len(selected) == 2
        assert set(selected) == {0, 1}

    @pytest.mark.asyncio
    async def test_find_available_gpus_default_on_failure(self):
        """Test that GPU selection returns default range on failure."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # All queries fail
        mock_fail = AsyncMock()
        mock_fail.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_fail.returncode = 1

        with patch('asyncio.create_subprocess_exec', return_value=mock_fail):
            selected = await gpu_service.find_available_gpus(count=3, exclude_deployed=False)

        # Should return default range [0, 1, 2]
        assert selected == [0, 1, 2]


class TestGpuServiceDeploymentTracking:
    """Tests for tracking GPUs used by deployments."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset GPU cache before each test."""
        from app.services.deployment.gpu_service import GpuService
        GpuService.reset_cache()
        yield

    @pytest.mark.asyncio
    async def test_get_gpus_used_by_deployments(self):
        """Test getting GPUs used by running deployments."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService(container_prefix="deployment")

        # Mock docker ps
        mock_docker_ps = AsyncMock()
        mock_docker_ps.communicate = AsyncMock(return_value=(b"deployment-abc123\ndeployment-def456", b""))
        mock_docker_ps.returncode = 0

        # Mock docker inspect for each container
        mock_inspect1 = AsyncMock()
        mock_inspect1.communicate = AsyncMock(return_value=(b"[0 1]", b""))  # Uses GPUs 0 and 1
        mock_inspect1.returncode = 0

        mock_inspect2 = AsyncMock()
        mock_inspect2.communicate = AsyncMock(return_value=(b"[2]", b""))  # Uses GPU 2
        mock_inspect2.returncode = 0

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_exec.side_effect = [mock_docker_ps, mock_inspect1, mock_inspect2]
            used_gpus = await gpu_service.get_gpus_used_by_deployments()

        assert used_gpus == {0, 1, 2}

    @pytest.mark.asyncio
    async def test_get_gpus_used_by_deployments_empty(self):
        """Test when no deployments are running."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock docker ps returning no containers
        mock_docker_ps = AsyncMock()
        mock_docker_ps.communicate = AsyncMock(return_value=(b"", b""))
        mock_docker_ps.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_docker_ps):
            used_gpus = await gpu_service.get_gpus_used_by_deployments()

        assert used_gpus == set()

    @pytest.mark.asyncio
    async def test_get_gpus_used_by_deployments_handles_error(self):
        """Test graceful handling of errors when getting used GPUs."""
        from app.services.deployment.gpu_service import GpuService

        gpu_service = GpuService()

        # Mock docker ps failing
        mock_fail = AsyncMock()
        mock_fail.communicate = AsyncMock(side_effect=Exception("Docker not available"))

        with patch('asyncio.create_subprocess_exec', return_value=mock_fail):
            used_gpus = await gpu_service.get_gpus_used_by_deployments()

        # Should return empty set on error, not raise
        assert used_gpus == set()


class TestGpuInfo:
    """Tests for GpuInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test creating GpuInfo instances."""
        from app.services.deployment.gpu_service import GpuInfo

        gpu = GpuInfo(
            index=0,
            memory_used_mb=5000,
            memory_total_mb=24576,
            memory_free_mb=19576,
            utilization_percent=45
        )

        assert gpu.index == 0
        assert gpu.memory_used_mb == 5000
        assert gpu.memory_total_mb == 24576
        assert gpu.memory_free_mb == 19576
        assert gpu.utilization_percent == 45


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
