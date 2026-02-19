"""
GPU detection and selection service.

Provides utilities for detecting GPU availability, querying GPU info,
and selecting optimal GPUs for deployments.

This module consolidates GPU-related logic that was previously duplicated
across multiple functions in the deployment executor.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """Information about a single GPU."""
    index: int
    memory_used_mb: int
    memory_total_mb: int
    memory_free_mb: int
    utilization_percent: int


class GpuService:
    """
    Service for GPU detection, querying, and selection.

    Provides methods to:
    - Detect GPU availability on the host
    - Query current GPU usage/memory
    - Select optimal GPUs for new deployments
    - Track which GPUs are used by existing deployments

    This class uses caching for GPU detection since that rarely changes
    during a process's lifetime.
    """

    # Class-level cache for GPU detection (rarely changes)
    _gpu_available: Optional[bool] = None
    _gpu_count: Optional[int] = None

    # Default timeouts
    DIRECT_COMMAND_TIMEOUT = 10  # seconds for direct nvidia-smi
    DOCKER_COMMAND_TIMEOUT = 30  # seconds for Docker-based nvidia-smi (includes image pull)

    def __init__(self, container_prefix: str = "deployment"):
        """
        Initialize the GPU service.

        Args:
            container_prefix: Prefix used for deployment container names
                            (used to identify which GPUs are in use)
        """
        self.container_prefix = container_prefix

    async def detect_gpu(self) -> Tuple[bool, int]:
        """
        Detect if GPUs are available on the Docker host.

        Checks for nvidia runtime in Docker and uses nvidia-smi to count GPUs.
        Results are cached at the class level since GPU availability rarely changes.

        Returns:
            Tuple of (gpu_available, gpu_count)
        """
        # Return cached result if available
        if GpuService._gpu_available is not None:
            return GpuService._gpu_available, GpuService._gpu_count or 0

        gpu_available = False
        gpu_count = 0

        # Method 1: Check if nvidia runtime is available via docker info
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "info", "--format", "{{.Runtimes}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.DIRECT_COMMAND_TIMEOUT
            )

            if process.returncode == 0 and stdout:
                runtimes = stdout.decode().strip().lower()
                if "nvidia" in runtimes:
                    gpu_available = True
                    logger.info("NVIDIA Docker runtime detected")
        except Exception as e:
            logger.debug(f"docker info check failed: {e}")

        # Method 2: Get GPU count via nvidia-smi
        if gpu_available:
            gpu_count = await self._count_gpus()

        GpuService._gpu_available = gpu_available
        GpuService._gpu_count = gpu_count if gpu_available else 0
        logger.info(f"GPU detection complete: available={gpu_available}, count={gpu_count}")
        return gpu_available, gpu_count

    async def _count_gpus(self) -> int:
        """
        Count available GPUs using nvidia-smi.

        Tries direct nvidia-smi first, falls back to Docker-based approach.

        Returns:
            Number of GPUs detected, or 1 if nvidia runtime exists but count fails
        """
        gpu_count = 0

        # Try direct nvidia-smi first
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.DIRECT_COMMAND_TIMEOUT
            )

            if process.returncode == 0 and stdout:
                gpu_list = stdout.decode().strip().split('\n')
                gpu_count = len([g for g in gpu_list if g.strip()])
                logger.info(f"Detected {gpu_count} GPU(s) via nvidia-smi: {gpu_list}")
                return gpu_count
        except Exception as e:
            logger.debug(f"Direct nvidia-smi failed: {e}, trying via Docker")

        # Fallback: try nvidia-smi via Docker
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm", "--gpus", "all",
                "nvidia/cuda:12.1.0-base-ubuntu22.04",
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.DOCKER_COMMAND_TIMEOUT
            )

            if process.returncode == 0 and stdout:
                gpu_list = stdout.decode().strip().split('\n')
                gpu_count = len([g for g in gpu_list if g.strip()])
                logger.info(f"Detected {gpu_count} GPU(s) via Docker: {gpu_list}")
                return gpu_count
        except Exception as e:
            logger.debug(f"Docker nvidia-smi failed: {e}, assuming 1 GPU")

        # nvidia runtime exists, assume at least 1 GPU
        return 1

    async def query_gpu_info(self) -> List[GpuInfo]:
        """
        Query detailed information about all GPUs.

        Uses nvidia-smi to get memory usage and utilization for each GPU.
        Tries direct command first, falls back to Docker-based approach.

        Returns:
            List of GpuInfo objects, one per GPU. Empty list if query fails.
        """
        stdout = await self._run_nvidia_smi_query(
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        )

        if not stdout:
            return []

        gpu_info = []
        for line in stdout.decode().strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    util = int(parts[3])
                    mem_free = mem_total - mem_used
                    gpu_info.append(GpuInfo(
                        index=idx,
                        memory_used_mb=mem_used,
                        memory_total_mb=mem_total,
                        memory_free_mb=mem_free,
                        utilization_percent=util,
                    ))
                except ValueError:
                    continue

        return gpu_info

    async def _run_nvidia_smi_query(self, *args: str) -> Optional[bytes]:
        """
        Run nvidia-smi with the given arguments.

        Tries direct execution first, falls back to Docker-based approach.

        Args:
            *args: Arguments to pass to nvidia-smi

        Returns:
            stdout bytes if successful, None otherwise
        """
        # Try direct nvidia-smi first
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.DIRECT_COMMAND_TIMEOUT
            )
            if process.returncode == 0 and stdout:
                return stdout
        except Exception as e:
            logger.debug(f"Direct nvidia-smi failed: {e}, trying via Docker")

        # Fallback: run nvidia-smi via Docker container
        try:
            docker_cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "nvidia/cuda:12.1.0-base-ubuntu22.04",
                "nvidia-smi", *args
            ]
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.DOCKER_COMMAND_TIMEOUT
            )
            if process.returncode == 0 and stdout:
                return stdout
        except Exception as e:
            logger.debug(f"Docker nvidia-smi failed: {e}")

        return None

    async def get_gpus_used_by_deployments(self) -> set:
        """
        Get GPUs currently assigned to running deployment containers.

        Inspects all running containers matching the deployment prefix
        to determine which GPU device IDs are in use.

        Returns:
            Set of GPU indices already in use by deployments
        """
        used_gpus = set()
        try:
            # List all deployment containers
            process = await asyncio.create_subprocess_exec(
                "docker", "ps", "--filter", f"name={self.container_prefix}",
                "--format", "{{.Names}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.DIRECT_COMMAND_TIMEOUT
            )

            if process.returncode != 0 or not out:
                return used_gpus

            container_names = out.decode().strip().split('\n')
            for name in container_names:
                if not name:
                    continue

                # Get GPU assignment for each container
                inspect_proc = await asyncio.create_subprocess_exec(
                    "docker", "inspect", name,
                    "--format", "{{range .HostConfig.DeviceRequests}}{{.DeviceIDs}}{{end}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                inspect_out, _ = await asyncio.wait_for(
                    inspect_proc.communicate(), timeout=5
                )

                if inspect_proc.returncode == 0 and inspect_out:
                    # Parse [0 1 2 3] format
                    gpu_str = inspect_out.decode().strip().strip('[]')
                    if gpu_str:
                        for gpu_id in gpu_str.split():
                            try:
                                used_gpus.add(int(gpu_id))
                            except ValueError:
                                pass

            logger.info(f"GPUs already used by deployments: {sorted(used_gpus)}")
        except Exception as e:
            logger.warning(f"Failed to get GPUs used by deployments: {e}")

        return used_gpus

    async def find_available_gpu(self) -> Optional[int]:
        """
        Find the GPU with lowest utilization/memory usage.

        Queries all GPUs and returns the device ID of the one with
        lowest utilization and most free memory.

        Returns:
            GPU device ID (0, 1, 2, etc.) or 0 as default if query fails
        """
        gpus = await self.find_available_gpus(count=1)
        return gpus[0] if gpus else 0

    async def find_available_gpus(
        self,
        count: int,
        exclude_deployed: bool = True
    ) -> List[int]:
        """
        Find the N GPUs with lowest utilization/memory usage.

        Optionally excludes GPUs already assigned to other running
        deployment containers.

        Args:
            count: Number of GPUs to select
            exclude_deployed: If True, exclude GPUs used by other deployments

        Returns:
            List of GPU device IDs (e.g., [4, 5] for 2 GPUs)
        """
        # Get GPUs used by other deployments
        used_gpus = set()
        if exclude_deployed:
            used_gpus = await self.get_gpus_used_by_deployments()

        # Query GPU info
        gpu_info = await self.query_gpu_info()

        if not gpu_info:
            logger.warning(f"Could not query GPU info, defaulting to GPUs 0-{count-1}")
            return list(range(count))

        # Exclude GPUs used by other deployments
        available_gpus = [g for g in gpu_info if g.index not in used_gpus]
        logger.info(
            f"Available GPUs after excluding deployed: {[g.index for g in available_gpus]} "
            f"(excluded: {sorted(used_gpus)})"
        )

        if len(available_gpus) < count:
            logger.warning(
                f"Only {len(available_gpus)} GPUs available after excluding deployed, "
                f"requested {count}. Falling back to all GPUs."
            )
            # Fall back to all GPUs if not enough available
            available_gpus = gpu_info

        if len(available_gpus) < count:
            logger.warning(f"Only {len(available_gpus)} GPUs total, requested {count}")
            return [g.index for g in available_gpus]

        # Sort by: lowest utilization first, then most free memory
        available_gpus.sort(key=lambda g: (g.utilization_percent, -g.memory_free_mb))
        selected = available_gpus[:count]

        logger.info(
            f"Selected {count} GPUs: {[g.index for g in selected]} "
            f"(free memory: {[g.memory_free_mb for g in selected]}MB)"
        )

        # Log all GPU states for debugging
        for g in gpu_info:
            logger.debug(
                f"GPU {g.index}: {g.utilization_percent}% util, "
                f"{g.memory_used_mb}MB/{g.memory_total_mb}MB used"
            )

        return [g.index for g in selected]

    @classmethod
    def reset_cache(cls):
        """
        Reset the cached GPU detection results.

        Useful for testing or when GPU configuration changes.
        """
        cls._gpu_available = None
        cls._gpu_count = None


# Singleton instance for shared use
gpu_service = GpuService()
