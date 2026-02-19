"""
Service for Docker build execution.

Handles:
- Docker command construction and execution
- Build logging
- Process management
"""
import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of a Docker build execution."""
    success: bool
    return_code: int
    log_path: str
    error_message: Optional[str] = None


@dataclass
class BuildCommand:
    """Docker build command configuration."""
    image_tag: str
    dockerfile_path: str
    context_path: str
    build_args: Dict[str, str] = field(default_factory=dict)
    network: str = "host"


class BuildExecutorService:
    """
    Service for Docker build execution.

    Responsibilities:
    - Construct Docker build commands
    - Execute builds with proper logging
    - Handle process lifecycle
    """

    def __init__(self, logs_dir: Optional[str] = None):
        """
        Initialize BuildExecutorService.

        Args:
            logs_dir: Directory for build logs
        """
        self.logs_dir = logs_dir or settings.DOCKER_LOGS_DIR
        os.makedirs(self.logs_dir, exist_ok=True)

    def get_log_path(self, build_id: str) -> str:
        """
        Get log file path for a build.

        Args:
            build_id: Build identifier

        Returns:
            Path to log file
        """
        return os.path.join(self.logs_dir, f"{build_id}.log")

    def build_command(self, config: BuildCommand) -> List[str]:
        """
        Construct Docker build command.

        Args:
            config: Build command configuration

        Returns:
            List of command arguments
        """
        cmd = [
            "docker", "build",
            "--progress=plain",  # Verbose output for better debugging
            "--network", config.network,
            "-t", config.image_tag,
            "-f", config.dockerfile_path,
        ]

        # Add build args
        for key, value in config.build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        # Add context path last
        cmd.append(config.context_path)

        return cmd

    async def execute(
        self,
        build_id: str,
        config: BuildCommand,
        timeout: Optional[int] = None,
    ) -> BuildResult:
        """
        Execute Docker build.

        Args:
            build_id: Build identifier for logging
            config: Build command configuration
            timeout: Optional timeout in seconds

        Returns:
            BuildResult with success status and log path

        Raises:
            BuildError: If build fails or times out
        """
        log_path = self.get_log_path(build_id)
        cmd = self.build_command(config)

        logger.info(f"Starting Docker build: {' '.join(cmd)}")

        try:
            with open(log_path, "a") as log_file:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=log_file,
                )

                if timeout:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        process.kill()
                        return BuildResult(
                            success=False,
                            return_code=-1,
                            log_path=log_path,
                            error_message=f"Build timed out after {timeout} seconds",
                        )
                else:
                    await process.wait()

            if process.returncode != 0:
                return BuildResult(
                    success=False,
                    return_code=process.returncode,
                    log_path=log_path,
                    error_message=f"Docker build failed with code {process.returncode}",
                )

            logger.info(f"Docker build completed successfully: {config.image_tag}")
            return BuildResult(
                success=True,
                return_code=0,
                log_path=log_path,
            )

        except Exception as e:
            logger.error(f"Docker build execution error: {e}")
            return BuildResult(
                success=False,
                return_code=-1,
                log_path=log_path,
                error_message=str(e),
            )

    async def stream_logs(
        self,
        log_path: str,
        follow: bool = False,
    ):
        """
        Stream build logs.

        Args:
            log_path: Path to log file
            follow: If True, follow log file (like tail -f)

        Yields:
            Log lines as they become available
        """
        if not os.path.exists(log_path):
            return

        with open(log_path, "r") as f:
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip("\n")
                elif follow:
                    await asyncio.sleep(0.1)
                else:
                    break


# Singleton instance
executor_service = BuildExecutorService()
