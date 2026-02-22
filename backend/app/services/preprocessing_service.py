"""Generic model preprocessing via Docker containers.

Runs a user-specified Docker container as a preprocessing step on model files.
Not specific to any model family — works for format conversion, quantization,
tokenizer generation, etc.
"""
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    success: bool
    log_path: str
    error_message: Optional[str] = None


class PreprocessingService:
    """Run a preprocessing step on model files via Docker container."""

    def __init__(self, logs_dir: Optional[str] = None):
        self.logs_dir = logs_dir or settings.DOCKER_LOGS_DIR

    async def run(
        self,
        job_id: str,
        image: str,
        command: list[str],
        mounts: dict[str, str],
        gpu: bool = False,
    ) -> PreprocessingResult:
        """
        Run `docker run` with the given image, command, and volume mounts.
        Streams stdout/stderr to a log file.

        Args:
            job_id: Unique ID for log file naming
            image: Docker image to run
            command: Command + args
            mounts: {host_path: container_path} volume mounts
            gpu: Whether to allocate GPU

        Returns:
            PreprocessingResult with success/failure + log path
        """
        log_path = os.path.join(self.logs_dir, f"preprocess_{job_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        docker_cmd = ["docker", "run", "--rm"]
        if gpu:
            docker_cmd += ["--gpus", "all"]
        for host_path, container_path in mounts.items():
            docker_cmd += ["-v", f"{host_path}:{container_path}"]
        docker_cmd += [image] + command

        logger.info(f"Running preprocessing: {' '.join(docker_cmd)}")

        try:
            with open(log_path, "w") as log_file:
                log_file.write(f"Running: {' '.join(docker_cmd)}\n\n")
                process = subprocess.Popen(
                    docker_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                process.wait()

            if process.returncode != 0:
                return PreprocessingResult(
                    success=False,
                    log_path=log_path,
                    error_message=f"Container exited with code {process.returncode}",
                )
            return PreprocessingResult(success=True, log_path=log_path)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return PreprocessingResult(
                success=False,
                log_path=log_path,
                error_message=str(e),
            )


preprocessing_service = PreprocessingService()
