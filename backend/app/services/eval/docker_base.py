"""
Base class for Docker-based evaluators.

Provides common functionality for running evaluations in Docker containers:
- Container lifecycle management
- Standard progress parsing (EVAL_PROGRESS: N/M)
- Timeout handling
- Output capture

Subclasses only need to implement:
- docker_image: The Docker image to use
- build_docker_args(): Arguments to pass to the container
- parse_metrics(): Extract metrics from container output
"""
import asyncio
import logging
import re
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.eval.base import (
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationResult,
    Evaluator,
    ProgressCallback,
)

logger = logging.getLogger(__name__)


class DockerEvaluator(Evaluator):
    """
    Base class for evaluators that run in Docker containers.

    Subclasses must define:
    - docker_image: str - The Docker image name
    - build_docker_args(config) -> List[str] - Container arguments
    - parse_metrics(stdout) -> Dict[str, Any] - Extract metrics from output

    Optional overrides:
    - docker_build_context: str - Path to Dockerfile if image needs building
    - primary_metric_name: str - Name of the primary metric (default: from parse_metrics)
    - secondary_metric_name: str - Name of the secondary metric
    """

    # Subclasses must set this
    docker_image: str = ""

    # Optional: path to build context if image doesn't exist
    docker_build_path: Optional[str] = None
    dockerfile_path: Optional[str] = None

    @abstractmethod
    def build_docker_args(self, config: EvaluationConfig) -> List[str]:
        """
        Build command-line arguments for the Docker container.

        Args:
            config: Evaluation configuration

        Returns:
            List of arguments to pass to the container
        """
        pass

    @abstractmethod
    def parse_metrics(self, stdout: str) -> Dict[str, Any]:
        """
        Parse metrics from container stdout.

        Args:
            stdout: Complete stdout from the container

        Returns:
            Dict with at least 'primary_metric' and 'primary_metric_name'.
            Optional: 'secondary_metric', 'secondary_metric_name', 'samples_evaluated', etc.
        """
        pass

    def validate_config(self, config: EvaluationConfig) -> tuple[bool, Optional[str]]:
        """Default validation - requires endpoint_url and model_name."""
        if not config.endpoint_url:
            return False, "endpoint_url is required"
        if not config.model_name:
            return False, "model_name is required"
        return True, None

    async def _ensure_docker_image(self) -> tuple[bool, Optional[str]]:
        """Ensure the Docker image exists, build if necessary."""
        try:
            # Check if image exists
            check_cmd = ["docker", "images", "-q", self.docker_image]
            process = await asyncio.create_subprocess_exec(
                *check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)

            if stdout.strip():
                return True, None

            # Build if we have a build path
            if self.docker_build_path and self.dockerfile_path:
                logger.info(f"Building Docker image: {self.docker_image}")
                build_cmd = [
                    "docker", "build",
                    "-t", self.docker_image,
                    "-f", self.dockerfile_path,
                    self.docker_build_path,
                ]
                build_proc = await asyncio.create_subprocess_exec(
                    *build_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, build_err = await asyncio.wait_for(
                    build_proc.communicate(), timeout=600
                )
                if build_proc.returncode != 0:
                    error_msg = build_err.decode() if build_err else "Build failed"
                    return False, f"Failed to build image: {error_msg[:500]}"
                logger.info("Docker image built successfully")
                return True, None

            return False, f"Docker image {self.docker_image} not found"

        except asyncio.TimeoutError:
            return False, "Timeout while preparing Docker image"
        except Exception as e:
            return False, f"Failed to prepare Docker image: {str(e)}"

    async def evaluate(
        self,
        config: EvaluationConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> EvaluationResult:
        """
        Run evaluation in Docker container.

        Handles:
        - Image availability check
        - Container execution with timeout
        - Progress parsing (EVAL_PROGRESS: N/M format)
        - Metrics extraction
        """
        # Validate config
        is_valid, error = self.validate_config(config)
        if not is_valid:
            return EvaluationResult(success=False, error_message=error)

        started_at = datetime.utcnow()

        # Ensure Docker image is available
        image_ok, image_error = await self._ensure_docker_image()
        if not image_ok:
            return EvaluationResult(
                success=False,
                error_message=image_error,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        # Build Docker command
        timeout = int(config.timeout * 15)  # Allow more time for batch processing
        docker_args = self.build_docker_args(config)
        cmd = [
            "docker", "run",
            "--rm",
            "--network", "host",
            "-e", "PYTHONUNBUFFERED=1",
            self.docker_image,
        ] + docker_args

        logger.info(f"Running eval Docker: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_lines = []
            stderr_text = ""
            total_samples = config.limit if config.limit > 0 else 500

            async def read_stderr():
                nonlocal stderr_text
                data = await process.stderr.read()
                stderr_text = data.decode() if data else ""

            stderr_task = asyncio.create_task(read_stderr())

            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            process.stdout.readline(),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                        raise

                    if not line:
                        break

                    line_text = line.decode().rstrip()
                    stdout_lines.append(line_text)

                    if line_text:
                        logger.info(f"[eval-docker] {line_text}")

                    # Parse total samples from header if available
                    total_match = re.search(r'(\d+) total samples', line_text)
                    if total_match:
                        total_samples = int(total_match.group(1))

                    # Parse progress updates
                    if progress_callback:
                        current = None
                        matched_total = None

                        # Standard format: EVAL_PROGRESS: N/M
                        std_match = re.search(r'EVAL_PROGRESS:\s*(\d+)/(\d+)', line_text)
                        if std_match:
                            current = int(std_match.group(1))
                            matched_total = int(std_match.group(2))

                        # Fallback: legacy (N/M @ rate) format
                        if current is None:
                            legacy_match = re.search(r'\((\d+)/(\d+)\s+@', line_text)
                            if legacy_match:
                                current = int(legacy_match.group(1))
                                matched_total = int(legacy_match.group(2))

                        # Last fallback: [  N] ERROR format for error lines
                        if current is None:
                            bracket_match = re.search(r'\[\s*(\d+)\]\s*ERROR', line_text)
                            if bracket_match:
                                current = int(bracket_match.group(1)) + 1

                        if current is not None:
                            if matched_total is not None:
                                total_samples = matched_total
                            result = progress_callback(current, total_samples)
                            if asyncio.iscoroutine(result):
                                await result

                await process.wait()
                await stderr_task
            except Exception:
                process.kill()
                await stderr_task
                raise

            stdout_text = '\n'.join(stdout_lines)
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            # Parse metrics from output
            metrics_dict = self.parse_metrics(stdout_text)

            if process.returncode != 0:
                return EvaluationResult(
                    success=False,
                    error_message=f"Eval docker exited with code {process.returncode}: {stderr_text[:500]}",
                    metrics=self._build_metrics(metrics_dict) if metrics_dict else None,
                    duration_seconds=duration,
                    started_at=started_at,
                    completed_at=completed_at,
                )

            if not metrics_dict or 'primary_metric' not in metrics_dict:
                return EvaluationResult(
                    success=False,
                    error_message="Could not parse metrics from Docker output",
                    duration_seconds=duration,
                    started_at=started_at,
                    completed_at=completed_at,
                )

            logger.info(
                f"Evaluation complete: {metrics_dict.get('primary_metric_name', 'metric')}="
                f"{metrics_dict.get('primary_metric', 0):.4f}"
            )

            return EvaluationResult(
                success=True,
                metrics=self._build_metrics(metrics_dict),
                duration_seconds=duration,
                started_at=started_at,
                completed_at=completed_at,
            )

        except asyncio.TimeoutError:
            return EvaluationResult(
                success=False,
                error_message=f"Eval docker timed out after {timeout}s",
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Eval docker failed: {e}")
            return EvaluationResult(
                success=False,
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    def _build_metrics(self, metrics_dict: Dict[str, Any]) -> EvaluationMetrics:
        """Convert metrics dict to EvaluationMetrics."""
        return EvaluationMetrics(
            primary_metric=metrics_dict.get('primary_metric', 0.0),
            primary_metric_name=metrics_dict.get('primary_metric_name', 'metric'),
            secondary_metric=metrics_dict.get('secondary_metric'),
            secondary_metric_name=metrics_dict.get('secondary_metric_name'),
            samples_evaluated=metrics_dict.get('samples_evaluated', 0),
            samples_with_errors=metrics_dict.get('samples_with_errors', 0),
            extra_metrics=metrics_dict.get('extra_metrics', {}),
        )
