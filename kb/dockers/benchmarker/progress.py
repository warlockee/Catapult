"""
HTTP callback client for reporting benchmark progress to the backend.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ProgressReporter:
    """
    Reports benchmark progress to the backend via HTTP callbacks.

    The backend receives progress updates and stores them in the database,
    which the frontend polls to display real-time progress.
    """

    def __init__(
        self,
        callback_base_url: str,
        benchmark_id: str,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        """
        Initialize the progress reporter.

        Args:
            callback_base_url: Base URL for callbacks (e.g., http://backend:8000/internal/benchmarks)
            benchmark_id: ID of the benchmark being executed
            timeout: HTTP timeout for callbacks
            max_retries: Number of retries for failed callbacks
        """
        self.callback_base_url = callback_base_url.rstrip("/")
        self.benchmark_id = benchmark_id
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _send_callback(self, endpoint: str, payload: Dict[str, Any]) -> bool:
        """Send a callback with retry logic."""
        url = f"{self.callback_base_url}/{self.benchmark_id}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                if not self._client:
                    self._client = httpx.AsyncClient(timeout=self.timeout)

                response = await self._client.post(url, json=payload)

                if response.status_code in (200, 201, 202):
                    return True

                logger.warning(
                    f"Callback failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"HTTP {response.status_code} from {url}"
                )

            except httpx.TimeoutException:
                logger.warning(
                    f"Callback timeout (attempt {attempt + 1}/{self.max_retries}): {url}"
                )
            except Exception as e:
                logger.warning(
                    f"Callback error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))

        logger.error(f"Callback failed after {self.max_retries} attempts: {url}")
        return False

    async def report_stage(
        self,
        stage: str,
        progress: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Report progress for the current stage.

        Args:
            stage: Current stage name (e.g., "discovering_model", "health_check")
            progress: Progress within stage (e.g., "3/5")
            data: Additional data for the stage

        Returns:
            True if callback succeeded
        """
        payload = {
            "stage": stage,
            "progress": progress,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        return await self._send_callback("progress", payload)

    async def report_stage_complete(
        self,
        stage: str,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Report that a stage has completed.

        Args:
            stage: Stage name that completed
            success: Whether the stage succeeded
            data: Stage result data (metrics, etc.)

        Returns:
            True if callback succeeded
        """
        payload = {
            "stage": stage,
            "success": success,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        return await self._send_callback("stage_complete", payload)

    async def report_complete(
        self,
        success: bool,
        results: Dict[str, Any],
    ) -> bool:
        """
        Report that the benchmark has completed.

        Args:
            success: Whether the benchmark succeeded
            results: Final benchmark results (all metrics)

        Returns:
            True if callback succeeded
        """
        payload = {
            "success": success,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return await self._send_callback("complete", payload)

    async def report_error(
        self,
        error: str,
        stage: Optional[str] = None,
    ) -> bool:
        """
        Report an error during benchmark execution.

        Args:
            error: Error message
            stage: Stage where error occurred (if known)

        Returns:
            True if callback succeeded
        """
        payload = {
            "error": error,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return await self._send_callback("error", payload)


class NoOpProgressReporter:
    """A no-op progress reporter for when callbacks are disabled."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def report_stage(self, stage: str, progress: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> bool:
        logger.info(f"[PROGRESS] Stage: {stage}, Progress: {progress}, Data: {data}")
        return True

    async def report_stage_complete(self, stage: str, success: bool, data: Optional[Dict[str, Any]] = None) -> bool:
        logger.info(f"[STAGE COMPLETE] Stage: {stage}, Success: {success}, Data: {data}")
        return True

    async def report_complete(self, success: bool, results: Dict[str, Any]) -> bool:
        logger.info(f"[COMPLETE] Success: {success}, Results: {results}")
        return True

    async def report_error(self, error: str, stage: Optional[str] = None) -> bool:
        logger.error(f"[ERROR] Stage: {stage}, Error: {error}")
        return True
