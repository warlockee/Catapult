"""
Benchmark service for running HTTP load tests and inference benchmarks against deployments.

This service provides:
- HTTP load testing (latency, throughput, error rates)
- Inference benchmarks (TTFT, TPS) for OpenAI-compatible APIs
- Model-type-aware benchmarking (text, audio, multimodal)
"""
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DeploymentNotFoundError
from app.models.benchmark import Benchmark
from app.models.deployment import Deployment
from app.models.model import Model
from app.models.release import Release
from app.schemas.benchmark import BenchmarkSummary

logger = logging.getLogger(__name__)


def get_inference_endpoint(server_type: str) -> tuple[str, str, dict]:
    """
    Get the inference endpoint and test payload for a given server type.

    Returns:
        Tuple of (endpoint_path, method, test_payload)
    """
    server_type_lower = (server_type or "").lower()

    if server_type_lower == "audio-generation":
        # TTS models use /v1/audio/speech
        return (
            "/v1/audio/speech",
            "POST",
            {
                "model": "default",
                "input": "Hello, this is a benchmark test.",
                "voice": "en_woman"
            }
        )
    elif server_type_lower in ("audio-understanding", "asr", "asr-vllm"):
        # ASR/audio understanding models (use chat completions, WER eval handles audio)
        return (
            "/v1/chat/completions",
            "POST",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        )
    elif server_type_lower == "embedding":
        # Embedding models
        return (
            "/v1/embeddings",
            "POST",
            {
                "model": "default",
                "input": "Hello, this is a test."
            }
        )
    else:
        # Default: vllm/text models use chat completions
        return (
            "/v1/chat/completions",
            "POST",
            {
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
                "stream": True
            }
        )


class ModelType(str, Enum):
    """Type of model being benchmarked."""
    TEXT = "text"           # Standard text LLM
    AUDIO = "audio"         # Audio/speech model
    MULTIMODAL = "multimodal"  # Multimodal (text + audio/image)
    UNKNOWN = "unknown"     # Unknown type


def detect_model_type(model_id: str, server_type: Optional[str] = None) -> ModelType:
    """
    Detect model type from model ID and/or server_type.

    server_type is authoritative if provided (from model metadata).
    Falls back to model_id heuristics if server_type is not available.

    Args:
        model_id: Model identifier from /v1/models or request body
        server_type: Server type from model metadata (authoritative)

    Returns:
        Detected ModelType
    """
    # server_type is authoritative - use it first
    if server_type:
        server_type_lower = server_type.lower()
        if server_type_lower in ('audio-generation', 'audio-understanding', 'tts', 'asr', 'asr-vllm'):
            return ModelType.AUDIO
        if server_type_lower in ('embedding', 'embeddings'):
            return ModelType.TEXT  # Embeddings are non-streaming like audio
        if server_type_lower in ('multimodal', 'vision'):
            return ModelType.MULTIMODAL
        # vllm and other text types
        if server_type_lower in ('vllm', 'text', 'llm'):
            return ModelType.TEXT

    # Fallback to model_id heuristics
    if not model_id:
        return ModelType.TEXT

    model_id_lower = model_id.lower()

    # Audio models (including generation and understanding)
    audio_keywords = [
        'audio-generation', 'audio-understanding', 'audio_generation', 'audio_understanding',
        'audio', 'speech', 'tts', 'asr', 'voice', 'whisper'
    ]
    if any(kw in model_id_lower for kw in audio_keywords):
        # Check if it's multimodal (audio + text)
        if any(kw in model_id_lower for kw in ['multimodal', 'omni', 'unified']):
            return ModelType.MULTIMODAL
        return ModelType.AUDIO

    # Vision/multimodal models
    if any(kw in model_id_lower for kw in ['vision', 'image', 'multimodal', 'omni', '-vl', '_vl']):
        return ModelType.MULTIMODAL

    # Default to text
    return ModelType.TEXT


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    endpoint_url: str
    endpoint_path: str = "/health"
    method: str = "GET"
    concurrent_requests: int = 10
    total_requests: int = 100
    timeout_seconds: float = 30.0
    request_body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class RequestResult:
    """Result of a single HTTP request."""
    success: bool
    latency_ms: float
    status_code: Optional[int] = None
    error: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of a single inference request."""
    success: bool
    ttft_ms: float  # Time to first token
    total_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    error: Optional[str] = None




@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    # Latency metrics (ms)
    latency_avg_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    total_requests_sent: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # Inference metrics (TTFT/TPS)
    ttft_avg_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tokens_per_second_avg: float = 0.0
    tokens_per_second_min: float = 0.0
    tokens_per_second_max: float = 0.0
    total_tokens_generated: int = 0
    model_id: Optional[str] = None

    # Note: WER/CER metrics moved to separate Evaluation service

    # Timing
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Errors
    error_message: Optional[str] = None
    success: bool = True


class BenchmarkService:
    """Service for running HTTP benchmarks."""

    def __init__(self):
        self.default_headers = {
            "User-Agent": "ModelRegistry-Benchmark/1.0",
            "Accept": "application/json",
        }

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        config: BenchmarkConfig,
    ) -> RequestResult:
        """Make a single HTTP request and measure latency."""
        url = f"{config.endpoint_url.rstrip('/')}{config.endpoint_path}"
        start_time = time.perf_counter()

        try:
            if config.method.upper() == "GET":
                response = await client.get(url, headers=config.headers)
            elif config.method.upper() == "POST":
                response = await client.post(
                    url,
                    json=config.request_body,
                    headers=config.headers,
                )
            elif config.method.upper() == "PUT":
                response = await client.put(
                    url,
                    json=config.request_body,
                    headers=config.headers,
                )
            elif config.method.upper() == "DELETE":
                response = await client.delete(url, headers=config.headers)
            else:
                response = await client.request(
                    config.method.upper(),
                    url,
                    json=config.request_body,
                    headers=config.headers,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000
            success = 200 <= response.status_code < 400

            return RequestResult(
                success=success,
                latency_ms=latency_ms,
                status_code=response.status_code,
                error=None if success else f"HTTP {response.status_code}",
            )

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency_ms,
                error="Timeout",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _run_worker(
        self,
        client: httpx.AsyncClient,
        config: BenchmarkConfig,
        semaphore: asyncio.Semaphore,
        results: List[RequestResult],
    ) -> None:
        """Worker that makes requests with concurrency control."""
        async with semaphore:
            result = await self._make_request(client, config)
            results.append(result)

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a benchmark against the configured endpoint.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult with aggregated metrics
        """
        logger.info(
            f"Starting benchmark: {config.total_requests} requests, "
            f"{config.concurrent_requests} concurrent to {config.endpoint_url}{config.endpoint_path}"
        )

        results: List[RequestResult] = []
        semaphore = asyncio.Semaphore(config.concurrent_requests)

        headers = {**self.default_headers, **config.headers}
        config.headers = headers

        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                tasks = [
                    self._run_worker(client, config, semaphore, results)
                    for _ in range(config.total_requests)
                ]
                await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                success=False,
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        end_time = time.perf_counter()
        completed_at = datetime.utcnow()
        duration_seconds = end_time - start_time

        # Calculate metrics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        latencies = [r.latency_ms for r in results]

        if not latencies:
            return BenchmarkResult(
                success=False,
                error_message="No requests completed",
                started_at=started_at,
                completed_at=completed_at,
            )

        result = BenchmarkResult(
            latency_avg_ms=statistics.mean(latencies),
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_p50_ms=self._calculate_percentile(latencies, 50),
            latency_p90_ms=self._calculate_percentile(latencies, 90),
            latency_p95_ms=self._calculate_percentile(latencies, 95),
            latency_p99_ms=self._calculate_percentile(latencies, 99),
            requests_per_second=len(results) / duration_seconds if duration_seconds > 0 else 0,
            total_requests_sent=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(results) * 100) if results else 0,
            duration_seconds=duration_seconds,
            started_at=started_at,
            completed_at=completed_at,
            success=True,
        )

        logger.info(
            f"Benchmark completed: {result.requests_per_second:.2f} req/s, "
            f"avg latency {result.latency_avg_ms:.2f}ms, "
            f"error rate {result.error_rate:.1f}%"
        )

        return result

    async def _discover_model_id(self, endpoint_url: str, timeout: float = 10.0) -> Optional[str]:
        """
        Discover the model ID from an OpenAI-compatible /v1/models endpoint.

        Args:
            endpoint_url: Base URL of the deployment
            timeout: Request timeout

        Returns:
            Model ID string or None if not found
        """
        url = f"{endpoint_url.rstrip('/')}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self.default_headers)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        model_id = data["data"][0].get("id")
                        logger.info(f"Discovered model ID: {model_id}")
                        return model_id
        except Exception as e:
            logger.debug(f"Failed to discover model ID from {url}: {e}")
        return None

    async def _run_inference_request(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        model_id: str,
        prompt: str,
        max_tokens: int = 100,
        timeout: float = 60.0,
        endpoint_path: str = "/v1/chat/completions",
        base_payload: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """
        Run a single streaming inference request and measure TTFT/TPS.

        Args:
            client: HTTP client
            endpoint_url: Base URL of the deployment
            model_id: Model ID to use
            prompt: Prompt text
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
            endpoint_path: API endpoint path (default: /v1/chat/completions)
            base_payload: Optional base payload to use (will add stream=True)

        Returns:
            InferenceResult with timing metrics
        """
        url = f"{endpoint_url.rstrip('/')}{endpoint_path}"

        # Build payload - use base_payload if provided, otherwise default chat format
        if base_payload:
            payload = {**base_payload, "stream": True}
            # Always use the discovered model_id (overrides frontend's model name)
            if "model" in payload:
                payload["model"] = model_id
            # Update the prompt/content in the payload based on format
            if "messages" in payload:
                # Chat completions format
                payload["messages"] = [{"role": "user", "content": prompt}]
            elif "prompt" in payload:
                # Text completions format (/v1/completions)
                payload["prompt"] = prompt
        else:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            }

        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0

        try:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={**self.default_headers, "Accept": "text/event-stream"},
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    return InferenceResult(
                        success=False,
                        ttft_ms=0,
                        total_time_ms=(time.perf_counter() - start_time) * 1000,
                        tokens_generated=0,
                        tokens_per_second=0,
                        error=f"HTTP {response.status_code}",
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break

                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Count token if "content" key exists (even if empty - for audio models)
                            # or if there's actual content
                            if "content" in delta:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                tokens_generated += 1
                    except json.JSONDecodeError:
                        continue

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms

            # Calculate TPS (tokens per second) from first token to end
            generation_time = end_time - first_token_time if first_token_time else end_time - start_time
            tps = tokens_generated / generation_time if generation_time > 0 else 0

            return InferenceResult(
                success=True,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                tokens_generated=tokens_generated,
                tokens_per_second=tps,
            )

        except httpx.TimeoutException:
            return InferenceResult(
                success=False,
                ttft_ms=0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                tokens_generated=tokens_generated,
                tokens_per_second=0,
                error="Timeout",
            )
        except Exception as e:
            return InferenceResult(
                success=False,
                ttft_ms=0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                tokens_generated=tokens_generated,
                tokens_per_second=0,
                error=str(e),
            )

    async def run_inference_benchmark(
        self,
        endpoint_url: str,
        num_requests: int = 5,
        timeout_seconds: float = 60.0,
    ) -> BenchmarkResult:
        """
        Run inference benchmark to measure TTFT and TPS.

        This runs actual LLM inference requests with streaming to measure:
        - TTFT (Time To First Token): How long until the first token is received
        - TPS (Tokens Per Second): Token generation speed

        Args:
            endpoint_url: Base URL of the deployment
            num_requests: Number of inference requests to run
            timeout_seconds: Timeout per request

        Returns:
            BenchmarkResult with inference metrics
        """
        logger.info(f"Starting inference benchmark: {num_requests} requests to {endpoint_url}")

        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        # Discover model ID
        model_id = await self._discover_model_id(endpoint_url, timeout=10.0)
        if not model_id:
            logger.warning("Could not discover model ID, using fallback")
            model_id = "default"

        # Test prompts for inference
        prompts = [
            "Hello, how are you today?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
            "What is the capital of France?",
            "Count from 1 to 10.",
        ]

        inference_results: List[InferenceResult] = []

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                for i in range(num_requests):
                    prompt = prompts[i % len(prompts)]
                    result = await self._run_inference_request(
                        client,
                        endpoint_url,
                        model_id,
                        prompt,
                        max_tokens=100,
                        timeout=timeout_seconds,
                    )
                    inference_results.append(result)
                    logger.debug(
                        f"Inference {i+1}/{num_requests}: "
                        f"TTFT={result.ttft_ms:.2f}ms, "
                        f"TPS={result.tokens_per_second:.2f}, "
                        f"tokens={result.tokens_generated}"
                    )

        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            return BenchmarkResult(
                success=False,
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                model_id=model_id,
            )

        end_time = time.perf_counter()
        completed_at = datetime.utcnow()
        duration_seconds = end_time - start_time

        # Calculate metrics
        successful = [r for r in inference_results if r.success]
        failed = [r for r in inference_results if not r.success]

        if not successful:
            error_msgs = [r.error for r in failed if r.error]
            return BenchmarkResult(
                success=False,
                error_message=f"All inference requests failed: {'; '.join(error_msgs[:3])}",
                started_at=started_at,
                completed_at=completed_at,
                model_id=model_id,
            )

        # TTFT metrics
        ttfts = [r.ttft_ms for r in successful]
        tps_values = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]
        total_tokens = sum(r.tokens_generated for r in successful)
        latencies = [r.total_time_ms for r in successful]

        result = BenchmarkResult(
            # Latency metrics (total request time)
            latency_avg_ms=statistics.mean(latencies) if latencies else 0,
            latency_min_ms=min(latencies) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            latency_p50_ms=self._calculate_percentile(latencies, 50),
            latency_p90_ms=self._calculate_percentile(latencies, 90),
            latency_p95_ms=self._calculate_percentile(latencies, 95),
            latency_p99_ms=self._calculate_percentile(latencies, 99),
            # Throughput
            requests_per_second=len(inference_results) / duration_seconds if duration_seconds > 0 else 0,
            total_requests_sent=len(inference_results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(inference_results) * 100) if inference_results else 0,
            # TTFT metrics
            ttft_avg_ms=statistics.mean(ttfts) if ttfts else 0,
            ttft_min_ms=min(ttfts) if ttfts else 0,
            ttft_max_ms=max(ttfts) if ttfts else 0,
            ttft_p50_ms=self._calculate_percentile(ttfts, 50),
            ttft_p90_ms=self._calculate_percentile(ttfts, 90),
            ttft_p95_ms=self._calculate_percentile(ttfts, 95),
            ttft_p99_ms=self._calculate_percentile(ttfts, 99),
            # TPS metrics
            tokens_per_second_avg=statistics.mean(tps_values) if tps_values else 0,
            tokens_per_second_min=min(tps_values) if tps_values else 0,
            tokens_per_second_max=max(tps_values) if tps_values else 0,
            total_tokens_generated=total_tokens,
            model_id=model_id,
            # Timing
            duration_seconds=duration_seconds,
            started_at=started_at,
            completed_at=completed_at,
            success=True,
        )

        logger.info(
            f"Inference benchmark completed: "
            f"TTFT avg={result.ttft_avg_ms:.2f}ms, "
            f"TPS avg={result.tokens_per_second_avg:.2f}, "
            f"total tokens={result.total_tokens_generated}"
        )

        return result

    async def _run_stress_test(
        self,
        endpoint_url: str,
        model_id: str,
        concurrency: int = 5,
        total_requests: int = 20,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Run concurrent requests stress test. Returns dict with metrics."""
        url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
        start = time.perf_counter()
        latencies: List[float] = []

        async def make_request(i: int) -> bool:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "max_tokens": 50,
                "stream": False,
            }
            req_start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=self.default_headers)
                    latencies.append((time.perf_counter() - req_start) * 1000)
                    return response.status_code == 200
            except Exception:
                return False

        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(i: int) -> bool:
            async with semaphore:
                return await make_request(i)

        try:
            tasks = [bounded_request(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks)
            success_count = sum(1 for r in results if r)
            duration_s = time.perf_counter() - start

            return {
                "requests_per_second": total_requests / duration_s if duration_s > 0 else 0,
                "successful": success_count,
                "failed": total_requests - success_count,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            }
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return {"requests_per_second": 0, "successful": 0, "failed": total_requests}

    async def run_full_verification(
        self,
        endpoint_url: str,
        run_stress_test: bool = True,
        stress_concurrency: int = 5,
        stress_requests: int = 20,
        timeout_seconds: float = 60.0,
    ) -> BenchmarkResult:
        """
        Run verification suite: health → inference → TTFT/TPS → stress test.

        Args:
            endpoint_url: Base URL of the deployment
            run_stress_test: Whether to run stress test
            stress_concurrency: Concurrent requests for stress test
            stress_requests: Total requests for stress test
            timeout_seconds: Timeout per request

        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Starting verification for {endpoint_url}")

        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        # Discover model ID
        model_id = await self._discover_model_id(endpoint_url, timeout=10.0)
        if not model_id:
            logger.warning("Could not discover model ID, using fallback")
            model_id = "default"

        # Detect model type
        model_type = detect_model_type(model_id)
        logger.info(f"Detected model type: {model_type.value} for model {model_id}")

        # Test 1: Health check
        logger.info("Running health check...")
        health_ok = await self._check_health(endpoint_url)
        if not health_ok:
            return BenchmarkResult(
                success=False,
                error_message="Health check failed",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                model_id=model_id,
            )

        # Test 2: Inference test (model-type-aware)
        logger.info("Running inference test...")
        inference_ok, inference_msg = await self._check_inference(
            endpoint_url, model_id, model_type, timeout_seconds
        )
        # For audio/multimodal models, continue even if inference test fails
        if not inference_ok and model_type == ModelType.TEXT:
            return BenchmarkResult(
                success=False,
                error_message=f"Inference test failed: {inference_msg}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                model_id=model_id,
            )

        # Test 3: TTFT/TPS benchmark (streaming)
        logger.info("Running TTFT/TPS benchmark...")
        inference_result = await self.run_inference_benchmark(
            endpoint_url=endpoint_url,
            num_requests=5,
            timeout_seconds=timeout_seconds,
        )

        if not inference_result.success:
            return inference_result

        # Test 4: Stress test (optional)
        stress_rps = 0.0
        if run_stress_test:
            logger.info(f"Running stress test ({stress_concurrency} concurrent, {stress_requests} total)...")
            stress_result = await self._run_stress_test(
                endpoint_url, model_id, stress_concurrency, stress_requests, timeout_seconds
            )
            stress_rps = stress_result.get("requests_per_second", 0)

        end_time = time.perf_counter()

        result = BenchmarkResult(
            latency_avg_ms=inference_result.latency_avg_ms,
            latency_min_ms=inference_result.latency_min_ms,
            latency_max_ms=inference_result.latency_max_ms,
            latency_p50_ms=inference_result.latency_p50_ms,
            latency_p90_ms=inference_result.latency_p90_ms,
            latency_p95_ms=inference_result.latency_p95_ms,
            latency_p99_ms=inference_result.latency_p99_ms,
            requests_per_second=stress_rps if stress_rps > 0 else inference_result.requests_per_second,
            total_requests_sent=inference_result.total_requests_sent + (stress_requests if run_stress_test else 0),
            successful_requests=inference_result.successful_requests,
            failed_requests=inference_result.failed_requests,
            error_rate=inference_result.error_rate,
            ttft_avg_ms=inference_result.ttft_avg_ms,
            ttft_min_ms=inference_result.ttft_min_ms,
            ttft_max_ms=inference_result.ttft_max_ms,
            ttft_p50_ms=inference_result.ttft_p50_ms,
            ttft_p90_ms=inference_result.ttft_p90_ms,
            ttft_p95_ms=inference_result.ttft_p95_ms,
            ttft_p99_ms=inference_result.ttft_p99_ms,
            tokens_per_second_avg=inference_result.tokens_per_second_avg,
            tokens_per_second_min=inference_result.tokens_per_second_min,
            tokens_per_second_max=inference_result.tokens_per_second_max,
            total_tokens_generated=inference_result.total_tokens_generated,
            model_id=model_id,
            duration_seconds=end_time - start_time,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            success=True,
        )

        logger.info(
            f"Verification completed: TTFT avg={result.ttft_avg_ms:.2f}ms, "
            f"TPS avg={result.tokens_per_second_avg:.2f}, model={model_id}"
        )

        return result

    async def _check_health(self, endpoint_url: str) -> bool:
        """Quick health check - try multiple endpoints to verify service is up."""
        # Try multiple health check endpoints in order of preference
        health_endpoints = [
            "/v1/models",     # Standard OpenAI-compatible
            "/health",        # Common health endpoint
            "/v1/health",     # vLLM-style health
            "/",              # Root endpoint
        ]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for endpoint in health_endpoints:
                    try:
                        response = await client.get(
                            f"{endpoint_url.rstrip('/')}{endpoint}",
                            headers=self.default_headers,
                        )
                        if response.status_code == 200:
                            logger.info(f"Health check passed via {endpoint}")
                            return True
                    except Exception as e:
                        logger.debug(f"Health check failed for {endpoint}: {e}")
                        continue

                # All endpoints failed
                logger.error("Health check failed - no endpoints responded")
                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _check_inference(
        self,
        endpoint_url: str,
        model_id: str,
        model_type: ModelType,
        timeout: float,
        selected_endpoint: Optional[str] = None,
        selected_payload: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        Quick inference check - verify model responds to a simple prompt.

        Adapts the test based on model type:
        - TEXT: Simple text chat completion
        - AUDIO/MULTIMODAL: Just verify endpoint responds (skip content check)

        Args:
            endpoint_url: Base URL of the deployment
            model_id: Model identifier
            model_type: Detected model type
            timeout: Request timeout
            selected_endpoint: Optional user-selected endpoint to test
            selected_payload: Optional payload for the selected endpoint

        Returns:
            Tuple of (success, message)
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # If user selected a specific endpoint, test that one
                if selected_endpoint and selected_endpoint.startswith("/v1/"):
                    payload = selected_payload or {"model": model_id}
                    try:
                        response = await client.post(
                            f"{endpoint_url.rstrip('/')}{selected_endpoint}",
                            json=payload,
                            headers=self.default_headers,
                        )
                        if response.status_code == 200:
                            return True, f"Inference OK via {selected_endpoint}"
                        return False, f"HTTP {response.status_code} from {selected_endpoint}"
                    except Exception as e:
                        return False, f"Failed to reach {selected_endpoint}: {e}"

                if model_type == ModelType.TEXT:
                    # Standard text model - send simple chat completion
                    response = await client.post(
                        f"{endpoint_url.rstrip('/')}/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 10,
                        },
                        headers=self.default_headers,
                    )
                    if response.status_code == 200:
                        return True, "Text inference OK"
                    return False, f"HTTP {response.status_code}"
                elif model_type == ModelType.AUDIO:
                    # Audio model - try audio-specific endpoints first
                    # TTS models use /v1/audio/speech, other audio models may use /v1/inference
                    inference_endpoints = [
                        ("/v1/audio/speech", {"model": model_id, "input": "Hello", "voice": "en_woman"}),
                        ("/v1/inference", {"input": "Hello, this is a test."}),
                        ("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "test"}], "max_tokens": 10}),
                    ]
                    for endpoint, payload in inference_endpoints:
                        try:
                            response = await client.post(
                                f"{endpoint_url.rstrip('/')}{endpoint}",
                                json=payload,
                                headers=self.default_headers,
                            )
                            if response.status_code == 200:
                                return True, f"Audio inference OK via {endpoint}"
                        except Exception:
                            continue
                    return False, "Audio inference endpoints not responsive"
                else:
                    # Multimodal - try chat completions first, then inference
                    response = await client.post(
                        f"{endpoint_url.rstrip('/')}/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 10,
                        },
                        headers=self.default_headers,
                    )
                    # Accept any response that's not a server error
                    if response.status_code < 500:
                        return True, f"Endpoint responsive (model_type={model_type.value})"
                    return False, f"Server error: HTTP {response.status_code}"
        except Exception as e:
            logger.warning(f"Inference check failed: {e}")
            return False, str(e)

    async def create_benchmark(
        self,
        db: AsyncSession,
        deployment_id: UUID,
        endpoint_path: str = "/health",
        method: str = "GET",
        concurrent_requests: int = 10,
        total_requests: int = 100,
        timeout_seconds: float = 30.0,
        request_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        execution_mode: str = "docker",
    ) -> Benchmark:
        """
        Create a new benchmark record.

        Args:
            db: Database session
            deployment_id: ID of the deployment to benchmark
            endpoint_path: Path to benchmark (e.g., /health)
            method: HTTP method
            concurrent_requests: Number of concurrent requests
            total_requests: Total number of requests to make
            timeout_seconds: Timeout per request
            request_body: Optional request body for POST/PUT
            headers: Optional custom headers

        Returns:
            Created Benchmark record
        """
        # Verify deployment exists and is running
        result = await db.execute(
            select(Deployment).where(Deployment.id == deployment_id)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            raise DeploymentNotFoundError(str(deployment_id))

        if not deployment.endpoint_url:
            raise ValueError("Deployment has no endpoint URL configured")

        if deployment.status != "running":
            raise ValueError(f"Deployment is not running (status: {deployment.status})")

        benchmark = Benchmark(
            deployment_id=deployment_id,
            endpoint_path=endpoint_path,
            method=method.upper(),
            concurrent_requests=concurrent_requests,
            total_requests=total_requests,
            timeout_seconds=timeout_seconds,
            execution_mode=execution_mode,
            status="pending",
            meta_data={
                "request_body": request_body,
                "headers": headers,
            },
        )

        db.add(benchmark)
        await db.commit()
        await db.refresh(benchmark)

        return benchmark

    async def create_production_benchmark(
        self,
        db: AsyncSession,
        endpoint_url: str,
        production_endpoint_id: Optional[int] = None,
        endpoint_path: str = "/health",
        method: str = "GET",
        concurrent_requests: int = 10,
        total_requests: int = 100,
        timeout_seconds: float = 30.0,
        request_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        execution_mode: str = "docker",
    ) -> Benchmark:
        """
        Create a benchmark for a production endpoint (external URL).

        Args:
            db: Database session
            endpoint_url: Full URL to benchmark (e.g., http://hostname:port)
            production_endpoint_id: Optional EID from production endpoints
            endpoint_path: Path to benchmark (e.g., /v1/chat/completions)
            method: HTTP method
            concurrent_requests: Number of concurrent requests
            total_requests: Total number of requests to make
            timeout_seconds: Timeout per request
            request_body: Optional request body for POST/PUT
            headers: Optional custom headers

        Returns:
            Created Benchmark record
        """
        if not endpoint_url:
            raise ValueError("endpoint_url is required for production benchmarks")

        benchmark = Benchmark(
            deployment_id=None,  # No deployment for production endpoints
            endpoint_url=endpoint_url,
            production_endpoint_id=production_endpoint_id,
            endpoint_path=endpoint_path,
            method=method.upper(),
            concurrent_requests=concurrent_requests,
            total_requests=total_requests,
            timeout_seconds=timeout_seconds,
            execution_mode=execution_mode,
            status="pending",
            meta_data={
                "request_body": request_body,
                "headers": headers,
            },
        )

        db.add(benchmark)
        await db.commit()
        await db.refresh(benchmark)

        return benchmark

    async def get_production_benchmark_summary(
        self,
        db: AsyncSession,
        production_endpoint_id: int,
    ) -> BenchmarkSummary:
        """
        Get benchmark summary for a production endpoint.

        Args:
            db: Database session
            production_endpoint_id: EID from production endpoints

        Returns:
            BenchmarkSummary with latest benchmark stats
        """
        # Get latest completed benchmark for this production endpoint
        result = await db.execute(
            select(Benchmark)
            .where(Benchmark.production_endpoint_id == production_endpoint_id)
            .where(Benchmark.status == "completed")
            .order_by(Benchmark.created_at.desc())
            .limit(1)
        )
        benchmark = result.scalar_one_or_none()

        if not benchmark:
            return BenchmarkSummary(has_data=False)

        # Extract model_type and benchmark_endpoint from stages_completed
        model_type = None
        benchmark_endpoint = None
        stages = benchmark.stages_completed or []
        for stage in stages:
            if stage.get("stage") == "discovering_model":
                model_type = stage.get("model_type")
            elif stage.get("stage") in ("inference_benchmark", "inference_test", "ttft_benchmark"):
                if stage.get("endpoint"):
                    benchmark_endpoint = stage.get("endpoint")

        # Fallback: use endpoint_path from benchmark if not found in stages
        if not benchmark_endpoint:
            benchmark_endpoint = benchmark.endpoint_path

        return BenchmarkSummary(
            has_data=True,
            last_run_at=benchmark.completed_at or benchmark.created_at,
            status=benchmark.status,
            model_id=benchmark.model_id,
            model_type=model_type,
            benchmark_endpoint=benchmark_endpoint,
            latency_avg_ms=benchmark.latency_avg_ms,
            latency_p50_ms=benchmark.latency_p50_ms,
            latency_p95_ms=benchmark.latency_p95_ms,
            latency_p99_ms=benchmark.latency_p99_ms,
            requests_per_second=benchmark.requests_per_second,
            total_requests=benchmark.total_requests_sent,
            error_rate=benchmark.error_rate,
            ttft_avg_ms=benchmark.ttft_avg_ms,
            ttft_p50_ms=benchmark.ttft_p50_ms,
            ttft_p95_ms=benchmark.ttft_p95_ms,
            tokens_per_second_avg=benchmark.tokens_per_second_avg,
            total_tokens_generated=benchmark.total_tokens_generated,
        )

    async def _update_benchmark_stage(
        self,
        db: AsyncSession,
        benchmark: Benchmark,
        stage: str,
        progress: Optional[str] = None,
        completed_stage: Optional[dict] = None,
    ) -> None:
        """
        Update benchmark progress tracking.

        Args:
            db: Database session
            benchmark: Benchmark record to update
            stage: Current stage name
            progress: Progress within stage (e.g., "3/5")
            completed_stage: Stage result to add to stages_completed
        """
        benchmark.current_stage = stage
        benchmark.stage_progress = progress

        if completed_stage:
            stages = list(benchmark.stages_completed or [])
            stages.append(completed_stage)
            benchmark.stages_completed = stages

        await db.commit()

    async def execute_benchmark(
        self,
        db: AsyncSession,
        benchmark_id: UUID,
    ) -> Benchmark:
        """
        Execute a benchmark and update the record with results.

        Tracks progress through stages:
        1. discovering_model - Get model ID from endpoint
        2. health_check - Verify endpoint is healthy
        3. inference_test - Quick inference sanity check
        4. ttft_benchmark - TTFT/TPS measurement (5 requests)
        5. stress_test - Concurrent load test
        6. finalizing - Computing final results

        Args:
            db: Database session
            benchmark_id: ID of the benchmark to execute

        Returns:
            Updated Benchmark record with results
        """
        # Get benchmark with deployment
        result = await db.execute(
            select(Benchmark).where(Benchmark.id == benchmark_id)
        )
        benchmark = result.scalar_one_or_none()

        if not benchmark:
            raise ValueError(f"Benchmark not found: {benchmark_id}")

        # Determine endpoint URL - either from benchmark directly (production) or from deployment
        endpoint_url = None
        server_type = None

        if benchmark.endpoint_url:
            # Production endpoint benchmark - URL is stored directly
            endpoint_url = benchmark.endpoint_url
            # Get server_type from production endpoint's backend_type
            server_type = None
            if benchmark.production_endpoint_id:
                try:
                    from app.services.production_deployment_service import production_deployment_service
                    prod_endpoint = await production_deployment_service.get_endpoint_by_eid(
                        benchmark.production_endpoint_id
                    )
                    if prod_endpoint:
                        server_type = prod_endpoint.get("backend_type")
                        logger.info(f"Using backend_type from production endpoint: {server_type}")
                except Exception as e:
                    logger.warning(f"Failed to get production endpoint backend_type: {e}")
        elif benchmark.deployment_id:
            # Local deployment benchmark - get URL from deployment
            result = await db.execute(
                select(Deployment).where(Deployment.id == benchmark.deployment_id)
            )
            deployment = result.scalar_one_or_none()

            if not deployment or not deployment.endpoint_url:
                benchmark.status = "failed"
                benchmark.error_message = "Deployment not found or has no endpoint"
                await db.commit()
                return benchmark

            endpoint_url = deployment.endpoint_url

            # Get server_type from deployment -> release -> model
            if deployment.release_id:
                result = await db.execute(
                    select(Release).where(Release.id == deployment.release_id)
                )
                release = result.scalar_one_or_none()
                if release:
                    result = await db.execute(
                        select(Model).where(Model.id == release.image_id)
                    )
                    model = result.scalar_one_or_none()
                    if model:
                        server_type = model.server_type
        else:
            benchmark.status = "failed"
            benchmark.error_message = "Benchmark has no target (no endpoint_url or deployment_id)"
            await db.commit()
            return benchmark

        # Update status to running
        benchmark.status = "running"
        benchmark.started_at = datetime.utcnow()
        benchmark.stages_completed = []
        await db.commit()
        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        # Determine inference endpoint based on server_type
        inference_endpoint, inference_method, inference_payload = get_inference_endpoint(server_type)

        # Auto-detect endpoint for deployments if not explicitly specified
        # If deployment_id is set and endpoint_path is default "/health", use server_type-based detection
        if benchmark.deployment_id and benchmark.endpoint_path == "/health":
            # Auto-detect: use server_type-based endpoint for inference benchmark
            benchmark.endpoint_path = inference_endpoint
            benchmark.method = inference_method
            if not benchmark.meta_data:
                benchmark.meta_data = {}
            if "request_body" not in benchmark.meta_data:
                benchmark.meta_data["request_body"] = inference_payload
            logger.info(f"Auto-detected endpoint {inference_endpoint} for deployment benchmark based on server_type {server_type}")
            await db.commit()

        # Check if this is an inference benchmark
        is_inference_benchmark = (
            benchmark.method.upper() == "POST" and
            "/v1/" in benchmark.endpoint_path
        )

        if not is_inference_benchmark:
            # Run regular HTTP load benchmark (no detailed progress tracking)
            await self._update_benchmark_stage(db, benchmark, "load_test", "starting")
            config = BenchmarkConfig(
                endpoint_url=endpoint_url,
                endpoint_path=benchmark.endpoint_path,
                method=benchmark.method,
                concurrent_requests=benchmark.concurrent_requests,
                total_requests=benchmark.total_requests,
                timeout_seconds=benchmark.timeout_seconds,
                request_body=benchmark.meta_data.get("request_body"),
                headers=benchmark.meta_data.get("headers") or {},
            )
            bench_result = await self.run_benchmark(config)
            await self._update_benchmark_stage(
                db, benchmark, "completed", None,
                {"stage": "load_test", "success": bench_result.success}
            )
        else:
            # Run inference benchmark with detailed progress tracking
            target_id = benchmark.deployment_id or f"prod-endpoint-{benchmark.production_endpoint_id}"
            logger.info(f"Running inference benchmark for {target_id}")

            # Stage 1: Discover model ID and detect model type
            await self._update_benchmark_stage(db, benchmark, "discovering_model")
            model_id = await self._discover_model_id(endpoint_url, timeout=10.0)

            # If model discovery failed, try to get model name from request_body
            model_name_for_detection = model_id
            request_body = benchmark.meta_data.get("request_body") if benchmark.meta_data else None
            if request_body and isinstance(request_body, dict):
                request_model = request_body.get("model")
                if request_model:
                    # Use request model for type detection (more reliable)
                    model_name_for_detection = request_model
                    if not model_id:
                        model_id = request_model

            if not model_id:
                model_id = "default"

            # Detect model type - server_type is authoritative, fall back to model name heuristics
            model_type = detect_model_type(model_name_for_detection or model_id, server_type=server_type)

            benchmark.model_id = model_id
            await self._update_benchmark_stage(
                db, benchmark, "discovering_model", None,
                {
                    "stage": "discovering_model",
                    "success": True,
                    "model_id": model_id,
                    "model_type": model_type.value,
                    "server_type": server_type,
                    "detected_from": "server_type" if server_type else ("request_body" if model_name_for_detection != model_id else "endpoint"),
                }
            )
            logger.info(f"Detected model type: {model_type.value} for model {model_id} (server_type={server_type}, model_name={model_name_for_detection})")

            # Stage 2: Health check
            await self._update_benchmark_stage(db, benchmark, "health_check")
            health_ok = await self._check_health(endpoint_url)
            await self._update_benchmark_stage(
                db, benchmark, "health_check", None,
                {"stage": "health_check", "success": health_ok}
            )
            if not health_ok:
                benchmark.status = "failed"
                benchmark.error_message = "Health check failed"
                benchmark.current_stage = "failed"
                benchmark.completed_at = datetime.utcnow()
                await db.commit()
                return benchmark

            # Determine user-selected endpoint and payload for inference test and benchmark
            user_selected_endpoint = benchmark.endpoint_path
            selected_inference_endpoint = None
            selected_inference_payload = None

            if user_selected_endpoint and user_selected_endpoint.startswith("/v1/"):
                # User selected a specific endpoint - use it
                selected_inference_endpoint = user_selected_endpoint
                request_body = benchmark.meta_data.get("request_body") if benchmark.meta_data else None
                if request_body:
                    # Use frontend payload but override model with discovered model_id
                    selected_inference_payload = dict(request_body)
                    if "model" in selected_inference_payload:
                        selected_inference_payload["model"] = model_id
                elif user_selected_endpoint == "/v1/audio/speech":
                    selected_inference_payload = {
                        "model": model_id,
                        "input": "Hello, this is a benchmark test.",
                        "voice": "en_woman"
                    }
                elif user_selected_endpoint == "/v1/completions":
                    # Text completions format (non-chat)
                    selected_inference_payload = {
                        "model": model_id,
                        "prompt": "Hello, how are you?",
                        "max_tokens": 50,
                    }
                elif user_selected_endpoint == "/v1/embeddings":
                    selected_inference_payload = {
                        "model": model_id,
                        "input": "Hello, this is a test.",
                    }
                elif user_selected_endpoint == "/v1/inference":
                    # Generic inference endpoint (used by audio/custom models)
                    selected_inference_payload = {
                        "input": "Hello, this is a benchmark test for the inference endpoint.",
                        "parameters": {},
                    }
                elif user_selected_endpoint == "/v1/generate":
                    # Text generation endpoint
                    selected_inference_payload = {
                        "model": model_id,
                        "prompt": "Hello, how are you?",
                        "max_tokens": 50,
                    }
                else:
                    # Default: chat completions format
                    selected_inference_payload = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 50,
                    }
            elif model_type == ModelType.AUDIO and server_type is None:
                # TTS/Audio models use /v1/audio/speech endpoint
                selected_inference_endpoint = "/v1/audio/speech"
                selected_inference_payload = {
                    "model": model_id,
                    "input": "Hello, this is a benchmark test.",
                    "voice": "en_woman"
                }
            elif model_type in (ModelType.AUDIO, ModelType.MULTIMODAL):
                # Use the server_type-based endpoint or fallback
                selected_inference_endpoint = inference_endpoint
                selected_inference_payload = inference_payload

            # Stage 3: Inference test (model-type-aware, non-fatal for audio/multimodal)
            await self._update_benchmark_stage(db, benchmark, "inference_test")
            inference_ok, inference_msg = await self._check_inference(
                endpoint_url, model_id, model_type, benchmark.timeout_seconds,
                selected_endpoint=selected_inference_endpoint,
                selected_payload=selected_inference_payload,
            )
            await self._update_benchmark_stage(
                db, benchmark, "inference_test", None,
                {"stage": "inference_test", "success": inference_ok, "message": inference_msg, "endpoint": selected_inference_endpoint}
            )
            # For text models, fail if inference test failed
            if not inference_ok and model_type == ModelType.TEXT:
                benchmark.status = "failed"
                benchmark.error_message = "Inference test failed"
                benchmark.current_stage = "failed"
                benchmark.completed_at = datetime.utcnow()
                await db.commit()
                return benchmark

            # Determine if endpoint supports streaming (TTFT/TPS) or just load test
                streaming_endpoints = ["/v1/chat/completions", "/v1/completions"]
                supports_streaming = selected_inference_endpoint in streaming_endpoints

                # For non-streaming endpoints (audio/speech, embeddings, etc.), run load test only
                if not supports_streaming:
                    await self._update_benchmark_stage(
                        db, benchmark, "inference_benchmark", "0/20",
                    )

                    config = BenchmarkConfig(
                        endpoint_url=endpoint_url,
                        endpoint_path=selected_inference_endpoint,
                        method="POST",
                        concurrent_requests=benchmark.concurrent_requests,
                        total_requests=min(benchmark.total_requests, 20),
                        timeout_seconds=benchmark.timeout_seconds,
                        request_body=selected_inference_payload,
                    )
                    api_result = await self.run_benchmark(config)

                    await self._update_benchmark_stage(
                        db, benchmark, "inference_benchmark", None,
                        {
                            "stage": "inference_benchmark",
                            "success": api_result.success,
                            "endpoint": selected_inference_endpoint,
                            "model_type": model_type.value,
                            "latency_avg_ms": api_result.latency_avg_ms,
                            "latency_p95_ms": api_result.latency_p95_ms,
                            "requests_per_second": api_result.requests_per_second,
                            "error_rate": api_result.error_rate,
                        }
                    )

                    # Build result with API benchmark metrics (no TTFT/TPS for non-streaming)
                    bench_result = BenchmarkResult(
                        latency_avg_ms=api_result.latency_avg_ms,
                        latency_min_ms=api_result.latency_min_ms,
                        latency_max_ms=api_result.latency_max_ms,
                        latency_p50_ms=api_result.latency_p50_ms,
                        latency_p90_ms=api_result.latency_p90_ms,
                        latency_p95_ms=api_result.latency_p95_ms,
                        latency_p99_ms=api_result.latency_p99_ms,
                        requests_per_second=api_result.requests_per_second,
                        total_requests_sent=api_result.total_requests_sent,
                        successful_requests=api_result.successful_requests,
                        failed_requests=api_result.failed_requests,
                        error_rate=api_result.error_rate,
                        ttft_avg_ms=0,
                        tokens_per_second_avg=0,
                        total_tokens_generated=0,
                        model_id=model_id,
                        duration_seconds=time.perf_counter() - start_time,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        success=api_result.success,
                        error_message=api_result.error_message,
                    )
                else:
                    # Stage 4: TTFT/TPS benchmark (with per-request progress) - for streaming endpoints
                    await self._update_benchmark_stage(db, benchmark, "ttft_benchmark", "0/5")
                    inference_result = await self._run_inference_benchmark_with_progress(
                        db, benchmark, endpoint_url, model_id,
                        num_requests=5, timeout_seconds=benchmark.timeout_seconds,
                        endpoint_path=selected_inference_endpoint,
                        request_payload=selected_inference_payload,
                    )

                    # Update benchmark with TTFT/TPS results
                    benchmark.ttft_avg_ms = inference_result.ttft_avg_ms
                    benchmark.ttft_min_ms = inference_result.ttft_min_ms
                    benchmark.ttft_max_ms = inference_result.ttft_max_ms
                    benchmark.ttft_p50_ms = inference_result.ttft_p50_ms
                    benchmark.ttft_p90_ms = inference_result.ttft_p90_ms
                    benchmark.ttft_p95_ms = inference_result.ttft_p95_ms
                    benchmark.ttft_p99_ms = inference_result.ttft_p99_ms
                    benchmark.tokens_per_second_avg = inference_result.tokens_per_second_avg
                    benchmark.tokens_per_second_min = inference_result.tokens_per_second_min
                    benchmark.tokens_per_second_max = inference_result.tokens_per_second_max
                    benchmark.total_tokens_generated = inference_result.total_tokens_generated

                    await self._update_benchmark_stage(
                        db, benchmark, "ttft_benchmark", None,
                        {
                            "stage": "ttft_benchmark",
                            "success": inference_result.success,
                            "endpoint": selected_inference_endpoint,
                            "ttft_avg_ms": inference_result.ttft_avg_ms,
                            "tokens_per_second_avg": inference_result.tokens_per_second_avg,
                            "total_tokens": inference_result.total_tokens_generated,
                        }
                    )

                    if not inference_result.success:
                        benchmark.status = "failed"
                        benchmark.error_message = inference_result.error_message
                        benchmark.current_stage = "failed"
                        benchmark.completed_at = datetime.utcnow()
                        await db.commit()
                        return benchmark

                    # Stage 5: Stress test (optional) - only for TEXT models
                    stress_rps = 0.0
                    run_stress = benchmark.concurrent_requests > 1
                    if run_stress:
                        await self._update_benchmark_stage(
                            db, benchmark, "stress_test",
                            f"0/{benchmark.total_requests}"
                        )
                        stress_result = await self._run_stress_test(
                            endpoint_url, model_id,
                            benchmark.concurrent_requests,
                            benchmark.total_requests,
                            benchmark.timeout_seconds
                        )
                        stress_rps = stress_result.get("requests_per_second", 0)
                        await self._update_benchmark_stage(
                            db, benchmark, "stress_test", None,
                            {
                                "stage": "stress_test",
                                "success": stress_result.get("failed", 0) == 0,
                                "requests_per_second": stress_rps,
                                "successful": stress_result.get("successful", 0),
                                "failed": stress_result.get("failed", 0),
                            }
                        )

                    # Build final result for TEXT models
                    bench_result = BenchmarkResult(
                        latency_avg_ms=inference_result.latency_avg_ms,
                        latency_min_ms=inference_result.latency_min_ms,
                        latency_max_ms=inference_result.latency_max_ms,
                        latency_p50_ms=inference_result.latency_p50_ms,
                        latency_p90_ms=inference_result.latency_p90_ms,
                        latency_p95_ms=inference_result.latency_p95_ms,
                        latency_p99_ms=inference_result.latency_p99_ms,
                        requests_per_second=stress_rps if stress_rps > 0 else inference_result.requests_per_second,
                        total_requests_sent=inference_result.total_requests_sent + (benchmark.total_requests if run_stress else 0),
                        successful_requests=inference_result.successful_requests,
                        failed_requests=inference_result.failed_requests,
                        error_rate=inference_result.error_rate,
                        ttft_avg_ms=inference_result.ttft_avg_ms,
                        ttft_min_ms=inference_result.ttft_min_ms,
                        ttft_max_ms=inference_result.ttft_max_ms,
                        ttft_p50_ms=inference_result.ttft_p50_ms,
                        ttft_p90_ms=inference_result.ttft_p90_ms,
                        ttft_p95_ms=inference_result.ttft_p95_ms,
                        ttft_p99_ms=inference_result.ttft_p99_ms,
                        tokens_per_second_avg=inference_result.tokens_per_second_avg,
                        tokens_per_second_min=inference_result.tokens_per_second_min,
                        tokens_per_second_max=inference_result.tokens_per_second_max,
                        total_tokens_generated=inference_result.total_tokens_generated,
                        model_id=model_id,
                        duration_seconds=time.perf_counter() - start_time,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        success=True,
                    )

            # Stage 6: Finalizing (for all model types)
            await self._update_benchmark_stage(db, benchmark, "finalizing")

        # Update record with final results
        benchmark.status = "completed" if bench_result.success else "failed"
        benchmark.current_stage = "completed" if bench_result.success else "failed"
        benchmark.stage_progress = None
        benchmark.error_message = bench_result.error_message
        benchmark.latency_avg_ms = bench_result.latency_avg_ms
        benchmark.latency_min_ms = bench_result.latency_min_ms
        benchmark.latency_max_ms = bench_result.latency_max_ms
        benchmark.latency_p50_ms = bench_result.latency_p50_ms
        benchmark.latency_p90_ms = bench_result.latency_p90_ms
        benchmark.latency_p95_ms = bench_result.latency_p95_ms
        benchmark.latency_p99_ms = bench_result.latency_p99_ms
        benchmark.requests_per_second = bench_result.requests_per_second
        benchmark.total_requests_sent = bench_result.total_requests_sent
        benchmark.successful_requests = bench_result.successful_requests
        benchmark.failed_requests = bench_result.failed_requests
        benchmark.error_rate = bench_result.error_rate
        benchmark.duration_seconds = bench_result.duration_seconds
        benchmark.started_at = bench_result.started_at
        benchmark.completed_at = bench_result.completed_at

        # Update inference-specific metrics (TTFT/TPS)
        benchmark.ttft_avg_ms = bench_result.ttft_avg_ms
        benchmark.ttft_min_ms = bench_result.ttft_min_ms
        benchmark.ttft_max_ms = bench_result.ttft_max_ms
        benchmark.ttft_p50_ms = bench_result.ttft_p50_ms
        benchmark.ttft_p90_ms = bench_result.ttft_p90_ms
        benchmark.ttft_p95_ms = bench_result.ttft_p95_ms
        benchmark.ttft_p99_ms = bench_result.ttft_p99_ms
        benchmark.tokens_per_second_avg = bench_result.tokens_per_second_avg
        benchmark.tokens_per_second_min = bench_result.tokens_per_second_min
        benchmark.tokens_per_second_max = bench_result.tokens_per_second_max
        benchmark.total_tokens_generated = bench_result.total_tokens_generated
        benchmark.model_id = bench_result.model_id

        await db.commit()
        await db.refresh(benchmark)

        return benchmark

    async def _run_inference_benchmark_with_progress(
        self,
        db: AsyncSession,
        benchmark: Benchmark,
        endpoint_url: str,
        model_id: str,
        num_requests: int = 5,
        timeout_seconds: float = 60.0,
        endpoint_path: str = "/v1/chat/completions",
        request_payload: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        Run inference benchmark with progress updates.

        Updates benchmark.stage_progress after each request completes.

        Args:
            endpoint_path: API endpoint to benchmark (default: /v1/chat/completions)
            request_payload: Base payload for requests (will add stream=True)
        """
        logger.info(f"Starting inference benchmark: {num_requests} requests to {endpoint_url}{endpoint_path}")

        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        prompts = [
            "Hello, how are you today?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
            "What is the capital of France?",
            "Count from 1 to 10.",
        ]

        inference_results: List[InferenceResult] = []

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                for i in range(num_requests):
                    # Update progress
                    await self._update_benchmark_stage(
                        db, benchmark, "ttft_benchmark", f"{i}/{num_requests}"
                    )

                    prompt = prompts[i % len(prompts)]
                    result = await self._run_inference_request(
                        client, endpoint_url, model_id, prompt,
                        max_tokens=100, timeout=timeout_seconds,
                        endpoint_path=endpoint_path,
                        base_payload=request_payload,
                    )
                    inference_results.append(result)

                    logger.debug(
                        f"Inference {i+1}/{num_requests}: "
                        f"TTFT={result.ttft_ms:.2f}ms, "
                        f"TPS={result.tokens_per_second:.2f}, "
                        f"tokens={result.tokens_generated}"
                    )

                # Update final progress
                await self._update_benchmark_stage(
                    db, benchmark, "ttft_benchmark", f"{num_requests}/{num_requests}"
                )

        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            return BenchmarkResult(
                success=False,
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                model_id=model_id,
            )

        end_time = time.perf_counter()
        completed_at = datetime.utcnow()
        duration_seconds = end_time - start_time

        # Calculate metrics
        successful = [r for r in inference_results if r.success]
        failed = [r for r in inference_results if not r.success]

        if not successful:
            error_msgs = [r.error for r in failed if r.error]
            return BenchmarkResult(
                success=False,
                error_message=f"All inference requests failed: {'; '.join(error_msgs[:3])}",
                started_at=started_at,
                completed_at=completed_at,
                model_id=model_id,
            )

        # TTFT metrics
        ttfts = [r.ttft_ms for r in successful]
        tps_values = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]
        total_tokens = sum(r.tokens_generated for r in successful)
        latencies = [r.total_time_ms for r in successful]

        return BenchmarkResult(
            latency_avg_ms=statistics.mean(latencies) if latencies else 0,
            latency_min_ms=min(latencies) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            latency_p50_ms=self._calculate_percentile(latencies, 50),
            latency_p90_ms=self._calculate_percentile(latencies, 90),
            latency_p95_ms=self._calculate_percentile(latencies, 95),
            latency_p99_ms=self._calculate_percentile(latencies, 99),
            requests_per_second=len(inference_results) / duration_seconds if duration_seconds > 0 else 0,
            total_requests_sent=len(inference_results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(inference_results) * 100) if inference_results else 0,
            ttft_avg_ms=statistics.mean(ttfts) if ttfts else 0,
            ttft_min_ms=min(ttfts) if ttfts else 0,
            ttft_max_ms=max(ttfts) if ttfts else 0,
            ttft_p50_ms=self._calculate_percentile(ttfts, 50),
            ttft_p90_ms=self._calculate_percentile(ttfts, 90),
            ttft_p95_ms=self._calculate_percentile(ttfts, 95),
            ttft_p99_ms=self._calculate_percentile(ttfts, 99),
            tokens_per_second_avg=statistics.mean(tps_values) if tps_values else 0,
            tokens_per_second_min=min(tps_values) if tps_values else 0,
            tokens_per_second_max=max(tps_values) if tps_values else 0,
            total_tokens_generated=total_tokens,
            model_id=model_id,
            duration_seconds=duration_seconds,
            started_at=started_at,
            completed_at=completed_at,
            success=True,
        )

    async def get_latest_benchmark(
        self,
        db: AsyncSession,
        deployment_id: UUID,
    ) -> Optional[Benchmark]:
        """
        Get the most recent completed benchmark for a deployment.

        Args:
            db: Database session
            deployment_id: ID of the deployment

        Returns:
            Latest Benchmark or None
        """
        result = await db.execute(
            select(Benchmark)
            .where(Benchmark.deployment_id == deployment_id)
            .where(Benchmark.status == "completed")
            .order_by(desc(Benchmark.completed_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_benchmarks(
        self,
        db: AsyncSession,
        deployment_id: UUID,
        limit: int = 10,
    ) -> List[Benchmark]:
        """
        List benchmarks for a deployment.

        Args:
            db: Database session
            deployment_id: ID of the deployment
            limit: Maximum number of benchmarks to return

        Returns:
            List of Benchmark records
        """
        result = await db.execute(
            select(Benchmark)
            .where(Benchmark.deployment_id == deployment_id)
            .order_by(desc(Benchmark.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def cleanup_orphaned_jobs(self, db: AsyncSession) -> int:
        """
        Mark any benchmarks stuck in 'running' or 'pending' status as cancelled.

        This should be called on startup to clean up jobs that were interrupted
        by a service restart.

        Returns:
            Number of orphaned jobs cleaned up
        """
        from sqlalchemy import update

        result = await db.execute(
            update(Benchmark)
            .where(Benchmark.status.in_(['running', 'pending']))
            .values(
                status='cancelled',
                error_message='Cancelled - orphaned after service restart',
                completed_at=datetime.utcnow(),
            )
            .returning(Benchmark.id)
        )
        orphaned_ids = result.fetchall()
        await db.commit()

        count = len(orphaned_ids)
        if count > 0:
            logger.info(f"Cleaned up {count} orphaned benchmark jobs")
        return count


# Singleton instance
benchmark_service = BenchmarkService()
