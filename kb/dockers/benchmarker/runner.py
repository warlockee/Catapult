"""
Main benchmark execution logic.

Supports multiple modes:
1. Generic mode: Direct load testing on any API endpoint
2. OpenAI mode: Full pipeline with model discovery, TTFT/TPS for OpenAI-compatible APIs
3. File upload mode: Load testing endpoints that require file uploads (e.g., /transcribe)
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .audio_sample import get_sample_audio
from .metrics import (
    BenchmarkConfig,
    BenchmarkResult,
    InferenceResult,
    ModelType,
    RequestResult,
    aggregate_latency_metrics,
    aggregate_ttft_metrics,
    aggregate_tps_metrics,
    calculate_percentile,
    detect_model_type,
    get_inference_endpoint,
)
from .progress import NoOpProgressReporter, ProgressReporter

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks against ML model endpoints."""

    # Endpoints that require file uploads
    FILE_UPLOAD_ENDPOINTS = {
        "/transcribe",
        "/transcribe/batch",
        "/stt",
        "/asr",
        "/v1/audio/transcriptions",
    }

    def __init__(self, progress_reporter: Optional[ProgressReporter] = None):
        self.progress = progress_reporter or NoOpProgressReporter()
        self.default_headers = {
            "User-Agent": "Benchmarker/1.0",
            "Accept": "application/json",
        }
        # Cache sample audio for file upload benchmarks
        self._sample_audio: Optional[Tuple[bytes, str]] = None

    def _requires_file_upload(self, endpoint_path: str) -> bool:
        """Check if endpoint requires file upload."""
        path_lower = endpoint_path.lower()
        return any(path_lower.startswith(ep) or path_lower.endswith(ep)
                   for ep in self.FILE_UPLOAD_ENDPOINTS)

    def _get_sample_audio(self) -> Tuple[bytes, str]:
        """Get cached sample audio for file upload benchmarks."""
        if self._sample_audio is None:
            self._sample_audio = get_sample_audio()
        return self._sample_audio

    async def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run benchmark based on configuration.

        If endpoint_path is a custom path (not /v1/*), runs in generic mode.
        Otherwise runs full OpenAI-compatible pipeline.
        """
        # Determine mode based on endpoint
        is_openai_endpoint = config.endpoint_path.startswith("/v1/")

        if is_openai_endpoint and config.server_type in (None, "vllm", "text", "llm"):
            return await self._run_openai_pipeline(config)
        else:
            return await self._run_generic_pipeline(config)

    async def _run_generic_pipeline(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run generic load test on any API endpoint.

        Simply sends requests to the specified endpoint and measures latency/throughput.
        """
        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        # Stage 1: Health check (try the endpoint itself or /health)
        await self.progress.report_stage("health_check")
        health_ok = await self._check_health_generic(config.endpoint_url)
        await self.progress.report_stage_complete("health_check", success=health_ok)

        if not health_ok:
            error = "Health check failed"
            await self.progress.report_error(error, stage="health_check")
            return BenchmarkResult(
                success=False,
                error_message=error,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        # Stage 2: Load test
        await self.progress.report_stage("load_test", f"0/{config.total_requests}")

        results = await self._run_load_test(
            config=config,
            num_requests=config.total_requests,
            concurrency=config.concurrent_requests,
        )

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        latencies = [r.latency_ms for r in results]

        if not latencies:
            error = "No requests completed"
            await self.progress.report_error(error, stage="load_test")
            return BenchmarkResult(
                success=False,
                error_message=error,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        duration = time.perf_counter() - start_time
        latency_metrics = aggregate_latency_metrics(latencies)

        await self.progress.report_stage_complete(
            "load_test",
            success=True,
            data={
                "endpoint": config.endpoint_path,
                "total_requests": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "latency_avg_ms": latency_metrics["avg_ms"],
                "requests_per_second": len(results) / duration if duration > 0 else 0,
            }
        )

        # Finalize
        await self.progress.report_stage("finalizing")

        result = BenchmarkResult(
            latency_avg_ms=latency_metrics["avg_ms"],
            latency_min_ms=latency_metrics["min_ms"],
            latency_max_ms=latency_metrics["max_ms"],
            latency_p50_ms=latency_metrics["p50_ms"],
            latency_p90_ms=latency_metrics["p90_ms"],
            latency_p95_ms=latency_metrics["p95_ms"],
            latency_p99_ms=latency_metrics["p99_ms"],
            requests_per_second=len(results) / duration if duration > 0 else 0,
            total_requests_sent=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(results) * 100) if results else 0,
            duration_seconds=duration,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            success=True,
        )

        await self.progress.report_complete(
            success=True,
            results=self._result_to_dict(result),
        )

        return result

    async def _run_load_test(
        self,
        config: BenchmarkConfig,
        num_requests: int,
        concurrency: int,
    ) -> List[RequestResult]:
        """Run concurrent load test on the endpoint."""
        url = f"{config.endpoint_url.rstrip('/')}{config.endpoint_path}"
        results: List[RequestResult] = []
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0

        # Check if this is a file upload endpoint
        is_file_upload = self._requires_file_upload(config.endpoint_path)
        is_batch_endpoint = "batch" in config.endpoint_path.lower()
        if is_file_upload:
            audio_data, audio_filename = self._get_sample_audio()
            logger.info(f"Using file upload mode for {config.endpoint_path} with {len(audio_data)} byte audio sample (batch={is_batch_endpoint})")

        async def make_request(i: int) -> RequestResult:
            nonlocal completed
            async with semaphore:
                req_start = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                        if config.method.upper() == "GET":
                            response = await client.get(url, headers=self.default_headers)
                        elif is_file_upload:
                            # Send as multipart/form-data with audio file
                            # Use "files" for batch endpoints, "file" for single
                            if is_batch_endpoint:
                                files = [("files", (audio_filename, audio_data, "audio/wav"))]
                            else:
                                files = {"file": (audio_filename, audio_data, "audio/wav")}
                            response = await client.post(
                                url,
                                files=files,
                                headers={"User-Agent": "Benchmarker/1.0"},
                            )
                        else:
                            response = await client.request(
                                config.method.upper(),
                                url,
                                json=config.request_body,
                                headers={**self.default_headers, **(config.headers or {})},
                            )
                        latency_ms = (time.perf_counter() - req_start) * 1000
                        success = 200 <= response.status_code < 400

                        completed += 1
                        if completed % max(1, num_requests // 10) == 0:
                            await self.progress.report_stage("load_test", f"{completed}/{num_requests}")

                        return RequestResult(
                            success=success,
                            latency_ms=latency_ms,
                            status_code=response.status_code,
                            error=None if success else f"HTTP {response.status_code}",
                        )
                except httpx.TimeoutException:
                    return RequestResult(
                        success=False,
                        latency_ms=(time.perf_counter() - req_start) * 1000,
                        error="Timeout",
                    )
                except Exception as e:
                    return RequestResult(
                        success=False,
                        latency_ms=(time.perf_counter() - req_start) * 1000,
                        error=str(e),
                    )

        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        await self.progress.report_stage("load_test", f"{num_requests}/{num_requests}")
        return list(results)

    async def _check_health_generic(self, endpoint_url: str) -> bool:
        """Check health via /health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try /health first
                try:
                    response = await client.get(
                        f"{endpoint_url.rstrip('/')}/health",
                        headers=self.default_headers,
                    )
                    if response.status_code == 200:
                        logger.info("Health check passed via /health")
                        return True
                except Exception:
                    pass

                # Try root
                try:
                    response = await client.get(
                        endpoint_url.rstrip('/'),
                        headers=self.default_headers,
                    )
                    if response.status_code < 500:
                        logger.info("Health check passed via root")
                        return True
                except Exception:
                    pass

                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _run_openai_pipeline(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run full OpenAI-compatible benchmark pipeline.

        Includes model discovery, inference test, TTFT/TPS measurement, stress test.
        """
        started_at = datetime.utcnow()
        start_time = time.perf_counter()

        # Stage 1: Discover model ID
        await self.progress.report_stage("discovering_model")
        model_id = await self._discover_model_id(config.endpoint_url)
        if not model_id:
            model_id = "default"

        model_type = detect_model_type(model_id, server_type=config.server_type)
        await self.progress.report_stage_complete(
            "discovering_model",
            success=True,
            data={
                "model_id": model_id,
                "model_type": model_type.value,
                "server_type": config.server_type,
            }
        )
        logger.info(f"Discovered model: {model_id} (type: {model_type.value})")

        # Stage 2: Health check
        await self.progress.report_stage("health_check")
        health_ok = await self._check_health(config.endpoint_url)
        await self.progress.report_stage_complete("health_check", success=health_ok)

        if not health_ok:
            error = "Health check failed"
            await self.progress.report_error(error, stage="health_check")
            return BenchmarkResult(
                success=False,
                error_message=error,
                model_id=model_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        # Determine inference endpoint
        inference_endpoint, _, inference_payload = get_inference_endpoint(config.server_type)
        if config.endpoint_path and config.endpoint_path.startswith("/v1/"):
            inference_endpoint = config.endpoint_path
        if config.request_body:
            inference_payload = config.request_body
        inference_payload["model"] = model_id

        # Stage 3: Inference test
        await self.progress.report_stage("inference_test")
        inference_ok, inference_msg = await self._check_inference(
            config.endpoint_url, model_id, model_type, config.timeout_seconds,
            selected_endpoint=inference_endpoint,
            selected_payload=inference_payload,
        )
        await self.progress.report_stage_complete(
            "inference_test",
            success=inference_ok,
            data={"message": inference_msg, "endpoint": inference_endpoint}
        )

        if not inference_ok and model_type == ModelType.TEXT:
            error = f"Inference test failed: {inference_msg}"
            await self.progress.report_error(error, stage="inference_test")
            return BenchmarkResult(
                success=False,
                error_message=error,
                model_id=model_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        # Check if endpoint supports streaming
        streaming_endpoints = ["/v1/chat/completions", "/v1/completions"]
        supports_streaming = inference_endpoint in streaming_endpoints

        if supports_streaming:
            # Stage 4: TTFT/TPS benchmark
            bench_result = await self._run_ttft_benchmark(
                config, model_id, inference_endpoint, inference_payload, start_time, started_at
            )

            if not bench_result.success:
                return bench_result

            # Stage 5: Stress test (optional)
            if config.concurrent_requests > 1:
                stress_result = await self._run_stress_test(
                    config, model_id, inference_endpoint
                )
                if stress_result.get("requests_per_second", 0) > 0:
                    bench_result.requests_per_second = stress_result["requests_per_second"]
                    bench_result.total_requests_sent += config.total_requests
        else:
            # Non-streaming: run load test
            bench_result = await self._run_api_load_test(
                config, model_id, inference_endpoint, inference_payload, start_time, started_at
            )

        # Stage 6: Finalizing
        await self.progress.report_stage("finalizing")
        bench_result.model_id = model_id
        bench_result.duration_seconds = time.perf_counter() - start_time
        bench_result.completed_at = datetime.utcnow()

        await self.progress.report_complete(
            success=bench_result.success,
            results=self._result_to_dict(bench_result),
        )

        return bench_result

    def _result_to_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to dictionary for JSON serialization."""
        return {
            "latency_avg_ms": result.latency_avg_ms,
            "latency_min_ms": result.latency_min_ms,
            "latency_max_ms": result.latency_max_ms,
            "latency_p50_ms": result.latency_p50_ms,
            "latency_p90_ms": result.latency_p90_ms,
            "latency_p95_ms": result.latency_p95_ms,
            "latency_p99_ms": result.latency_p99_ms,
            "requests_per_second": result.requests_per_second,
            "total_requests_sent": result.total_requests_sent,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "error_rate": result.error_rate,
            "ttft_avg_ms": result.ttft_avg_ms,
            "ttft_min_ms": result.ttft_min_ms,
            "ttft_max_ms": result.ttft_max_ms,
            "ttft_p50_ms": result.ttft_p50_ms,
            "ttft_p90_ms": result.ttft_p90_ms,
            "ttft_p95_ms": result.ttft_p95_ms,
            "ttft_p99_ms": result.ttft_p99_ms,
            "tokens_per_second_avg": result.tokens_per_second_avg,
            "tokens_per_second_min": result.tokens_per_second_min,
            "tokens_per_second_max": result.tokens_per_second_max,
            "total_tokens_generated": result.total_tokens_generated,
            "model_id": result.model_id,
            "duration_seconds": result.duration_seconds,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "success": result.success,
            "error_message": result.error_message,
        }

    async def _discover_model_id(self, endpoint_url: str, timeout: float = 10.0) -> Optional[str]:
        """Discover the model ID from /v1/models endpoint."""
        url = f"{endpoint_url.rstrip('/')}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self.default_headers)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        return data["data"][0].get("id")
        except Exception as e:
            logger.debug(f"Failed to discover model ID: {e}")
        return None

    async def _check_health(self, endpoint_url: str) -> bool:
        """Check endpoint health via multiple endpoints."""
        health_endpoints = ["/v1/models", "/health", "/v1/health", "/"]

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
        """Quick inference check - verify model responds."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
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
                else:
                    # For non-text models, just check endpoint responds
                    response = await client.post(
                        f"{endpoint_url.rstrip('/')}/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 10,
                        },
                        headers=self.default_headers,
                    )
                    if response.status_code < 500:
                        return True, f"Endpoint responsive (model_type={model_type.value})"
                    return False, f"Server error: HTTP {response.status_code}"
        except Exception as e:
            logger.warning(f"Inference check failed: {e}")
            return False, str(e)

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
        """Run a single streaming inference request and measure TTFT/TPS."""
        url = f"{endpoint_url.rstrip('/')}{endpoint_path}"

        if base_payload:
            payload = {**base_payload, "stream": True}
            if "model" in payload:
                payload["model"] = model_id
            if "messages" in payload:
                payload["messages"] = [{"role": "user", "content": prompt}]
            elif "prompt" in payload:
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
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                tokens_generated += 1
                    except json.JSONDecodeError:
                        continue

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms
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

    async def _run_ttft_benchmark(
        self,
        config: BenchmarkConfig,
        model_id: str,
        endpoint_path: str,
        request_payload: Dict[str, Any],
        start_time: float,
        started_at: datetime,
        num_requests: int = 5,
    ) -> BenchmarkResult:
        """Run TTFT/TPS benchmark with progress updates."""
        await self.progress.report_stage("ttft_benchmark", f"0/{num_requests}")

        prompts = [
            "Hello, how are you today?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
            "What is the capital of France?",
            "Count from 1 to 10.",
        ]

        inference_results: List[InferenceResult] = []

        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                for i in range(num_requests):
                    await self.progress.report_stage("ttft_benchmark", f"{i}/{num_requests}")

                    prompt = prompts[i % len(prompts)]
                    result = await self._run_inference_request(
                        client, config.endpoint_url, model_id, prompt,
                        max_tokens=100, timeout=config.timeout_seconds,
                        endpoint_path=endpoint_path,
                        base_payload=request_payload,
                    )
                    inference_results.append(result)

                    logger.debug(
                        f"Inference {i+1}/{num_requests}: "
                        f"TTFT={result.ttft_ms:.2f}ms, TPS={result.tokens_per_second:.2f}"
                    )

                await self.progress.report_stage("ttft_benchmark", f"{num_requests}/{num_requests}")

        except Exception as e:
            error = str(e)
            await self.progress.report_error(error, stage="ttft_benchmark")
            return BenchmarkResult(
                success=False,
                error_message=error,
                model_id=model_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        successful = [r for r in inference_results if r.success]
        failed = [r for r in inference_results if not r.success]

        if not successful:
            error_msgs = [r.error for r in failed if r.error]
            error = f"All inference requests failed: {'; '.join(error_msgs[:3])}"
            await self.progress.report_error(error, stage="ttft_benchmark")
            return BenchmarkResult(
                success=False,
                error_message=error,
                model_id=model_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        ttfts = [r.ttft_ms for r in successful]
        tps_values = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]
        total_tokens = sum(r.tokens_generated for r in successful)
        latencies = [r.total_time_ms for r in successful]

        latency_metrics = aggregate_latency_metrics(latencies)
        ttft_metrics = aggregate_ttft_metrics(ttfts)
        tps_metrics = aggregate_tps_metrics(tps_values)

        duration = time.perf_counter() - start_time

        await self.progress.report_stage_complete(
            "ttft_benchmark",
            success=True,
            data={
                "ttft_avg_ms": ttft_metrics["avg_ms"],
                "tokens_per_second_avg": tps_metrics["avg"],
                "total_tokens": total_tokens,
                "endpoint": endpoint_path,
            }
        )

        return BenchmarkResult(
            latency_avg_ms=latency_metrics["avg_ms"],
            latency_min_ms=latency_metrics["min_ms"],
            latency_max_ms=latency_metrics["max_ms"],
            latency_p50_ms=latency_metrics["p50_ms"],
            latency_p90_ms=latency_metrics["p90_ms"],
            latency_p95_ms=latency_metrics["p95_ms"],
            latency_p99_ms=latency_metrics["p99_ms"],
            requests_per_second=len(inference_results) / duration if duration > 0 else 0,
            total_requests_sent=len(inference_results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(inference_results) * 100) if inference_results else 0,
            ttft_avg_ms=ttft_metrics["avg_ms"],
            ttft_min_ms=ttft_metrics["min_ms"],
            ttft_max_ms=ttft_metrics["max_ms"],
            ttft_p50_ms=ttft_metrics["p50_ms"],
            ttft_p90_ms=ttft_metrics["p90_ms"],
            ttft_p95_ms=ttft_metrics["p95_ms"],
            ttft_p99_ms=ttft_metrics["p99_ms"],
            tokens_per_second_avg=tps_metrics["avg"],
            tokens_per_second_min=tps_metrics["min"],
            tokens_per_second_max=tps_metrics["max"],
            total_tokens_generated=total_tokens,
            model_id=model_id,
            duration_seconds=duration,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            success=True,
        )

    async def _run_stress_test(
        self,
        config: BenchmarkConfig,
        model_id: str,
        endpoint_path: str,
    ) -> Dict[str, Any]:
        """Run concurrent requests stress test."""
        await self.progress.report_stage("stress_test", f"0/{config.total_requests}")

        url = f"{config.endpoint_url.rstrip('/')}{endpoint_path}"
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
                async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                    response = await client.post(url, json=payload, headers=self.default_headers)
                    latencies.append((time.perf_counter() - req_start) * 1000)
                    return response.status_code == 200
            except Exception:
                return False

        semaphore = asyncio.Semaphore(config.concurrent_requests)

        async def bounded_request(i: int) -> bool:
            async with semaphore:
                return await make_request(i)

        try:
            tasks = [bounded_request(i) for i in range(config.total_requests)]
            results = await asyncio.gather(*tasks)
            success_count = sum(1 for r in results if r)
            duration_s = time.perf_counter() - start

            result = {
                "requests_per_second": config.total_requests / duration_s if duration_s > 0 else 0,
                "successful": success_count,
                "failed": config.total_requests - success_count,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            }

            await self.progress.report_stage_complete(
                "stress_test",
                success=result["failed"] == 0,
                data=result
            )

            return result

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            await self.progress.report_error(str(e), stage="stress_test")
            return {"requests_per_second": 0, "successful": 0, "failed": config.total_requests}

    async def _run_api_load_test(
        self,
        config: BenchmarkConfig,
        model_id: str,
        endpoint_path: str,
        request_payload: Dict[str, Any],
        start_time: float,
        started_at: datetime,
    ) -> BenchmarkResult:
        """Run load test for non-streaming endpoints."""
        await self.progress.report_stage("load_test", f"0/{config.total_requests}")

        url = f"{config.endpoint_url.rstrip('/')}{endpoint_path}"
        results: List[RequestResult] = []
        semaphore = asyncio.Semaphore(config.concurrent_requests)

        async def make_request() -> RequestResult:
            req_start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                    response = await client.post(
                        url,
                        json=request_payload,
                        headers=self.default_headers,
                    )
                    latency_ms = (time.perf_counter() - req_start) * 1000
                    success = 200 <= response.status_code < 400
                    return RequestResult(
                        success=success,
                        latency_ms=latency_ms,
                        status_code=response.status_code,
                        error=None if success else f"HTTP {response.status_code}",
                    )
            except httpx.TimeoutException:
                return RequestResult(
                    success=False,
                    latency_ms=(time.perf_counter() - req_start) * 1000,
                    error="Timeout",
                )
            except Exception as e:
                return RequestResult(
                    success=False,
                    latency_ms=(time.perf_counter() - req_start) * 1000,
                    error=str(e),
                )

        async def bounded_request() -> RequestResult:
            async with semaphore:
                return await make_request()

        try:
            total = min(config.total_requests, 20)
            tasks = [bounded_request() for _ in range(total)]
            results = await asyncio.gather(*tasks)

            await self.progress.report_stage("load_test", f"{total}/{total}")

        except Exception as e:
            error = str(e)
            await self.progress.report_error(error, stage="load_test")
            return BenchmarkResult(
                success=False,
                error_message=error,
                model_id=model_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        latencies = [r.latency_ms for r in results]

        if not latencies:
            error = "No requests completed"
            await self.progress.report_error(error, stage="load_test")
            return BenchmarkResult(
                success=False,
                error_message=error,
                model_id=model_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        latency_metrics = aggregate_latency_metrics(latencies)
        duration = time.perf_counter() - start_time

        await self.progress.report_stage_complete(
            "load_test",
            success=True,
            data={
                "latency_avg_ms": latency_metrics["avg_ms"],
                "latency_p95_ms": latency_metrics["p95_ms"],
                "requests_per_second": len(results) / duration if duration > 0 else 0,
                "error_rate": (len(failed) / len(results) * 100) if results else 0,
                "endpoint": endpoint_path,
            }
        )

        return BenchmarkResult(
            latency_avg_ms=latency_metrics["avg_ms"],
            latency_min_ms=latency_metrics["min_ms"],
            latency_max_ms=latency_metrics["max_ms"],
            latency_p50_ms=latency_metrics["p50_ms"],
            latency_p90_ms=latency_metrics["p90_ms"],
            latency_p95_ms=latency_metrics["p95_ms"],
            latency_p99_ms=latency_metrics["p99_ms"],
            requests_per_second=len(results) / duration if duration > 0 else 0,
            total_requests_sent=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(results) * 100) if results else 0,
            ttft_avg_ms=0,
            tokens_per_second_avg=0,
            total_tokens_generated=0,
            model_id=model_id,
            duration_seconds=duration,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            success=True,
        )
