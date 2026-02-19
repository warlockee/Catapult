"""
Metric calculation utilities for benchmark results.
"""
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class ModelType(str, Enum):
    """Type of model being benchmarked."""
    TEXT = "text"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


def detect_model_type(model_id: str, server_type: Optional[str] = None) -> ModelType:
    """
    Detect model type from model ID and/or server_type.

    server_type is authoritative if provided (from model metadata).
    Falls back to model_id heuristics if server_type is not available.
    """
    if server_type:
        server_type_lower = server_type.lower()
        if server_type_lower in ('audio-generation', 'audio-understanding', 'tts', 'asr', 'asr-vllm'):
            return ModelType.AUDIO
        if server_type_lower in ('embedding', 'embeddings'):
            return ModelType.TEXT
        if server_type_lower in ('multimodal', 'vision'):
            return ModelType.MULTIMODAL
        if server_type_lower in ('vllm', 'text', 'llm'):
            return ModelType.TEXT

    if not model_id:
        return ModelType.TEXT

    model_id_lower = model_id.lower()

    audio_keywords = [
        'audio-generation', 'audio-understanding', 'audio_generation', 'audio_understanding',
        'audio', 'speech', 'tts', 'asr', 'voice', 'whisper'
    ]
    if any(kw in model_id_lower for kw in audio_keywords):
        if any(kw in model_id_lower for kw in ['multimodal', 'omni', 'unified']):
            return ModelType.MULTIMODAL
        return ModelType.AUDIO

    if any(kw in model_id_lower for kw in ['vision', 'image', 'multimodal', 'omni', '-vl', '_vl']):
        return ModelType.MULTIMODAL

    return ModelType.TEXT


def get_inference_endpoint(server_type: str) -> tuple[str, str, dict]:
    """
    Get the inference endpoint and test payload for a given server type.

    Returns:
        Tuple of (endpoint_path, method, test_payload)
    """
    server_type_lower = (server_type or "").lower()

    if server_type_lower == "audio-generation":
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
        return (
            "/v1/embeddings",
            "POST",
            {
                "model": "default",
                "input": "Hello, this is a test."
            }
        )
    else:
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
    server_type: Optional[str] = None


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
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    latency_avg_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    requests_per_second: float = 0.0
    total_requests_sent: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
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
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    success: bool = True


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def aggregate_latency_metrics(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency metrics from a list of latencies."""
    if not latencies:
        return {
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    return {
        "avg_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": calculate_percentile(latencies, 50),
        "p90_ms": calculate_percentile(latencies, 90),
        "p95_ms": calculate_percentile(latencies, 95),
        "p99_ms": calculate_percentile(latencies, 99),
    }


def aggregate_ttft_metrics(ttfts: List[float]) -> Dict[str, float]:
    """Calculate TTFT metrics from a list of TTFT values."""
    return aggregate_latency_metrics(ttfts)


def aggregate_tps_metrics(tps_values: List[float]) -> Dict[str, float]:
    """Calculate TPS metrics from a list of TPS values."""
    if not tps_values:
        return {
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "avg": statistics.mean(tps_values),
        "min": min(tps_values),
        "max": max(tps_values),
    }
