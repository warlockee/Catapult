"""
Pydantic schemas for Benchmark.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID


class BenchmarkStatus(str, Enum):
    """Status of a benchmark run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkCreate(BaseModel):
    """Schema for creating a benchmark."""
    deployment_id: Optional[UUID] = None
    endpoint_url: Optional[str] = Field(default=None, max_length=500)  # For production endpoints
    production_endpoint_id: Optional[int] = None  # EID from production endpoints
    endpoint_path: str = Field(default="/health", max_length=500)
    method: str = Field(default="GET", pattern="^(GET|POST|PUT|DELETE|PATCH)$")
    concurrent_requests: int = Field(default=10, ge=1, le=100)
    total_requests: int = Field(default=100, ge=1, le=10000)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    request_body: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    # Execution mode: 'docker' runs via benchmarker container, 'inline' runs directly
    execution_mode: str = Field(default="docker", pattern="^(docker|inline)$")

    # Note: ASR evaluation (WER/CER) moved to separate /evaluations endpoint

    @property
    def has_target(self) -> bool:
        """Check if benchmark has a valid target."""
        return bool(self.deployment_id or self.endpoint_url)


class BenchmarkResponse(BaseModel):
    """Schema for benchmark response."""
    id: UUID
    deployment_id: Optional[UUID] = None
    endpoint_url: Optional[str] = None
    production_endpoint_id: Optional[int] = None
    endpoint_path: str
    method: str
    concurrent_requests: int
    total_requests: int
    timeout_seconds: float
    status: str
    error_message: Optional[str] = None
    # Execution mode and container tracking
    execution_mode: Optional[str] = None  # 'docker' or 'inline'
    container_id: Optional[str] = None
    log_path: Optional[str] = None

    # Progress tracking
    current_stage: Optional[str] = None
    stage_progress: Optional[str] = None  # e.g., "3/5"
    stages_completed: List[Dict[str, Any]] = Field(default_factory=list)

    # Latency metrics (milliseconds)
    latency_avg_ms: Optional[float] = None
    latency_min_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    # Throughput metrics
    requests_per_second: Optional[float] = None
    total_requests_sent: Optional[int] = None
    successful_requests: Optional[int] = None
    failed_requests: Optional[int] = None
    error_rate: Optional[float] = None

    # Inference metrics (TTFT/TPS)
    ttft_avg_ms: Optional[float] = None
    ttft_min_ms: Optional[float] = None
    ttft_max_ms: Optional[float] = None
    ttft_p50_ms: Optional[float] = None
    ttft_p90_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tokens_per_second_avg: Optional[float] = None
    tokens_per_second_min: Optional[float] = None
    tokens_per_second_max: Optional[float] = None
    total_tokens_generated: Optional[int] = None
    model_id: Optional[str] = None

    # Note: WER/CER metrics moved to separate /evaluations endpoint

    # Timing
    duration_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True

    @classmethod
    def from_benchmark(cls, benchmark) -> "BenchmarkResponse":
        """Convert a Benchmark ORM instance to response schema."""
        return cls(
            id=benchmark.id,
            deployment_id=benchmark.deployment_id,
            endpoint_url=benchmark.endpoint_url,
            production_endpoint_id=benchmark.production_endpoint_id,
            endpoint_path=benchmark.endpoint_path,
            method=benchmark.method,
            concurrent_requests=benchmark.concurrent_requests,
            total_requests=benchmark.total_requests,
            timeout_seconds=benchmark.timeout_seconds,
            status=benchmark.status,
            error_message=benchmark.error_message,
            execution_mode=benchmark.execution_mode,
            container_id=benchmark.container_id,
            log_path=benchmark.log_path,
            current_stage=benchmark.current_stage,
            stage_progress=benchmark.stage_progress,
            stages_completed=benchmark.stages_completed or [],
            latency_avg_ms=benchmark.latency_avg_ms,
            latency_min_ms=benchmark.latency_min_ms,
            latency_max_ms=benchmark.latency_max_ms,
            latency_p50_ms=benchmark.latency_p50_ms,
            latency_p90_ms=benchmark.latency_p90_ms,
            latency_p95_ms=benchmark.latency_p95_ms,
            latency_p99_ms=benchmark.latency_p99_ms,
            requests_per_second=benchmark.requests_per_second,
            total_requests_sent=benchmark.total_requests_sent,
            successful_requests=benchmark.successful_requests,
            failed_requests=benchmark.failed_requests,
            error_rate=benchmark.error_rate,
            ttft_avg_ms=benchmark.ttft_avg_ms,
            ttft_min_ms=benchmark.ttft_min_ms,
            ttft_max_ms=benchmark.ttft_max_ms,
            ttft_p50_ms=benchmark.ttft_p50_ms,
            ttft_p90_ms=benchmark.ttft_p90_ms,
            ttft_p95_ms=benchmark.ttft_p95_ms,
            ttft_p99_ms=benchmark.ttft_p99_ms,
            tokens_per_second_avg=benchmark.tokens_per_second_avg,
            tokens_per_second_min=benchmark.tokens_per_second_min,
            tokens_per_second_max=benchmark.tokens_per_second_max,
            total_tokens_generated=benchmark.total_tokens_generated,
            model_id=benchmark.model_id,
            duration_seconds=benchmark.duration_seconds,
            started_at=benchmark.started_at,
            completed_at=benchmark.completed_at,
            created_at=benchmark.created_at,
            metadata=benchmark.meta_data or {},
        )


class BenchmarkSummary(BaseModel):
    """Summary of benchmark stats for display."""
    has_data: bool = False

    # Latest benchmark info
    last_run_at: Optional[datetime] = None
    status: Optional[str] = None
    model_id: Optional[str] = None
    model_type: Optional[str] = None  # text, audio, multimodal
    benchmark_endpoint: Optional[str] = None  # The API endpoint that was benchmarked

    # Latency (ms)
    latency_avg_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    # Throughput
    requests_per_second: Optional[float] = None
    total_requests: Optional[int] = None
    error_rate: Optional[float] = None

    # Inference metrics (TTFT/TPS)
    ttft_avg_ms: Optional[float] = None
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    tokens_per_second_avg: Optional[float] = None
    total_tokens_generated: Optional[int] = None

    # Note: WER/CER metrics moved to separate /evaluations endpoint
