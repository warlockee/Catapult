"""
Benchmark model for storing deployment benchmark results.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Benchmark(Base):
    """Benchmark result for a deployment."""

    __tablename__ = "benchmarks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=True,  # Now nullable - can be None for production endpoint benchmarks
        index=True
    )
    # For production endpoints (external URLs)
    endpoint_url = Column(String(500), nullable=True)  # Direct URL to benchmark
    production_endpoint_id = Column(Integer, nullable=True, index=True)  # EID from production

    # Benchmark configuration
    endpoint_path = Column(String(500), nullable=False, default="/health")
    method = Column(String(10), nullable=False, default="GET")
    concurrent_requests = Column(Integer, nullable=False, default=10)
    total_requests = Column(Integer, nullable=False, default=100)
    timeout_seconds = Column(Float, nullable=False, default=30.0)

    # Execution mode: 'docker' (via benchmarker container) or 'inline' (direct execution)
    execution_mode = Column(String(20), nullable=False, default="docker")
    container_id = Column(String(64), nullable=True)  # Docker container ID when execution_mode='docker'
    log_path = Column(String(500), nullable=True)  # Path to benchmark log file

    # Status
    status = Column(String(20), nullable=False, default="pending", index=True)
    # 'pending', 'running', 'completed', 'failed'
    error_message = Column(String(1000), nullable=True)

    # Progress tracking
    current_stage = Column(String(50), nullable=True)
    # Stages: 'discovering_model', 'health_check', 'inference_test',
    #         'ttft_benchmark', 'stress_test', 'finalizing'
    stage_progress = Column(String(50), nullable=True)  # e.g., "3/5 requests"
    stages_completed = Column(JSONB, default=list, nullable=False)  # List of completed stages

    # Latency metrics (in milliseconds)
    latency_avg_ms = Column(Float, nullable=True)
    latency_min_ms = Column(Float, nullable=True)
    latency_max_ms = Column(Float, nullable=True)
    latency_p50_ms = Column(Float, nullable=True)
    latency_p90_ms = Column(Float, nullable=True)
    latency_p95_ms = Column(Float, nullable=True)
    latency_p99_ms = Column(Float, nullable=True)

    # Throughput metrics
    requests_per_second = Column(Float, nullable=True)
    total_requests_sent = Column(Integer, nullable=True)
    successful_requests = Column(Integer, nullable=True)
    failed_requests = Column(Integer, nullable=True)
    error_rate = Column(Float, nullable=True)  # Percentage 0-100

    # Inference metrics (TTFT/TPS)
    ttft_avg_ms = Column(Float, nullable=True)  # Time To First Token (average)
    ttft_min_ms = Column(Float, nullable=True)
    ttft_max_ms = Column(Float, nullable=True)
    ttft_p50_ms = Column(Float, nullable=True)
    ttft_p90_ms = Column(Float, nullable=True)
    ttft_p95_ms = Column(Float, nullable=True)
    ttft_p99_ms = Column(Float, nullable=True)
    tokens_per_second_avg = Column(Float, nullable=True)  # TPS (average)
    tokens_per_second_min = Column(Float, nullable=True)
    tokens_per_second_max = Column(Float, nullable=True)
    total_tokens_generated = Column(Integer, nullable=True)
    model_id = Column(String(255), nullable=True)  # Model ID from endpoint

    # Note: WER/CER metrics moved to separate Evaluation model (app.models.evaluation)

    # Timing
    duration_seconds = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Additional data (request/response samples, etc.)
    meta_data = Column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    deployment = relationship("Deployment", backref="benchmarks")
