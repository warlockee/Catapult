"""
Evaluation model for storing quality evaluation results.

Separate from Benchmark (performance metrics) for clean separation of concerns.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Evaluation(Base):
    """Quality evaluation result for a deployment or endpoint."""

    __tablename__ = "evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to deployment (optional - can evaluate standalone endpoints)
    deployment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deployments.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    # For production endpoints (standalone URL evaluation)
    production_endpoint_id = Column(Integer, nullable=True, index=True)
    endpoint_url = Column(String(500), nullable=True)  # Direct URL for standalone evaluation

    # Evaluation type and config
    evaluation_type = Column(String(50), nullable=False, index=True)
    # 'asr', 'llm', 'vision', 'custom'
    evaluator_name = Column(String(100), nullable=False)
    # e.g., 'ASRWERCEREvaluator'

    # Status
    status = Column(String(20), nullable=False, default="pending", index=True)
    # 'pending', 'running', 'completed', 'failed'
    error_message = Column(String(1000), nullable=True)

    # Progress tracking
    current_stage = Column(String(50), nullable=True)
    stage_progress = Column(String(50), nullable=True)  # e.g., "50/100 samples"

    # Common metrics (all evaluations may populate these)
    primary_metric = Column(Float, nullable=True)
    primary_metric_name = Column(String(50), nullable=True)  # e.g., 'wer', 'bleu'
    secondary_metric = Column(Float, nullable=True)
    secondary_metric_name = Column(String(50), nullable=True)

    # ASR-specific metrics (denormalized for query performance)
    wer = Column(Float, nullable=True)  # Word Error Rate (0.0-1.0)
    cer = Column(Float, nullable=True)  # Character Error Rate (0.0-1.0)

    # Sample counts
    samples_total = Column(Integer, nullable=True)
    samples_evaluated = Column(Integer, nullable=True)
    samples_with_errors = Column(Integer, nullable=True)
    no_speech_count = Column(Integer, nullable=True)  # ASR-specific

    # Dataset info
    dataset_path = Column(String(500), nullable=True)
    dataset_name = Column(String(255), nullable=True)

    # Configuration used
    config = Column(JSONB, default=dict, nullable=False)
    # For ASR: {"limit": 100, "language": "English", "vad_config": {...}}

    # Results storage
    results = Column(JSONB, default=dict, nullable=False)
    # Detailed results: sample-level scores, error breakdown, etc.

    # Timing
    duration_seconds = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    deployment = relationship("Deployment", backref="evaluations")
