"""
Pydantic schemas for Evaluation.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID


class EvaluationType(str, Enum):
    """Type of evaluation."""
    ASR = "asr"
    LLM = "llm"
    VISION = "vision"
    CUSTOM = "custom"


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationCreate(BaseModel):
    """Schema for creating an evaluation."""
    endpoint_url: str = Field(..., max_length=500)
    model_name: str = Field(..., max_length=255)
    dataset_path: Optional[str] = Field(default=None, max_length=500, description="Path to evaluation dataset. If not provided, uses embedded dataset in eval container.")
    model_type: str = Field(default="asr", max_length=50)
    deployment_id: Optional[UUID] = None
    production_endpoint_id: Optional[int] = None
    limit: int = Field(default=0, ge=0, le=10000, description="0 = all samples")
    language: Optional[str] = Field(default="English", max_length=50)


class EvaluationResponse(BaseModel):
    """Schema for evaluation response."""
    id: UUID
    deployment_id: Optional[UUID] = None
    production_endpoint_id: Optional[int] = None
    endpoint_url: Optional[str] = None

    # Type and status
    evaluation_type: str
    evaluator_name: str
    status: str
    error_message: Optional[str] = None

    # Progress
    current_stage: Optional[str] = None
    stage_progress: Optional[str] = None

    # Metrics
    primary_metric: Optional[float] = None
    primary_metric_name: Optional[str] = None
    secondary_metric: Optional[float] = None
    secondary_metric_name: Optional[str] = None

    # ASR-specific (denormalized for convenience)
    wer: Optional[float] = None
    cer: Optional[float] = None

    # Sample counts
    samples_total: Optional[int] = None
    samples_evaluated: Optional[int] = None
    samples_with_errors: Optional[int] = None
    no_speech_count: Optional[int] = None

    # Dataset info
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None

    # Configuration and results
    config: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    duration_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True

    @classmethod
    def from_evaluation(cls, evaluation) -> "EvaluationResponse":
        """Convert an Evaluation ORM instance to response schema."""
        return cls(
            id=evaluation.id,
            deployment_id=evaluation.deployment_id,
            production_endpoint_id=evaluation.production_endpoint_id,
            endpoint_url=evaluation.endpoint_url,
            evaluation_type=evaluation.evaluation_type,
            evaluator_name=evaluation.evaluator_name,
            status=evaluation.status,
            error_message=evaluation.error_message,
            current_stage=evaluation.current_stage,
            stage_progress=evaluation.stage_progress,
            primary_metric=evaluation.primary_metric,
            primary_metric_name=evaluation.primary_metric_name,
            secondary_metric=evaluation.secondary_metric,
            secondary_metric_name=evaluation.secondary_metric_name,
            wer=evaluation.wer,
            cer=evaluation.cer,
            samples_total=evaluation.samples_total,
            samples_evaluated=evaluation.samples_evaluated,
            samples_with_errors=evaluation.samples_with_errors,
            no_speech_count=evaluation.no_speech_count,
            dataset_path=evaluation.dataset_path,
            dataset_name=evaluation.dataset_name,
            config=evaluation.config or {},
            results=evaluation.results or {},
            duration_seconds=evaluation.duration_seconds,
            started_at=evaluation.started_at,
            completed_at=evaluation.completed_at,
            created_at=evaluation.created_at,
        )


class EvaluationSummary(BaseModel):
    """Summary of evaluation for display in benchmark results."""
    has_data: bool = False
    evaluation_type: Optional[str] = None
    evaluator_name: Optional[str] = None
    status: Optional[str] = None

    # Main metrics
    primary_metric: Optional[float] = None
    primary_metric_name: Optional[str] = None
    secondary_metric: Optional[float] = None
    secondary_metric_name: Optional[str] = None

    # ASR-specific
    wer: Optional[float] = None
    cer: Optional[float] = None
    samples_evaluated: Optional[int] = None
    no_speech_count: Optional[int] = None

    # Dataset
    dataset_path: Optional[str] = None
