"""
Abstract base class for quality evaluators.

Defines the interface that all evaluators must implement.

## Adding a New Evaluator

1. Create a new file in `app/services/eval/<type>/` (e.g., `llm/evaluator.py`)
2. Implement the `Evaluator` base class
3. Use `@register_evaluator('type')` decorator to register

Example:
```python
from app.services.eval.base import Evaluator, EvaluationConfig, EvaluationResult
from app.services.eval.factory import register_evaluator

@register_evaluator('llm')
class LLMEvaluator(Evaluator):
    @property
    def evaluation_type(self) -> str:
        return "llm"

    @property
    def evaluator_name(self) -> str:
        return "LLMEvaluator"

    async def evaluate(self, config, progress_callback=None) -> EvaluationResult:
        # Your evaluation logic here
        # Call progress_callback(current, total) to report progress
        pass
```

## Docker-based Evaluators

For evaluators that run in Docker containers, extend `DockerEvaluator`:
```python
from app.services.eval.docker_base import DockerEvaluator

@register_evaluator('mytype')
class MyDockerEvaluator(DockerEvaluator):
    docker_image = "my-eval:latest"

    def build_docker_command(self, config):
        return ["--endpoint", config.endpoint_url, ...]

    def parse_metrics(self, stdout: str) -> dict:
        # Parse your metrics from stdout
        return {"accuracy": 0.95}
```

## Standard Progress Format

All eval containers should output progress in this format:
```
EVAL_PROGRESS: current/total
```

Example: `EVAL_PROGRESS: 42/100`
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    endpoint_url: str
    model_name: str
    dataset_path: str = ""
    limit: int = 0  # 0 = all samples
    offset: int = 0
    timeout: float = 120.0
    language: Optional[str] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Result metrics from an evaluation."""
    primary_metric: float
    primary_metric_name: str
    secondary_metric: Optional[float] = None
    secondary_metric_name: Optional[str] = None
    samples_evaluated: int = 0
    samples_with_errors: int = 0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete result of an evaluation run."""
    success: bool
    metrics: Optional[EvaluationMetrics] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    sample_results: List[Dict[str, Any]] = field(default_factory=list)


# Progress callback type: (current, total) -> awaitable
ProgressCallback = Callable[[int, int], Awaitable[None]]


class Evaluator(ABC):
    """
    Abstract base class for quality evaluators.

    Implementations must provide methods for:
    - Running evaluation against an endpoint
    - Reporting evaluation type/name
    - Validating configuration
    """

    @property
    @abstractmethod
    def evaluation_type(self) -> str:
        """Return the evaluation type (e.g., 'asr', 'llm', 'vision')."""
        pass

    @property
    @abstractmethod
    def evaluator_name(self) -> str:
        """Return the evaluator name (e.g., 'ASRWERCEREvaluator')."""
        pass

    @abstractmethod
    async def evaluate(
        self,
        config: EvaluationConfig,
        progress_callback: Optional[ProgressCallback] = None,
        evaluation_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run evaluation against the configured endpoint.

        Args:
            config: Evaluation configuration
            progress_callback: Optional async callback(current, total) for progress
            evaluation_id: Optional ID for cancellation support

        Returns:
            EvaluationResult with metrics or error
        """
        pass

    @abstractmethod
    def validate_config(self, config: EvaluationConfig) -> tuple[bool, Optional[str]]:
        """
        Validate configuration before running.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message_if_invalid)
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Return list of metrics this evaluator produces."""
        pass
