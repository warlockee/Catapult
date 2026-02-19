"""
Evaluation services package.

Provides evaluation orchestration and evaluator factory pattern.

## Quick Start - Adding a New Evaluator

1. Create a new directory: `app/services/eval/<type>/`
2. Create `__init__.py` and `evaluator.py`
3. Implement your evaluator (see base.py for docs)
4. Import it here to register

Example for a new "llm" evaluator:

```python
# In app/services/eval/llm/evaluator.py
from app.services.eval.docker_base import DockerEvaluator
from app.services.eval.factory import register_evaluator

@register_evaluator('llm')
class LLMDockerEvaluator(DockerEvaluator):
    docker_image = "my-llm-eval:latest"

    def build_docker_args(self, config):
        return ["--endpoint", config.endpoint_url, ...]

    def parse_metrics(self, stdout):
        return {'primary_metric': 0.95, 'primary_metric_name': 'accuracy'}
```

Then add to this file:
```python
from app.services.eval.llm import LLMDockerEvaluator
```
"""
# Import evaluators to trigger registration
# Add new evaluators here
from app.services.eval.asr import ASRDockerEvaluator
from app.services.eval.base import (
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationResult,
    Evaluator,
    ProgressCallback,
)
from app.services.eval.docker_base import DockerEvaluator
from app.services.eval.evaluation_service import EvaluationService, evaluation_service
from app.services.eval.factory import EvaluatorFactory, add_type_alias, register_evaluator

__all__ = [
    # Base classes
    "Evaluator",
    "DockerEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationMetrics",
    "ProgressCallback",
    # Factory
    "EvaluatorFactory",
    "register_evaluator",
    "add_type_alias",
    # Service
    "EvaluationService",
    "evaluation_service",
    # Evaluators
    "ASRDockerEvaluator",
]
