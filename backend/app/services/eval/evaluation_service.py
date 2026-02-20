"""
Evaluation Service - orchestrates quality evaluations for model endpoints.

This is the main entry point for running evaluations. It:
1. Uses EvaluatorFactory to get the appropriate evaluator
2. Updates existing Evaluation DB records with progress and results
"""
import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation
from app.services.eval.base import EvaluationConfig, EvaluationResult
from app.services.eval.factory import EvaluatorFactory

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for orchestrating quality evaluations."""

    async def run_evaluation(
        self,
        db: AsyncSession,
        endpoint_url: str,
        model_name: str,
        dataset_path: str,
        model_type: str,
        server_type: Optional[str] = None,
        deployment_id: Optional[UUID] = None,
        production_endpoint_id: Optional[int] = None,
        evaluation_id: Optional[UUID] = None,
        limit: int = 0,
        timeout: float = 120.0,
        language: Optional[str] = "English",
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> Optional[Evaluation]:
        """
        Run an evaluation on an endpoint.

        Args:
            db: Database session
            endpoint_url: Model endpoint URL
            model_name: Model name/ID
            dataset_path: Path to evaluation dataset
            model_type: Type of model (e.g., 'audio', 'text')
            server_type: Server type for more specific evaluator selection
            deployment_id: Optional deployment ID
            production_endpoint_id: Optional production endpoint ID
            evaluation_id: Optional existing evaluation ID to update
            limit: Max samples to evaluate (0 = all)
            timeout: Request timeout in seconds
            language: Language for evaluation
            progress_callback: Optional callback(current, total)

        Returns:
            Evaluation record if evaluation was run, None if no evaluator available
        """
        # Get evaluator from factory
        evaluator = EvaluatorFactory.create(model_type, server_type)

        if evaluator is None:
            logger.debug(f"No evaluator available for model_type={model_type}, server_type={server_type}")
            return None

        logger.info(f"Running {evaluator.evaluation_type} evaluation with {evaluator.evaluator_name}")

        # Get existing evaluation record or create new one
        if evaluation_id:
            result = await db.execute(
                select(Evaluation).where(Evaluation.id == evaluation_id)
            )
            evaluation = result.scalar_one_or_none()
            if not evaluation:
                logger.warning(f"Evaluation {evaluation_id} not found")
                return None
            # Update existing record
            evaluation.status = "running"
            evaluation.evaluator_name = evaluator.evaluator_name
            evaluation.evaluation_type = evaluator.evaluation_type
        else:
            # Create new Evaluation record
            evaluation = Evaluation(
                deployment_id=deployment_id,
                production_endpoint_id=production_endpoint_id,
                endpoint_url=endpoint_url,
                evaluation_type=evaluator.evaluation_type,
                evaluator_name=evaluator.evaluator_name,
                status="running",
                dataset_path=dataset_path,
                config={
                    "limit": limit,
                    "language": language,
                    "model_name": model_name,
                },
            )
            db.add(evaluation)

        await db.commit()
        await db.refresh(evaluation)

        # Progress callback wrapper
        async def eval_progress(current: int, total: int):
            evaluation.stage_progress = f"{current}/{total}"
            await db.commit()
            if progress_callback:
                await progress_callback(current, total)

        try:
            # Build config and run evaluation
            config = EvaluationConfig(
                endpoint_url=endpoint_url,
                model_name=model_name,
                dataset_path=dataset_path,
                limit=limit,
                timeout=timeout,
                language=language,
            )

            result = await evaluator.evaluate(
                config,
                progress_callback=eval_progress,
                evaluation_id=str(evaluation.id) if evaluation else None,
            )

            # Update Evaluation record with results
            self._update_evaluation_from_result(evaluation, result)
            await db.commit()

            if result.metrics:
                logger.info(
                    f"Evaluation complete: {evaluator.evaluation_type} "
                    f"{result.metrics.primary_metric_name}={result.metrics.primary_metric:.4f}"
                )

            return evaluation

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation.status = "failed"
            evaluation.error_message = str(e)
            evaluation.completed_at = datetime.utcnow()
            await db.commit()
            return evaluation

    def _update_evaluation_from_result(
        self,
        evaluation: Evaluation,
        result: EvaluationResult,
    ) -> None:
        """Update Evaluation record from EvaluationResult."""
        evaluation.status = "completed" if result.success else "failed"
        evaluation.error_message = result.error_message
        evaluation.duration_seconds = result.duration_seconds
        evaluation.started_at = result.started_at
        evaluation.completed_at = result.completed_at

        if result.metrics:
            evaluation.primary_metric = result.metrics.primary_metric
            evaluation.primary_metric_name = result.metrics.primary_metric_name
            evaluation.secondary_metric = result.metrics.secondary_metric
            evaluation.secondary_metric_name = result.metrics.secondary_metric_name
            evaluation.samples_evaluated = result.metrics.samples_evaluated
            evaluation.samples_with_errors = result.metrics.samples_with_errors

            # Denormalized fields based on metric names (not type)
            # This allows new evaluators to use these fields without modifying this service
            if result.metrics.primary_metric_name == "wer":
                evaluation.wer = result.metrics.primary_metric
            if result.metrics.secondary_metric_name == "cer":
                evaluation.cer = result.metrics.secondary_metric
            if "no_speech_count" in result.metrics.extra_metrics:
                evaluation.no_speech_count = result.metrics.extra_metrics.get("no_speech_count")

            # Store detailed results (limit size)
            evaluation.results = {
                "sample_results": result.sample_results[:100] if result.sample_results else [],
                "extra_metrics": result.metrics.extra_metrics,
            }


# Singleton instance
evaluation_service = EvaluationService()
