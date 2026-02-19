"""
Evaluation API endpoints - quality metrics (WER/CER) separate from benchmarks (latency/throughput).
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import DeploymentNotFoundError
from app.core.security import require_operator, verify_api_key
from app.models.api_key import ApiKey
from app.models.evaluation import Evaluation
from app.repositories.deployment_repository import DeploymentRepository
from app.repositories.evaluation_repository import EvaluationRepository
from app.schemas.evaluation import (
    EvaluationCreate,
    EvaluationResponse,
    EvaluationSummary,
)
from app.services.eval.evaluation_service import evaluation_service

router = APIRouter()


def _evaluation_to_summary(evaluation: Optional[Evaluation]) -> EvaluationSummary:
    """Convert Evaluation to summary response."""
    if not evaluation:
        return EvaluationSummary(has_data=False)

    return EvaluationSummary(
        has_data=True,
        evaluation_type=evaluation.evaluation_type,
        evaluator_name=evaluation.evaluator_name,
        status=evaluation.status,
        primary_metric=evaluation.primary_metric,
        primary_metric_name=evaluation.primary_metric_name,
        secondary_metric=evaluation.secondary_metric,
        secondary_metric_name=evaluation.secondary_metric_name,
        wer=evaluation.wer,
        cer=evaluation.cer,
        samples_evaluated=evaluation.samples_evaluated,
        no_speech_count=evaluation.no_speech_count,
        dataset_path=evaluation.dataset_path,
    )


async def _run_evaluation_task(
    evaluation_id: UUID,
    endpoint_url: str,
    model_name: str,
    dataset_path: Optional[str],
    model_type: str,
    deployment_id: Optional[UUID],
    production_endpoint_id: Optional[int],
    limit: int,
    language: str,
):
    """Background task to run evaluation."""
    import logging
    import sys
    logger = logging.getLogger(__name__)
    print(f"[EVAL] Starting evaluation task for {evaluation_id}", file=sys.stderr, flush=True)
    logger.info(f"Starting evaluation task for {evaluation_id}")

    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        print("[EVAL] Got DB session", file=sys.stderr, flush=True)
        repo = EvaluationRepository(db)
        evaluation = await repo.get_by_id(evaluation_id)
        print(f"[EVAL] Got evaluation record: {evaluation is not None}", file=sys.stderr, flush=True)
        if not evaluation:
            print("[EVAL] Evaluation not found, returning", file=sys.stderr, flush=True)
            return

        try:
            print(f"[EVAL] Running evaluation for endpoint_url={endpoint_url}, model_type={model_type}", file=sys.stderr, flush=True)
            logger.info(f"Running evaluation for endpoint_url={endpoint_url}, model_type={model_type}")
            # Run evaluation using the service - pass evaluation_id to update existing record
            await evaluation_service.run_evaluation(
                db=db,
                endpoint_url=endpoint_url,
                model_name=model_name,
                dataset_path=dataset_path or "",  # Empty string if not provided
                model_type=model_type,
                deployment_id=deployment_id,
                production_endpoint_id=production_endpoint_id,
                evaluation_id=evaluation_id,
                limit=limit,
                language=language,
            )
            logger.info(f"Evaluation {evaluation_id} completed")
        except Exception as e:
            logger.error(f"Evaluation {evaluation_id} failed: {e}")
            await repo.mark_failed(evaluation_id, str(e))


@router.post("", response_model=EvaluationResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_evaluation(
    eval_data: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> EvaluationResponse:
    """
    Create and run a quality evaluation (WER/CER for ASR models).

    This is separate from benchmarks which measure performance (latency, throughput).
    Evaluations measure model quality metrics.
    """
    # Validate endpoint
    endpoint_url = eval_data.endpoint_url
    deployment_id = eval_data.deployment_id
    production_endpoint_id = eval_data.production_endpoint_id

    # If deployment_id provided, get endpoint from deployment
    if deployment_id:
        deployment_repo = DeploymentRepository(db)
        deployment = await deployment_repo.get_by_id(deployment_id)
        if not deployment:
            raise DeploymentNotFoundError(str(deployment_id))
        endpoint_url = deployment.endpoint_url

    # Create pending evaluation record
    eval_repo = EvaluationRepository(db)
    evaluation = Evaluation(
        deployment_id=deployment_id,
        production_endpoint_id=production_endpoint_id,
        endpoint_url=endpoint_url,
        evaluation_type=eval_data.model_type,
        evaluator_name="pending",
        status="pending",
        dataset_path=eval_data.dataset_path,
        config={
            "limit": eval_data.limit,
            "language": eval_data.language,
            "model_name": eval_data.model_name,
        },
    )
    evaluation = await eval_repo.create(evaluation)

    # Queue background task
    background_tasks.add_task(
        _run_evaluation_task,
        evaluation_id=evaluation.id,
        endpoint_url=endpoint_url,
        model_name=eval_data.model_name,
        dataset_path=eval_data.dataset_path,
        model_type=eval_data.model_type,
        deployment_id=deployment_id,
        production_endpoint_id=production_endpoint_id,
        limit=eval_data.limit,
        language=eval_data.language or "English",
    )

    return EvaluationResponse.from_evaluation(evaluation)


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> EvaluationResponse:
    """Get evaluation by ID."""
    repo = EvaluationRepository(db)
    evaluation = await repo.get_by_id_or_raise(evaluation_id)
    return EvaluationResponse.from_evaluation(evaluation)


@router.post("/{evaluation_id}/cancel", response_model=EvaluationResponse)
async def cancel_evaluation(
    evaluation_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_operator),
) -> EvaluationResponse:
    """Cancel a running evaluation."""
    repo = EvaluationRepository(db)
    evaluation = await repo.mark_cancelled(evaluation_id)
    return EvaluationResponse.from_evaluation(evaluation)


@router.get("/deployment/{deployment_id}", response_model=List[EvaluationResponse])
async def list_deployment_evaluations(
    deployment_id: UUID,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> List[EvaluationResponse]:
    """List evaluations for a deployment."""
    repo = EvaluationRepository(db)
    evaluations = await repo.list_by_deployment(deployment_id, limit)
    return [EvaluationResponse.from_evaluation(e) for e in evaluations]


@router.get("/deployment/{deployment_id}/summary", response_model=EvaluationSummary)
async def get_deployment_evaluation_summary(
    deployment_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> EvaluationSummary:
    """Get latest evaluation summary for a deployment."""
    repo = EvaluationRepository(db)
    evaluation = await repo.get_latest_completed_by_deployment(deployment_id)
    return _evaluation_to_summary(evaluation)


@router.get("/production-endpoint/{endpoint_id}", response_model=List[EvaluationResponse])
async def list_production_endpoint_evaluations(
    endpoint_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> List[EvaluationResponse]:
    """List evaluations for a production endpoint."""
    repo = EvaluationRepository(db)
    evaluations = await repo.list_by_production_endpoint(endpoint_id, limit)
    return [EvaluationResponse.from_evaluation(e) for e in evaluations]


@router.get("/production-endpoint/{endpoint_id}/summary", response_model=EvaluationSummary)
async def get_production_endpoint_evaluation_summary(
    endpoint_id: int,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(verify_api_key),
) -> EvaluationSummary:
    """Get latest evaluation summary for a production endpoint."""
    repo = EvaluationRepository(db)
    evaluation = await repo.get_latest_completed_by_production_endpoint(endpoint_id)
    return _evaluation_to_summary(evaluation)
