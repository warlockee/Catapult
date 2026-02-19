"""
Repository for Evaluation entity database operations.
"""
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository
from app.models.evaluation import Evaluation
from app.core.exceptions import (
    EvaluationNotFoundError,
    EvaluationNotCancellableError,
)


class EvaluationRepository(BaseRepository[Evaluation]):
    """Repository for Evaluation database operations."""

    model = Evaluation

    async def get_by_id_or_raise(self, id: UUID) -> Evaluation:
        """
        Get an evaluation by ID, raising exception if not found.

        Args:
            id: Evaluation UUID

        Returns:
            Evaluation entity

        Raises:
            EvaluationNotFoundError: If evaluation not found
        """
        evaluation = await self.get_by_id(id)
        if not evaluation:
            raise EvaluationNotFoundError(str(id))
        return evaluation

    async def list_by_deployment(
        self,
        deployment_id: UUID,
        limit: int = 10,
    ) -> List[Evaluation]:
        """
        List evaluations for a deployment, ordered by creation date desc.

        Args:
            deployment_id: Deployment UUID
            limit: Maximum number of results

        Returns:
            List of evaluations
        """
        result = await self.db.execute(
            select(Evaluation)
            .where(Evaluation.deployment_id == deployment_id)
            .order_by(desc(Evaluation.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_production_endpoint(
        self,
        endpoint_id: int,
        limit: int = 10,
    ) -> List[Evaluation]:
        """
        List evaluations for a production endpoint, ordered by creation date desc.

        Args:
            endpoint_id: Production endpoint ID
            limit: Maximum number of results

        Returns:
            List of evaluations
        """
        result = await self.db.execute(
            select(Evaluation)
            .where(Evaluation.production_endpoint_id == endpoint_id)
            .order_by(desc(Evaluation.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_latest_completed_by_deployment(
        self,
        deployment_id: UUID,
        with_wer: bool = True,
    ) -> Optional[Evaluation]:
        """
        Get the latest completed evaluation for a deployment.

        Args:
            deployment_id: Deployment UUID
            with_wer: If True, only return evaluations with WER data

        Returns:
            Latest completed evaluation or None
        """
        query = (
            select(Evaluation)
            .where(Evaluation.deployment_id == deployment_id)
            .where(Evaluation.status == "completed")
            .order_by(desc(Evaluation.created_at))
            .limit(1)
        )
        if with_wer:
            query = query.where(Evaluation.wer.isnot(None))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_latest_completed_by_production_endpoint(
        self,
        endpoint_id: int,
        with_wer: bool = True,
    ) -> Optional[Evaluation]:
        """
        Get the latest completed evaluation for a production endpoint.

        Args:
            endpoint_id: Production endpoint ID
            with_wer: If True, only return evaluations with WER data

        Returns:
            Latest completed evaluation or None
        """
        query = (
            select(Evaluation)
            .where(Evaluation.production_endpoint_id == endpoint_id)
            .where(Evaluation.status == "completed")
            .order_by(desc(Evaluation.created_at))
            .limit(1)
        )
        if with_wer:
            query = query.where(Evaluation.wer.isnot(None))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def mark_cancelled(
        self,
        evaluation_id: UUID,
    ) -> Evaluation:
        """
        Cancel an evaluation.

        Args:
            evaluation_id: Evaluation UUID

        Returns:
            Updated evaluation

        Raises:
            EvaluationNotFoundError: If evaluation not found
            EvaluationNotCancellableError: If evaluation cannot be cancelled
        """
        evaluation = await self.get_by_id_or_raise(evaluation_id)

        if evaluation.status not in ("pending", "running"):
            raise EvaluationNotCancellableError(str(evaluation_id), evaluation.status)

        evaluation.status = "cancelled"

        await self.db.commit()
        return evaluation

    async def mark_failed(
        self,
        evaluation_id: UUID,
        error_message: str,
    ) -> Evaluation:
        """
        Mark evaluation as failed with error.

        Args:
            evaluation_id: Evaluation UUID
            error_message: Error description

        Returns:
            Updated evaluation

        Raises:
            EvaluationNotFoundError: If evaluation not found
        """
        evaluation = await self.get_by_id_or_raise(evaluation_id)

        if evaluation.status in ("pending", "running"):
            evaluation.status = "failed"
            evaluation.error_message = error_message
            await self.db.commit()

        return evaluation
