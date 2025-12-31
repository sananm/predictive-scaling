"""
Scaling API routes.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import get_db
from src.storage.repositories import ScalingDecisionsRepository

router = APIRouter(prefix="/scaling", tags=["Scaling"])


class ScalingDecisionResponse(BaseModel):
    """Response model for a scaling decision."""

    id: UUID
    created_at: datetime
    service_name: str
    strategy: str
    status: str
    current_instances: int
    target_instances: int
    reasoning: str | None
    confidence_score: float | None
    estimated_hourly_cost: float | None
    estimated_savings: float | None
    executed_at: datetime | None
    completed_at: datetime | None


class ScalingDecisionDetailResponse(ScalingDecisionResponse):
    """Detailed response with full configuration."""

    current_capacity: float | None
    target_capacity: float | None
    execution_duration_seconds: float | None
    rollback_config: dict[str, Any] | None
    verification_criteria: dict[str, Any] | None
    verification_result: dict[str, Any] | None
    full_config: dict[str, Any]


class ScalingDecisionListResponse(BaseModel):
    """Response for list of scaling decisions."""

    decisions: list[ScalingDecisionResponse]
    count: int


class InfrastructureStatus(BaseModel):
    """Current infrastructure status."""

    service_name: str
    current_instances: int
    desired_instances: int
    min_instances: int
    max_instances: int
    cpu_utilization: float | None
    memory_utilization: float | None
    status: str


class ApprovalResponse(BaseModel):
    """Response for approval/rejection actions."""

    id: UUID
    status: str
    message: str


@router.get("/status", response_model=InfrastructureStatus)
async def get_infrastructure_status(
    service_name: str = "default",
) -> InfrastructureStatus:
    """Get current infrastructure state."""
    # TODO: Implement actual K8s/cloud status retrieval
    return InfrastructureStatus(
        service_name=service_name,
        current_instances=3,
        desired_instances=3,
        min_instances=1,
        max_instances=50,
        cpu_utilization=0.45,
        memory_utilization=0.60,
        status="stable",
    )


@router.get("/decisions", response_model=ScalingDecisionListResponse)
async def list_scaling_decisions(
    service_name: str | None = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> ScalingDecisionListResponse:
    """List recent scaling decisions."""
    repo = ScalingDecisionsRepository(db)
    decisions = await repo.get_recent(service_name, limit)

    return ScalingDecisionListResponse(
        decisions=[
            ScalingDecisionResponse(
                id=d.id,
                created_at=d.created_at,
                service_name=d.service_name,
                strategy=d.strategy,
                status=d.status,
                current_instances=d.current_instances,
                target_instances=d.target_instances,
                reasoning=d.reasoning,
                confidence_score=d.confidence_score,
                estimated_hourly_cost=d.estimated_hourly_cost,
                estimated_savings=d.estimated_savings,
                executed_at=d.executed_at,
                completed_at=d.completed_at,
            )
            for d in decisions
        ],
        count=len(decisions),
    )


@router.get("/decisions/{decision_id}", response_model=ScalingDecisionDetailResponse)
async def get_scaling_decision(
    decision_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ScalingDecisionDetailResponse:
    """Get details of a specific scaling decision."""
    repo = ScalingDecisionsRepository(db)
    decision = await repo.get_by_id(decision_id)

    if not decision:
        raise HTTPException(status_code=404, detail="Scaling decision not found")

    return ScalingDecisionDetailResponse(
        id=decision.id,
        created_at=decision.created_at,
        service_name=decision.service_name,
        strategy=decision.strategy,
        status=decision.status,
        current_instances=decision.current_instances,
        current_capacity=decision.current_capacity,
        target_instances=decision.target_instances,
        target_capacity=decision.target_capacity,
        reasoning=decision.reasoning,
        confidence_score=decision.confidence_score,
        estimated_hourly_cost=decision.estimated_hourly_cost,
        estimated_savings=decision.estimated_savings,
        executed_at=decision.executed_at,
        completed_at=decision.completed_at,
        execution_duration_seconds=decision.execution_duration_seconds,
        rollback_config=decision.rollback_config,
        verification_criteria=decision.verification_criteria,
        verification_result=decision.verification_result,
        full_config=decision.full_config or {},
    )


@router.post("/decisions/{decision_id}/approve", response_model=ApprovalResponse)
async def approve_scaling_decision(
    decision_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ApprovalResponse:
    """Approve a pending scaling decision."""
    repo = ScalingDecisionsRepository(db)
    decision = await repo.get_by_id(decision_id)

    if not decision:
        raise HTTPException(status_code=404, detail="Scaling decision not found")

    if decision.status != "pending":
        raise HTTPException(status_code=400, detail=f"Decision is not pending: {decision.status}")

    await repo.update_status(decision_id, "approved")

    # TODO: Trigger execution

    return ApprovalResponse(
        id=decision_id,
        status="approved",
        message="Scaling decision approved and queued for execution",
    )


@router.post("/decisions/{decision_id}/reject", response_model=ApprovalResponse)
async def reject_scaling_decision(
    decision_id: UUID,
    reason: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> ApprovalResponse:
    """Reject a pending scaling decision."""
    repo = ScalingDecisionsRepository(db)
    decision = await repo.get_by_id(decision_id)

    if not decision:
        raise HTTPException(status_code=404, detail="Scaling decision not found")

    if decision.status != "pending":
        raise HTTPException(status_code=400, detail=f"Decision is not pending: {decision.status}")

    await repo.update_status(decision_id, "rejected", reasoning=reason)

    return ApprovalResponse(
        id=decision_id,
        status="rejected",
        message="Scaling decision rejected",
    )


@router.post("/rollback", response_model=ApprovalResponse)
async def trigger_rollback(
    service_name: str = "default",
    db: AsyncSession = Depends(get_db),
) -> ApprovalResponse:
    """Trigger a manual rollback to the previous state."""
    # TODO: Implement rollback logic
    # 1. Get the most recent completed decision
    # 2. Get its rollback_config
    # 3. Execute rollback

    raise HTTPException(status_code=501, detail="Rollback not yet implemented")
