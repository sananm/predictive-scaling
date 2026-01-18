"""
Scaling API routes.
"""

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from src.execution.kubernetes import KubernetesConfig, KubernetesExecutor
from src.services.scaling import get_scaling_service
from src.storage.database import get_db
from src.storage.repositories import ScalingDecisionsRepository
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global K8s executor instance
_k8s_executor: KubernetesExecutor | None = None

# Track last scaling action timestamp for status display
_last_scaling_action: datetime | None = None

# Simulated state for demo purposes (tracks replica count changes from evaluate endpoint)
# This allows the UI to show "responses" even if we don't have a real scalable K8s cluster
_simulated_state: dict[str, Any] = {
    "current_instances": 3,
    "last_updated": None,
}


def get_k8s_executor() -> KubernetesExecutor:
    """Get or create the Kubernetes executor."""
    global _k8s_executor
    if _k8s_executor is None:
        # Get deployment name from settings or environment
        import os

        deployment_name = os.environ.get("K8S_DEPLOYMENT_NAME", "sample-app")
        namespace = os.environ.get("K8S_NAMESPACE", "default")

        _k8s_executor = KubernetesExecutor(
            KubernetesConfig(
                namespace=namespace,
                deployment_name=deployment_name,
                wait_for_rollout=True,
                rollout_timeout_seconds=300.0,
            )
        )
    return _k8s_executor


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


async def execute_scaling_in_background(
    decision_id: UUID,
    target_instances: int,
    db_session_maker: Any,
) -> None:
    """Execute scaling in the background."""
    try:
        executor = get_k8s_executor()
        from src.execution.base import ScalingAction

        action = ScalingAction(
            action_id=str(decision_id),
            target_count=target_instances,
            timeout_seconds=300.0,
        )

        result = await executor.scale(action)

        # Update decision status in database
        async with db_session_maker() as db:
            repo = ScalingDecisionsRepository(db)
            if result.status.value == "completed":
                await repo.update_status(decision_id, "completed")
                logger.info("Scaling completed successfully", decision_id=str(decision_id))
            else:
                await repo.update_status(
                    decision_id, "failed", reasoning=result.error_message
                )
                logger.error(
                    "Scaling failed",
                    decision_id=str(decision_id),
                    error=result.error_message,
                )
            await db.commit()

    except Exception as e:
        logger.error("Background scaling failed", decision_id=str(decision_id), error=str(e))


@router.post("/decisions/{decision_id}/approve", response_model=ApprovalResponse)
async def approve_scaling_decision(
    decision_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ApprovalResponse:
    """Approve a pending scaling decision and trigger execution."""
    repo = ScalingDecisionsRepository(db)
    decision = await repo.get_by_id(decision_id)

    if not decision:
        raise HTTPException(status_code=404, detail="Scaling decision not found")

    if decision.status != "pending":
        raise HTTPException(status_code=400, detail=f"Decision is not pending: {decision.status}")

    # Update status to approved/executing
    await repo.update_status(decision_id, "executing")
    await db.commit()

    # Trigger execution in background
    from src.storage.database import async_session_maker

    background_tasks.add_task(
        execute_scaling_in_background,
        decision_id,
        decision.target_instances,
        async_session_maker,
    )

    logger.info(
        "Scaling decision approved and execution triggered",
        decision_id=str(decision_id),
        target_instances=decision.target_instances,
    )

    return ApprovalResponse(
        id=decision_id,
        status="executing",
        message=f"Scaling to {decision.target_instances} instances started",
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


class RollbackResponse(BaseModel):
    """Response for rollback action."""

    success: bool
    message: str
    previous_instances: int | None = None
    current_instances: int | None = None


@router.post("/rollback", response_model=RollbackResponse)
async def trigger_rollback(
    service_name: str = "default",
    db: AsyncSession = Depends(get_db),
) -> RollbackResponse:
    """
    Trigger a manual rollback to the previous state.

    Finds the most recent completed scaling decision and reverts
    the deployment to its previous instance count.
    """
    try:
        repo = ScalingDecisionsRepository(db)

        # Get the most recent completed decision for this service
        decisions = await repo.get_recent(service_name, limit=5)
        completed_decision = None

        for decision in decisions:
            if decision.status == "completed":
                completed_decision = decision
                break

        if not completed_decision:
            raise HTTPException(
                status_code=404,
                detail="No completed scaling decision found to rollback",
            )

        # The rollback target is the current_instances from the last decision
        rollback_target = completed_decision.current_instances

        logger.info(
            "Initiating rollback",
            service_name=service_name,
            current=completed_decision.target_instances,
            target=rollback_target,
        )

        # Execute rollback using K8s executor
        executor = get_k8s_executor()
        from src.execution.base import ScalingAction

        action = ScalingAction(
            action_id=f"rollback-{completed_decision.id}",
            target_count=rollback_target,
            timeout_seconds=300.0,
        )

        result = await executor.scale(action)

        if result.status.value == "completed":
            logger.info(
                "Rollback completed successfully",
                service_name=service_name,
                instances=rollback_target,
            )

            return RollbackResponse(
                success=True,
                message=f"Rolled back to {rollback_target} instances",
                previous_instances=completed_decision.target_instances,
                current_instances=rollback_target,
            )
        else:
            logger.error("Rollback failed", error=result.error_message)
            return RollbackResponse(
                success=False,
                message=f"Rollback failed: {result.error_message}",
                previous_instances=completed_decision.target_instances,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rollback error", error=str(e))
        return RollbackResponse(
            success=False,
            message=f"Rollback error: {str(e)}",
        )



@router.get("/decisions", response_model=ScalingDecisionListResponse)
async def get_scaling_decisions(
    service_name: str | None = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> ScalingDecisionListResponse:
    """Get recent scaling decisions."""
    repo = ScalingDecisionsRepository(db)
    # The repository method signature is get_recent(service_name, limit)
    decisions = await repo.get_recent(service_name=service_name, limit=limit)
    
    response_decisions = []
    for d in decisions:
        response_decisions.append(
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
        )
        
    return ScalingDecisionListResponse(
        decisions=response_decisions,
        count=len(response_decisions)
    )


@router.get("/status", response_model=InfrastructureStatus)
async def get_infrastructure_status(
    service_name: str = "default",
) -> InfrastructureStatus:
    """Get infrastructure state (mix of real K8s and simulation for demo)."""
    global _simulated_state
    import math
    import time
    
    # Generate dynamic mock values for CPU/memory if real metrics unavailable
    t = time.time()
    base_cpu = 45
    base_memory = 55
    
    cpu_variation = math.sin(t / 30) * 15 + math.sin(t / 7) * 5
    memory_variation = math.cos(t / 25) * 10 + math.sin(t / 11) * 5
    
    cpu_util = max(0.15, min(0.95, (base_cpu + cpu_variation) / 100))
    memory_util = max(0.25, min(0.90, (base_memory + memory_variation) / 100))
    
    try:
        executor = get_k8s_executor()
        # Try to get real state, but fallback to simulation if connection/cluster fails
        # or if we want to show the "simulated" result of actions that can't run locally
        try:
             state = await executor.get_current_state()
             instance_count = state.instance_count
             
             # DEMO MODE: Always prefer simulated state if it has been updated by evaluate_scaling
             # This ensures Live Status reflects the result of simulation actions
             simulated_count = _simulated_state.get("current_instances", 3)
             simulated_updated = _simulated_state.get("last_updated") is not None
             
             if simulated_updated or (simulated_count != instance_count and instance_count <= 3):
                 instance_count = simulated_count
        except Exception:
             # Fallback entirely to simulation
             instance_count = _simulated_state.get("current_instances", 3)
             state = None

        # Determine status based on load
        if cpu_util > 0.80:
            status = "warning"
        elif cpu_util > 0.70:
             status = "scaling"
        else:
             status = "stable"
                
        # Check if we recently scaled (show "scaling" status briefly)
        if _last_scaling_action:
            elapsed = (datetime.utcnow() - _last_scaling_action).total_seconds()
            if elapsed < 30:
                status = "scaling"

        # Start building response
        return InfrastructureStatus(
            service_name=service_name,
            current_instances=instance_count,
            desired_instances=instance_count,
            min_instances=1,
            max_instances=50,
            cpu_utilization=round(cpu_util, 3),
            memory_utilization=round(memory_util, 3),
            status=status,
        )
        
    except Exception as e:
        logger.warning("Failed to get status", error=str(e))
        return InfrastructureStatus(
             service_name=service_name,
             current_instances=_simulated_state.get("current_instances", 3),
             desired_instances=_simulated_state.get("current_instances", 3),
             min_instances=1,
             max_instances=50,
             cpu_utilization=0.45,
             memory_utilization=0.55,
             status="stable"
        )


class EvaluateScalingRequest(BaseModel):
    """Request to evaluate scaling with a simulated scenario."""
    multiplier: float = 3.0
    duration_minutes: int = 30
    service_name: str = "default"


class EvaluateScalingResponse(BaseModel):
    """Response from scaling evaluation."""
    success: bool
    message: str
    action: str
    current_instances: int
    target_instances: int
    reasoning: str | None = None
    decision_id: str | None = None
    prediction_p50: float | None = None
    prediction_p90: float | None = None
    confidence: float | None = None


@router.post("/evaluate", response_model=EvaluateScalingResponse)
async def evaluate_scaling(
    request: EvaluateScalingRequest,
    db: AsyncSession = Depends(get_db),
) -> EvaluateScalingResponse:
    """
    Evaluate scaling needs using the REAL prediction pipeline.
    """
    from src.services.prediction import get_prediction_service, PredictionHorizon
    from src.services.scaling import get_scaling_service, ScalingAction
    from src.storage.repositories import BusinessEventsRepository, ScalingDecisionsRepository
    from src.storage.models import BusinessEvent
    from uuid import uuid4, UUID
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update, select
    
    global _simulated_state
    
    try:
        # 1. Create Business Event to simulate load impact
        event_repo = BusinessEventsRepository(db)
        
        # Deactivate any existing simulation events to ensure "last click wins"
        # This prevents a 5x spike from persisting when user clicks "Scale Down"
        await db.execute(
            update(BusinessEvent)
            .where(
                BusinessEvent.source == "simulation",
                BusinessEvent.is_active == True
            )
            .values(is_active=False)
        )
        
        event_id = uuid4()
        now = datetime.now(timezone.utc)
        
        event_type = "flash_sale" if request.multiplier > 1.0 else "maintenance"
        event_name = f"Simulated {'Spike' if request.multiplier > 1.0 else 'Drop'} ({request.multiplier}x)"
        
        await event_repo.create(
            id=event_id,
            name=event_name,
            event_type=event_type,
            source="simulation",
            start_time=now,
            end_time=now + timedelta(minutes=request.duration_minutes),
            expected_impact_multiplier=request.multiplier,
            is_active=True
        )
        await db.commit()
        logger.info(f"Created simulation event: {event_name} id={event_id} multiplier={request.multiplier}")
        
        # 2. Get Current State (Use Simulated if available to allow 'stacking' simulations)
        current_instances = _simulated_state.get("current_instances", 3)
        
        # Calculate mock utilization
        import time, math
        t = time.time()
        base_cpu = 45
        cpu_variation = math.sin(t / 30) * 15 + math.sin(t / 7) * 5
        current_utilization = max(0.15, min(0.95, (base_cpu + cpu_variation) / 100))
        
        logger.info(f"Evaluating scaling: current_instances={current_instances}, utilization={current_utilization}")

        # 3. Run Real Predictions
        prediction_service = get_prediction_service()
        # Force refresh of cache or logic might be needed here if it caches 'no event' state?
        # Assuming run_predictions_for_service queries active events fresh.
        predictions = await prediction_service.run_predictions_for_service(
             request.service_name, 
             horizons=[PredictionHorizon.MEDIUM]
        )
        prediction = predictions[0] if predictions else None
        
        if prediction:
             logger.info(f"Prediction result: p50={prediction.p50}, p90={prediction.p90}")
        else:
             logger.warning("No prediction returned!")
        
        # 4. Evaluate Scaling via Decision Engine
        scaling_service = get_scaling_service()
        action = await scaling_service.evaluate_scaling(
            request.service_name,
            current_instances=current_instances,
            current_utilization=current_utilization,
            prediction=prediction,
            bypass_cooldown=True,  # Always bypass cooldown for demo UI
        )
        
        global _last_scaling_action
        _last_scaling_action = datetime.utcnow()
        
        # SIMULATION UPDATE: Update our fake state so the UI reflects the ML decision
        if action:
             _simulated_state["current_instances"] = action.target_count
             _simulated_state["last_updated"] = datetime.utcnow()
             logger.info(f"Updated simulated state to {action.target_count} instances based on action {action.action_type}")
             
             # PERSISTENCE: Save decision to DB for history
             try:
                 decision_repo = ScalingDecisionsRepository(db)
                 await decision_repo.create(
                     id=UUID(action.action_id),
                     service_name=action.service_name,
                     strategy=action.action_type.value,
                     status=action.status.value,
                     current_instances=action.current_count,
                     target_instances=action.target_count,
                     reasoning=action.reason,
                     confidence_score=action.confidence,
                     estimated_hourly_cost=action.estimated_cost_change,
                     created_at=action.created_at
                 )
                 await db.commit()
                 logger.info(f"Persisted scaling decision {action.action_id}")
             except Exception as e:
                 logger.error(f"Failed to persist decision: {e}")

        if action:
            return EvaluateScalingResponse(
                success=True,
                message=f"Scaling decision: {action.action_type.value} to {action.target_count} instances.",
                action=action.action_type.value,
                current_instances=action.current_count,
                target_instances=action.target_count,
                reasoning=action.reason,
                decision_id=action.action_id,
                prediction_p50=prediction.p50 if prediction else None,
                prediction_p90=prediction.p90 if prediction else None,
                confidence=action.confidence
            )
        else:
            return EvaluateScalingResponse(
                success=True,
                message="Evaluation complete: No scaling action needed currently.",
                action="no_change",
                current_instances=current_instances,
                target_instances=current_instances,
                reasoning="Current capacity is sufficient for predicted load.",
                prediction_p50=prediction.p50 if prediction else None,
                prediction_p90=prediction.p90 if prediction else None,
                confidence=prediction.confidence if prediction else None
            )

    except Exception as e:
        logger.error("Scaling evaluation failed", error=str(e))
        return EvaluateScalingResponse(
            success=False,
            message=f"Evaluation failed: {str(e)}",
            action="error",
            current_instances=0,
            target_instances=0
        )
