"""
Predictions API routes.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.decision.candidates import CandidateConfig, CandidateGenerator
from src.decision.capacity_model import CapacityModel
from src.decision.engine import DecisionEngine, InfrastructureState, PredictionInput
from src.services.prediction import PredictionHorizon, get_prediction_service
from src.storage.database import get_db
from src.storage.repositories import PredictionsRepository, ScalingDecisionsRepository
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class PredictionResponse(BaseModel):
    """Response model for a prediction."""

    id: UUID
    created_at: datetime
    service_name: str
    model_name: str
    model_version: str
    horizon_minutes: int
    target_timestamp: datetime
    prediction_p10: float
    prediction_p50: float
    prediction_p90: float
    metadata: dict[str, Any]


class PredictionsListResponse(BaseModel):
    """Response model for multiple predictions."""

    predictions: list[PredictionResponse]
    count: int


class TriggerResponse(BaseModel):
    """Response for prediction trigger."""

    status: str
    message: str


@router.get("/current", response_model=PredictionsListResponse)
async def get_current_predictions(
    service_name: str = "default",
    db: AsyncSession = Depends(get_db),
) -> PredictionsListResponse:
    """Get the latest predictions for all horizons."""
    repo = PredictionsRepository(db)

    # Get latest predictions for different horizons
    horizons = [15, 60, 360, 1440, 10080]  # 15min, 1hr, 6hr, 24hr, 7days
    predictions = []

    for horizon in horizons:
        pred = await repo.get_latest_for_horizon(service_name, horizon)
        if pred:
            predictions.append(
                PredictionResponse(
                    id=pred.id,
                    created_at=pred.created_at,
                    service_name=pred.service_name,
                    model_name=pred.model_name,
                    model_version=pred.model_version,
                    horizon_minutes=pred.horizon_minutes,
                    target_timestamp=pred.target_timestamp,
                    prediction_p10=pred.prediction_p10,
                    prediction_p50=pred.prediction_p50,
                    prediction_p90=pred.prediction_p90,
                    metadata=pred.prediction_metadata,
                )
            )

    return PredictionsListResponse(predictions=predictions, count=len(predictions))


@router.get("/{horizon}", response_model=PredictionResponse | None)
async def get_prediction_by_horizon(
    horizon: int,
    service_name: str = "default",
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse | None:
    """Get the latest prediction for a specific horizon (in minutes)."""
    repo = PredictionsRepository(db)
    pred = await repo.get_latest_for_horizon(service_name, horizon)

    if not pred:
        return None

    return PredictionResponse(
        id=pred.id,
        created_at=pred.created_at,
        service_name=pred.service_name,
        model_name=pred.model_name,
        model_version=pred.model_version,
        horizon_minutes=pred.horizon_minutes,
        target_timestamp=pred.target_timestamp,
        prediction_p10=pred.prediction_p10,
        prediction_p50=pred.prediction_p50,
        prediction_p90=pred.prediction_p90,
        metadata=pred.prediction_metadata,
    )


@router.get("/history", response_model=PredictionsListResponse)
async def get_prediction_history(
    service_name: str,
    target_timestamp: datetime,
    tolerance_minutes: int = 5,
    db: AsyncSession = Depends(get_db),
) -> PredictionsListResponse:
    """Get all predictions that targeted a specific timestamp."""
    repo = PredictionsRepository(db)
    preds = await repo.get_predictions_for_timestamp(
        service_name, target_timestamp, tolerance_minutes
    )

    predictions = [
        PredictionResponse(
            id=p.id,
            created_at=p.created_at,
            service_name=p.service_name,
            model_name=p.model_name,
            model_version=p.model_version,
            horizon_minutes=p.horizon_minutes,
            target_timestamp=p.target_timestamp,
            prediction_p10=p.prediction_p10,
            prediction_p50=p.prediction_p50,
            prediction_p90=p.prediction_p90,
            metadata=p.metadata,
        )
        for p in preds
    ]

    return PredictionsListResponse(predictions=predictions, count=len(predictions))


class TriggerResponseExtended(BaseModel):
    """Extended response for prediction trigger."""

    status: str
    message: str
    predictions_count: int = 0
    predictions: list[dict[str, Any]] = []
    scaling_decision: dict[str, Any] | None = None


async def get_current_infrastructure_state(service_name: str) -> InfrastructureState:
    """Get current infrastructure state from Kubernetes or simulation."""
    # Get actual RPS from request tracker
    from src.api.middleware import get_current_rps

    current_rps = get_current_rps()
    # Ensure minimum RPS for calculations (avoid divide by zero issues)
    if current_rps < 1.0:
        current_rps = 1.0

    logger.info("Current RPS from tracker", rps=round(current_rps, 2))

    # Import simulated state from scaling module for demo consistency
    try:
        from src.api.routes.scaling import _simulated_state
        simulated_count = _simulated_state.get("current_instances", 3)
        simulated_updated = _simulated_state.get("last_updated") is not None
    except ImportError:
        simulated_count = 3
        simulated_updated = False

    try:
        from src.api.routes.scaling import get_k8s_executor

        executor = get_k8s_executor()
        state = await executor.get_current_state()
        
        # Use simulated state if it has been updated (demo mode)
        instance_count = state.instance_count
        if simulated_updated or (simulated_count != instance_count and instance_count <= 3):
            instance_count = simulated_count

        return InfrastructureState(
            instance_type="t3.medium",  # Default for demo
            instance_count=instance_count,
            spot_percentage=0.0,
            current_rps=current_rps,
            current_utilization=state.metadata.get("cpu_utilization", 0.45),
            healthy_instances=instance_count,
        )
    except Exception as e:
        logger.warning("Failed to get K8s state, using simulated state", error=str(e))
        return InfrastructureState(
            instance_type="t3.medium",
            instance_count=simulated_count,  # Use simulated, not hardcoded 3
            spot_percentage=0.0,
            current_rps=current_rps,
            current_utilization=0.45,
            healthy_instances=simulated_count,
        )



def convert_to_prediction_inputs(predictions: list[dict[str, Any]]) -> list[PredictionInput]:
    """Convert prediction results to DecisionEngine inputs."""
    inputs = []
    for pred in predictions:
        # Get horizon in minutes from the prediction
        horizon_map = {"short": 15, "medium": 60, "long": 1440}
        horizon_minutes = horizon_map.get(pred.get("horizon", "short"), 15)

        inputs.append(
            PredictionInput(
                horizon_minutes=horizon_minutes,
                p10=pred.get("p10", 80),
                p50=pred.get("p50", 100),
                p90=pred.get("p90", 120),
                confidence=pred.get("confidence", 0.85),
                timestamp=datetime.now(UTC),
            )
        )
    return inputs


@router.post("/trigger", response_model=TriggerResponseExtended)
async def trigger_prediction(
    service_name: str = "default",
    db: AsyncSession = Depends(get_db),
) -> TriggerResponseExtended:
    """
    Manually trigger a prediction run and create scaling decision if needed.

    Runs ML models to generate predictions for the specified service
    across short-term (15-60 min), medium-term (1-24 hr), and long-term (1-7 day) horizons.
    Then evaluates predictions to determine if scaling action is needed.
    """
    logger.info("Triggering prediction", service_name=service_name)

    try:
        # Run predictions synchronously to return results immediately
        prediction_service = get_prediction_service()

        # Run predictions across all horizons
        results = await prediction_service.run_predictions_for_service(
            service_name=service_name,
            horizons=[PredictionHorizon.SHORT, PredictionHorizon.MEDIUM, PredictionHorizon.LONG],
        )

        predictions = [r.to_dict() for r in results]

        logger.info(
            "Predictions completed",
            service_name=service_name,
            predictions_count=len(predictions),
        )

        # Now evaluate predictions with DecisionEngine
        scaling_decision_dict = None
        try:
            # Get current infrastructure state
            current_state = await get_current_infrastructure_state(service_name)

            # Convert predictions to DecisionEngine format
            prediction_inputs = convert_to_prediction_inputs(predictions)

            # Run decision engine with realistic capacity model for demo
            # Override default RPS per vCPU (500 is too high for demo)
            capacity_model = CapacityModel()
            # Set low capacity for demo: ~25 RPS per vCPU
            # t3.medium (2 vCPU) = 25 * 2 * 0.8 = 40 RPS per instance
            # 3 instances = 120 RPS capacity
            # This makes traffic spikes more impactful
            capacity_model.DEFAULT_RPS_PER_VCPU = 25

            # Create candidate generator that only considers horizontal scaling
            # (same instance type, different count)
            candidate_config = CandidateConfig(
                min_instances=1,
                max_instances=20,
                spot_percentages=[0.0],  # No spot for demo simplicity
                capacity_multipliers=[1.0, 1.2, 1.5, 2.0],  # Include 2x for larger spikes
            )
            candidate_generator = CandidateGenerator(
                config=candidate_config,
                capacity_model=capacity_model,
            )

            engine = DecisionEngine(
                capacity_model=capacity_model,
                candidate_generator=candidate_generator,
            )
            decision = engine.decide(
                current_state=current_state,
                predictions=prediction_inputs,
                force_evaluation=True,  # Always evaluate for demo
                # Constrain to horizontal scaling only (same instance type)
                allowed_instance_types=[current_state.instance_type],
            )

            if decision:
                # Extract values from decision structure
                current_instances = decision.current_state.instance_count
                target_instances = decision.target_config.instance_count
                strategy = decision.scaling_plan.strategy_type.value if decision.scaling_plan else "predictive"
                confidence = 1.0 - decision.risk_assessment.overall_score if decision.risk_assessment else 0.85

                logger.info(
                    "Scaling decision generated",
                    strategy=strategy,
                    target=target_instances,
                    current=current_instances,
                )

                # Determine action type
                if target_instances > current_instances:
                    action = "scale_up"
                elif target_instances < current_instances:
                    action = "scale_down"
                else:
                    action = "no_change"

                # Store decision in database
                from src.storage.models import ScalingDecision as ScalingDecisionDB

                db_decision = ScalingDecisionDB(
                    id=uuid4(),
                    service_name=service_name,
                    strategy=strategy,
                    status="pending",
                    current_instances=current_instances,
                    target_instances=target_instances,
                    current_capacity=current_state.current_rps,
                    target_capacity=target_instances * 100,  # Estimated
                    reasoning=decision.reasoning[:500] if decision.reasoning else "",
                    confidence_score=confidence,
                    estimated_hourly_cost=decision.score.cost_score if decision.score else 0,
                    estimated_savings=0,
                    full_config={
                        "decision_id": decision.id,
                        "target_instances": target_instances,
                        "current_instances": current_instances,
                    },
                )
                db.add(db_decision)
                await db.commit()

                scaling_decision_dict = {
                    "id": str(db_decision.id),
                    "action": action,
                    "current_instances": current_instances,
                    "target_instances": target_instances,
                    "reasoning": decision.reasoning,
                    "confidence": confidence,
                    "status": "pending",
                }

        except Exception as e:
            logger.warning("Decision engine evaluation failed", error=str(e))

        return TriggerResponseExtended(
            status="completed",
            message=f"Generated {len(predictions)} predictions for service: {service_name}",
            predictions_count=len(predictions),
            predictions=predictions,
            scaling_decision=scaling_decision_dict,
        )

    except Exception as e:
        logger.error("Prediction trigger failed", error=str(e), service_name=service_name)
        return TriggerResponseExtended(
            status="error",
            message=f"Prediction failed: {str(e)}",
            predictions_count=0,
            predictions=[],
        )
