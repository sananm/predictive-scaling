"""
Predictions API routes.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import get_db
from src.storage.repositories import PredictionsRepository

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
                    metadata=pred.metadata,
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
        metadata=pred.metadata,
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


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_prediction(
    service_name: str = "default",
) -> TriggerResponse:
    """Manually trigger a prediction run."""
    # TODO: Implement prediction trigger via message queue or direct call
    return TriggerResponse(
        status="pending",
        message=f"Prediction run triggered for service: {service_name}",
    )
