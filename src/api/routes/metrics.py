"""
Metrics API routes.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import get_db
from src.storage.repositories import MetricsRepository

router = APIRouter(prefix="/metrics", tags=["Metrics"])


class MetricInput(BaseModel):
    """Input model for metric ingestion."""

    timestamp: datetime
    service_name: str
    metric_name: str
    value: float
    labels: dict[str, Any] = Field(default_factory=dict)


class MetricResponse(BaseModel):
    """Response model for a single metric."""

    timestamp: datetime
    service_name: str
    metric_name: str
    value: float
    labels: dict[str, Any]


class MetricsListResponse(BaseModel):
    """Response model for multiple metrics."""

    metrics: list[MetricResponse]
    count: int


class AggregatedMetricResponse(BaseModel):
    """Response model for aggregated metrics."""

    bucket: datetime
    avg_value: float
    min_value: float
    max_value: float
    count: int


@router.get("", response_model=MetricsListResponse)
async def get_metrics(
    service_name: str,
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    db: AsyncSession = Depends(get_db),
) -> MetricsListResponse:
    """Query stored metrics with filters."""
    repo = MetricsRepository(db)
    metrics = await repo.get_by_time_range(service_name, metric_name, start_time, end_time)

    return MetricsListResponse(
        metrics=[
            MetricResponse(
                timestamp=m.timestamp,
                service_name=m.service_name,
                metric_name=m.metric_name,
                value=m.value,
                labels=m.labels,
            )
            for m in metrics
        ],
        count=len(metrics),
    )


@router.get("/aggregated", response_model=list[AggregatedMetricResponse])
async def get_aggregated_metrics(
    service_name: str,
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    bucket_interval: str = Query(default="1 minute", description="Time bucket interval"),
    db: AsyncSession = Depends(get_db),
) -> list[AggregatedMetricResponse]:
    """Get aggregated metrics using TimescaleDB time buckets."""
    repo = MetricsRepository(db)
    results = await repo.get_aggregated(
        service_name, metric_name, start_time, end_time, bucket_interval
    )

    return [
        AggregatedMetricResponse(
            bucket=r["bucket"],
            avg_value=r["avg_value"],
            min_value=r["min_value"],
            max_value=r["max_value"],
            count=r["count"],
        )
        for r in results
    ]


@router.post("", status_code=201)
async def ingest_metrics(
    metrics: list[MetricInput],
    db: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Ingest metrics (for testing purposes)."""
    repo = MetricsRepository(db)
    count = await repo.insert_batch([m.model_dump() for m in metrics])
    return {"ingested": count}


@router.get("/latest", response_model=MetricResponse | None)
async def get_latest_metric(
    service_name: str,
    metric_name: str,
    db: AsyncSession = Depends(get_db),
) -> MetricResponse | None:
    """Get the most recent value for a metric."""
    repo = MetricsRepository(db)
    metric = await repo.get_latest(service_name, metric_name)

    if not metric:
        return None

    return MetricResponse(
        timestamp=metric.timestamp,
        service_name=metric.service_name,
        metric_name=metric.metric_name,
        value=metric.value,
        labels=metric.labels,
    )
