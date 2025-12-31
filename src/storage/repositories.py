"""
Async repository layer for database operations.

Provides clean abstraction over SQLAlchemy for CRUD operations.
"""

from datetime import datetime, timedelta
from typing import Any, Generic, Sequence, TypeVar
from uuid import UUID

import pandas as pd
from sqlalchemy import Select, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    AlertLog,
    BusinessEvent,
    CostTracking,
    Feature,
    Metric,
    ModelPerformance,
    Prediction,
    ScalingDecision,
)

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    model: type[T]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, id: UUID) -> T | None:
        """Get a single record by ID."""
        result = await self.session.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def create(self, **kwargs: Any) -> T:
        """Create a new record."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def create_many(self, items: list[dict[str, Any]]) -> list[T]:
        """Create multiple records."""
        instances = [self.model(**item) for item in items]
        self.session.add_all(instances)
        await self.session.flush()
        return instances

    async def update(self, id: UUID, **kwargs: Any) -> T | None:
        """Update a record by ID."""
        instance = await self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            await self.session.flush()
        return instance

    async def delete(self, id: UUID) -> bool:
        """Delete a record by ID."""
        result = await self.session.execute(delete(self.model).where(self.model.id == id))
        return result.rowcount > 0


class MetricsRepository:
    """Repository for metrics time-series data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert(
        self,
        timestamp: datetime,
        service_name: str,
        metric_name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> Metric:
        """Insert a single metric."""
        metric = Metric(
            timestamp=timestamp,
            service_name=service_name,
            metric_name=metric_name,
            value=value,
            labels=labels or {},
        )
        self.session.add(metric)
        await self.session.flush()
        return metric

    async def insert_batch(self, metrics: list[dict[str, Any]]) -> int:
        """Insert multiple metrics efficiently."""
        instances = [Metric(**m) for m in metrics]
        self.session.add_all(instances)
        await self.session.flush()
        return len(instances)

    async def get_by_time_range(
        self,
        service_name: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Metric]:
        """Get metrics in a time range."""
        result = await self.session.execute(
            select(Metric)
            .where(
                Metric.service_name == service_name,
                Metric.metric_name == metric_name,
                Metric.timestamp >= start_time,
                Metric.timestamp <= end_time,
            )
            .order_by(Metric.timestamp)
        )
        return result.scalars().all()

    async def get_latest(
        self,
        service_name: str,
        metric_name: str,
    ) -> Metric | None:
        """Get the most recent metric value."""
        result = await self.session.execute(
            select(Metric)
            .where(
                Metric.service_name == service_name,
                Metric.metric_name == metric_name,
            )
            .order_by(Metric.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_aggregated(
        self,
        service_name: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        bucket_interval: str = "1 minute",
    ) -> list[dict[str, Any]]:
        """Get aggregated metrics using TimescaleDB time_bucket."""
        query = text(
            """
            SELECT
                time_bucket(:interval, timestamp) AS bucket,
                AVG(value) AS avg_value,
                MIN(value) AS min_value,
                MAX(value) AS max_value,
                COUNT(*) AS count
            FROM metrics
            WHERE service_name = :service_name
              AND metric_name = :metric_name
              AND timestamp >= :start_time
              AND timestamp <= :end_time
            GROUP BY bucket
            ORDER BY bucket
            """
        )
        result = await self.session.execute(
            query,
            {
                "interval": bucket_interval,
                "service_name": service_name,
                "metric_name": metric_name,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        return [dict(row._mapping) for row in result]

    async def to_dataframe(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        metric_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame for ML pipelines."""
        query = select(Metric).where(
            Metric.service_name == service_name,
            Metric.timestamp >= start_time,
            Metric.timestamp <= end_time,
        )
        if metric_names:
            query = query.where(Metric.metric_name.in_(metric_names))
        query = query.order_by(Metric.timestamp)

        result = await self.session.execute(query)
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        # Pivot to have metrics as columns
        data = [
            {"timestamp": r.timestamp, "metric": r.metric_name, "value": r.value} for r in rows
        ]
        df = pd.DataFrame(data)
        df = df.pivot(index="timestamp", columns="metric", values="value")
        return df


class FeaturesRepository:
    """Repository for computed features."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert(
        self,
        timestamp: datetime,
        service_name: str,
        feature_set_version: str,
        features: dict[str, Any],
    ) -> Feature:
        """Insert computed features."""
        feature = Feature(
            timestamp=timestamp,
            service_name=service_name,
            feature_set_version=feature_set_version,
            features=features,
        )
        self.session.add(feature)
        await self.session.flush()
        return feature

    async def get_latest(
        self,
        service_name: str,
        count: int = 1,
    ) -> Sequence[Feature]:
        """Get the most recent features."""
        result = await self.session.execute(
            select(Feature)
            .where(Feature.service_name == service_name)
            .order_by(Feature.timestamp.desc())
            .limit(count)
        )
        return result.scalars().all()

    async def get_by_time_range(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[Feature]:
        """Get features in a time range."""
        result = await self.session.execute(
            select(Feature)
            .where(
                Feature.service_name == service_name,
                Feature.timestamp >= start_time,
                Feature.timestamp <= end_time,
            )
            .order_by(Feature.timestamp)
        )
        return result.scalars().all()

    async def to_dataframe(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Convert features to DataFrame for ML."""
        features = await self.get_by_time_range(service_name, start_time, end_time)
        if not features:
            return pd.DataFrame()

        data = []
        for f in features:
            row = {"timestamp": f.timestamp, **f.features}
            data.append(row)

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df


class PredictionsRepository(BaseRepository[Prediction]):
    """Repository for predictions."""

    model = Prediction

    async def insert(
        self,
        service_name: str,
        model_name: str,
        model_version: str,
        horizon_minutes: int,
        target_timestamp: datetime,
        prediction_p10: float,
        prediction_p50: float,
        prediction_p90: float,
        metadata: dict[str, Any] | None = None,
    ) -> Prediction:
        """Insert a prediction."""
        return await self.create(
            service_name=service_name,
            model_name=model_name,
            model_version=model_version,
            horizon_minutes=horizon_minutes,
            target_timestamp=target_timestamp,
            prediction_p10=prediction_p10,
            prediction_p50=prediction_p50,
            prediction_p90=prediction_p90,
            metadata=metadata or {},
        )

    async def get_latest_for_horizon(
        self,
        service_name: str,
        horizon_minutes: int,
    ) -> Prediction | None:
        """Get the most recent prediction for a specific horizon."""
        result = await self.session.execute(
            select(Prediction)
            .where(
                Prediction.service_name == service_name,
                Prediction.horizon_minutes == horizon_minutes,
            )
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_predictions_for_timestamp(
        self,
        service_name: str,
        target_timestamp: datetime,
        tolerance_minutes: int = 5,
    ) -> Sequence[Prediction]:
        """Get all predictions targeting a specific timestamp."""
        start = target_timestamp - timedelta(minutes=tolerance_minutes)
        end = target_timestamp + timedelta(minutes=tolerance_minutes)

        result = await self.session.execute(
            select(Prediction)
            .where(
                Prediction.service_name == service_name,
                Prediction.target_timestamp >= start,
                Prediction.target_timestamp <= end,
            )
            .order_by(Prediction.created_at.desc())
        )
        return result.scalars().all()


class ScalingDecisionsRepository(BaseRepository[ScalingDecision]):
    """Repository for scaling decisions."""

    model = ScalingDecision

    async def get_recent(
        self,
        service_name: str | None = None,
        limit: int = 10,
    ) -> Sequence[ScalingDecision]:
        """Get recent scaling decisions."""
        query = select(ScalingDecision).order_by(ScalingDecision.created_at.desc()).limit(limit)
        if service_name:
            query = query.where(ScalingDecision.service_name == service_name)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_pending(self) -> Sequence[ScalingDecision]:
        """Get all pending scaling decisions."""
        result = await self.session.execute(
            select(ScalingDecision)
            .where(ScalingDecision.status == "pending")
            .order_by(ScalingDecision.created_at)
        )
        return result.scalars().all()

    async def update_status(
        self,
        id: UUID,
        status: str,
        **kwargs: Any,
    ) -> ScalingDecision | None:
        """Update decision status with optional additional fields."""
        return await self.update(id, status=status, **kwargs)


class BusinessEventsRepository(BaseRepository[BusinessEvent]):
    """Repository for business events."""

    model = BusinessEvent

    async def get_active_events(
        self,
        at_time: datetime | None = None,
    ) -> Sequence[BusinessEvent]:
        """Get events that are currently active."""
        now = at_time or datetime.utcnow()
        result = await self.session.execute(
            select(BusinessEvent)
            .where(
                BusinessEvent.is_active == True,
                BusinessEvent.start_time <= now,
                BusinessEvent.end_time >= now,
            )
            .order_by(BusinessEvent.start_time)
        )
        return result.scalars().all()

    async def get_upcoming_events(
        self,
        hours_ahead: int = 24,
    ) -> Sequence[BusinessEvent]:
        """Get events starting within the specified hours."""
        now = datetime.utcnow()
        future = now + timedelta(hours=hours_ahead)

        result = await self.session.execute(
            select(BusinessEvent)
            .where(
                BusinessEvent.is_active == True,
                BusinessEvent.start_time >= now,
                BusinessEvent.start_time <= future,
            )
            .order_by(BusinessEvent.start_time)
        )
        return result.scalars().all()


class ModelPerformanceRepository(BaseRepository[ModelPerformance]):
    """Repository for model performance tracking."""

    model = ModelPerformance

    async def record_performance(
        self,
        model_name: str,
        model_version: str,
        horizon_minutes: int,
        mae: float,
        mape: float,
        rmse: float,
        sample_count: int,
        evaluation_window_hours: int,
        coverage_80: float | None = None,
        coverage_90: float | None = None,
    ) -> ModelPerformance:
        """Record model performance metrics."""
        return await self.create(
            model_name=model_name,
            model_version=model_version,
            horizon_minutes=horizon_minutes,
            mae=mae,
            mape=mape,
            rmse=rmse,
            sample_count=sample_count,
            evaluation_window_hours=evaluation_window_hours,
            coverage_80=coverage_80,
            coverage_90=coverage_90,
        )

    async def get_latest_performance(
        self,
        model_name: str,
        horizon_minutes: int | None = None,
    ) -> ModelPerformance | None:
        """Get the most recent performance record for a model."""
        query = (
            select(ModelPerformance)
            .where(ModelPerformance.model_name == model_name)
            .order_by(ModelPerformance.recorded_at.desc())
            .limit(1)
        )
        if horizon_minutes is not None:
            query = query.where(ModelPerformance.horizon_minutes == horizon_minutes)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()


class CostTrackingRepository:
    """Repository for cost tracking."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_cost(
        self,
        timestamp: datetime,
        service_name: str,
        actual_cost_hourly: float,
        actual_instances: int,
        simulated_reactive_cost: float | None = None,
        simulated_reactive_instances: int | None = None,
    ) -> CostTracking:
        """Record cost data point."""
        savings = None
        if simulated_reactive_cost is not None:
            savings = simulated_reactive_cost - actual_cost_hourly

        cost = CostTracking(
            timestamp=timestamp,
            service_name=service_name,
            actual_cost_hourly=actual_cost_hourly,
            actual_instances=actual_instances,
            simulated_reactive_cost=simulated_reactive_cost,
            simulated_reactive_instances=simulated_reactive_instances,
            estimated_savings=savings,
        )
        self.session.add(cost)
        await self.session.flush()
        return cost

    async def get_total_savings(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """Calculate total savings in a time period."""
        result = await self.session.execute(
            select(func.sum(CostTracking.estimated_savings)).where(
                CostTracking.service_name == service_name,
                CostTracking.timestamp >= start_time,
                CostTracking.timestamp <= end_time,
            )
        )
        return result.scalar() or 0.0


class AlertLogsRepository(BaseRepository[AlertLog]):
    """Repository for alert logs."""

    model = AlertLog

    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        service_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AlertLog:
        """Create a new alert."""
        return await self.create(
            alert_type=alert_type,
            severity=severity,
            message=message,
            service_name=service_name,
            context=context or {},
        )

    async def get_unacknowledged(
        self,
        severity: str | None = None,
    ) -> Sequence[AlertLog]:
        """Get unacknowledged alerts."""
        query = select(AlertLog).where(AlertLog.acknowledged == False)
        if severity:
            query = query.where(AlertLog.severity == severity)
        query = query.order_by(AlertLog.created_at.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def acknowledge(
        self,
        id: UUID,
        acknowledged_by: str,
    ) -> AlertLog | None:
        """Acknowledge an alert."""
        return await self.update(
            id,
            acknowledged=True,
            acknowledged_at=datetime.utcnow(),
            acknowledged_by=acknowledged_by,
        )

    async def resolve(
        self,
        id: UUID,
        resolution_notes: str | None = None,
    ) -> AlertLog | None:
        """Mark an alert as resolved."""
        return await self.update(
            id,
            resolved=True,
            resolved_at=datetime.utcnow(),
            resolution_notes=resolution_notes,
        )
