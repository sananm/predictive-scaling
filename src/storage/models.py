"""
Database models for the predictive scaling system.

Uses SQLAlchemy 2.0 with async support and TimescaleDB hypertables.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def generate_uuid() -> uuid.UUID:
    return uuid.uuid4()


class ScalingStrategy(str, Enum):
    """Scaling strategy types."""

    GRADUAL_RAMP = "gradual_ramp"
    PREEMPTIVE_BURST = "preemptive_burst"
    EMERGENCY_SCALE = "emergency_scale"
    SCALE_DOWN = "scale_down"


class ScalingStatus(str, Enum):
    """Scaling decision status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Time-Series Tables (TimescaleDB Hypertables)
# ============================================================================


class Metric(Base):
    """Raw metrics collected from Prometheus and other sources."""

    __tablename__ = "metrics"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    service_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    labels: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_metrics_service_time", "service_name", "timestamp"),
        Index("idx_metrics_name_time", "metric_name", "timestamp"),
    )


class Feature(Base):
    """Computed features for ML models."""

    __tablename__ = "features"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    service_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    feature_set_version: Mapped[str] = mapped_column(String(50), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index("idx_features_service_time", "service_name", "timestamp"),
    )


class Prediction(Base):
    """Model predictions with uncertainty bounds."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=generate_uuid
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    service_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    target_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    prediction_p10: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_p50: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_p90: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_predictions_service_target", "service_name", "target_timestamp"),
    )


# ============================================================================
# Operational Tables
# ============================================================================


class ScalingDecision(Base):
    """Scaling decisions made by the decision engine."""

    __tablename__ = "scaling_decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=generate_uuid
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    service_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default=ScalingStatus.PENDING)

    # Current state
    current_instances: Mapped[int] = mapped_column(Integer, nullable=False)
    current_capacity: Mapped[float] = mapped_column(Float, nullable=True)

    # Target state
    target_instances: Mapped[int] = mapped_column(Integer, nullable=False)
    target_capacity: Mapped[float] = mapped_column(Float, nullable=True)

    # Reasoning
    reasoning: Mapped[str] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Cost estimates
    estimated_hourly_cost: Mapped[float] = mapped_column(Float, nullable=True)
    estimated_savings: Mapped[float] = mapped_column(Float, nullable=True)

    # Execution tracking
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Rollback info
    rollback_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Verification
    verification_criteria: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    verification_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Full configuration
    full_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class BusinessEvent(Base):
    """Business events that impact traffic predictions."""

    __tablename__ = "business_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=generate_uuid
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Impact estimation
    expected_impact_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    actual_impact_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Source and metadata
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class ModelPerformance(Base):
    """Model accuracy tracking over time."""

    __tablename__ = "model_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=generate_uuid
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Accuracy metrics
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)

    # Coverage metrics (prediction interval accuracy)
    coverage_80: Mapped[float] = mapped_column(Float, nullable=True)
    coverage_90: Mapped[float] = mapped_column(Float, nullable=True)

    # Sample info
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_window_hours: Mapped[int] = mapped_column(Integer, nullable=False)


class CostTracking(Base):
    """Cost tracking for savings calculation."""

    __tablename__ = "cost_tracking"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=generate_uuid
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    service_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Actual costs
    actual_cost_hourly: Mapped[float] = mapped_column(Float, nullable=False)
    actual_instances: Mapped[int] = mapped_column(Integer, nullable=False)

    # Simulated reactive costs (what it would have cost without prediction)
    simulated_reactive_cost: Mapped[float] = mapped_column(Float, nullable=True)
    simulated_reactive_instances: Mapped[int] = mapped_column(Integer, nullable=True)

    # Savings
    estimated_savings: Mapped[float] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("idx_cost_service_time", "service_name", "timestamp"),
    )


class AlertLog(Base):
    """Alert history and acknowledgment tracking."""

    __tablename__ = "alert_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=generate_uuid
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    service_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Context
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Acknowledgment
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Resolution
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
