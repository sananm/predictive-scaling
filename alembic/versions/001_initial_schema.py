"""Initial schema with TimescaleDB hypertables.

Revision ID: 001
Revises:
Create Date: 2024-01-01

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # Create metrics table
    op.create_table(
        "metrics",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("service_name", sa.String(255), nullable=False),
        sa.Column("metric_name", sa.String(255), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("labels", postgresql.JSONB(), nullable=True, default={}),
        sa.PrimaryKeyConstraint("timestamp", "service_name", "metric_name"),
    )
    op.create_index("idx_metrics_service_time", "metrics", ["service_name", "timestamp"])
    op.create_index("idx_metrics_name_time", "metrics", ["metric_name", "timestamp"])

    # Create features table
    op.create_table(
        "features",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("service_name", sa.String(255), nullable=False),
        sa.Column("feature_set_version", sa.String(50), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("timestamp", "service_name"),
    )
    op.create_index("idx_features_service_time", "features", ["service_name", "timestamp"])

    # Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("service_name", sa.String(255), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("target_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_p10", sa.Float(), nullable=False),
        sa.Column("prediction_p50", sa.Float(), nullable=False),
        sa.Column("prediction_p90", sa.Float(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True, default={}),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_predictions_created", "predictions", ["created_at"])
    op.create_index("idx_predictions_service", "predictions", ["service_name"])
    op.create_index("idx_predictions_target", "predictions", ["target_timestamp"])
    op.create_index(
        "idx_predictions_service_target", "predictions", ["service_name", "target_timestamp"]
    )

    # Create scaling_decisions table
    op.create_table(
        "scaling_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("service_name", sa.String(255), nullable=False),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="pending"),
        sa.Column("current_instances", sa.Integer(), nullable=False),
        sa.Column("current_capacity", sa.Float(), nullable=True),
        sa.Column("target_instances", sa.Integer(), nullable=False),
        sa.Column("target_capacity", sa.Float(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("estimated_hourly_cost", sa.Float(), nullable=True),
        sa.Column("estimated_savings", sa.Float(), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_duration_seconds", sa.Float(), nullable=True),
        sa.Column("rollback_config", postgresql.JSONB(), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verification_criteria", postgresql.JSONB(), nullable=True),
        sa.Column("verification_result", postgresql.JSONB(), nullable=True),
        sa.Column("full_config", postgresql.JSONB(), nullable=True, default={}),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_scaling_decisions_created", "scaling_decisions", ["created_at"])
    op.create_index("idx_scaling_decisions_service", "scaling_decisions", ["service_name"])

    # Create business_events table
    op.create_table(
        "business_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_impact_multiplier", sa.Float(), default=1.0),
        sa.Column("actual_impact_multiplier", sa.Float(), nullable=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True, default={}),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_business_events_type", "business_events", ["event_type"])
    op.create_index("idx_business_events_start", "business_events", ["start_time"])

    # Create model_performance table
    op.create_table(
        "model_performance",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("mae", sa.Float(), nullable=False),
        sa.Column("mape", sa.Float(), nullable=False),
        sa.Column("rmse", sa.Float(), nullable=False),
        sa.Column("coverage_80", sa.Float(), nullable=True),
        sa.Column("coverage_90", sa.Float(), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("evaluation_window_hours", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_model_performance_recorded", "model_performance", ["recorded_at"])
    op.create_index("idx_model_performance_model", "model_performance", ["model_name"])

    # Create cost_tracking table
    op.create_table(
        "cost_tracking",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("service_name", sa.String(255), nullable=False),
        sa.Column("actual_cost_hourly", sa.Float(), nullable=False),
        sa.Column("actual_instances", sa.Integer(), nullable=False),
        sa.Column("simulated_reactive_cost", sa.Float(), nullable=True),
        sa.Column("simulated_reactive_instances", sa.Integer(), nullable=True),
        sa.Column("estimated_savings", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id", "timestamp"),
    )
    op.create_index("idx_cost_service_time", "cost_tracking", ["service_name", "timestamp"])

    # Create alert_logs table
    op.create_table(
        "alert_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("alert_type", sa.String(100), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("service_name", sa.String(255), nullable=True),
        sa.Column("context", postgresql.JSONB(), nullable=True, default={}),
        sa.Column("acknowledged", sa.Boolean(), default=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.Column("resolved", sa.Boolean(), default=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_alert_logs_created", "alert_logs", ["created_at"])
    op.create_index("idx_alert_logs_type", "alert_logs", ["alert_type"])

    # Convert time-series tables to TimescaleDB hypertables
    op.execute("SELECT create_hypertable('metrics', 'timestamp', if_not_exists => TRUE);")
    op.execute("SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);")
    op.execute("SELECT create_hypertable('cost_tracking', 'timestamp', if_not_exists => TRUE);")

    # Set up retention policies (keep 30 days of raw metrics, 90 days of features)
    op.execute(
        "SELECT add_retention_policy('metrics', INTERVAL '30 days', if_not_exists => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('features', INTERVAL '90 days', if_not_exists => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('cost_tracking', INTERVAL '365 days', if_not_exists => TRUE);"
    )


def downgrade() -> None:
    # Remove retention policies
    op.execute("SELECT remove_retention_policy('metrics', if_exists => TRUE);")
    op.execute("SELECT remove_retention_policy('features', if_exists => TRUE);")
    op.execute("SELECT remove_retention_policy('cost_tracking', if_exists => TRUE);")

    # Drop tables
    op.drop_table("alert_logs")
    op.drop_table("cost_tracking")
    op.drop_table("model_performance")
    op.drop_table("business_events")
    op.drop_table("scaling_decisions")
    op.drop_table("predictions")
    op.drop_table("features")
    op.drop_table("metrics")
