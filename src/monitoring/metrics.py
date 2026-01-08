"""
Prometheus Metrics Exporter for the predictive scaling system.

Responsibilities:
- Define and export system metrics
- Track prediction accuracy
- Monitor scaling operations
- Expose metrics for Prometheus scraping
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetricCategory(str, Enum):
    """Categories of metrics."""

    PREDICTION = "prediction"
    SCALING = "scaling"
    EXECUTION = "execution"
    SYSTEM = "system"
    COST = "cost"


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    description: str
    category: MetricCategory
    labels: list[str] = field(default_factory=list)


class ScalingMetrics:
    """
    Prometheus metrics for the predictive scaling system.

    Exposes metrics for:
    - Prediction accuracy and latency
    - Scaling decisions and executions
    - System health and performance
    - Cost tracking
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        prefix: str = "predictive_scaler",
    ) -> None:
        """
        Initialize metrics.

        Args:
            registry: Prometheus registry (uses default if None)
            prefix: Prefix for all metric names
        """
        self._registry = registry or CollectorRegistry()
        self._prefix = prefix
        self._initialized = False

        # Initialize all metrics
        self._init_prediction_metrics()
        self._init_scaling_metrics()
        self._init_execution_metrics()
        self._init_system_metrics()
        self._init_cost_metrics()

        self._initialized = True
        logger.info("Metrics initialized", prefix=prefix)

    def _metric_name(self, name: str) -> str:
        """Generate full metric name with prefix."""
        return f"{self._prefix}_{name}"

    def _init_prediction_metrics(self) -> None:
        """Initialize prediction-related metrics."""
        # Prediction counters
        self.predictions_total = Counter(
            self._metric_name("predictions_total"),
            "Total number of predictions made",
            ["service", "horizon", "model"],
            registry=self._registry,
        )

        self.prediction_errors_total = Counter(
            self._metric_name("prediction_errors_total"),
            "Total number of prediction errors",
            ["service", "horizon", "error_type"],
            registry=self._registry,
        )

        # Prediction accuracy
        self.prediction_mae = Gauge(
            self._metric_name("prediction_mae"),
            "Mean Absolute Error of predictions",
            ["service", "horizon", "model"],
            registry=self._registry,
        )

        self.prediction_mape = Gauge(
            self._metric_name("prediction_mape"),
            "Mean Absolute Percentage Error of predictions",
            ["service", "horizon", "model"],
            registry=self._registry,
        )

        self.prediction_coverage = Gauge(
            self._metric_name("prediction_coverage"),
            "Prediction interval coverage (should be ~0.8 for 80% CI)",
            ["service", "horizon"],
            registry=self._registry,
        )

        # Prediction latency
        self.prediction_latency_seconds = Histogram(
            self._metric_name("prediction_latency_seconds"),
            "Time taken to generate predictions",
            ["service", "horizon", "model"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        # Current predictions
        self.current_prediction = Gauge(
            self._metric_name("current_prediction"),
            "Current predicted value",
            ["service", "horizon", "quantile"],
            registry=self._registry,
        )

        self.prediction_confidence = Gauge(
            self._metric_name("prediction_confidence"),
            "Confidence score of current prediction",
            ["service", "horizon"],
            registry=self._registry,
        )

    def _init_scaling_metrics(self) -> None:
        """Initialize scaling decision metrics."""
        # Decision counters
        self.scaling_decisions_total = Counter(
            self._metric_name("scaling_decisions_total"),
            "Total number of scaling decisions made",
            ["service", "decision_type", "approved"],
            registry=self._registry,
        )

        self.scaling_decisions_pending = Gauge(
            self._metric_name("scaling_decisions_pending"),
            "Number of scaling decisions pending approval",
            ["service"],
            registry=self._registry,
        )

        # Instance counts
        self.current_instances = Gauge(
            self._metric_name("current_instances"),
            "Current number of instances",
            ["service"],
            registry=self._registry,
        )

        self.target_instances = Gauge(
            self._metric_name("target_instances"),
            "Target number of instances from last decision",
            ["service"],
            registry=self._registry,
        )

        self.recommended_instances = Gauge(
            self._metric_name("recommended_instances"),
            "Recommended number of instances",
            ["service", "horizon"],
            registry=self._registry,
        )

        # Risk metrics
        self.risk_score = Gauge(
            self._metric_name("risk_score"),
            "Current risk score (0-1)",
            ["service", "risk_type"],
            registry=self._registry,
        )

        self.sla_violation_probability = Gauge(
            self._metric_name("sla_violation_probability"),
            "Probability of SLA violation",
            ["service"],
            registry=self._registry,
        )

    def _init_execution_metrics(self) -> None:
        """Initialize execution metrics."""
        # Execution counters
        self.executions_total = Counter(
            self._metric_name("executions_total"),
            "Total number of scaling executions",
            ["service", "executor_type", "status"],
            registry=self._registry,
        )

        self.rollbacks_total = Counter(
            self._metric_name("rollbacks_total"),
            "Total number of rollbacks",
            ["service", "reason", "strategy"],
            registry=self._registry,
        )

        # Execution timing
        self.execution_duration_seconds = Histogram(
            self._metric_name("execution_duration_seconds"),
            "Time taken to execute scaling action",
            ["service", "executor_type"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
            registry=self._registry,
        )

        self.verification_duration_seconds = Histogram(
            self._metric_name("verification_duration_seconds"),
            "Time taken to verify scaling action",
            ["service"],
            buckets=(5, 10, 30, 60, 120, 300),
            registry=self._registry,
        )

        # Execution state
        self.execution_in_progress = Gauge(
            self._metric_name("execution_in_progress"),
            "Number of executions currently in progress",
            ["service", "executor_type"],
            registry=self._registry,
        )

        self.last_execution_timestamp = Gauge(
            self._metric_name("last_execution_timestamp"),
            "Timestamp of last execution",
            ["service", "status"],
            registry=self._registry,
        )

        # Verification metrics
        self.verification_checks_total = Counter(
            self._metric_name("verification_checks_total"),
            "Total number of verification checks",
            ["service", "check_type", "passed"],
            registry=self._registry,
        )

    def _init_system_metrics(self) -> None:
        """Initialize system health metrics."""
        # Component health
        self.component_health = Gauge(
            self._metric_name("component_health"),
            "Health status of system components (1=healthy, 0=unhealthy)",
            ["component"],
            registry=self._registry,
        )

        self.component_last_heartbeat = Gauge(
            self._metric_name("component_last_heartbeat"),
            "Timestamp of last heartbeat from component",
            ["component"],
            registry=self._registry,
        )

        # Data freshness
        self.data_age_seconds = Gauge(
            self._metric_name("data_age_seconds"),
            "Age of most recent data",
            ["service", "data_type"],
            registry=self._registry,
        )

        self.metrics_received_total = Counter(
            self._metric_name("metrics_received_total"),
            "Total number of metrics received",
            ["service", "collector"],
            registry=self._registry,
        )

        # Feature engineering
        self.features_computed_total = Counter(
            self._metric_name("features_computed_total"),
            "Total number of feature computations",
            ["service"],
            registry=self._registry,
        )

        self.feature_computation_seconds = Histogram(
            self._metric_name("feature_computation_seconds"),
            "Time taken to compute features",
            ["service"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        # Model info
        self.model_info = Info(
            self._metric_name("model"),
            "Information about loaded models",
            registry=self._registry,
        )

        self.model_last_trained = Gauge(
            self._metric_name("model_last_trained_timestamp"),
            "Timestamp when model was last trained",
            ["model_name"],
            registry=self._registry,
        )

    def _init_cost_metrics(self) -> None:
        """Initialize cost tracking metrics."""
        # Cost tracking
        self.hourly_cost = Gauge(
            self._metric_name("hourly_cost_dollars"),
            "Current hourly cost in dollars",
            ["service"],
            registry=self._registry,
        )

        self.daily_cost = Gauge(
            self._metric_name("daily_cost_dollars"),
            "Accumulated daily cost in dollars",
            ["service"],
            registry=self._registry,
        )

        self.cost_savings = Gauge(
            self._metric_name("cost_savings_dollars"),
            "Estimated cost savings from predictive scaling",
            ["service", "period"],
            registry=self._registry,
        )

        self.over_provision_cost = Gauge(
            self._metric_name("over_provision_cost_dollars"),
            "Cost of over-provisioned resources",
            ["service"],
            registry=self._registry,
        )

        self.sla_violation_cost = Gauge(
            self._metric_name("sla_violation_cost_dollars"),
            "Cost of SLA violations",
            ["service"],
            registry=self._registry,
        )

    # ==========================================================================
    # Recording methods
    # ==========================================================================

    def record_prediction(
        self,
        service: str,
        horizon: int,
        model: str,
        p10: float,
        p50: float,
        p90: float,
        confidence: float,
        latency_seconds: float,
    ) -> None:
        """Record a prediction."""
        horizon_str = str(horizon)

        self.predictions_total.labels(
            service=service, horizon=horizon_str, model=model
        ).inc()

        self.prediction_latency_seconds.labels(
            service=service, horizon=horizon_str, model=model
        ).observe(latency_seconds)

        self.current_prediction.labels(
            service=service, horizon=horizon_str, quantile="p10"
        ).set(p10)
        self.current_prediction.labels(
            service=service, horizon=horizon_str, quantile="p50"
        ).set(p50)
        self.current_prediction.labels(
            service=service, horizon=horizon_str, quantile="p90"
        ).set(p90)

        self.prediction_confidence.labels(
            service=service, horizon=horizon_str
        ).set(confidence)

    def record_prediction_error(
        self,
        service: str,
        horizon: int,
        error_type: str,
    ) -> None:
        """Record a prediction error."""
        self.prediction_errors_total.labels(
            service=service, horizon=str(horizon), error_type=error_type
        ).inc()

    def record_prediction_accuracy(
        self,
        service: str,
        horizon: int,
        model: str,
        mae: float,
        mape: float,
        coverage: float,
    ) -> None:
        """Record prediction accuracy metrics."""
        horizon_str = str(horizon)

        self.prediction_mae.labels(
            service=service, horizon=horizon_str, model=model
        ).set(mae)

        self.prediction_mape.labels(
            service=service, horizon=horizon_str, model=model
        ).set(mape)

        self.prediction_coverage.labels(
            service=service, horizon=horizon_str
        ).set(coverage)

    def record_scaling_decision(
        self,
        service: str,
        decision_type: str,
        current_instances: int,
        target_instances: int,
        approved: bool,
        risk_score: float,
        sla_violation_prob: float,
    ) -> None:
        """Record a scaling decision."""
        self.scaling_decisions_total.labels(
            service=service,
            decision_type=decision_type,
            approved=str(approved).lower(),
        ).inc()

        self.current_instances.labels(service=service).set(current_instances)
        self.target_instances.labels(service=service).set(target_instances)

        self.risk_score.labels(service=service, risk_type="overall").set(risk_score)
        self.sla_violation_probability.labels(service=service).set(sla_violation_prob)

    def record_execution(
        self,
        service: str,
        executor_type: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a scaling execution."""
        self.executions_total.labels(
            service=service, executor_type=executor_type, status=status
        ).inc()

        self.execution_duration_seconds.labels(
            service=service, executor_type=executor_type
        ).observe(duration_seconds)

        self.last_execution_timestamp.labels(
            service=service, status=status
        ).set(time.time())

    def record_rollback(
        self,
        service: str,
        reason: str,
        strategy: str,
    ) -> None:
        """Record a rollback."""
        self.rollbacks_total.labels(
            service=service, reason=reason, strategy=strategy
        ).inc()

    def record_verification(
        self,
        service: str,
        check_type: str,
        passed: bool,
        duration_seconds: float | None = None,
    ) -> None:
        """Record a verification check."""
        self.verification_checks_total.labels(
            service=service, check_type=check_type, passed=str(passed).lower()
        ).inc()

        if duration_seconds is not None:
            self.verification_duration_seconds.labels(service=service).observe(
                duration_seconds
            )

    def set_component_health(self, component: str, healthy: bool) -> None:
        """Set health status of a component."""
        self.component_health.labels(component=component).set(1 if healthy else 0)
        self.component_last_heartbeat.labels(component=component).set(time.time())

    def record_metrics_received(self, service: str, collector: str, count: int) -> None:
        """Record metrics received from a collector."""
        self.metrics_received_total.labels(
            service=service, collector=collector
        ).inc(count)

    def set_data_age(self, service: str, data_type: str, age_seconds: float) -> None:
        """Set the age of data."""
        self.data_age_seconds.labels(service=service, data_type=data_type).set(
            age_seconds
        )

    def record_feature_computation(
        self, service: str, duration_seconds: float
    ) -> None:
        """Record a feature computation."""
        self.features_computed_total.labels(service=service).inc()
        self.feature_computation_seconds.labels(service=service).observe(
            duration_seconds
        )

    def set_model_info(self, info: dict[str, str]) -> None:
        """Set model information."""
        self.model_info.info(info)

    def set_cost_metrics(
        self,
        service: str,
        hourly_cost: float,
        daily_cost: float,
        savings: float,
        over_provision: float,
        sla_violation: float,
    ) -> None:
        """Set cost tracking metrics."""
        self.hourly_cost.labels(service=service).set(hourly_cost)
        self.daily_cost.labels(service=service).set(daily_cost)
        self.cost_savings.labels(service=service, period="daily").set(savings)
        self.over_provision_cost.labels(service=service).set(over_provision)
        self.sla_violation_cost.labels(service=service).set(sla_violation)

    # ==========================================================================
    # Export methods
    # ==========================================================================

    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest(self._registry)

    def get_content_type(self) -> str:
        """Get the content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST

    def get_registry(self) -> CollectorRegistry:
        """Get the Prometheus registry."""
        return self._registry


# Global metrics instance
_metrics: ScalingMetrics | None = None


def get_metrics() -> ScalingMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = ScalingMetrics()
    return _metrics


def init_metrics(
    registry: CollectorRegistry | None = None,
    prefix: str = "predictive_scaler",
) -> ScalingMetrics:
    """Initialize the global metrics instance."""
    global _metrics
    _metrics = ScalingMetrics(registry=registry, prefix=prefix)
    return _metrics
