"""
Monitoring and Alerting for predictive scaling system.

This module provides:
- Prometheus metrics export
- Alert management and notifications
- Audit logging for compliance
- Health check system
- Prediction accuracy tracking
- Cost tracking and savings calculation
"""

from src.monitoring.metrics import (
    MetricCategory,
    ScalingMetrics,
    get_metrics,
    init_metrics,
)
from src.monitoring.alerts import (
    Alert,
    AlertCategory,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertState,
    NotificationChannel,
)
from src.monitoring.audit import (
    AuditActor,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditQuery,
    get_audit_logger,
    init_audit_logger,
)
from src.monitoring.health import (
    ComponentType,
    HealthCheck,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    get_health_checker,
    init_health_checker,
)
from src.monitoring.accuracy import (
    AccuracyAlert,
    AccuracyMetrics,
    AccuracyTracker,
    PredictionRecord,
    get_accuracy_tracker,
    init_accuracy_tracker,
)
from src.monitoring.cost import (
    CostPeriod,
    CostRecord,
    CostSummary,
    CostTracker,
    InstanceCost,
    SavingsRecord,
    get_cost_tracker,
    init_cost_tracker,
)


__all__ = [
    # Metrics
    "MetricCategory",
    "ScalingMetrics",
    "get_metrics",
    "init_metrics",
    # Alerts
    "Alert",
    "AlertCategory",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",
    # Audit
    "AuditActor",
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "AuditQuery",
    "get_audit_logger",
    "init_audit_logger",
    # Health
    "ComponentType",
    "HealthCheck",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
    "SystemHealth",
    "get_health_checker",
    "init_health_checker",
    # Accuracy Tracking
    "AccuracyAlert",
    "AccuracyMetrics",
    "AccuracyTracker",
    "PredictionRecord",
    "get_accuracy_tracker",
    "init_accuracy_tracker",
    # Cost Tracking
    "CostPeriod",
    "CostRecord",
    "CostSummary",
    "CostTracker",
    "InstanceCost",
    "SavingsRecord",
    "get_cost_tracker",
    "init_cost_tracker",
]
