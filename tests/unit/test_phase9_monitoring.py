"""
Unit tests for Phase 9: Monitoring and Alerting.

Tests cover:
- Prometheus metrics exporter
- Alert manager and rules
- Audit logger
- Health check system
"""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

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
)
from src.monitoring.health import (
    ComponentType,
    HealthCheck,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
    SystemHealth,
)


# =============================================================================
# Prometheus Metrics Tests
# =============================================================================


class TestScalingMetrics:
    """Tests for ScalingMetrics."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return CollectorRegistry()

    @pytest.fixture
    def metrics(self, registry):
        """Create metrics instance."""
        return ScalingMetrics(registry=registry, prefix="test")

    def test_create_metrics(self, metrics):
        """Test creating metrics instance."""
        assert metrics is not None
        assert metrics._initialized is True

    def test_record_prediction(self, metrics):
        """Test recording a prediction."""
        metrics.record_prediction(
            service="api",
            horizon=60,
            model="transformer",
            p10=100.0,
            p50=150.0,
            p90=200.0,
            confidence=0.85,
            latency_seconds=0.5,
        )

        # Metrics should be recorded (check doesn't raise)
        output = metrics.generate_metrics()
        assert b"test_predictions_total" in output
        assert b"test_current_prediction" in output

    def test_record_prediction_accuracy(self, metrics):
        """Test recording prediction accuracy."""
        metrics.record_prediction_accuracy(
            service="api",
            horizon=60,
            model="gradient_boosting",
            mae=10.5,
            mape=0.08,
            coverage=0.82,
        )

        output = metrics.generate_metrics()
        assert b"test_prediction_mae" in output
        assert b"test_prediction_mape" in output

    def test_record_scaling_decision(self, metrics):
        """Test recording a scaling decision."""
        metrics.record_scaling_decision(
            service="api",
            decision_type="scale_up",
            current_instances=5,
            target_instances=8,
            approved=True,
            risk_score=0.15,
            sla_violation_prob=0.02,
        )

        output = metrics.generate_metrics()
        assert b"test_scaling_decisions_total" in output
        assert b"test_current_instances" in output

    def test_record_execution(self, metrics):
        """Test recording an execution."""
        metrics.record_execution(
            service="api",
            executor_type="kubernetes",
            status="completed",
            duration_seconds=45.0,
        )

        output = metrics.generate_metrics()
        assert b"test_executions_total" in output
        assert b"test_execution_duration_seconds" in output

    def test_record_rollback(self, metrics):
        """Test recording a rollback."""
        metrics.record_rollback(
            service="api",
            reason="verification_failed",
            strategy="immediate",
        )

        output = metrics.generate_metrics()
        assert b"test_rollbacks_total" in output

    def test_record_verification(self, metrics):
        """Test recording verification."""
        metrics.record_verification(
            service="api",
            check_type="health",
            passed=True,
            duration_seconds=30.0,
        )

        output = metrics.generate_metrics()
        assert b"test_verification_checks_total" in output

    def test_set_component_health(self, metrics):
        """Test setting component health."""
        metrics.set_component_health("database", True)
        metrics.set_component_health("kafka", False)

        output = metrics.generate_metrics()
        assert b"test_component_health" in output

    def test_set_cost_metrics(self, metrics):
        """Test setting cost metrics."""
        metrics.set_cost_metrics(
            service="api",
            hourly_cost=10.0,
            daily_cost=240.0,
            savings=50.0,
            over_provision=5.0,
            sla_violation=0.0,
        )

        output = metrics.generate_metrics()
        assert b"test_hourly_cost_dollars" in output
        assert b"test_daily_cost_dollars" in output

    def test_generate_metrics_output(self, metrics):
        """Test generating Prometheus format output."""
        metrics.record_prediction(
            service="api",
            horizon=60,
            model="test",
            p10=100.0,
            p50=150.0,
            p90=200.0,
            confidence=0.9,
            latency_seconds=0.1,
        )

        output = metrics.generate_metrics()
        assert isinstance(output, bytes)
        assert len(output) > 0

    def test_get_content_type(self, metrics):
        """Test getting content type."""
        content_type = metrics.get_content_type()
        assert "text/plain" in content_type or "openmetrics" in content_type


# =============================================================================
# Alert Manager Tests
# =============================================================================


class TestAlertRule:
    """Tests for AlertRule."""

    def test_create_rule(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="test_rule",
            description="Test alert rule",
            category=AlertCategory.PREDICTION,
            severity=AlertSeverity.WARNING,
            condition=lambda ctx: ctx.get("value", 0) > 100,
        )

        assert rule.name == "test_rule"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.enabled is True

    def test_evaluate_condition_true(self):
        """Test evaluating condition that returns True."""
        rule = AlertRule(
            name="test",
            description="Test",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            condition=lambda ctx: ctx.get("value", 0) > 50,
        )

        assert rule.evaluate({"value": 100}) is True

    def test_evaluate_condition_false(self):
        """Test evaluating condition that returns False."""
        rule = AlertRule(
            name="test",
            description="Test",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            condition=lambda ctx: ctx.get("value", 0) > 50,
        )

        assert rule.evaluate({"value": 25}) is False

    def test_evaluate_condition_error(self):
        """Test evaluating condition that raises exception."""
        rule = AlertRule(
            name="test",
            description="Test",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            condition=lambda ctx: ctx["missing_key"] > 50,
        )

        # Should return False on error
        assert rule.evaluate({}) is False


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_properties(self):
        """Test Alert properties."""
        rule = AlertRule(
            name="test",
            description="Test",
            category=AlertCategory.SLA,
            severity=AlertSeverity.CRITICAL,
            condition=lambda ctx: True,
        )

        alert = Alert(
            alert_id="alert-001",
            rule=rule,
            state=AlertState.FIRING,
            labels={"service": "api"},
            annotations={"summary": "Test alert"},
        )

        assert alert.is_firing is True
        assert alert.duration_seconds >= 0

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        rule = AlertRule(
            name="test",
            description="Test",
            category=AlertCategory.SCALING,
            severity=AlertSeverity.ERROR,
            condition=lambda ctx: True,
        )

        alert = Alert(
            alert_id="alert-002",
            rule=rule,
            state=AlertState.RESOLVED,
            labels={},
            annotations={},
            resolved_at=datetime.now(timezone.utc),
        )

        data = alert.to_dict()
        assert data["alert_id"] == "alert-002"
        assert data["state"] == "resolved"
        assert data["severity"] == "error"


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager(self):
        """Create alert manager."""
        return AlertManager(evaluation_interval_seconds=1.0)

    def test_create_manager(self, manager):
        """Test creating alert manager."""
        assert manager is not None
        # Should have default rules
        assert len(manager._rules) > 0

    def test_add_rule(self, manager):
        """Test adding a custom rule."""
        rule = AlertRule(
            name="custom_rule",
            description="Custom test rule",
            category=AlertCategory.COST,
            severity=AlertSeverity.WARNING,
            condition=lambda ctx: ctx.get("cost", 0) > 1000,
        )

        manager.add_rule(rule)
        assert "custom_rule" in manager._rules

    def test_remove_rule(self, manager):
        """Test removing a rule."""
        rule = AlertRule(
            name="removable_rule",
            description="Test",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            condition=lambda ctx: False,
        )

        manager.add_rule(rule)
        assert manager.remove_rule("removable_rule") is True
        assert "removable_rule" not in manager._rules

    @pytest.mark.asyncio
    async def test_evaluate_fires_alert(self, manager):
        """Test that evaluation fires alerts."""
        rule = AlertRule(
            name="fire_test",
            description="Test firing",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition=lambda ctx: ctx.get("should_fire", False),
            for_duration_seconds=0,  # Immediate
        )
        manager.add_rule(rule)

        # Set context that triggers alert
        manager.update_context({"should_fire": True})

        # Evaluate
        changed = await manager.evaluate()

        assert len(changed) > 0
        firing = manager.get_firing_alerts()
        assert any(a.rule.name == "fire_test" for a in firing)

    @pytest.mark.asyncio
    async def test_evaluate_resolves_alert(self, manager):
        """Test that evaluation resolves alerts."""
        rule = AlertRule(
            name="resolve_test",
            description="Test resolving",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            condition=lambda ctx: ctx.get("active", False),
            for_duration_seconds=0,
        )
        manager.add_rule(rule)

        # Fire alert
        manager.update_context({"active": True})
        await manager.evaluate()

        # Resolve alert
        manager.update_context({"active": False})
        changed = await manager.evaluate()

        # Should have a resolved alert
        resolved = [a for a in changed if a.state == AlertState.RESOLVED]
        assert len(resolved) > 0

    def test_add_notification_channel(self, manager):
        """Test adding notification channel."""
        channel = NotificationChannel(
            name="test_webhook",
            channel_type="webhook",
            config={"url": "https://example.com/webhook"},
            severities=[AlertSeverity.CRITICAL, AlertSeverity.ERROR],
        )

        manager.add_channel(channel)
        assert "test_webhook" in manager._channels

    def test_get_alerts_by_severity(self, manager):
        """Test getting alerts by severity."""
        alerts = manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        assert isinstance(alerts, list)

    def test_get_alerts_by_category(self, manager):
        """Test getting alerts by category."""
        alerts = manager.get_alerts_by_category(AlertCategory.SLA)
        assert isinstance(alerts, list)

    def test_get_stats(self, manager):
        """Test getting alert statistics."""
        stats = manager.get_stats()

        assert "total_rules" in stats
        assert "firing_alerts" in stats
        assert "by_severity" in stats


# =============================================================================
# Audit Logger Tests
# =============================================================================


class TestAuditEvent:
    """Tests for AuditEvent."""

    def test_create_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_id="evt-001",
            event_type=AuditEventType.DECISION_APPROVED,
            timestamp=datetime.now(timezone.utc),
            actor_type=AuditActor.USER,
            actor_id="user-123",
            service_name="api",
            resource_type="decision",
            resource_id="dec-456",
            action="Approved scaling decision",
            outcome="success",
        )

        assert event.event_id == "evt-001"
        assert event.actor_type == AuditActor.USER

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_id="evt-002",
            event_type=AuditEventType.EXECUTION_COMPLETED,
            timestamp=datetime.now(timezone.utc),
            actor_type=AuditActor.SYSTEM,
            actor_id="system",
            service_name="api",
            resource_type="execution",
            resource_id="exec-001",
            action="Scaling completed",
            outcome="success",
            details={"instances": 5},
        )

        data = event.to_dict()
        assert data["event_id"] == "evt-002"
        assert data["event_type"] == "execution.completed"
        assert data["details"]["instances"] == 5

    def test_event_to_json(self):
        """Test converting event to JSON."""
        event = AuditEvent(
            event_id="evt-003",
            event_type=AuditEventType.ALERT_FIRED,
            timestamp=datetime.now(timezone.utc),
            actor_type=AuditActor.SYSTEM,
            actor_id="system",
            service_name="api",
            resource_type="alert",
            resource_id="alert-001",
            action="Alert fired",
            outcome="success",
        )

        json_str = event.to_json()
        assert isinstance(json_str, str)
        assert "evt-003" in json_str

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "event_id": "evt-004",
            "event_type": "decision.rejected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor_type": "user",
            "actor_id": "admin",
            "action": "Rejected decision",
            "outcome": "failure",
        }

        event = AuditEvent.from_dict(data)
        assert event.event_id == "evt-004"
        assert event.event_type == AuditEventType.DECISION_REJECTED


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def audit_logger(self):
        """Create audit logger with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AuditLogger(
                storage_path=Path(tmpdir),
                max_memory_events=100,
                enable_file_logging=True,
            )

    def test_create_logger(self, audit_logger):
        """Test creating audit logger."""
        assert audit_logger is not None

    def test_log_event(self, audit_logger):
        """Test logging an event."""
        event = audit_logger.log(
            event_type=AuditEventType.PREDICTION_GENERATED,
            action="Generated prediction",
            service_name="api",
        )

        assert event.event_id is not None
        assert event.event_type == AuditEventType.PREDICTION_GENERATED

    def test_log_with_details(self, audit_logger):
        """Test logging event with details."""
        event = audit_logger.log(
            event_type=AuditEventType.EXECUTION_STARTED,
            action="Started scaling",
            actor_type=AuditActor.API,
            actor_id="client-123",
            service_name="api",
            details={"target_count": 10},
        )

        assert event.details["target_count"] == 10
        assert event.actor_type == AuditActor.API

    def test_query_events(self, audit_logger):
        """Test querying events."""
        # Log some events
        audit_logger.log(
            event_type=AuditEventType.DECISION_GENERATED,
            action="Decision 1",
            service_name="api",
        )
        audit_logger.log(
            event_type=AuditEventType.DECISION_APPROVED,
            action="Decision approved",
            service_name="api",
        )

        # Query
        results = audit_logger.query(
            AuditQuery(
                event_types=[AuditEventType.DECISION_GENERATED],
                limit=10,
            )
        )

        assert len(results) >= 1
        assert all(e.event_type == AuditEventType.DECISION_GENERATED for e in results)

    def test_get_recent(self, audit_logger):
        """Test getting recent events."""
        for i in range(5):
            audit_logger.log(
                event_type=AuditEventType.API_ACCESS,
                action=f"Request {i}",
            )

        recent = audit_logger.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_by_resource(self, audit_logger):
        """Test getting events by resource."""
        audit_logger.log(
            event_type=AuditEventType.EXECUTION_COMPLETED,
            action="Execution done",
            resource_type="execution",
            resource_id="exec-001",
        )

        events = audit_logger.get_by_resource("execution", "exec-001")
        assert len(events) >= 1

    def test_log_prediction_convenience(self, audit_logger):
        """Test prediction logging convenience method."""
        event = audit_logger.log_prediction(
            service_name="api",
            horizon_minutes=60,
            model_name="transformer",
            prediction_id="pred-001",
            p50=150.0,
            confidence=0.85,
            latency_ms=100.0,
        )

        assert event.event_type == AuditEventType.PREDICTION_GENERATED
        assert event.details["horizon_minutes"] == 60

    def test_log_decision_convenience(self, audit_logger):
        """Test decision logging convenience method."""
        event = audit_logger.log_decision(
            service_name="api",
            decision_id="dec-001",
            decision_type="scale_up",
            current_instances=5,
            target_instances=8,
            requires_approval=True,
            confidence=0.9,
            reasoning=["High predicted load"],
        )

        assert event.event_type == AuditEventType.DECISION_GENERATED
        assert event.details["target_instances"] == 8

    def test_log_execution_completed(self, audit_logger):
        """Test execution completion logging."""
        event = audit_logger.log_execution_completed(
            service_name="api",
            action_id="act-001",
            executor_type="kubernetes",
            duration_seconds=45.0,
            final_count=8,
        )

        assert event.event_type == AuditEventType.EXECUTION_COMPLETED
        assert event.details["final_count"] == 8

    def test_get_failures(self, audit_logger):
        """Test getting failure events."""
        audit_logger.log(
            event_type=AuditEventType.EXECUTION_FAILED,
            action="Failed execution",
            outcome="failure",
        )

        failures = audit_logger.get_failures()
        assert len(failures) >= 1
        assert all(e.outcome == "failure" for e in failures)

    def test_get_stats(self, audit_logger):
        """Test getting audit statistics."""
        audit_logger.log(
            event_type=AuditEventType.SYSTEM_STARTED,
            action="System started",
        )

        stats = audit_logger.get_stats()
        assert "total_events" in stats
        assert "by_type" in stats
        assert "by_outcome" in stats


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_healthy_result(self):
        """Test healthy check result."""
        result = HealthCheckResult(
            component_name="database",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=5.0,
        )

        assert result.is_healthy is True

    def test_unhealthy_result(self):
        """Test unhealthy check result."""
        result = HealthCheckResult(
            component_name="kafka",
            component_type=ComponentType.MESSAGE_QUEUE,
            status=HealthStatus.UNHEALTHY,
            message="Connection failed",
            error="Connection refused",
        )

        assert result.is_healthy is False

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            component_name="api",
            component_type=ComponentType.EXTERNAL_API,
            status=HealthStatus.DEGRADED,
            message="Slow response",
            latency_ms=1500.0,
        )

        data = result.to_dict()
        assert data["component_name"] == "api"
        assert data["status"] == "degraded"
        assert data["latency_ms"] == 1500.0


class TestHealthCheck:
    """Tests for HealthCheck."""

    @pytest.mark.asyncio
    async def test_execute_sync_check(self):
        """Test executing synchronous health check."""
        check = HealthCheck(
            name="sync_check",
            component_type=ComponentType.INTERNAL,
            check_fn=lambda: True,
            timeout_seconds=1.0,
        )

        result = await check.execute()
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_execute_async_check(self):
        """Test executing async health check."""

        async def async_check():
            await asyncio.sleep(0.01)
            return {"status": HealthStatus.HEALTHY, "message": "Async OK"}

        check = HealthCheck(
            name="async_check",
            component_type=ComponentType.INTERNAL,
            check_fn=async_check,
            timeout_seconds=1.0,
        )

        result = await check.execute()
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Async OK"

    @pytest.mark.asyncio
    async def test_execute_check_timeout(self):
        """Test health check timeout."""

        async def slow_check():
            await asyncio.sleep(10.0)
            return True

        check = HealthCheck(
            name="slow_check",
            component_type=ComponentType.EXTERNAL_API,
            check_fn=slow_check,
            timeout_seconds=0.1,
        )

        result = await check.execute()
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_check_error(self):
        """Test health check with error."""

        def error_check():
            raise ValueError("Check failed")

        check = HealthCheck(
            name="error_check",
            component_type=ComponentType.DATABASE,
            check_fn=error_check,
            timeout_seconds=1.0,
        )

        result = await check.execute()
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.error


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.fixture
    def checker(self):
        """Create health checker."""
        return HealthChecker(version="1.0.0-test")

    def test_create_checker(self, checker):
        """Test creating health checker."""
        assert checker is not None
        assert checker._version == "1.0.0-test"

    def test_add_check(self, checker):
        """Test adding a health check."""
        check = HealthCheck(
            name="test_check",
            component_type=ComponentType.INTERNAL,
            check_fn=lambda: True,
        )

        checker.add_check(check)
        assert "test_check" in checker._checks

    def test_remove_check(self, checker):
        """Test removing a health check."""
        check = HealthCheck(
            name="removable",
            component_type=ComponentType.INTERNAL,
            check_fn=lambda: True,
        )

        checker.add_check(check)
        assert checker.remove_check("removable") is True
        assert "removable" not in checker._checks

    @pytest.mark.asyncio
    async def test_check_component(self, checker):
        """Test checking a single component."""
        check = HealthCheck(
            name="single_check",
            component_type=ComponentType.CACHE,
            check_fn=lambda: True,
        )
        checker.add_check(check)

        result = await checker.check_component("single_check")
        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_all(self, checker):
        """Test checking all components."""
        checker.add_check(
            HealthCheck(
                name="check1",
                component_type=ComponentType.DATABASE,
                check_fn=lambda: True,
                critical=True,
            )
        )
        checker.add_check(
            HealthCheck(
                name="check2",
                component_type=ComponentType.CACHE,
                check_fn=lambda: True,
                critical=False,
            )
        )

        health = await checker.check_all()
        assert health.status == HealthStatus.HEALTHY
        assert health.healthy_count == 2

    @pytest.mark.asyncio
    async def test_check_all_degraded(self, checker):
        """Test degraded status when non-critical fails."""
        checker.add_check(
            HealthCheck(
                name="critical_check",
                component_type=ComponentType.DATABASE,
                check_fn=lambda: True,
                critical=True,
            )
        )
        checker.add_check(
            HealthCheck(
                name="non_critical_check",
                component_type=ComponentType.CACHE,
                check_fn=lambda: False,
                critical=False,
            )
        )

        health = await checker.check_all()
        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_unhealthy(self, checker):
        """Test unhealthy status when critical fails."""
        checker.add_check(
            HealthCheck(
                name="failing_critical",
                component_type=ComponentType.DATABASE,
                check_fn=lambda: False,
                critical=True,
            )
        )

        health = await checker.check_all()
        assert health.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_is_live(self, checker):
        """Test liveness probe."""
        result = await checker.is_live()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_ready(self, checker):
        """Test readiness probe."""
        checker.add_check(
            HealthCheck(
                name="ready_check",
                component_type=ComponentType.DATABASE,
                check_fn=lambda: True,
                critical=True,
            )
        )

        result = await checker.is_ready()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_ready_false_when_critical_fails(self, checker):
        """Test readiness returns False when critical fails."""
        checker.add_check(
            HealthCheck(
                name="failing_check",
                component_type=ComponentType.DATABASE,
                check_fn=lambda: False,
                critical=True,
            )
        )

        result = await checker.is_ready()
        assert result is False

    def test_uptime(self, checker):
        """Test uptime tracking."""
        uptime = checker.uptime_seconds
        assert uptime >= 0

    def test_create_database_check(self):
        """Test database check factory."""
        check = HealthChecker.create_database_check(
            name="postgres",
            connection_fn=lambda: True,
        )

        assert check.component_type == ComponentType.DATABASE
        assert check.critical is True

    def test_create_cache_check(self):
        """Test cache check factory."""
        check = HealthChecker.create_cache_check(
            name="redis",
            ping_fn=lambda: True,
        )

        assert check.component_type == ComponentType.CACHE
        assert check.critical is False

    def test_create_kafka_check(self):
        """Test Kafka check factory."""
        check = HealthChecker.create_kafka_check(
            name="kafka",
            check_fn=lambda: True,
        )

        assert check.component_type == ComponentType.MESSAGE_QUEUE
        assert check.critical is True

    def test_get_stats(self, checker):
        """Test getting health checker statistics."""
        checker.add_check(
            HealthCheck(
                name="stat_check",
                component_type=ComponentType.INTERNAL,
                check_fn=lambda: True,
            )
        )

        stats = checker.get_stats()
        assert "total_checks" in stats
        assert "enabled_checks" in stats
        assert "by_status" in stats


class TestSystemHealth:
    """Tests for SystemHealth."""

    def test_system_health_counts(self):
        """Test healthy/unhealthy counts."""
        components = [
            HealthCheckResult(
                component_name="db",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                message="OK",
            ),
            HealthCheckResult(
                component_name="cache",
                component_type=ComponentType.CACHE,
                status=HealthStatus.UNHEALTHY,
                message="Failed",
            ),
        ]

        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            components=components,
            checked_at=datetime.now(timezone.utc),
        )

        assert health.healthy_count == 1
        assert health.unhealthy_count == 1

    def test_system_health_to_dict(self):
        """Test converting system health to dictionary."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[],
            checked_at=datetime.now(timezone.utc),
            version="1.0.0",
            uptime_seconds=3600,
        )

        data = health.to_dict()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["uptime_seconds"] == 3600


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_import_metrics(self):
        """Test importing metrics module."""
        from src.monitoring import (
            ScalingMetrics,
            get_metrics,
            init_metrics,
        )

        assert ScalingMetrics is not None

    def test_import_alerts(self):
        """Test importing alerts module."""
        from src.monitoring import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            Alert,
        )

        assert AlertManager is not None
        assert AlertSeverity.CRITICAL is not None

    def test_import_audit(self):
        """Test importing audit module."""
        from src.monitoring import (
            AuditLogger,
            AuditEvent,
            AuditEventType,
        )

        assert AuditLogger is not None
        assert AuditEventType.EXECUTION_COMPLETED is not None

    def test_import_health(self):
        """Test importing health module."""
        from src.monitoring import (
            HealthChecker,
            HealthCheck,
            HealthStatus,
        )

        assert HealthChecker is not None
        assert HealthStatus.HEALTHY is not None
