"""
Alert Manager for the predictive scaling system.

Responsibilities:
- Define alert rules and thresholds
- Evaluate conditions and trigger alerts
- Manage alert lifecycle (firing, resolved)
- Send notifications via multiple channels
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    """State of an alert."""

    PENDING = "pending"  # Condition met, waiting for duration
    FIRING = "firing"  # Alert is active
    RESOLVED = "resolved"  # Condition no longer met


class AlertCategory(str, Enum):
    """Categories of alerts."""

    PREDICTION = "prediction"
    SCALING = "scaling"
    EXECUTION = "execution"
    SYSTEM = "system"
    COST = "cost"
    SLA = "sla"


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    condition: Callable[[dict[str, Any]], bool]
    for_duration_seconds: float = 0  # How long condition must be true
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the alert condition."""
        try:
            return self.condition(context)
        except Exception as e:
            logger.error(
                "Alert condition evaluation failed",
                rule=self.name,
                error=str(e),
            )
            return False


@dataclass
class Alert:
    """An instance of an alert."""

    alert_id: str
    rule: AlertRule
    state: AlertState
    labels: dict[str, str]
    annotations: dict[str, str]
    value: Any = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    fired_at: datetime | None = None
    resolved_at: datetime | None = None
    last_evaluated_at: datetime | None = None
    notification_sent: bool = False

    @property
    def duration_seconds(self) -> float:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.now(UTC)
        return (end_time - self.started_at).total_seconds()

    @property
    def is_firing(self) -> bool:
        """Check if alert is currently firing."""
        return self.state == AlertState.FIRING

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule.name,
            "category": self.rule.category.value,
            "severity": self.rule.severity.value,
            "state": self.state.value,
            "labels": self.labels,
            "annotations": self.annotations,
            "value": self.value,
            "started_at": self.started_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class NotificationChannel:
    """A notification channel configuration."""

    name: str
    channel_type: str  # "webhook", "email", "slack", "pagerduty"
    config: dict[str, Any]
    severities: list[AlertSeverity] = field(
        default_factory=lambda: list(AlertSeverity)
    )
    enabled: bool = True


class AlertManager:
    """
    Manager for alert rules and notifications.

    Evaluates alert conditions, manages alert lifecycle,
    and sends notifications through configured channels.
    """

    def __init__(
        self,
        evaluation_interval_seconds: float = 30.0,
    ) -> None:
        """
        Initialize alert manager.

        Args:
            evaluation_interval_seconds: How often to evaluate alert rules
        """
        self._rules: dict[str, AlertRule] = {}
        self._alerts: dict[str, Alert] = {}
        self._channels: dict[str, NotificationChannel] = {}
        self._callbacks: list[Callable] = []
        self._evaluation_interval = evaluation_interval_seconds
        self._running = False
        self._context: dict[str, Any] = {}
        self._pending_alerts: dict[str, datetime] = {}  # rule_name -> pending_since

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        # Prediction accuracy degradation
        self.add_rule(
            AlertRule(
                name="prediction_accuracy_degraded",
                description="Prediction accuracy has degraded significantly",
                category=AlertCategory.PREDICTION,
                severity=AlertSeverity.WARNING,
                condition=lambda ctx: ctx.get("prediction_mape", 0) > 0.2,
                for_duration_seconds=300,
                annotations={
                    "summary": "Prediction MAPE > 20%",
                    "runbook": "Check model performance and retrain if needed",
                },
            )
        )

        # High SLA violation probability
        self.add_rule(
            AlertRule(
                name="sla_violation_risk_high",
                description="High probability of SLA violation",
                category=AlertCategory.SLA,
                severity=AlertSeverity.ERROR,
                condition=lambda ctx: ctx.get("sla_violation_probability", 0) > 0.05,
                for_duration_seconds=60,
                annotations={
                    "summary": "SLA violation probability > 5%",
                    "runbook": "Consider immediate scale-up",
                },
            )
        )

        # Critical SLA violation probability
        self.add_rule(
            AlertRule(
                name="sla_violation_risk_critical",
                description="Critical probability of SLA violation",
                category=AlertCategory.SLA,
                severity=AlertSeverity.CRITICAL,
                condition=lambda ctx: ctx.get("sla_violation_probability", 0) > 0.1,
                for_duration_seconds=0,  # Immediate
                annotations={
                    "summary": "SLA violation probability > 10%",
                    "runbook": "Immediate action required - scale up now",
                },
            )
        )

        # Scaling execution failure
        self.add_rule(
            AlertRule(
                name="scaling_execution_failed",
                description="Scaling execution has failed",
                category=AlertCategory.EXECUTION,
                severity=AlertSeverity.ERROR,
                condition=lambda ctx: ctx.get("last_execution_failed", False),
                for_duration_seconds=0,
                annotations={
                    "summary": "Scaling execution failed",
                    "runbook": "Check executor logs and infrastructure state",
                },
            )
        )

        # Multiple rollbacks
        self.add_rule(
            AlertRule(
                name="excessive_rollbacks",
                description="Too many rollbacks in short period",
                category=AlertCategory.EXECUTION,
                severity=AlertSeverity.WARNING,
                condition=lambda ctx: ctx.get("rollbacks_last_hour", 0) >= 3,
                for_duration_seconds=0,
                annotations={
                    "summary": "3+ rollbacks in the last hour",
                    "runbook": "Investigate root cause of scaling failures",
                },
            )
        )

        # Data freshness
        self.add_rule(
            AlertRule(
                name="stale_metrics_data",
                description="Metrics data is stale",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.WARNING,
                condition=lambda ctx: ctx.get("metrics_age_seconds", 0) > 300,
                for_duration_seconds=60,
                annotations={
                    "summary": "Metrics data older than 5 minutes",
                    "runbook": "Check collector connectivity",
                },
            )
        )

        # Cost anomaly
        self.add_rule(
            AlertRule(
                name="cost_anomaly",
                description="Unusual cost increase detected",
                category=AlertCategory.COST,
                severity=AlertSeverity.WARNING,
                condition=lambda ctx: ctx.get("cost_increase_percent", 0) > 50,
                for_duration_seconds=1800,  # 30 minutes
                annotations={
                    "summary": "Cost increased by > 50%",
                    "runbook": "Review recent scaling decisions",
                },
            )
        )

        # Prediction latency
        self.add_rule(
            AlertRule(
                name="prediction_latency_high",
                description="Prediction latency is too high",
                category=AlertCategory.PREDICTION,
                severity=AlertSeverity.WARNING,
                condition=lambda ctx: ctx.get("prediction_latency_p99", 0) > 5.0,
                for_duration_seconds=300,
                annotations={
                    "summary": "Prediction p99 latency > 5 seconds",
                    "runbook": "Check model complexity and hardware resources",
                },
            )
        )

        # Component unhealthy
        self.add_rule(
            AlertRule(
                name="component_unhealthy",
                description="System component is unhealthy",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.ERROR,
                condition=lambda ctx: any(
                    not healthy
                    for healthy in ctx.get("component_health", {}).values()
                ),
                for_duration_seconds=60,
                annotations={
                    "summary": "One or more components are unhealthy",
                    "runbook": "Check component logs and connectivity",
                },
            )
        )

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.name] = rule
        logger.debug("Alert rule added", rule=rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels[channel.name] = channel
        logger.info(
            "Notification channel added",
            channel=channel.name,
            type=channel.channel_type,
        )

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for alert state changes."""
        self._callbacks.append(callback)

    def update_context(self, context: dict[str, Any]) -> None:
        """Update the evaluation context."""
        self._context.update(context)

    def set_context(self, context: dict[str, Any]) -> None:
        """Set the evaluation context."""
        self._context = context.copy()

    async def evaluate(self) -> list[Alert]:
        """Evaluate all alert rules and return state changes."""
        changed_alerts = []
        now = datetime.now(UTC)

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            condition_met = rule.evaluate(self._context)
            alert_key = rule.name

            if condition_met:
                if alert_key not in self._alerts:
                    # New alert - check pending duration
                    if rule.for_duration_seconds > 0:
                        if alert_key not in self._pending_alerts:
                            self._pending_alerts[alert_key] = now
                            continue

                        pending_since = self._pending_alerts[alert_key]
                        if (now - pending_since).total_seconds() < rule.for_duration_seconds:
                            continue

                    # Create and fire alert
                    alert = Alert(
                        alert_id=f"{rule.name}_{now.timestamp()}",
                        rule=rule,
                        state=AlertState.FIRING,
                        labels=rule.labels.copy(),
                        annotations=rule.annotations.copy(),
                        value=self._context.get(rule.name),
                        started_at=self._pending_alerts.get(alert_key, now),
                        fired_at=now,
                        last_evaluated_at=now,
                    )
                    self._alerts[alert_key] = alert
                    self._pending_alerts.pop(alert_key, None)
                    changed_alerts.append(alert)

                    logger.warning(
                        "Alert fired",
                        alert=rule.name,
                        severity=rule.severity.value,
                    )

                    # Send notifications
                    await self._notify(alert)

                else:
                    # Update existing alert
                    self._alerts[alert_key].last_evaluated_at = now
            else:
                # Condition not met
                self._pending_alerts.pop(alert_key, None)

                if alert_key in self._alerts:
                    alert = self._alerts[alert_key]
                    if alert.state == AlertState.FIRING:
                        # Resolve alert
                        alert.state = AlertState.RESOLVED
                        alert.resolved_at = now
                        changed_alerts.append(alert)

                        logger.info(
                            "Alert resolved",
                            alert=rule.name,
                            duration_seconds=alert.duration_seconds,
                        )

                        # Send resolution notification
                        await self._notify(alert)

        # Notify callbacks
        for alert in changed_alerts:
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(
                        "Alert callback error",
                        callback=str(callback),
                        error=str(e),
                    )

        return changed_alerts

    async def _notify(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for channel in self._channels.values():
            if not channel.enabled:
                continue

            if alert.rule.severity not in channel.severities:
                continue

            try:
                if channel.channel_type == "webhook":
                    await self._send_webhook(channel, alert)
                elif channel.channel_type == "slack":
                    await self._send_slack(channel, alert)
                elif channel.channel_type == "pagerduty":
                    await self._send_pagerduty(channel, alert)

                alert.notification_sent = True

            except Exception as e:
                logger.error(
                    "Notification failed",
                    channel=channel.name,
                    alert=alert.rule.name,
                    error=str(e),
                )

    async def _send_webhook(
        self, channel: NotificationChannel, alert: Alert
    ) -> None:
        """Send webhook notification."""
        import httpx

        url = channel.config.get("url")
        if not url:
            return

        payload = {
            "alert": alert.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                json=payload,
                headers=channel.config.get("headers", {}),
                timeout=10.0,
            )

    async def _send_slack(
        self, channel: NotificationChannel, alert: Alert
    ) -> None:
        """Send Slack notification."""
        import httpx

        webhook_url = channel.config.get("webhook_url")
        if not webhook_url:
            return

        # Format Slack message
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000",
        }.get(alert.rule.severity, "#808080")

        state_emoji = "🔥" if alert.is_firing else "✅"

        attachment = {
            "color": color,
            "title": f"{state_emoji} {alert.rule.name}",
            "text": alert.annotations.get("summary", alert.rule.description),
            "fields": [
                {"title": "Severity", "value": alert.rule.severity.value, "short": True},
                {"title": "State", "value": alert.state.value, "short": True},
                {"title": "Category", "value": alert.rule.category.value, "short": True},
            ],
            "footer": "Predictive Scaler",
            "ts": int(datetime.now(UTC).timestamp()),
        }

        if alert.annotations.get("runbook"):
            attachment["fields"].append({
                "title": "Runbook",
                "value": alert.annotations["runbook"],
                "short": False,
            })

        payload = {"attachments": [attachment]}

        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=10.0)

    async def _send_pagerduty(
        self, channel: NotificationChannel, alert: Alert
    ) -> None:
        """Send PagerDuty notification."""
        import httpx

        routing_key = channel.config.get("routing_key")
        if not routing_key:
            return

        event_action = "trigger" if alert.is_firing else "resolve"

        payload = {
            "routing_key": routing_key,
            "event_action": event_action,
            "dedup_key": alert.rule.name,
            "payload": {
                "summary": alert.annotations.get("summary", alert.rule.description),
                "severity": alert.rule.severity.value,
                "source": "predictive-scaler",
                "component": alert.rule.category.value,
                "custom_details": alert.to_dict(),
            },
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10.0,
            )

    async def start(self) -> None:
        """Start the alert evaluation loop."""
        if self._running:
            return

        self._running = True
        logger.info("Alert manager started")

        while self._running:
            try:
                await self.evaluate()
            except Exception as e:
                logger.error("Alert evaluation error", error=str(e))

            await asyncio.sleep(self._evaluation_interval)

    async def stop(self) -> None:
        """Stop the alert evaluation loop."""
        self._running = False
        logger.info("Alert manager stopped")

    def get_firing_alerts(self) -> list[Alert]:
        """Get all currently firing alerts."""
        return [a for a in self._alerts.values() if a.state == AlertState.FIRING]

    def get_alert(self, alert_key: str) -> Alert | None:
        """Get an alert by key."""
        return self._alerts.get(alert_key)

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts by severity."""
        return [
            a for a in self._alerts.values()
            if a.rule.severity == severity and a.state == AlertState.FIRING
        ]

    def get_alerts_by_category(self, category: AlertCategory) -> list[Alert]:
        """Get alerts by category."""
        return [
            a for a in self._alerts.values()
            if a.rule.category == category and a.state == AlertState.FIRING
        ]

    def acknowledge_alert(self, alert_key: str) -> bool:
        """Acknowledge an alert (prevents repeat notifications)."""
        if alert_key in self._alerts:
            self._alerts[alert_key].notification_sent = True
            return True
        return False

    def silence_rule(self, rule_name: str, duration_seconds: float) -> None:
        """Temporarily disable a rule."""
        if rule_name in self._rules:
            self._rules[rule_name].enabled = False
            # Schedule re-enabling
            asyncio.get_event_loop().call_later(
                duration_seconds,
                lambda: setattr(self._rules[rule_name], "enabled", True),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        firing = [a for a in self._alerts.values() if a.state == AlertState.FIRING]
        resolved = [a for a in self._alerts.values() if a.state == AlertState.RESOLVED]

        by_severity = {}
        for severity in AlertSeverity:
            by_severity[severity.value] = len(
                [a for a in firing if a.rule.severity == severity]
            )

        by_category = {}
        for category in AlertCategory:
            by_category[category.value] = len(
                [a for a in firing if a.rule.category == category]
            )

        return {
            "total_rules": len(self._rules),
            "enabled_rules": len([r for r in self._rules.values() if r.enabled]),
            "firing_alerts": len(firing),
            "resolved_alerts": len(resolved),
            "pending_alerts": len(self._pending_alerts),
            "by_severity": by_severity,
            "by_category": by_category,
            "notification_channels": len(self._channels),
        }
