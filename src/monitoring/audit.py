"""
Audit Logger for the predictive scaling system.

Responsibilities:
- Record all scaling decisions and actions
- Track who/what initiated changes
- Provide audit trail for compliance
- Support querying historical events
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Prediction events
    PREDICTION_GENERATED = "prediction.generated"
    PREDICTION_ERROR = "prediction.error"
    MODEL_LOADED = "model.loaded"
    MODEL_TRAINED = "model.trained"

    # Decision events
    DECISION_GENERATED = "decision.generated"
    DECISION_APPROVED = "decision.approved"
    DECISION_REJECTED = "decision.rejected"
    DECISION_AUTO_APPROVED = "decision.auto_approved"
    DECISION_EXPIRED = "decision.expired"

    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    ROLLBACK_STARTED = "rollback.started"
    ROLLBACK_COMPLETED = "rollback.completed"
    ROLLBACK_FAILED = "rollback.failed"

    # Verification events
    VERIFICATION_STARTED = "verification.started"
    VERIFICATION_PASSED = "verification.passed"
    VERIFICATION_FAILED = "verification.failed"
    VERIFICATION_TIMEOUT = "verification.timeout"

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    CONFIG_CHANGED = "config.changed"
    ALERT_FIRED = "alert.fired"
    ALERT_RESOLVED = "alert.resolved"

    # Security events
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    PERMISSION_DENIED = "permission.denied"
    API_ACCESS = "api.access"


class AuditActor(str, Enum):
    """Types of actors that can trigger events."""

    SYSTEM = "system"  # Automated system action
    USER = "user"  # Human user action
    API = "api"  # External API call
    SCHEDULER = "scheduler"  # Scheduled task
    WEBHOOK = "webhook"  # Incoming webhook


@dataclass
class AuditEvent:
    """An audit event record."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    actor_type: AuditActor
    actor_id: str
    service_name: str | None
    resource_type: str | None
    resource_id: str | None
    action: str
    outcome: str  # "success", "failure", "pending"
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_type": self.actor_type.value,
            "actor_id": self.actor_id,
            "service_name": self.service_name,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "metadata": self.metadata,
            "request_id": self.request_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            actor_type=AuditActor(data["actor_type"]),
            actor_id=data["actor_id"],
            service_name=data.get("service_name"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            action=data["action"],
            outcome=data["outcome"],
            details=data.get("details", {}),
            metadata=data.get("metadata", {}),
            request_id=data.get("request_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
        )


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""

    start_time: datetime | None = None
    end_time: datetime | None = None
    event_types: list[AuditEventType] | None = None
    actor_types: list[AuditActor] | None = None
    actor_ids: list[str] | None = None
    service_names: list[str] | None = None
    resource_types: list[str] | None = None
    resource_ids: list[str] | None = None
    outcomes: list[str] | None = None
    limit: int = 100
    offset: int = 0


class AuditLogger:
    """
    Logger for audit events.

    Provides a comprehensive audit trail of all system actions
    for compliance, debugging, and security purposes.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        max_memory_events: int = 10000,
        enable_file_logging: bool = True,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            storage_path: Path for audit log files
            max_memory_events: Maximum events to keep in memory
            enable_file_logging: Whether to write to files
        """
        self._storage_path = storage_path or Path("./audit_logs")
        self._max_memory_events = max_memory_events
        self._enable_file_logging = enable_file_logging
        self._events: list[AuditEvent] = []
        self._callbacks: list[callable] = []

        if self._enable_file_logging:
            self._storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Audit logger initialized",
            storage_path=str(self._storage_path),
            file_logging=enable_file_logging,
        )

    def add_callback(self, callback: callable) -> None:
        """Add a callback for new audit events."""
        self._callbacks.append(callback)

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str = "success",
        actor_type: AuditActor = AuditActor.SYSTEM,
        actor_id: str = "system",
        service_name: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Description of the action
            outcome: Outcome of the action
            actor_type: Type of actor
            actor_id: Identifier of the actor
            service_name: Service being affected
            resource_type: Type of resource
            resource_id: ID of resource
            details: Additional event details
            metadata: Extra metadata
            request_id: Request correlation ID
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            The created AuditEvent
        """
        event = AuditEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            actor_type=actor_type,
            actor_id=actor_id,
            service_name=service_name,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            metadata=metadata or {},
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Store in memory
        self._events.append(event)

        # Trim if needed
        if len(self._events) > self._max_memory_events:
            self._events = self._events[-self._max_memory_events:]

        # Write to file
        if self._enable_file_logging:
            self._write_to_file(event)

        # Log to standard logger
        log_level = "info" if outcome == "success" else "warning"
        getattr(logger, log_level)(
            f"Audit: {event_type.value}",
            action=action,
            outcome=outcome,
            actor=f"{actor_type.value}:{actor_id}",
            resource=f"{resource_type}:{resource_id}" if resource_type else None,
        )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Audit callback error", error=str(e))

        return event

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to audit log file."""
        # Organize by date
        date_str = event.timestamp.strftime("%Y-%m-%d")
        file_path = self._storage_path / f"audit_{date_str}.jsonl"

        try:
            with open(file_path, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(
                "Failed to write audit log",
                file=str(file_path),
                error=str(e),
            )

    def query(self, query: AuditQuery) -> list[AuditEvent]:
        """
        Query audit events.

        Args:
            query: Query parameters

        Returns:
            List of matching events
        """
        results = []

        for event in reversed(self._events):  # Most recent first
            # Apply filters
            if query.start_time and event.timestamp < query.start_time:
                continue
            if query.end_time and event.timestamp > query.end_time:
                continue
            if query.event_types and event.event_type not in query.event_types:
                continue
            if query.actor_types and event.actor_type not in query.actor_types:
                continue
            if query.actor_ids and event.actor_id not in query.actor_ids:
                continue
            if query.service_names and event.service_name not in query.service_names:
                continue
            if query.resource_types and event.resource_type not in query.resource_types:
                continue
            if query.resource_ids and event.resource_id not in query.resource_ids:
                continue
            if query.outcomes and event.outcome not in query.outcomes:
                continue

            results.append(event)

            if len(results) >= query.limit + query.offset:
                break

        return results[query.offset : query.offset + query.limit]

    def get_recent(self, limit: int = 100) -> list[AuditEvent]:
        """Get most recent events."""
        return list(reversed(self._events[-limit:]))

    def get_by_resource(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Get events for a specific resource."""
        return self.query(
            AuditQuery(
                resource_types=[resource_type],
                resource_ids=[resource_id],
                limit=limit,
            )
        )

    def get_by_actor(
        self,
        actor_id: str,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Get events by a specific actor."""
        return self.query(
            AuditQuery(
                actor_ids=[actor_id],
                limit=limit,
            )
        )

    def get_failures(self, limit: int = 100) -> list[AuditEvent]:
        """Get recent failure events."""
        return self.query(
            AuditQuery(
                outcomes=["failure"],
                limit=limit,
            )
        )

    # ==========================================================================
    # Convenience methods for common events
    # ==========================================================================

    def log_prediction(
        self,
        service_name: str,
        horizon_minutes: int,
        model_name: str,
        prediction_id: str,
        p50: float,
        confidence: float,
        latency_ms: float,
    ) -> AuditEvent:
        """Log a prediction event."""
        return self.log(
            event_type=AuditEventType.PREDICTION_GENERATED,
            action=f"Generated {horizon_minutes}min prediction for {service_name}",
            service_name=service_name,
            resource_type="prediction",
            resource_id=prediction_id,
            details={
                "horizon_minutes": horizon_minutes,
                "model_name": model_name,
                "p50": p50,
                "confidence": confidence,
                "latency_ms": latency_ms,
            },
        )

    def log_decision(
        self,
        service_name: str,
        decision_id: str,
        decision_type: str,
        current_instances: int,
        target_instances: int,
        requires_approval: bool,
        confidence: float,
        reasoning: list[str],
    ) -> AuditEvent:
        """Log a scaling decision event."""
        return self.log(
            event_type=AuditEventType.DECISION_GENERATED,
            action=f"Generated {decision_type} decision for {service_name}",
            service_name=service_name,
            resource_type="decision",
            resource_id=decision_id,
            details={
                "decision_type": decision_type,
                "current_instances": current_instances,
                "target_instances": target_instances,
                "requires_approval": requires_approval,
                "confidence": confidence,
                "reasoning": reasoning,
            },
        )

    def log_approval(
        self,
        decision_id: str,
        approver_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> AuditEvent:
        """Log a decision approval/rejection event."""
        event_type = (
            AuditEventType.DECISION_APPROVED
            if approved
            else AuditEventType.DECISION_REJECTED
        )
        return self.log(
            event_type=event_type,
            action=f"Decision {'approved' if approved else 'rejected'}",
            actor_type=AuditActor.USER,
            actor_id=approver_id,
            resource_type="decision",
            resource_id=decision_id,
            outcome="success" if approved else "failure",
            details={"reason": reason} if reason else {},
        )

    def log_execution_started(
        self,
        service_name: str,
        action_id: str,
        executor_type: str,
        target_count: int,
        current_count: int,
    ) -> AuditEvent:
        """Log execution start event."""
        return self.log(
            event_type=AuditEventType.EXECUTION_STARTED,
            action=f"Started scaling {service_name} from {current_count} to {target_count}",
            service_name=service_name,
            resource_type="execution",
            resource_id=action_id,
            outcome="pending",
            details={
                "executor_type": executor_type,
                "current_count": current_count,
                "target_count": target_count,
            },
        )

    def log_execution_completed(
        self,
        service_name: str,
        action_id: str,
        executor_type: str,
        duration_seconds: float,
        final_count: int,
    ) -> AuditEvent:
        """Log execution completion event."""
        return self.log(
            event_type=AuditEventType.EXECUTION_COMPLETED,
            action=f"Completed scaling {service_name} to {final_count} instances",
            service_name=service_name,
            resource_type="execution",
            resource_id=action_id,
            details={
                "executor_type": executor_type,
                "duration_seconds": duration_seconds,
                "final_count": final_count,
            },
        )

    def log_execution_failed(
        self,
        service_name: str,
        action_id: str,
        executor_type: str,
        error_message: str,
    ) -> AuditEvent:
        """Log execution failure event."""
        return self.log(
            event_type=AuditEventType.EXECUTION_FAILED,
            action=f"Failed to scale {service_name}",
            service_name=service_name,
            resource_type="execution",
            resource_id=action_id,
            outcome="failure",
            details={
                "executor_type": executor_type,
                "error_message": error_message,
            },
        )

    def log_rollback(
        self,
        service_name: str,
        action_id: str,
        reason: str,
        success: bool,
        restored_count: int | None = None,
    ) -> AuditEvent:
        """Log rollback event."""
        event_type = (
            AuditEventType.ROLLBACK_COMPLETED
            if success
            else AuditEventType.ROLLBACK_FAILED
        )
        return self.log(
            event_type=event_type,
            action=f"Rollback for {service_name}: {reason}",
            service_name=service_name,
            resource_type="rollback",
            resource_id=action_id,
            outcome="success" if success else "failure",
            details={
                "reason": reason,
                "restored_count": restored_count,
            },
        )

    def log_verification(
        self,
        service_name: str,
        action_id: str,
        passed: bool,
        checks_passed: list[str],
        checks_failed: list[str],
        duration_seconds: float,
    ) -> AuditEvent:
        """Log verification event."""
        event_type = (
            AuditEventType.VERIFICATION_PASSED
            if passed
            else AuditEventType.VERIFICATION_FAILED
        )
        return self.log(
            event_type=event_type,
            action=f"Verification {'passed' if passed else 'failed'} for {service_name}",
            service_name=service_name,
            resource_type="verification",
            resource_id=action_id,
            outcome="success" if passed else "failure",
            details={
                "checks_passed": checks_passed,
                "checks_failed": checks_failed,
                "duration_seconds": duration_seconds,
            },
        )

    def log_alert(
        self,
        alert_name: str,
        severity: str,
        fired: bool,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log alert event."""
        event_type = AuditEventType.ALERT_FIRED if fired else AuditEventType.ALERT_RESOLVED
        return self.log(
            event_type=event_type,
            action=f"Alert {alert_name} {'fired' if fired else 'resolved'}",
            resource_type="alert",
            resource_id=alert_name,
            details={
                "severity": severity,
                **(details or {}),
            },
        )

    def log_config_change(
        self,
        actor_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
    ) -> AuditEvent:
        """Log configuration change event."""
        return self.log(
            event_type=AuditEventType.CONFIG_CHANGED,
            action=f"Configuration changed: {config_key}",
            actor_type=AuditActor.USER,
            actor_id=actor_id,
            resource_type="config",
            resource_id=config_key,
            details={
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )

    def log_api_access(
        self,
        method: str,
        path: str,
        status_code: int,
        user_id: str | None = None,
        request_id: str | None = None,
        ip_address: str | None = None,
        duration_ms: float | None = None,
    ) -> AuditEvent:
        """Log API access event."""
        return self.log(
            event_type=AuditEventType.API_ACCESS,
            action=f"{method} {path}",
            actor_type=AuditActor.API if user_id is None else AuditActor.USER,
            actor_id=user_id or "anonymous",
            outcome="success" if status_code < 400 else "failure",
            request_id=request_id,
            ip_address=ip_address,
            details={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        by_type: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        by_actor_type: dict[str, int] = {}

        for event in self._events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_outcome[event.outcome] = by_outcome.get(event.outcome, 0) + 1
            by_actor_type[event.actor_type.value] = (
                by_actor_type.get(event.actor_type.value, 0) + 1
            )

        return {
            "total_events": len(self._events),
            "by_type": by_type,
            "by_outcome": by_outcome,
            "by_actor_type": by_actor_type,
            "oldest_event": (
                self._events[0].timestamp.isoformat() if self._events else None
            ),
            "newest_event": (
                self._events[-1].timestamp.isoformat() if self._events else None
            ),
        }


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def init_audit_logger(
    storage_path: Path | None = None,
    max_memory_events: int = 10000,
    enable_file_logging: bool = True,
) -> AuditLogger:
    """Initialize the global audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(
        storage_path=storage_path,
        max_memory_events=max_memory_events,
        enable_file_logging=enable_file_logging,
    )
    return _audit_logger
