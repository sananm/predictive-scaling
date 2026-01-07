"""
Verification System for scaling operations.

Responsibilities:
- Wait for target replica count
- Check pod/instance health
- Verify capacity via metrics
- Check latency remains within SLO
- Timeout handling with automatic rollback trigger
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from src.execution.base import (
    BaseExecutor,
    ScalingAction,
    VerificationResult,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VerificationCheckType(str, Enum):
    """Types of verification checks."""

    INSTANCE_COUNT = "instance_count"
    HEALTH_CHECK = "health_check"
    LATENCY_CHECK = "latency_check"
    ERROR_RATE_CHECK = "error_rate_check"
    CAPACITY_CHECK = "capacity_check"
    CUSTOM_CHECK = "custom_check"


class VerificationStatus(str, Enum):
    """Status of verification."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class VerificationCheck:
    """A single verification check."""

    check_type: VerificationCheckType
    name: str
    passed: bool
    message: str
    value: Any = None
    threshold: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VerificationConfig:
    """Configuration for verification."""

    # Timing
    timeout_seconds: float = 300.0
    poll_interval_seconds: float = 10.0
    stabilization_seconds: float = 30.0

    # Instance checks
    require_all_healthy: bool = True
    min_healthy_percentage: float = 0.95

    # Latency thresholds
    max_latency_p50_ms: float = 200.0
    max_latency_p99_ms: float = 500.0

    # Error rate thresholds
    max_error_rate: float = 0.01  # 1%

    # Capacity thresholds
    min_capacity_utilization: float = 0.0
    max_capacity_utilization: float = 0.90

    # Automatic rollback
    auto_rollback_on_failure: bool = True
    rollback_delay_seconds: float = 30.0


@dataclass
class VerificationSession:
    """A verification session for a scaling action."""

    action_id: str
    action: ScalingAction
    config: VerificationConfig
    status: VerificationStatus = VerificationStatus.PENDING
    checks: list[VerificationCheck] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    rollback_triggered: bool = False
    error_message: str | None = None

    @property
    def is_passed(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.PASSED

    @property
    def all_checks_passed(self) -> bool:
        """Check if all checks passed."""
        return all(check.passed for check in self.checks)

    @property
    def duration_seconds(self) -> float | None:
        """Get verification duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "rollback_triggered": self.rollback_triggered,
            "is_passed": self.is_passed,
            "all_checks_passed": self.all_checks_passed,
            "error_message": self.error_message,
        }


class VerificationSystem:
    """
    System for verifying scaling operations.

    Performs multiple checks to ensure scaling completed successfully
    and triggers automatic rollback if verification fails.
    """

    def __init__(
        self,
        executor: BaseExecutor,
        config: VerificationConfig | None = None,
    ) -> None:
        """
        Initialize verification system.

        Args:
            executor: The executor to verify
            config: Verification configuration
        """
        self.executor = executor
        self.config = config or VerificationConfig()
        self._sessions: dict[str, VerificationSession] = {}
        self._custom_checks: list[Callable] = []
        self._metrics_provider: Callable | None = None

    def set_metrics_provider(
        self, provider: Callable[[str], dict[str, float]]
    ) -> None:
        """
        Set a metrics provider for latency/error rate checks.

        Args:
            provider: Function that takes service name and returns metrics dict
        """
        self._metrics_provider = provider

    def add_custom_check(
        self, check: Callable[[ScalingAction], VerificationCheck]
    ) -> None:
        """
        Add a custom verification check.

        Args:
            check: Function that takes action and returns VerificationCheck
        """
        self._custom_checks.append(check)

    async def verify(
        self,
        action: ScalingAction,
        config: VerificationConfig | None = None,
    ) -> VerificationSession:
        """
        Verify a scaling action completed successfully.

        Args:
            action: The scaling action to verify
            config: Optional override configuration

        Returns:
            VerificationSession with results
        """
        cfg = config or self.config

        session = VerificationSession(
            action_id=action.action_id,
            action=action,
            config=cfg,
            status=VerificationStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
        )
        self._sessions[action.action_id] = session

        try:
            # Wait for initial stabilization
            await asyncio.sleep(cfg.stabilization_seconds)

            # Run verification loop with timeout
            start_time = datetime.now(timezone.utc)
            deadline = start_time + timedelta(seconds=cfg.timeout_seconds)

            while datetime.now(timezone.utc) < deadline:
                # Run all checks
                session.checks = await self._run_checks(action, cfg)

                # Check if all passed
                if session.all_checks_passed:
                    session.status = VerificationStatus.PASSED
                    session.completed_at = datetime.now(timezone.utc)

                    logger.info(
                        "Verification passed",
                        action_id=action.action_id,
                        checks_passed=len(session.checks),
                    )
                    return session

                # Wait before next check
                await asyncio.sleep(cfg.poll_interval_seconds)

            # Timeout
            session.status = VerificationStatus.TIMEOUT
            session.completed_at = datetime.now(timezone.utc)
            session.error_message = "Verification timeout"

            logger.warning(
                "Verification timeout",
                action_id=action.action_id,
                timeout_seconds=cfg.timeout_seconds,
            )

            # Trigger rollback if configured
            if cfg.auto_rollback_on_failure:
                await self._trigger_rollback(session)

            return session

        except Exception as e:
            session.status = VerificationStatus.FAILED
            session.completed_at = datetime.now(timezone.utc)
            session.error_message = str(e)

            logger.error(
                "Verification failed",
                action_id=action.action_id,
                error=str(e),
            )

            if cfg.auto_rollback_on_failure:
                await self._trigger_rollback(session)

            return session

    async def _run_checks(
        self,
        action: ScalingAction,
        config: VerificationConfig,
    ) -> list[VerificationCheck]:
        """Run all verification checks."""
        checks = []

        # Instance count check
        count_check = await self._check_instance_count(action)
        checks.append(count_check)

        # Health check
        health_check = await self._check_health(action, config)
        checks.append(health_check)

        # Latency check (if metrics provider available)
        if self._metrics_provider:
            latency_check = await self._check_latency(action, config)
            checks.append(latency_check)

            error_check = await self._check_error_rate(action, config)
            checks.append(error_check)

        # Custom checks
        for custom_check in self._custom_checks:
            try:
                check_result = custom_check(action)
                checks.append(check_result)
            except Exception as e:
                checks.append(
                    VerificationCheck(
                        check_type=VerificationCheckType.CUSTOM_CHECK,
                        name="custom_check",
                        passed=False,
                        message=f"Custom check failed: {str(e)}",
                    )
                )

        return checks

    async def _check_instance_count(
        self, action: ScalingAction
    ) -> VerificationCheck:
        """Check that instance count matches target."""
        try:
            state = await self.executor.get_current_state()

            passed = state.instance_count == action.target_count

            return VerificationCheck(
                check_type=VerificationCheckType.INSTANCE_COUNT,
                name="instance_count",
                passed=passed,
                message=(
                    f"Instance count: {state.instance_count} "
                    f"(target: {action.target_count})"
                ),
                value=state.instance_count,
                threshold=action.target_count,
            )

        except Exception as e:
            return VerificationCheck(
                check_type=VerificationCheckType.INSTANCE_COUNT,
                name="instance_count",
                passed=False,
                message=f"Failed to check instance count: {str(e)}",
            )

    async def _check_health(
        self,
        action: ScalingAction,
        config: VerificationConfig,
    ) -> VerificationCheck:
        """Check instance health."""
        try:
            state = await self.executor.get_current_state()

            if state.instance_count == 0:
                return VerificationCheck(
                    check_type=VerificationCheckType.HEALTH_CHECK,
                    name="health",
                    passed=False,
                    message="No instances available",
                    value=0,
                )

            healthy_percentage = state.healthy_count / state.instance_count

            if config.require_all_healthy:
                passed = state.is_healthy
            else:
                passed = healthy_percentage >= config.min_healthy_percentage

            return VerificationCheck(
                check_type=VerificationCheckType.HEALTH_CHECK,
                name="health",
                passed=passed,
                message=(
                    f"Healthy: {state.healthy_count}/{state.instance_count} "
                    f"({healthy_percentage:.0%})"
                ),
                value=healthy_percentage,
                threshold=config.min_healthy_percentage,
            )

        except Exception as e:
            return VerificationCheck(
                check_type=VerificationCheckType.HEALTH_CHECK,
                name="health",
                passed=False,
                message=f"Failed to check health: {str(e)}",
            )

    async def _check_latency(
        self,
        action: ScalingAction,
        config: VerificationConfig,
    ) -> VerificationCheck:
        """Check latency metrics."""
        if not self._metrics_provider:
            return VerificationCheck(
                check_type=VerificationCheckType.LATENCY_CHECK,
                name="latency",
                passed=True,
                message="No metrics provider configured (skipped)",
            )

        try:
            service_name = action.metadata.get("service_name", "default")
            metrics = self._metrics_provider(service_name)

            p50 = metrics.get("latency_p50_ms", 0)
            p99 = metrics.get("latency_p99_ms", 0)

            p50_ok = p50 <= config.max_latency_p50_ms
            p99_ok = p99 <= config.max_latency_p99_ms
            passed = p50_ok and p99_ok

            return VerificationCheck(
                check_type=VerificationCheckType.LATENCY_CHECK,
                name="latency",
                passed=passed,
                message=(
                    f"Latency p50={p50:.0f}ms (max {config.max_latency_p50_ms}), "
                    f"p99={p99:.0f}ms (max {config.max_latency_p99_ms})"
                ),
                value={"p50": p50, "p99": p99},
                threshold={
                    "p50": config.max_latency_p50_ms,
                    "p99": config.max_latency_p99_ms,
                },
            )

        except Exception as e:
            return VerificationCheck(
                check_type=VerificationCheckType.LATENCY_CHECK,
                name="latency",
                passed=False,
                message=f"Failed to check latency: {str(e)}",
            )

    async def _check_error_rate(
        self,
        action: ScalingAction,
        config: VerificationConfig,
    ) -> VerificationCheck:
        """Check error rate metrics."""
        if not self._metrics_provider:
            return VerificationCheck(
                check_type=VerificationCheckType.ERROR_RATE_CHECK,
                name="error_rate",
                passed=True,
                message="No metrics provider configured (skipped)",
            )

        try:
            service_name = action.metadata.get("service_name", "default")
            metrics = self._metrics_provider(service_name)

            error_rate = metrics.get("error_rate", 0)
            passed = error_rate <= config.max_error_rate

            return VerificationCheck(
                check_type=VerificationCheckType.ERROR_RATE_CHECK,
                name="error_rate",
                passed=passed,
                message=(
                    f"Error rate: {error_rate:.2%} "
                    f"(max {config.max_error_rate:.2%})"
                ),
                value=error_rate,
                threshold=config.max_error_rate,
            )

        except Exception as e:
            return VerificationCheck(
                check_type=VerificationCheckType.ERROR_RATE_CHECK,
                name="error_rate",
                passed=False,
                message=f"Failed to check error rate: {str(e)}",
            )

    async def _trigger_rollback(self, session: VerificationSession) -> None:
        """Trigger automatic rollback."""
        logger.warning(
            "Triggering automatic rollback",
            action_id=session.action_id,
            reason=session.error_message or "Verification failed",
        )

        await asyncio.sleep(session.config.rollback_delay_seconds)

        try:
            result = await self.executor.rollback(session.action_id)
            session.rollback_triggered = True

            if result.success:
                logger.info(
                    "Automatic rollback completed",
                    action_id=session.action_id,
                )
            else:
                logger.error(
                    "Automatic rollback failed",
                    action_id=session.action_id,
                    error=result.error_message,
                )

        except Exception as e:
            logger.error(
                "Rollback error",
                action_id=session.action_id,
                error=str(e),
            )

    def get_session(self, action_id: str) -> VerificationSession | None:
        """Get a verification session."""
        return self._sessions.get(action_id)

    def get_recent_sessions(self, limit: int = 10) -> list[VerificationSession]:
        """Get recent verification sessions."""
        sessions = list(self._sessions.values())
        sessions.sort(
            key=lambda s: s.started_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return sessions[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get verification statistics."""
        total = len(self._sessions)
        passed = sum(1 for s in self._sessions.values() if s.is_passed)
        failed = sum(
            1 for s in self._sessions.values()
            if s.status == VerificationStatus.FAILED
        )
        timeout = sum(
            1 for s in self._sessions.values()
            if s.status == VerificationStatus.TIMEOUT
        )
        rollbacks = sum(1 for s in self._sessions.values() if s.rollback_triggered)

        return {
            "total_verifications": total,
            "passed": passed,
            "failed": failed,
            "timeout": timeout,
            "rollbacks_triggered": rollbacks,
            "success_rate": passed / total if total > 0 else 0.0,
        }
