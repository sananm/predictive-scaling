"""
Rollback System for scaling operations.

Responsibilities:
- Coordinate rollback across executors
- Manage rollback policies and scheduling
- Track rollback history
- Handle multi-step rollback sequences
- Provide retry logic for failed rollbacks
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.execution.base import (
    BaseExecutor,
    RollbackResult,
    ScalingAction,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RollbackReason(str, Enum):
    """Reasons for triggering rollback."""

    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_TIMEOUT = "verification_timeout"
    HEALTH_CHECK_FAILED = "health_check_failed"
    LATENCY_EXCEEDED = "latency_exceeded"
    ERROR_RATE_EXCEEDED = "error_rate_exceeded"
    MANUAL_REQUEST = "manual_request"
    POLICY_VIOLATION = "policy_violation"
    EXECUTION_FAILED = "execution_failed"


class RollbackStrategy(str, Enum):
    """Strategies for rollback execution."""

    IMMEDIATE = "immediate"  # Rollback immediately
    DELAYED = "delayed"  # Wait before rollback
    GRADUAL = "gradual"  # Step-by-step rollback
    RETRY_FIRST = "retry_first"  # Retry operation before rollback


@dataclass
class RollbackPolicy:
    """Policy for automatic rollback."""

    strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE
    delay_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 10.0
    gradual_steps: int = 3
    gradual_interval_seconds: float = 30.0

    # Conditions that trigger rollback
    on_verification_failure: bool = True
    on_health_check_failure: bool = True
    on_latency_exceeded: bool = True
    on_error_rate_exceeded: bool = True

    # Safety settings
    require_confirmation: bool = False
    notify_on_rollback: bool = True


@dataclass
class RollbackRequest:
    """A request to rollback a scaling action."""

    request_id: str
    action_id: str
    reason: RollbackReason
    executor: BaseExecutor
    action: ScalingAction | None = None
    policy: RollbackPolicy = field(default_factory=RollbackPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RollbackRecord:
    """Record of a rollback operation."""

    request_id: str
    action_id: str
    reason: RollbackReason
    strategy: RollbackStrategy
    result: RollbackResult | None
    attempts: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if rollback succeeded."""
        return self.result is not None and self.result.success

    @property
    def duration_seconds(self) -> float | None:
        """Get rollback duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "action_id": self.action_id,
            "reason": self.reason.value,
            "strategy": self.strategy.value,
            "success": self.success,
            "attempts": self.attempts,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
        }


class RollbackManager:
    """
    Manager for coordinating rollback operations.

    Provides centralized rollback orchestration with support for
    different strategies, policies, and retry logic.
    """

    def __init__(
        self,
        default_policy: RollbackPolicy | None = None,
    ) -> None:
        """
        Initialize rollback manager.

        Args:
            default_policy: Default policy for rollbacks
        """
        self.default_policy = default_policy or RollbackPolicy()
        self._records: dict[str, RollbackRecord] = {}
        self._pending_requests: dict[str, RollbackRequest] = {}
        self._callbacks: list[callable] = []

    def add_callback(self, callback: callable) -> None:
        """
        Add a callback for rollback events.

        Args:
            callback: Function called with (request_id, event_type, data)
        """
        self._callbacks.append(callback)

    async def _notify(
        self,
        request_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Notify callbacks of rollback events."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(request_id, event_type, data)
                else:
                    callback(request_id, event_type, data)
            except Exception as e:
                logger.error(
                    "Rollback callback error",
                    callback=str(callback),
                    error=str(e),
                )

    async def request_rollback(
        self,
        request: RollbackRequest,
    ) -> RollbackRecord:
        """
        Request a rollback operation.

        Args:
            request: The rollback request

        Returns:
            RollbackRecord with results
        """
        policy = request.policy

        record = RollbackRecord(
            request_id=request.request_id,
            action_id=request.action_id,
            reason=request.reason,
            strategy=policy.strategy,
            result=None,
            metadata=request.metadata,
        )
        self._records[request.request_id] = record
        self._pending_requests[request.request_id] = request

        logger.info(
            "Rollback requested",
            request_id=request.request_id,
            action_id=request.action_id,
            reason=request.reason.value,
            strategy=policy.strategy.value,
        )

        await self._notify(
            request.request_id,
            "rollback_requested",
            {"reason": request.reason.value},
        )

        try:
            # Execute based on strategy
            if policy.strategy == RollbackStrategy.IMMEDIATE:
                await self._execute_immediate(request, record)
            elif policy.strategy == RollbackStrategy.DELAYED:
                await self._execute_delayed(request, record)
            elif policy.strategy == RollbackStrategy.GRADUAL:
                await self._execute_gradual(request, record)
            elif policy.strategy == RollbackStrategy.RETRY_FIRST:
                await self._execute_retry_first(request, record)

        except Exception as e:
            record.error_message = str(e)
            record.completed_at = datetime.now(UTC)
            logger.error(
                "Rollback failed",
                request_id=request.request_id,
                error=str(e),
            )
            await self._notify(
                request.request_id,
                "rollback_failed",
                {"error": str(e)},
            )

        finally:
            self._pending_requests.pop(request.request_id, None)

        return record

    async def _execute_immediate(
        self,
        request: RollbackRequest,
        record: RollbackRecord,
    ) -> None:
        """Execute immediate rollback."""
        record.started_at = datetime.now(UTC)
        record.attempts = 1

        result = await request.executor.rollback(request.action_id)
        record.result = result
        record.completed_at = datetime.now(UTC)

        if result.success:
            logger.info(
                "Immediate rollback completed",
                request_id=request.request_id,
                action_id=request.action_id,
            )
            await self._notify(
                request.request_id,
                "rollback_completed",
                {"success": True},
            )
        else:
            record.error_message = result.error_message
            logger.error(
                "Immediate rollback failed",
                request_id=request.request_id,
                error=result.error_message,
            )
            await self._notify(
                request.request_id,
                "rollback_failed",
                {"error": result.error_message},
            )

    async def _execute_delayed(
        self,
        request: RollbackRequest,
        record: RollbackRecord,
    ) -> None:
        """Execute delayed rollback."""
        policy = request.policy

        logger.info(
            "Delaying rollback",
            request_id=request.request_id,
            delay_seconds=policy.delay_seconds,
        )

        await self._notify(
            request.request_id,
            "rollback_delayed",
            {"delay_seconds": policy.delay_seconds},
        )

        await asyncio.sleep(policy.delay_seconds)

        # Execute after delay
        await self._execute_immediate(request, record)

    async def _execute_gradual(
        self,
        request: RollbackRequest,
        record: RollbackRecord,
    ) -> None:
        """Execute gradual step-by-step rollback."""
        policy = request.policy
        record.started_at = datetime.now(UTC)

        # Get current and target states
        rollback_state = request.executor.get_rollback_state(request.action_id)
        if rollback_state is None:
            record.error_message = "No rollback state found"
            record.completed_at = datetime.now(UTC)
            return

        current_state = await request.executor.get_current_state()
        current_count = current_state.instance_count
        target_count = rollback_state.instance_count

        if current_count == target_count:
            # Already at target
            record.result = RollbackResult(
                action_id=request.action_id,
                success=True,
                previous_state=current_state,
                restored_state=current_state,
            )
            record.completed_at = datetime.now(UTC)
            return

        # Calculate step size
        delta = target_count - current_count
        step_size = delta // policy.gradual_steps
        if step_size == 0:
            step_size = 1 if delta > 0 else -1

        logger.info(
            "Starting gradual rollback",
            request_id=request.request_id,
            current_count=current_count,
            target_count=target_count,
            steps=policy.gradual_steps,
        )

        # Execute steps
        for step in range(policy.gradual_steps):
            record.attempts += 1

            if step < policy.gradual_steps - 1:
                intermediate_target = current_count + (step + 1) * step_size
            else:
                intermediate_target = target_count

            logger.info(
                "Gradual rollback step",
                request_id=request.request_id,
                step=step + 1,
                target=intermediate_target,
            )

            await self._notify(
                request.request_id,
                "rollback_step",
                {"step": step + 1, "target": intermediate_target},
            )

            # Create intermediate action
            intermediate_action = ScalingAction(
                action_id=f"{request.action_id}_rollback_step_{step}",
                target_count=intermediate_target,
                current_count=current_count,
            )

            # Execute scaling
            result = await request.executor.scale(intermediate_action)
            if not result.is_success:
                record.error_message = f"Step {step + 1} failed: {result.error_message}"
                record.completed_at = datetime.now(UTC)
                return

            current_count = intermediate_target

            # Wait between steps
            if step < policy.gradual_steps - 1:
                await asyncio.sleep(policy.gradual_interval_seconds)

        # Clear rollback state
        request.executor.clear_rollback_state(request.action_id)

        # Build result
        final_state = await request.executor.get_current_state()
        record.result = RollbackResult(
            action_id=request.action_id,
            success=True,
            previous_state=rollback_state,
            restored_state=final_state,
        )
        record.completed_at = datetime.now(UTC)

        logger.info(
            "Gradual rollback completed",
            request_id=request.request_id,
        )
        await self._notify(
            request.request_id,
            "rollback_completed",
            {"success": True, "steps": policy.gradual_steps},
        )

    async def _execute_retry_first(
        self,
        request: RollbackRequest,
        record: RollbackRecord,
    ) -> None:
        """Retry the original action before rolling back."""
        policy = request.policy
        record.started_at = datetime.now(UTC)

        if request.action is None:
            # No action to retry, fall back to immediate rollback
            await self._execute_immediate(request, record)
            return

        # Try to retry the original action
        for attempt in range(policy.max_retries):
            record.attempts += 1

            logger.info(
                "Retrying original action",
                request_id=request.request_id,
                attempt=attempt + 1,
                max_retries=policy.max_retries,
            )

            await self._notify(
                request.request_id,
                "retry_attempt",
                {"attempt": attempt + 1, "max_retries": policy.max_retries},
            )

            result = await request.executor.scale(request.action)

            if result.is_success:
                # Verify the retry
                verification = await request.executor.verify(request.action)
                if verification.verified:
                    logger.info(
                        "Retry succeeded, rollback cancelled",
                        request_id=request.request_id,
                        attempt=attempt + 1,
                    )
                    await self._notify(
                        request.request_id,
                        "retry_succeeded",
                        {"attempt": attempt + 1},
                    )

                    # Build pseudo rollback result (action succeeded)
                    current_state = await request.executor.get_current_state()
                    record.result = RollbackResult(
                        action_id=request.action_id,
                        success=True,
                        previous_state=current_state,
                        restored_state=current_state,
                    )
                    record.completed_at = datetime.now(UTC)
                    record.metadata["retry_succeeded"] = True
                    return

            # Wait before next retry
            if attempt < policy.max_retries - 1:
                await asyncio.sleep(policy.retry_delay_seconds)

        # Retries exhausted, perform rollback
        logger.warning(
            "Retries exhausted, proceeding with rollback",
            request_id=request.request_id,
        )
        await self._notify(
            request.request_id,
            "retries_exhausted",
            {"max_retries": policy.max_retries},
        )

        await self._execute_immediate(request, record)

    async def cancel_rollback(self, request_id: str) -> bool:
        """
        Cancel a pending rollback request.

        Args:
            request_id: The request to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        if request_id in self._pending_requests:
            self._pending_requests.pop(request_id)
            logger.info("Rollback cancelled", request_id=request_id)
            await self._notify(request_id, "rollback_cancelled", {})
            return True
        return False

    def get_record(self, request_id: str) -> RollbackRecord | None:
        """Get a rollback record."""
        return self._records.get(request_id)

    def get_records_for_action(self, action_id: str) -> list[RollbackRecord]:
        """Get all rollback records for an action."""
        return [
            r for r in self._records.values()
            if r.action_id == action_id
        ]

    def get_recent_records(self, limit: int = 10) -> list[RollbackRecord]:
        """Get recent rollback records."""
        records = list(self._records.values())
        records.sort(
            key=lambda r: r.started_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return records[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get rollback statistics."""
        total = len(self._records)
        successful = sum(1 for r in self._records.values() if r.success)
        failed = sum(
            1 for r in self._records.values()
            if r.result is not None and not r.result.success
        )

        # Count by reason
        by_reason: dict[str, int] = {}
        for r in self._records.values():
            reason = r.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1

        # Count by strategy
        by_strategy: dict[str, int] = {}
        for r in self._records.values():
            strategy = r.strategy.value
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        return {
            "total_rollbacks": total,
            "successful": successful,
            "failed": failed,
            "pending": len(self._pending_requests),
            "success_rate": successful / total if total > 0 else 0.0,
            "by_reason": by_reason,
            "by_strategy": by_strategy,
        }

    def clear_history(self, before: datetime | None = None) -> int:
        """
        Clear rollback history.

        Args:
            before: Clear records before this timestamp (or all if None)

        Returns:
            Number of records cleared
        """
        if before is None:
            count = len(self._records)
            self._records.clear()
            return count

        to_remove = [
            rid for rid, record in self._records.items()
            if record.completed_at and record.completed_at < before
        ]
        for rid in to_remove:
            del self._records[rid]

        return len(to_remove)
