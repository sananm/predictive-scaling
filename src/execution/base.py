"""
Base Executor for infrastructure scaling.

Responsibilities:
- Define abstract interface for all executors
- Common scaling operation logic
- State management
- Error handling patterns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutorType(str, Enum):
    """Types of executors."""

    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    AWS = "aws"
    MOCK = "mock"


class ExecutionStatus(str, Enum):
    """Status of an execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    VERIFIED = "verified"


@dataclass
class InfrastructureState:
    """Current state of infrastructure."""

    executor_type: ExecutorType
    timestamp: datetime
    instance_count: int
    instance_type: str | None = None
    healthy_count: int = 0
    unhealthy_count: int = 0
    pending_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if infrastructure is healthy."""
        return self.healthy_count == self.instance_count and self.unhealthy_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "executor_type": self.executor_type.value,
            "timestamp": self.timestamp.isoformat(),
            "instance_count": self.instance_count,
            "instance_type": self.instance_type,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "pending_count": self.pending_count,
            "is_healthy": self.is_healthy,
            "metadata": self.metadata,
        }


@dataclass
class ScalingAction:
    """A scaling action to execute."""

    action_id: str
    target_count: int
    current_count: int
    instance_type: str | None = None
    strategy: str = "immediate"
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_scale_up(self) -> bool:
        """Check if this is a scale-up action."""
        return self.target_count > self.current_count

    @property
    def is_scale_down(self) -> bool:
        """Check if this is a scale-down action."""
        return self.target_count < self.current_count

    @property
    def scale_delta(self) -> int:
        """Get the number of instances to add/remove."""
        return self.target_count - self.current_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "target_count": self.target_count,
            "current_count": self.current_count,
            "instance_type": self.instance_type,
            "strategy": self.strategy,
            "timeout_seconds": self.timeout_seconds,
            "is_scale_up": self.is_scale_up,
            "is_scale_down": self.is_scale_down,
            "scale_delta": self.scale_delta,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """Result of a scaling execution."""

    action_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: datetime | None = None
    previous_state: InfrastructureState | None = None
    current_state: InfrastructureState | None = None
    error_message: str | None = None
    rollback_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Get execution duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status in (ExecutionStatus.COMPLETED, ExecutionStatus.VERIFIED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "previous_state": self.previous_state.to_dict() if self.previous_state else None,
            "current_state": self.current_state.to_dict() if self.current_state else None,
            "error_message": self.error_message,
            "rollback_available": self.rollback_available,
            "is_success": self.is_success,
            "metadata": self.metadata,
        }


@dataclass
class VerificationResult:
    """Result of scaling verification."""

    action_id: str
    verified: bool
    checks_passed: list[str]
    checks_failed: list[str]
    target_count: int
    actual_count: int
    healthy_count: int
    latency_ok: bool = True
    error_rate_ok: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "verified": self.verified,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "target_count": self.target_count,
            "actual_count": self.actual_count,
            "healthy_count": self.healthy_count,
            "latency_ok": self.latency_ok,
            "error_rate_ok": self.error_rate_ok,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    action_id: str
    success: bool
    previous_state: InfrastructureState
    restored_state: InfrastructureState | None = None
    error_message: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "success": self.success,
            "previous_state": self.previous_state.to_dict(),
            "restored_state": self.restored_state.to_dict() if self.restored_state else None,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseExecutor(ABC):
    """
    Abstract base class for infrastructure executors.

    All executors must implement:
    - scale(): Apply a scaling action
    - rollback(): Revert to previous state
    - verify(): Confirm scaling succeeded
    - get_current_state(): Read current infrastructure state
    """

    def __init__(self, executor_type: ExecutorType) -> None:
        """
        Initialize base executor.

        Args:
            executor_type: Type of this executor
        """
        self.executor_type = executor_type
        self._execution_history: list[ExecutionResult] = []
        self._rollback_states: dict[str, InfrastructureState] = {}

    @abstractmethod
    async def scale(self, action: ScalingAction) -> ExecutionResult:
        """
        Apply a scaling action.

        Args:
            action: The scaling action to execute

        Returns:
            ExecutionResult with status and details
        """
        pass

    @abstractmethod
    async def rollback(self, action_id: str) -> RollbackResult:
        """
        Rollback a previous scaling action.

        Args:
            action_id: ID of the action to rollback

        Returns:
            RollbackResult with status and details
        """
        pass

    @abstractmethod
    async def verify(self, action: ScalingAction) -> VerificationResult:
        """
        Verify that a scaling action completed successfully.

        Args:
            action: The scaling action to verify

        Returns:
            VerificationResult with verification details
        """
        pass

    @abstractmethod
    async def get_current_state(self) -> InfrastructureState:
        """
        Get current infrastructure state.

        Returns:
            Current InfrastructureState
        """
        pass

    def store_rollback_state(
        self,
        action_id: str,
        state: InfrastructureState,
    ) -> None:
        """Store state for potential rollback."""
        self._rollback_states[action_id] = state
        logger.debug(
            "Stored rollback state",
            action_id=action_id,
            instance_count=state.instance_count,
        )

    def get_rollback_state(self, action_id: str) -> InfrastructureState | None:
        """Get stored rollback state."""
        return self._rollback_states.get(action_id)

    def clear_rollback_state(self, action_id: str) -> None:
        """Clear stored rollback state."""
        if action_id in self._rollback_states:
            del self._rollback_states[action_id]

    def record_execution(self, result: ExecutionResult) -> None:
        """Record an execution result."""
        self._execution_history.append(result)
        # Keep only last 100 executions
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

    def get_execution_history(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution history."""
        return list(reversed(self._execution_history[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.is_success)
        failed = sum(
            1 for r in self._execution_history
            if r.status == ExecutionStatus.FAILED
        )
        rolled_back = sum(
            1 for r in self._execution_history
            if r.status == ExecutionStatus.ROLLED_BACK
        )

        return {
            "executor_type": self.executor_type.value,
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "rolled_back": rolled_back,
            "success_rate": successful / total if total > 0 else 0.0,
            "pending_rollbacks": len(self._rollback_states),
        }


class MockExecutor(BaseExecutor):
    """
    Mock executor for testing.

    Simulates scaling operations without making actual infrastructure changes.
    """

    def __init__(self) -> None:
        """Initialize mock executor."""
        super().__init__(ExecutorType.MOCK)
        self._current_count = 3
        self._instance_type = "mock.large"
        self._should_fail = False
        self._verification_delay = 0.0

    def set_current_count(self, count: int) -> None:
        """Set current instance count for testing."""
        self._current_count = count

    def set_should_fail(self, should_fail: bool) -> None:
        """Set whether operations should fail."""
        self._should_fail = should_fail

    async def scale(self, action: ScalingAction) -> ExecutionResult:
        """Mock scale operation."""
        started_at = datetime.now(UTC)

        # Store rollback state
        previous_state = await self.get_current_state()
        self.store_rollback_state(action.action_id, previous_state)

        if self._should_fail:
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                previous_state=previous_state,
                error_message="Mock failure",
            )
            self.record_execution(result)
            return result

        # Simulate scaling
        self._current_count = action.target_count

        current_state = await self.get_current_state()
        result = ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            previous_state=previous_state,
            current_state=current_state,
        )
        self.record_execution(result)

        logger.info(
            "Mock scale completed",
            action_id=action.action_id,
            previous_count=previous_state.instance_count,
            current_count=current_state.instance_count,
        )

        return result

    async def rollback(self, action_id: str) -> RollbackResult:
        """Mock rollback operation."""
        previous_state = self.get_rollback_state(action_id)

        if previous_state is None:
            return RollbackResult(
                action_id=action_id,
                success=False,
                previous_state=await self.get_current_state(),
                error_message="No rollback state found",
            )

        # Restore state
        self._current_count = previous_state.instance_count
        restored_state = await self.get_current_state()

        self.clear_rollback_state(action_id)

        logger.info(
            "Mock rollback completed",
            action_id=action_id,
            restored_count=restored_state.instance_count,
        )

        return RollbackResult(
            action_id=action_id,
            success=True,
            previous_state=previous_state,
            restored_state=restored_state,
        )

    async def verify(self, action: ScalingAction) -> VerificationResult:
        """Mock verification."""
        current_state = await self.get_current_state()

        checks_passed = []
        checks_failed = []

        # Check instance count
        if current_state.instance_count == action.target_count:
            checks_passed.append("instance_count")
        else:
            checks_failed.append("instance_count")

        # Check health
        if current_state.is_healthy:
            checks_passed.append("health")
        else:
            checks_failed.append("health")

        verified = len(checks_failed) == 0

        return VerificationResult(
            action_id=action.action_id,
            verified=verified,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            target_count=action.target_count,
            actual_count=current_state.instance_count,
            healthy_count=current_state.healthy_count,
        )

    async def get_current_state(self) -> InfrastructureState:
        """Get mock current state."""
        return InfrastructureState(
            executor_type=self.executor_type,
            timestamp=datetime.now(UTC),
            instance_count=self._current_count,
            instance_type=self._instance_type,
            healthy_count=self._current_count,
            unhealthy_count=0,
            pending_count=0,
        )
