"""
Scaling Service for executing and monitoring scaling operations.

Responsibilities:
- Monitor predictions and current state
- Trigger decision engine when action needed
- Execute approved decisions
- Monitor verification and trigger rollback if needed
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ScalingActionType(str, Enum):
    """Types of scaling actions."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"


class ScalingActionStatus(str, Enum):
    """Status of a scaling action."""

    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ScalingContext:
    """Context for scaling decision."""

    service_name: str
    current_instances: int
    current_utilization: float
    predicted_load: float
    predicted_load_p90: float
    prediction_confidence: float
    time_horizon_minutes: int
    current_cost_hourly: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ScalingAction:
    """A scaling action to be executed."""

    action_id: str
    service_name: str
    action_type: ScalingActionType
    current_count: int
    target_count: int
    reason: str
    confidence: float
    risk_score: float
    estimated_cost_change: float
    requires_approval: bool
    status: ScalingActionStatus = ScalingActionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    approved_at: datetime | None = None
    approved_by: str | None = None
    executed_at: datetime | None = None
    completed_at: datetime | None = None
    verification_result: dict[str, Any] | None = None
    error: str | None = None
    rollback_action_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "service_name": self.service_name,
            "action_type": self.action_type.value,
            "current_count": self.current_count,
            "target_count": self.target_count,
            "reason": self.reason,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "estimated_cost_change": self.estimated_cost_change,
            "requires_approval": self.requires_approval,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class ScalingService:
    """
    Background service for scaling operations.

    Monitors predictions, triggers scaling decisions,
    executes approved actions, and handles verification/rollback.
    """

    def __init__(
        self,
        decision_engine: Any = None,
        executor: Any = None,
        prediction_service: Any = None,
        audit_logger: Any = None,
        metrics: Any = None,
        auto_approve_threshold: float = 0.9,
        action_expiry_minutes: int = 30,
        verification_timeout_seconds: int = 300,
        cooldown_seconds: int = 10,  # Reduced from 300 for demo responsiveness
        max_concurrent_actions: int = 1,
    ) -> None:
        """
        Initialize scaling service.

        Args:
            decision_engine: Decision engine for scaling decisions
            executor: Executor for scaling actions
            prediction_service: Prediction service for load predictions
            audit_logger: Audit logger for tracking actions
            metrics: Metrics exporter
            auto_approve_threshold: Confidence threshold for auto-approval
            action_expiry_minutes: Minutes before pending action expires
            verification_timeout_seconds: Timeout for verification
            cooldown_seconds: Cooldown between scaling actions
            max_concurrent_actions: Maximum concurrent scaling actions
        """
        self._decision_engine = decision_engine
        self._executor = executor
        self._prediction_service = prediction_service
        self._audit_logger = audit_logger
        self._metrics = metrics
        self._auto_approve_threshold = auto_approve_threshold
        self._action_expiry = timedelta(minutes=action_expiry_minutes)
        self._verification_timeout = verification_timeout_seconds
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._max_concurrent = max_concurrent_actions

        self._running = False
        self._pending_actions: dict[str, ScalingAction] = {}
        self._executing_actions: dict[str, ScalingAction] = {}
        self._action_history: list[ScalingAction] = []
        self._last_action_time: dict[str, datetime] = {}
        self._callbacks: list[callable] = []

        # Statistics
        self._stats = {
            "total_evaluations": 0,
            "actions_created": 0,
            "actions_approved": 0,
            "actions_executed": 0,
            "actions_completed": 0,
            "actions_failed": 0,
            "rollbacks_triggered": 0,
        }

        logger.info("Scaling service initialized")

    def add_callback(self, callback: callable) -> None:
        """Add callback for scaling events."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the scaling service."""
        if self._running:
            return

        self._running = True
        logger.info("Scaling service started")

        # Start background tasks
        asyncio.create_task(self._monitor_pending_actions())
        asyncio.create_task(self._monitor_executing_actions())

    async def stop(self) -> None:
        """Stop the scaling service."""
        self._running = False
        logger.info("Scaling service stopped")

    async def _monitor_pending_actions(self) -> None:
        """Monitor and expire pending actions."""
        while self._running:
            try:
                now = datetime.now(UTC)
                expired = []

                for action_id, action in self._pending_actions.items():
                    if now - action.created_at > self._action_expiry:
                        expired.append(action_id)

                for action_id in expired:
                    action = self._pending_actions.pop(action_id)
                    action.status = ScalingActionStatus.EXPIRED
                    self._action_history.append(action)
                    logger.info("Action expired", action_id=action_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error("Error monitoring pending actions", error=str(e))
                await asyncio.sleep(10)

    async def _monitor_executing_actions(self) -> None:
        """Monitor executing actions for completion/failure."""
        while self._running:
            try:
                for _action_id, action in list(self._executing_actions.items()):
                    if action.status == ScalingActionStatus.VERIFYING:
                        # Check verification status
                        await self._check_verification(action)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error("Error monitoring executing actions", error=str(e))
                await asyncio.sleep(10)

    async def evaluate_scaling(
        self,
        service_name: str,
        current_instances: int,
        current_utilization: float,
        prediction: Any = None,
        bypass_cooldown: bool = False,
    ) -> ScalingAction | None:
        """
        Evaluate if scaling action is needed.

        Args:
            service_name: Service to evaluate
            current_instances: Current instance count
            current_utilization: Current utilization (0-1)
            prediction: Prediction result (optional)
            bypass_cooldown: Skip cooldown check for demo/testing

        Returns:
            ScalingAction if action is needed, None otherwise
        """
        self._stats["total_evaluations"] += 1

        # Check cooldown (can be bypassed for demo)
        if not bypass_cooldown and not self._can_scale(service_name):
            logger.debug("Scaling in cooldown", service=service_name)
            return None

        # Get predictions
        # prediction = None (Removed to avoid shadowing argument)
        if prediction is None and self._prediction_service:
            prediction = self._prediction_service.get_latest_prediction(
                service_name,
                horizon="medium",
            )

        # Build context
        context = ScalingContext(
            service_name=service_name,
            current_instances=current_instances,
            current_utilization=current_utilization,
            predicted_load=prediction.p50 if prediction else current_utilization * 100,
            predicted_load_p90=prediction.p90 if prediction else current_utilization * 120,
            prediction_confidence=prediction.confidence if prediction else 0.5,
            time_horizon_minutes=30,
            current_cost_hourly=current_instances * 0.1,  # Simplified cost
        )

        # Get decision from engine
        if self._decision_engine:
            decision = await self._invoke_decision_engine(context)
        else:
            decision = self._simple_scaling_decision(context)

        if decision is None or decision["action_type"] == "no_action":
            return None

        # Create action
        action = ScalingAction(
            action_id=str(uuid4()),
            service_name=service_name,
            action_type=ScalingActionType(decision["action_type"]),
            current_count=current_instances,
            target_count=decision["target_count"],
            reason=decision.get("reason", "Predicted load change"),
            confidence=decision.get("confidence", 0.8),
            risk_score=decision.get("risk_score", 0.2),
            estimated_cost_change=decision.get("cost_change", 0.0),
            requires_approval=decision.get("requires_approval", True),
        )

        # Auto-approve if confidence is high enough
        if (
            not action.requires_approval
            or action.confidence >= self._auto_approve_threshold
        ):
            action.status = ScalingActionStatus.APPROVED
            action.approved_at = datetime.now(UTC)
            action.approved_by = "auto"
            self._stats["actions_approved"] += 1

            # Execute immediately
            asyncio.create_task(self._execute_action(action))
        else:
            self._pending_actions[action.action_id] = action

        self._stats["actions_created"] += 1

        logger.info(
            "Scaling action created",
            action_id=action.action_id,
            service=service_name,
            type=action.action_type.value,
            target=action.target_count,
            confidence=action.confidence,
            auto_approved=action.status == ScalingActionStatus.APPROVED,
        )

        return action

    def _can_scale(self, service_name: str) -> bool:
        """Check if service can be scaled (not in cooldown)."""
        last_action = self._last_action_time.get(service_name)
        if last_action is None:
            return True
        return datetime.now(UTC) - last_action >= self._cooldown

    async def _invoke_decision_engine(
        self,
        context: ScalingContext,
    ) -> dict[str, Any] | None:
        """Invoke the decision engine."""
        if asyncio.iscoroutinefunction(self._decision_engine.decide):
            return await self._decision_engine.decide(context)
        else:
            return self._decision_engine.decide(context)

    def _simple_scaling_decision(
        self,
        context: ScalingContext,
    ) -> dict[str, Any] | None:
        """Simple scaling decision logic (fallback)."""
        target_utilization = 0.7
        headroom = 1.2

        # Calculate required capacity
        # For demo: use ~40 RPS per instance (matching predictions.py capacity model)
        rps_per_instance = 40.0
        required_capacity = context.predicted_load_p90 * headroom
        current_capacity = context.current_instances * rps_per_instance

        logger.debug(
            "Simple scaling decision",
            predicted_load=context.predicted_load,
            predicted_p90=context.predicted_load_p90,
            required_capacity=required_capacity,
            current_capacity=current_capacity,
            current_instances=context.current_instances,
        )

        if required_capacity > current_capacity:
            # Scale up
            new_instances = max(
                context.current_instances + 1,
                int(required_capacity / rps_per_instance) + 1,
            )
            return {
                "action_type": "scale_up",
                "target_count": new_instances,
                "reason": f"Predicted load {context.predicted_load_p90:.0f} RPS exceeds capacity {current_capacity:.0f} RPS",
                "confidence": context.prediction_confidence,
                "risk_score": 0.2,
                "requires_approval": new_instances > context.current_instances * 1.5,
            }
        elif (
            # Scale down based on utilization
            (context.current_utilization < target_utilization * 0.5 and context.current_instances > 1)
            or
            # Scale down based on predicted load: if we need less than 30% of current capacity
            # (less aggressive to allow high instance counts during demo)
            (required_capacity < current_capacity * 0.3 and context.current_instances > 1)
        ):
            # Scale down GRADUALLY - reduce by ~30-50% per action, not all at once
            # This prevents jumping from 37 to 1 in a single step
            predicted_instances = max(1, int(required_capacity / rps_per_instance) + 1)

            # Gradual reduction: reduce to max of (predicted, 50% of current)
            gradual_target = max(
                predicted_instances,
                int(context.current_instances * 0.5),  # At most halve each time
                1
            )
            new_instances = max(1, min(gradual_target, context.current_instances - 1))

            return {
                "action_type": "scale_down",
                "target_count": new_instances,
                "reason": f"Predicted load {context.predicted_load_p90:.0f} RPS is low, reducing from {context.current_instances} to {new_instances} instances",
                "confidence": context.prediction_confidence * 0.8,
                "risk_score": 0.3,
                "requires_approval": False,  # Auto-approve for demo
            }

        return None

    async def approve_action(
        self,
        action_id: str,
        approver: str,
    ) -> bool:
        """
        Approve a pending action.

        Args:
            action_id: Action to approve
            approver: Who is approving

        Returns:
            True if approved
        """
        if action_id not in self._pending_actions:
            return False

        action = self._pending_actions.pop(action_id)
        action.status = ScalingActionStatus.APPROVED
        action.approved_at = datetime.now(UTC)
        action.approved_by = approver

        self._stats["actions_approved"] += 1

        logger.info(
            "Action approved",
            action_id=action_id,
            approver=approver,
        )

        # Execute the action
        asyncio.create_task(self._execute_action(action))

        return True

    async def reject_action(
        self,
        action_id: str,
        reason: str,
    ) -> bool:
        """
        Reject a pending action.

        Args:
            action_id: Action to reject
            reason: Rejection reason

        Returns:
            True if rejected
        """
        if action_id not in self._pending_actions:
            return False

        action = self._pending_actions.pop(action_id)
        action.status = ScalingActionStatus.REJECTED
        action.error = reason
        self._action_history.append(action)

        logger.info(
            "Action rejected",
            action_id=action_id,
            reason=reason,
        )

        return True

    async def _execute_action(self, action: ScalingAction) -> None:
        """Execute a scaling action."""
        action.status = ScalingActionStatus.EXECUTING
        action.executed_at = datetime.now(UTC)
        self._executing_actions[action.action_id] = action

        logger.info(
            "Executing action",
            action_id=action.action_id,
            type=action.action_type.value,
            target=action.target_count,
        )

        try:
            # Execute via executor
            if self._executor:
                result = await self._invoke_executor(action)
            else:
                # Mock execution
                await asyncio.sleep(2)
                result = {"success": True}

            if result.get("success"):
                self._stats["actions_executed"] += 1
                action.status = ScalingActionStatus.VERIFYING

                # Start verification
                await self._verify_action(action)
            else:
                action.status = ScalingActionStatus.FAILED
                action.error = result.get("error", "Execution failed")
                self._stats["actions_failed"] += 1

        except Exception as e:
            action.status = ScalingActionStatus.FAILED
            action.error = str(e)
            self._stats["actions_failed"] += 1
            logger.error("Action execution failed", action_id=action.action_id, error=str(e))

        finally:
            if action.status in (
                ScalingActionStatus.COMPLETED,
                ScalingActionStatus.FAILED,
                ScalingActionStatus.ROLLED_BACK,
            ):
                self._executing_actions.pop(action.action_id, None)
                self._action_history.append(action)
                self._last_action_time[action.service_name] = datetime.now(UTC)

    async def _invoke_executor(self, action: ScalingAction) -> dict[str, Any]:
        """Invoke the executor."""
        if asyncio.iscoroutinefunction(self._executor.execute):
            return await self._executor.execute(action)
        else:
            return self._executor.execute(action)

    async def _verify_action(self, action: ScalingAction) -> None:
        """Verify a scaling action."""
        start_time = datetime.now(UTC)
        timeout = timedelta(seconds=self._verification_timeout)

        while datetime.now(UTC) - start_time < timeout:
            try:
                if self._executor:
                    result = await self._executor.verify(action)
                else:
                    # Mock verification
                    await asyncio.sleep(5)
                    result = {"verified": True, "checks_passed": ["health", "capacity"]}

                if result.get("verified"):
                    action.status = ScalingActionStatus.COMPLETED
                    action.completed_at = datetime.now(UTC)
                    action.verification_result = result
                    self._stats["actions_completed"] += 1

                    logger.info(
                        "Action verified and completed",
                        action_id=action.action_id,
                    )

                    # Notify callbacks
                    await self._notify_callbacks(action)
                    return

                elif result.get("failed"):
                    # Verification failed, trigger rollback
                    await self._rollback_action(action, "Verification failed")
                    return

                # Wait and retry
                await asyncio.sleep(10)

            except Exception as e:
                logger.error("Verification error", action_id=action.action_id, error=str(e))
                await asyncio.sleep(10)

        # Timeout - trigger rollback
        await self._rollback_action(action, "Verification timeout")

    async def _check_verification(self, action: ScalingAction) -> None:
        """Check verification status of an action."""
        # This is called by the monitor loop
        # The actual verification is handled in _verify_action
        pass

    async def _rollback_action(self, action: ScalingAction, reason: str) -> None:
        """Rollback a scaling action."""
        logger.warning(
            "Rolling back action",
            action_id=action.action_id,
            reason=reason,
        )

        self._stats["rollbacks_triggered"] += 1

        try:
            if self._executor:
                await self._executor.rollback(action)
            else:
                # Mock rollback
                await asyncio.sleep(2)

            action.status = ScalingActionStatus.ROLLED_BACK
            action.error = f"Rolled back: {reason}"

        except Exception as e:
            action.status = ScalingActionStatus.FAILED
            action.error = f"Rollback failed: {str(e)}"
            logger.error("Rollback failed", action_id=action.action_id, error=str(e))

        finally:
            self._executing_actions.pop(action.action_id, None)
            self._action_history.append(action)
            action.completed_at = datetime.now(UTC)

    async def _notify_callbacks(self, action: ScalingAction) -> None:
        """Notify callbacks of action completion."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(action)
                else:
                    callback(action)
            except Exception as e:
                logger.error("Callback error", error=str(e))

    def get_pending_actions(self) -> list[ScalingAction]:
        """Get all pending actions."""
        return list(self._pending_actions.values())

    def get_executing_actions(self) -> list[ScalingAction]:
        """Get all executing actions."""
        return list(self._executing_actions.values())

    def get_action_history(
        self,
        service_name: str | None = None,
        limit: int = 100,
    ) -> list[ScalingAction]:
        """Get action history."""
        history = self._action_history
        if service_name:
            history = [a for a in history if a.service_name == service_name]
        return list(reversed(history[-limit:]))

    def get_action(self, action_id: str) -> ScalingAction | None:
        """Get an action by ID."""
        if action_id in self._pending_actions:
            return self._pending_actions[action_id]
        if action_id in self._executing_actions:
            return self._executing_actions[action_id]
        for action in self._action_history:
            if action.action_id == action_id:
                return action
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "running": self._running,
            "pending_actions": len(self._pending_actions),
            "executing_actions": len(self._executing_actions),
            "total_evaluations": self._stats["total_evaluations"],
            "actions_created": self._stats["actions_created"],
            "actions_approved": self._stats["actions_approved"],
            "actions_executed": self._stats["actions_executed"],
            "actions_completed": self._stats["actions_completed"],
            "actions_failed": self._stats["actions_failed"],
            "rollbacks_triggered": self._stats["rollbacks_triggered"],
            "success_rate": (
                self._stats["actions_completed"] / self._stats["actions_executed"]
                if self._stats["actions_executed"] > 0
                else 0
            ),
        }


# Global instance
_scaling_service: ScalingService | None = None


def get_scaling_service() -> ScalingService:
    """Get the global scaling service."""
    global _scaling_service
    if _scaling_service is None:
        _scaling_service = ScalingService()
    return _scaling_service


def init_scaling_service(**kwargs) -> ScalingService:
    """Initialize the global scaling service."""
    global _scaling_service
    _scaling_service = ScalingService(**kwargs)
    return _scaling_service
