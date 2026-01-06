"""
Scaling Strategies for different scaling scenarios.

Responsibilities:
- Gradual Ramp: Slowly increase/decrease capacity over time
- Preemptive Burst: Quickly scale up before predicted spike
- Emergency Scale: Immediate scaling for unexpected load
- Scale Down: Carefully reduce capacity during low-traffic periods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyType(str, Enum):
    """Types of scaling strategies."""

    GRADUAL_RAMP = "gradual_ramp"
    PREEMPTIVE_BURST = "preemptive_burst"
    EMERGENCY_SCALE = "emergency_scale"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ScalingStep:
    """Single step in a scaling plan."""

    step_number: int
    target_instances: int
    scheduled_time: datetime
    duration_seconds: float
    description: str
    is_critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "target_instances": self.target_instances,
            "scheduled_time": self.scheduled_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "is_critical": self.is_critical,
        }


@dataclass
class ScalingPlan:
    """Complete scaling plan with multiple steps."""

    strategy_type: StrategyType
    current_instances: int
    target_instances: int
    steps: list[ScalingStep]
    total_duration_seconds: float
    estimated_completion: datetime
    reason: str
    rollback_target: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_scale_up(self) -> bool:
        """Check if this is a scale-up plan."""
        return self.target_instances > self.current_instances

    @property
    def is_scale_down(self) -> bool:
        """Check if this is a scale-down plan."""
        return self.target_instances < self.current_instances

    @property
    def scale_factor(self) -> float:
        """Get the scale factor."""
        if self.current_instances == 0:
            return float(self.target_instances)
        return self.target_instances / self.current_instances

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_type": self.strategy_type.value,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_seconds": self.total_duration_seconds,
            "estimated_completion": self.estimated_completion.isoformat(),
            "reason": self.reason,
            "rollback_target": self.rollback_target,
            "is_scale_up": self.is_scale_up,
            "is_scale_down": self.is_scale_down,
            "scale_factor": self.scale_factor,
            "metadata": self.metadata,
        }


@dataclass
class StrategyConfig:
    """Configuration for scaling strategies."""

    # Gradual ramp settings
    gradual_step_size: int = 2  # Instances per step
    gradual_step_interval_seconds: float = 60.0  # Time between steps

    # Preemptive burst settings
    burst_lead_time_seconds: float = 300.0  # 5 minutes before spike
    burst_step_size: int = 5  # Larger steps for burst

    # Emergency scale settings
    emergency_timeout_seconds: float = 30.0  # Fast timeout

    # Scale down settings
    scale_down_step_size: int = 1  # Conservative steps
    scale_down_interval_seconds: float = 120.0  # Slower scale down
    scale_down_cooldown_seconds: float = 300.0  # Wait before more scale down

    # General settings
    max_steps: int = 20
    min_instances: int = 1


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""

    @abstractmethod
    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        config: StrategyConfig,
        **kwargs: Any,
    ) -> ScalingPlan:
        """
        Create a scaling plan.

        Args:
            current_instances: Current instance count
            target_instances: Target instance count
            config: Strategy configuration
            **kwargs: Additional strategy-specific parameters

        Returns:
            ScalingPlan with steps to execute
        """
        pass

    @staticmethod
    def get_strategy_type() -> StrategyType:
        """Get the strategy type."""
        raise NotImplementedError


class GradualRampStrategy(ScalingStrategy):
    """
    Gradually ramp capacity up or down.

    Best for:
    - Predictable, moderate load changes
    - Cost optimization (avoid over-provisioning)
    - Minimizing disruption
    """

    @staticmethod
    def get_strategy_type() -> StrategyType:
        return StrategyType.GRADUAL_RAMP

    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        config: StrategyConfig,
        reason: str = "Gradual capacity adjustment",
        **kwargs: Any,
    ) -> ScalingPlan:
        """Create a gradual ramping plan."""
        steps = []
        now = datetime.now(timezone.utc)
        current = current_instances
        step_num = 1

        if target_instances > current_instances:
            # Scale up
            step_size = config.gradual_step_size
            while current < target_instances and step_num <= config.max_steps:
                next_target = min(current + step_size, target_instances)
                scheduled = now + timedelta(
                    seconds=(step_num - 1) * config.gradual_step_interval_seconds
                )

                steps.append(
                    ScalingStep(
                        step_number=step_num,
                        target_instances=next_target,
                        scheduled_time=scheduled,
                        duration_seconds=config.gradual_step_interval_seconds,
                        description=f"Ramp up to {next_target} instances",
                    )
                )

                current = next_target
                step_num += 1

        elif target_instances < current_instances:
            # Scale down (use scale_down settings)
            step_size = config.scale_down_step_size
            interval = config.scale_down_interval_seconds

            while current > target_instances and step_num <= config.max_steps:
                next_target = max(
                    current - step_size,
                    target_instances,
                    config.min_instances,
                )
                scheduled = now + timedelta(
                    seconds=(step_num - 1) * interval
                )

                steps.append(
                    ScalingStep(
                        step_number=step_num,
                        target_instances=next_target,
                        scheduled_time=scheduled,
                        duration_seconds=interval,
                        description=f"Ramp down to {next_target} instances",
                    )
                )

                current = next_target
                step_num += 1

        # Calculate total duration
        total_duration = (
            (len(steps) - 1) * config.gradual_step_interval_seconds
            if steps
            else 0
        )

        return ScalingPlan(
            strategy_type=StrategyType.GRADUAL_RAMP,
            current_instances=current_instances,
            target_instances=target_instances,
            steps=steps,
            total_duration_seconds=total_duration,
            estimated_completion=now + timedelta(seconds=total_duration),
            reason=reason,
            rollback_target=current_instances,
            metadata={"step_size": config.gradual_step_size},
        )


class PreemptiveBurstStrategy(ScalingStrategy):
    """
    Quickly scale up before a predicted demand spike.

    Best for:
    - Known upcoming events (marketing campaigns, launches)
    - Predicted traffic spikes
    - When timing is critical
    """

    @staticmethod
    def get_strategy_type() -> StrategyType:
        return StrategyType.PREEMPTIVE_BURST

    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        config: StrategyConfig,
        spike_time: datetime | None = None,
        reason: str = "Preemptive scaling for predicted spike",
        **kwargs: Any,
    ) -> ScalingPlan:
        """Create a preemptive burst plan."""
        steps = []
        now = datetime.now(timezone.utc)

        # Calculate when to start scaling
        if spike_time:
            lead_time = (spike_time - now).total_seconds()
            if lead_time < 0:
                # Spike already happened, treat as emergency
                lead_time = 0
        else:
            lead_time = config.burst_lead_time_seconds

        # Scale in larger, faster steps
        step_size = config.burst_step_size
        instances_needed = target_instances - current_instances

        if instances_needed <= 0:
            return ScalingPlan(
                strategy_type=StrategyType.PREEMPTIVE_BURST,
                current_instances=current_instances,
                target_instances=target_instances,
                steps=[],
                total_duration_seconds=0,
                estimated_completion=now,
                reason="No scaling needed",
                rollback_target=current_instances,
            )

        # Calculate number of steps
        num_steps = max(1, (instances_needed + step_size - 1) // step_size)
        num_steps = min(num_steps, config.max_steps)

        # Distribute time across steps
        time_per_step = lead_time / num_steps if num_steps > 0 else 0

        current = current_instances
        for i in range(num_steps):
            next_target = min(current + step_size, target_instances)
            scheduled = now + timedelta(seconds=i * time_per_step)

            steps.append(
                ScalingStep(
                    step_number=i + 1,
                    target_instances=next_target,
                    scheduled_time=scheduled,
                    duration_seconds=time_per_step,
                    description=f"Burst to {next_target} instances",
                    is_critical=True,
                )
            )

            current = next_target

        return ScalingPlan(
            strategy_type=StrategyType.PREEMPTIVE_BURST,
            current_instances=current_instances,
            target_instances=target_instances,
            steps=steps,
            total_duration_seconds=lead_time,
            estimated_completion=now + timedelta(seconds=lead_time),
            reason=reason,
            rollback_target=current_instances,
            metadata={
                "spike_time": spike_time.isoformat() if spike_time else None,
                "lead_time_seconds": lead_time,
            },
        )


class EmergencyScaleStrategy(ScalingStrategy):
    """
    Immediate scaling for unexpected high load.

    Best for:
    - Unexpected traffic spikes
    - System overload situations
    - When latency is degrading
    """

    @staticmethod
    def get_strategy_type() -> StrategyType:
        return StrategyType.EMERGENCY_SCALE

    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        config: StrategyConfig,
        reason: str = "Emergency scaling for high load",
        **kwargs: Any,
    ) -> ScalingPlan:
        """Create an emergency scaling plan (single step)."""
        now = datetime.now(timezone.utc)

        # Emergency scale is a single, immediate step
        step = ScalingStep(
            step_number=1,
            target_instances=target_instances,
            scheduled_time=now,
            duration_seconds=config.emergency_timeout_seconds,
            description=f"Emergency scale to {target_instances} instances",
            is_critical=True,
        )

        return ScalingPlan(
            strategy_type=StrategyType.EMERGENCY_SCALE,
            current_instances=current_instances,
            target_instances=target_instances,
            steps=[step],
            total_duration_seconds=config.emergency_timeout_seconds,
            estimated_completion=now + timedelta(
                seconds=config.emergency_timeout_seconds
            ),
            reason=reason,
            rollback_target=current_instances,
            metadata={"is_emergency": True},
        )


class ScaleDownStrategy(ScalingStrategy):
    """
    Carefully reduce capacity during low-traffic periods.

    Best for:
    - Off-peak hours
    - After traffic spikes subside
    - Cost optimization
    """

    @staticmethod
    def get_strategy_type() -> StrategyType:
        return StrategyType.SCALE_DOWN

    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        config: StrategyConfig,
        reason: str = "Scale down for cost optimization",
        **kwargs: Any,
    ) -> ScalingPlan:
        """Create a conservative scale-down plan."""
        steps = []
        now = datetime.now(timezone.utc)

        # Ensure we don't go below minimum
        target_instances = max(target_instances, config.min_instances)

        if target_instances >= current_instances:
            return ScalingPlan(
                strategy_type=StrategyType.SCALE_DOWN,
                current_instances=current_instances,
                target_instances=current_instances,
                steps=[],
                total_duration_seconds=0,
                estimated_completion=now,
                reason="No scale down needed",
                rollback_target=current_instances,
            )

        # Scale down conservatively
        step_size = config.scale_down_step_size
        interval = config.scale_down_interval_seconds

        current = current_instances
        step_num = 1

        while current > target_instances and step_num <= config.max_steps:
            next_target = max(current - step_size, target_instances)
            scheduled = now + timedelta(seconds=(step_num - 1) * interval)

            steps.append(
                ScalingStep(
                    step_number=step_num,
                    target_instances=next_target,
                    scheduled_time=scheduled,
                    duration_seconds=interval,
                    description=f"Scale down to {next_target} instances",
                )
            )

            current = next_target
            step_num += 1

        total_duration = (len(steps) - 1) * interval if steps else 0

        return ScalingPlan(
            strategy_type=StrategyType.SCALE_DOWN,
            current_instances=current_instances,
            target_instances=target_instances,
            steps=steps,
            total_duration_seconds=total_duration,
            estimated_completion=now + timedelta(seconds=total_duration),
            reason=reason,
            rollback_target=current_instances,
            metadata={
                "cooldown_seconds": config.scale_down_cooldown_seconds,
            },
        )


class MaintainStrategy(ScalingStrategy):
    """
    Maintain current capacity (no scaling).

    Best for:
    - Stable load periods
    - When within acceptable utilization range
    """

    @staticmethod
    def get_strategy_type() -> StrategyType:
        return StrategyType.MAINTAIN

    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        config: StrategyConfig,
        reason: str = "Maintaining current capacity",
        **kwargs: Any,
    ) -> ScalingPlan:
        """Create a no-op plan."""
        now = datetime.now(timezone.utc)

        return ScalingPlan(
            strategy_type=StrategyType.MAINTAIN,
            current_instances=current_instances,
            target_instances=current_instances,
            steps=[],
            total_duration_seconds=0,
            estimated_completion=now,
            reason=reason,
            rollback_target=current_instances,
        )


class StrategySelector:
    """
    Selects the appropriate scaling strategy based on conditions.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize strategy selector."""
        self.config = config or StrategyConfig()
        self._strategies: dict[StrategyType, ScalingStrategy] = {
            StrategyType.GRADUAL_RAMP: GradualRampStrategy(),
            StrategyType.PREEMPTIVE_BURST: PreemptiveBurstStrategy(),
            StrategyType.EMERGENCY_SCALE: EmergencyScaleStrategy(),
            StrategyType.SCALE_DOWN: ScaleDownStrategy(),
            StrategyType.MAINTAIN: MaintainStrategy(),
        }

    def select_strategy(
        self,
        current_instances: int,
        target_instances: int,
        current_utilization: float,
        time_until_spike: float | None = None,
        is_emergency: bool = False,
    ) -> StrategyType:
        """
        Select the best strategy for the given conditions.

        Args:
            current_instances: Current instance count
            target_instances: Target instance count
            current_utilization: Current capacity utilization (0-1)
            time_until_spike: Seconds until predicted spike (or None)
            is_emergency: Whether this is an emergency situation

        Returns:
            Selected strategy type
        """
        # No change needed
        if target_instances == current_instances:
            return StrategyType.MAINTAIN

        # Emergency situation
        if is_emergency or current_utilization > 0.95:
            logger.info("Selecting emergency scale strategy", utilization=current_utilization)
            return StrategyType.EMERGENCY_SCALE

        # Scale up scenarios
        if target_instances > current_instances:
            # Preemptive burst if spike is coming soon
            if time_until_spike is not None and time_until_spike < 600:  # < 10 minutes
                logger.info(
                    "Selecting preemptive burst strategy",
                    time_until_spike=time_until_spike,
                )
                return StrategyType.PREEMPTIVE_BURST

            # High utilization suggests urgency
            if current_utilization > 0.85:
                logger.info(
                    "Selecting preemptive burst due to high utilization",
                    utilization=current_utilization,
                )
                return StrategyType.PREEMPTIVE_BURST

            # Otherwise, gradual ramp
            return StrategyType.GRADUAL_RAMP

        # Scale down scenario
        return StrategyType.SCALE_DOWN

    def get_strategy(self, strategy_type: StrategyType) -> ScalingStrategy:
        """Get a strategy instance by type."""
        return self._strategies[strategy_type]

    def create_plan(
        self,
        current_instances: int,
        target_instances: int,
        strategy_type: StrategyType | None = None,
        current_utilization: float = 0.5,
        time_until_spike: float | None = None,
        is_emergency: bool = False,
        reason: str | None = None,
        **kwargs: Any,
    ) -> ScalingPlan:
        """
        Create a scaling plan using the appropriate strategy.

        Args:
            current_instances: Current instance count
            target_instances: Target instance count
            strategy_type: Override strategy selection (optional)
            current_utilization: Current utilization
            time_until_spike: Time until predicted spike
            is_emergency: Emergency flag
            reason: Reason for scaling
            **kwargs: Additional parameters for strategy

        Returns:
            ScalingPlan with steps
        """
        # Select strategy if not specified
        if strategy_type is None:
            strategy_type = self.select_strategy(
                current_instances,
                target_instances,
                current_utilization,
                time_until_spike,
                is_emergency,
            )

        strategy = self.get_strategy(strategy_type)

        # Default reason based on strategy
        if reason is None:
            reasons = {
                StrategyType.GRADUAL_RAMP: "Gradual capacity adjustment",
                StrategyType.PREEMPTIVE_BURST: "Preemptive scaling for predicted demand",
                StrategyType.EMERGENCY_SCALE: "Emergency scaling for high load",
                StrategyType.SCALE_DOWN: "Scale down for cost optimization",
                StrategyType.MAINTAIN: "Maintaining current capacity",
            }
            reason = reasons.get(strategy_type, "Scaling operation")

        return strategy.create_plan(
            current_instances,
            target_instances,
            self.config,
            reason=reason,
            **kwargs,
        )
