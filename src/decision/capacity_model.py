"""
Capacity Model for infrastructure scaling.

Responsibilities:
- Map instance types to capacity (requests per second)
- Account for warm-up time when scaling up
- Model capacity degradation under high load
- Support capacity testing/benchmarking to calibrate estimates
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.decision.cost_model import CloudProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InstanceCapacity:
    """Capacity specification for an instance type."""

    instance_type: str
    provider: CloudProvider
    base_rps: float  # Requests per second at normal load
    max_rps: float  # Maximum RPS before degradation
    warm_up_seconds: float  # Time to reach full capacity after start
    degradation_threshold: float = 0.8  # Load factor where degradation starts
    degradation_rate: float = 0.1  # Capacity reduction per 10% over threshold
    memory_headroom: float = 0.85  # Target memory utilization
    cpu_headroom: float = 0.70  # Target CPU utilization
    last_benchmarked: datetime | None = None

    def effective_rps(self, load_factor: float) -> float:
        """
        Calculate effective RPS accounting for load degradation.

        Args:
            load_factor: Current load as fraction of max_rps (0.0 to 1.0+)

        Returns:
            Effective capacity in RPS
        """
        if load_factor <= self.degradation_threshold:
            return self.base_rps

        # Calculate degradation
        excess_load = load_factor - self.degradation_threshold
        degradation_factor = 1.0 - (excess_load / 0.1) * self.degradation_rate
        degradation_factor = max(0.5, degradation_factor)  # Floor at 50%

        return self.base_rps * degradation_factor

    def warm_up_capacity(self, seconds_since_start: float) -> float:
        """
        Calculate capacity during warm-up period.

        Args:
            seconds_since_start: Time since instance started

        Returns:
            Available capacity as fraction of base_rps (0.0 to 1.0)
        """
        if seconds_since_start >= self.warm_up_seconds:
            return 1.0

        # Sigmoid-like warm-up curve
        progress = seconds_since_start / self.warm_up_seconds
        return 0.1 + 0.9 * (1 - np.exp(-3 * progress))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_type": self.instance_type,
            "provider": self.provider.value,
            "base_rps": self.base_rps,
            "max_rps": self.max_rps,
            "warm_up_seconds": self.warm_up_seconds,
            "degradation_threshold": self.degradation_threshold,
            "degradation_rate": self.degradation_rate,
            "memory_headroom": self.memory_headroom,
            "cpu_headroom": self.cpu_headroom,
            "last_benchmarked": (
                self.last_benchmarked.isoformat() if self.last_benchmarked else None
            ),
        }


@dataclass
class CapacityEstimate:
    """Capacity estimate for an infrastructure configuration."""

    total_base_rps: float
    total_max_rps: float
    effective_rps: float  # Accounting for current load
    warm_up_rps: float  # During warm-up period
    headroom_rps: float  # Available headroom
    utilization: float  # Current utilization (0.0 to 1.0)
    instances_warming: int
    instances_ready: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_base_rps": self.total_base_rps,
            "total_max_rps": self.total_max_rps,
            "effective_rps": self.effective_rps,
            "warm_up_rps": self.warm_up_rps,
            "headroom_rps": self.headroom_rps,
            "utilization": self.utilization,
            "instances_warming": self.instances_warming,
            "instances_ready": self.instances_ready,
        }


@dataclass
class ScalingRequirement:
    """Required capacity for scaling decisions."""

    target_rps: float
    min_instances: int
    recommended_instances: int
    max_instances: int
    headroom_factor: float
    confidence_level: str  # "p50", "p90", "p99"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_rps": self.target_rps,
            "min_instances": self.min_instances,
            "recommended_instances": self.recommended_instances,
            "max_instances": self.max_instances,
            "headroom_factor": self.headroom_factor,
            "confidence_level": self.confidence_level,
        }


@dataclass
class CapacityConfig:
    """Configuration for capacity model."""

    # Default headroom factors
    default_headroom: float = 1.2  # 20% headroom
    peak_headroom: float = 1.5  # 50% headroom during peaks

    # Instance limits
    min_instances: int = 1
    max_instances: int = 100

    # Warm-up settings
    default_warm_up_seconds: float = 60.0

    # Load thresholds
    high_load_threshold: float = 0.8
    critical_load_threshold: float = 0.95


class CapacityModel:
    """
    Capacity model for infrastructure sizing.

    Maps instance types to capacity and calculates required
    instances for target load levels.
    """

    # Default capacity estimates per vCPU (RPS)
    DEFAULT_RPS_PER_VCPU = 500

    # Capacity multipliers by instance family
    FAMILY_MULTIPLIERS: dict[str, float] = {
        "t3": 0.8,  # Burstable, lower sustained capacity
        "m5": 1.0,  # General purpose baseline
        "c5": 1.3,  # Compute optimized
        "r5": 0.9,  # Memory optimized (less CPU-efficient)
        "e2": 0.85,  # GCP E2
        "n1": 1.0,  # GCP N1
        "Standard_B": 0.8,  # Azure burstable
        "Standard_D": 1.0,  # Azure general purpose
    }

    # Warm-up times by instance family (seconds)
    WARM_UP_TIMES: dict[str, float] = {
        "t3": 30,
        "m5": 45,
        "c5": 45,
        "r5": 60,
        "e2": 30,
        "n1": 45,
        "Standard_B": 30,
        "Standard_D": 45,
    }

    def __init__(
        self,
        config: CapacityConfig | None = None,
        provider: CloudProvider = CloudProvider.AWS,
    ) -> None:
        """
        Initialize capacity model.

        Args:
            config: Capacity configuration
            provider: Cloud provider
        """
        self.config = config or CapacityConfig()
        self.provider = provider
        self._capacity_cache: dict[str, InstanceCapacity] = {}
        self._benchmark_results: dict[str, list[float]] = {}

    def get_capacity(self, instance_type: str) -> InstanceCapacity:
        """
        Get capacity specification for an instance type.

        Args:
            instance_type: Instance type name

        Returns:
            InstanceCapacity with capacity metrics
        """
        if instance_type in self._capacity_cache:
            return self._capacity_cache[instance_type]

        # Calculate capacity based on instance specs
        capacity = self._estimate_capacity(instance_type)
        self._capacity_cache[instance_type] = capacity
        return capacity

    def _estimate_capacity(self, instance_type: str) -> InstanceCapacity:
        """Estimate capacity for an instance type."""
        # Extract instance family and size
        family = self._get_instance_family(instance_type)
        vcpus = self._estimate_vcpus(instance_type)

        # Get multiplier for family
        multiplier = 1.0
        for family_prefix, mult in self.FAMILY_MULTIPLIERS.items():
            if family.startswith(family_prefix):
                multiplier = mult
                break

        # Calculate base RPS
        base_rps = vcpus * self.DEFAULT_RPS_PER_VCPU * multiplier

        # Get warm-up time
        warm_up = self.config.default_warm_up_seconds
        for family_prefix, wu_time in self.WARM_UP_TIMES.items():
            if family.startswith(family_prefix):
                warm_up = wu_time
                break

        return InstanceCapacity(
            instance_type=instance_type,
            provider=self.provider,
            base_rps=base_rps,
            max_rps=base_rps * 1.5,  # Max is 50% over base
            warm_up_seconds=warm_up,
        )

    def _get_instance_family(self, instance_type: str) -> str:
        """Extract instance family from type name."""
        if "." in instance_type:
            # AWS format: t3.large
            return instance_type.split(".")[0]
        elif instance_type.startswith("Standard_"):
            # Azure format: Standard_D2s_v3
            parts = instance_type.split("_")
            if len(parts) >= 2:
                return f"Standard_{parts[1][0]}"
        elif "-" in instance_type:
            # GCP format: e2-standard-2
            return instance_type.split("-")[0]
        return instance_type

    def _estimate_vcpus(self, instance_type: str) -> int:
        """Estimate vCPU count from instance type name."""
        # Try to extract number from instance type
        import re

        # AWS: t3.xlarge -> 4, t3.2xlarge -> 8
        if "micro" in instance_type:
            return 1
        elif "small" in instance_type:
            return 1
        elif "medium" in instance_type:
            return 2
        elif "large" in instance_type:
            # Check for multiplier (2xlarge, 4xlarge, etc.)
            match = re.search(r"(\d+)xlarge", instance_type)
            if match:
                multiplier = int(match.group(1))
                return multiplier * 4  # xlarge is typically 4 vCPU
            return 2 if "xlarge" not in instance_type else 4

        # GCP: e2-standard-4 -> 4
        match = re.search(r"-(\d+)$", instance_type)
        if match:
            return int(match.group(1))

        # Azure: Standard_D4s_v3 -> 4
        match = re.search(r"[A-Z](\d+)", instance_type)
        if match:
            return int(match.group(1))

        return 2  # Default

    def calculate_capacity(
        self,
        instance_type: str,
        instance_count: int,
        current_load: float = 0.0,
        warming_instances: int = 0,
        warm_up_progress: float = 0.0,
    ) -> CapacityEstimate:
        """
        Calculate total capacity for a configuration.

        Args:
            instance_type: Type of instances
            instance_count: Number of instances
            current_load: Current RPS load
            warming_instances: Number of instances still warming up
            warm_up_progress: Progress of warming instances (0.0 to 1.0)

        Returns:
            CapacityEstimate with capacity metrics
        """
        capacity = self.get_capacity(instance_type)
        ready_instances = instance_count - warming_instances

        # Calculate ready capacity
        total_base_rps = ready_instances * capacity.base_rps
        total_max_rps = ready_instances * capacity.max_rps

        # Calculate warming capacity
        warm_up_rps = 0.0
        if warming_instances > 0:
            warm_factor = capacity.warm_up_capacity(
                warm_up_progress * capacity.warm_up_seconds
            )
            warm_up_rps = warming_instances * capacity.base_rps * warm_factor

        # Calculate effective capacity under current load
        load_factor = current_load / total_base_rps if total_base_rps > 0 else 0
        effective_rps = (
            ready_instances * capacity.effective_rps(load_factor) + warm_up_rps
        )

        # Calculate headroom
        headroom_rps = effective_rps - current_load

        # Calculate utilization
        utilization = current_load / effective_rps if effective_rps > 0 else 1.0

        return CapacityEstimate(
            total_base_rps=total_base_rps + warm_up_rps,
            total_max_rps=total_max_rps,
            effective_rps=effective_rps,
            warm_up_rps=warm_up_rps,
            headroom_rps=max(0, headroom_rps),
            utilization=min(1.0, utilization),
            instances_warming=warming_instances,
            instances_ready=ready_instances,
        )

    def calculate_required_instances(
        self,
        instance_type: str,
        target_rps: float,
        headroom_factor: float | None = None,
        confidence_level: str = "p50",
    ) -> ScalingRequirement:
        """
        Calculate required instances for target load.

        Args:
            instance_type: Type of instances
            target_rps: Target requests per second
            headroom_factor: Headroom multiplier (default from config)
            confidence_level: Confidence level for sizing

        Returns:
            ScalingRequirement with instance counts
        """
        capacity = self.get_capacity(instance_type)
        headroom = headroom_factor or self.config.default_headroom

        # Adjust headroom based on confidence level
        confidence_multipliers = {
            "p50": 1.0,
            "p90": 1.2,
            "p99": 1.5,
        }
        confidence_mult = confidence_multipliers.get(confidence_level, 1.0)
        effective_headroom = headroom * confidence_mult

        # Calculate instances needed
        target_with_headroom = target_rps * effective_headroom
        instances_needed = int(np.ceil(target_with_headroom / capacity.base_rps))

        # Apply limits
        min_instances = max(self.config.min_instances, 1)
        max_instances = min(self.config.max_instances, instances_needed * 2)

        # Calculate range
        min_required = max(
            min_instances, int(np.ceil(target_rps / capacity.base_rps))
        )
        recommended = max(min_instances, instances_needed)
        max_required = min(
            max_instances,
            int(np.ceil(target_rps * self.config.peak_headroom / capacity.base_rps)),
        )

        return ScalingRequirement(
            target_rps=target_rps,
            min_instances=min_required,
            recommended_instances=recommended,
            max_instances=max(max_required, recommended),
            headroom_factor=effective_headroom,
            confidence_level=confidence_level,
        )

    def calculate_transition_time(
        self,
        instance_type: str,
        from_count: int,
        to_count: int,
    ) -> float:
        """
        Calculate time to transition between instance counts.

        Args:
            instance_type: Type of instances
            from_count: Current instance count
            to_count: Target instance count

        Returns:
            Estimated transition time in seconds
        """
        capacity = self.get_capacity(instance_type)

        if to_count <= from_count:
            # Scale down is fast (just terminate)
            return 30.0

        # Scale up requires warm-up
        new_instances = to_count - from_count

        # Instance launch time (API + provisioning)
        launch_time = 60.0  # ~1 minute for cloud instance

        # Warm-up time
        warm_up_time = capacity.warm_up_seconds

        # Pods/containers might be faster
        if new_instances <= 3:
            return launch_time + warm_up_time

        # Larger scale-ups may be staged
        return launch_time + warm_up_time + (new_instances // 5) * 15

    def update_from_benchmark(
        self,
        instance_type: str,
        measured_rps: float,
        load_level: float = 0.7,
    ) -> None:
        """
        Update capacity model with benchmark results.

        Args:
            instance_type: Type of instance benchmarked
            measured_rps: Measured RPS at given load
            load_level: Load level during benchmark (0.0 to 1.0)
        """
        # Store benchmark result
        if instance_type not in self._benchmark_results:
            self._benchmark_results[instance_type] = []
        self._benchmark_results[instance_type].append(measured_rps)

        # Update capacity estimate
        capacity = self.get_capacity(instance_type)

        # Average recent benchmarks
        recent_results = self._benchmark_results[instance_type][-10:]
        avg_rps = np.mean(recent_results)

        # Adjust base RPS based on benchmark
        adjusted_base = avg_rps / (1 - load_level * 0.2)  # Account for load impact

        # Update cache
        self._capacity_cache[instance_type] = InstanceCapacity(
            instance_type=instance_type,
            provider=self.provider,
            base_rps=adjusted_base,
            max_rps=adjusted_base * 1.5,
            warm_up_seconds=capacity.warm_up_seconds,
            degradation_threshold=capacity.degradation_threshold,
            degradation_rate=capacity.degradation_rate,
            last_benchmarked=datetime.now(timezone.utc),
        )

        logger.info(
            "Capacity updated from benchmark",
            instance_type=instance_type,
            measured_rps=measured_rps,
            adjusted_base_rps=adjusted_base,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get capacity model statistics."""
        return {
            "cached_capacities": len(self._capacity_cache),
            "benchmark_results": {
                k: len(v) for k, v in self._benchmark_results.items()
            },
            "provider": self.provider.value,
            "config": {
                "default_headroom": self.config.default_headroom,
                "peak_headroom": self.config.peak_headroom,
                "min_instances": self.config.min_instances,
                "max_instances": self.config.max_instances,
            },
        }
