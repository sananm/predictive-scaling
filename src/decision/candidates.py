"""
Candidate Generation for scaling decisions.

Responsibilities:
- Iterate over available instance types
- Calculate instances needed for different capacity targets
- Apply constraints (min/max instances)
- Generate variants with different spot percentages
- Filter infeasible candidates
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.decision.capacity_model import CapacityModel, CapacityConfig
from src.decision.cost_model import (
    CloudProvider,
    CostEstimate,
    CostModel,
    InfrastructureConfig,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateConfig:
    """Configuration for candidate generation."""

    # Instance constraints
    min_instances: int = 1
    max_instances: int = 100

    # Spot percentage variants to generate
    spot_percentages: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.5, 0.7]
    )

    # Capacity targets (relative to predicted load)
    capacity_multipliers: list[float] = field(
        default_factory=lambda: [1.0, 1.2, 1.5]  # 100%, 120%, 150% of predicted
    )

    # Instance type preferences (higher = more preferred)
    instance_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "m5.large": 1.0,
            "m5.xlarge": 0.9,
            "c5.large": 0.95,
            "c5.xlarge": 0.85,
            "t3.large": 0.8,
            "t3.xlarge": 0.75,
        }
    )

    # Filter settings
    max_candidates: int = 50
    min_headroom_factor: float = 1.1


@dataclass
class ScalingCandidate:
    """A candidate scaling configuration."""

    config: InfrastructureConfig
    cost_estimate: CostEstimate
    capacity_rps: float
    headroom_factor: float
    feasibility_score: float  # 0.0 to 1.0
    preference_score: float  # Based on instance type preferences
    is_feasible: bool
    infeasibility_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "cost_estimate": self.cost_estimate.to_dict(),
            "capacity_rps": self.capacity_rps,
            "headroom_factor": self.headroom_factor,
            "feasibility_score": self.feasibility_score,
            "preference_score": self.preference_score,
            "is_feasible": self.is_feasible,
            "infeasibility_reason": self.infeasibility_reason,
        }


@dataclass
class CandidateSet:
    """Set of scaling candidates for a decision."""

    candidates: list[ScalingCandidate]
    target_rps: float
    generated_at: str
    total_generated: int
    total_feasible: int

    @property
    def feasible_candidates(self) -> list[ScalingCandidate]:
        """Get only feasible candidates."""
        return [c for c in self.candidates if c.is_feasible]

    @property
    def cheapest_feasible(self) -> ScalingCandidate | None:
        """Get cheapest feasible candidate."""
        feasible = self.feasible_candidates
        if not feasible:
            return None
        return min(feasible, key=lambda c: c.cost_estimate.hourly_cost)

    @property
    def best_headroom(self) -> ScalingCandidate | None:
        """Get candidate with best headroom."""
        feasible = self.feasible_candidates
        if not feasible:
            return None
        return max(feasible, key=lambda c: c.headroom_factor)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "target_rps": self.target_rps,
            "generated_at": self.generated_at,
            "total_generated": self.total_generated,
            "total_feasible": self.total_feasible,
        }


class CandidateGenerator:
    """
    Generates and evaluates scaling candidates.

    Creates candidate configurations based on:
    - Available instance types
    - Capacity requirements
    - Cost optimization
    - Spot instance usage
    """

    def __init__(
        self,
        config: CandidateConfig | None = None,
        cost_model: CostModel | None = None,
        capacity_model: CapacityModel | None = None,
        provider: CloudProvider = CloudProvider.AWS,
    ) -> None:
        """
        Initialize candidate generator.

        Args:
            config: Candidate generation configuration
            cost_model: Cost model for pricing
            capacity_model: Capacity model for sizing
            provider: Cloud provider
        """
        self.config = config or CandidateConfig()
        self.cost_model = cost_model or CostModel(provider=provider)
        self.capacity_model = capacity_model or CapacityModel(
            config=CapacityConfig(
                min_instances=self.config.min_instances,
                max_instances=self.config.max_instances,
            ),
            provider=provider,
        )

    def generate(
        self,
        target_rps: float,
        current_instances: int = 0,
        current_instance_type: str | None = None,
        instance_types: list[str] | None = None,
    ) -> CandidateSet:
        """
        Generate scaling candidates for target load.

        Args:
            target_rps: Target requests per second
            current_instances: Current instance count
            current_instance_type: Current instance type
            instance_types: Instance types to consider (or all available)

        Returns:
            CandidateSet with ranked candidates
        """
        from datetime import datetime, timezone

        candidates = []

        # Get instance types to consider
        if instance_types is None:
            instance_types = self.cost_model.list_instance_types()

        # Prioritize current type if specified
        if current_instance_type and current_instance_type not in instance_types:
            instance_types = [current_instance_type] + instance_types

        # Generate candidates for each instance type and spot percentage
        for instance_type in instance_types:
            for spot_pct in self.config.spot_percentages:
                for capacity_mult in self.config.capacity_multipliers:
                    candidate = self._generate_candidate(
                        instance_type=instance_type,
                        spot_percentage=spot_pct,
                        target_rps=target_rps * capacity_mult,
                        original_target=target_rps,
                    )
                    if candidate:
                        candidates.append(candidate)

        # Sort by cost (cheapest first)
        candidates.sort(key=lambda c: c.cost_estimate.hourly_cost)

        # Limit to max candidates
        candidates = candidates[: self.config.max_candidates]

        # Count feasible
        total_feasible = sum(1 for c in candidates if c.is_feasible)

        return CandidateSet(
            candidates=candidates,
            target_rps=target_rps,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_generated=len(candidates),
            total_feasible=total_feasible,
        )

    def _generate_candidate(
        self,
        instance_type: str,
        spot_percentage: float,
        target_rps: float,
        original_target: float,
    ) -> ScalingCandidate | None:
        """Generate a single candidate configuration."""
        # Calculate required instances
        requirement = self.capacity_model.calculate_required_instances(
            instance_type=instance_type,
            target_rps=target_rps,
        )

        instance_count = requirement.recommended_instances

        # Apply constraints
        if instance_count < self.config.min_instances:
            instance_count = self.config.min_instances
        elif instance_count > self.config.max_instances:
            # Infeasible - too many instances needed
            return self._create_infeasible_candidate(
                instance_type,
                spot_percentage,
                instance_count,
                target_rps,
                f"Exceeds max instances ({self.config.max_instances})",
            )

        # Create configuration
        config = InfrastructureConfig(
            instance_type=instance_type,
            instance_count=instance_count,
            spot_percentage=spot_percentage,
            provider=self.cost_model.provider,
            region=self.cost_model.region,
        )

        # Calculate cost
        cost_estimate = self.cost_model.calculate_cost(config)

        # Calculate actual capacity
        capacity = self.capacity_model.get_capacity(instance_type)
        total_capacity = instance_count * capacity.base_rps

        # Calculate headroom
        headroom_factor = total_capacity / original_target if original_target > 0 else float("inf")

        # Check minimum headroom
        if headroom_factor < self.config.min_headroom_factor:
            return self._create_infeasible_candidate(
                instance_type,
                spot_percentage,
                instance_count,
                target_rps,
                f"Insufficient headroom ({headroom_factor:.2f}x < {self.config.min_headroom_factor}x)",
            )

        # Calculate scores
        feasibility_score = self._calculate_feasibility_score(
            headroom_factor, spot_percentage, instance_count
        )
        preference_score = self.config.instance_type_weights.get(instance_type, 0.5)

        return ScalingCandidate(
            config=config,
            cost_estimate=cost_estimate,
            capacity_rps=total_capacity,
            headroom_factor=headroom_factor,
            feasibility_score=feasibility_score,
            preference_score=preference_score,
            is_feasible=True,
        )

    def _create_infeasible_candidate(
        self,
        instance_type: str,
        spot_percentage: float,
        instance_count: int,
        target_rps: float,
        reason: str,
    ) -> ScalingCandidate:
        """Create an infeasible candidate for reference."""
        config = InfrastructureConfig(
            instance_type=instance_type,
            instance_count=min(instance_count, self.config.max_instances),
            spot_percentage=spot_percentage,
            provider=self.cost_model.provider,
            region=self.cost_model.region,
        )

        cost_estimate = self.cost_model.calculate_cost(config)
        capacity = self.capacity_model.get_capacity(instance_type)
        total_capacity = config.instance_count * capacity.base_rps

        return ScalingCandidate(
            config=config,
            cost_estimate=cost_estimate,
            capacity_rps=total_capacity,
            headroom_factor=total_capacity / target_rps if target_rps > 0 else 0,
            feasibility_score=0.0,
            preference_score=0.0,
            is_feasible=False,
            infeasibility_reason=reason,
        )

    def _calculate_feasibility_score(
        self,
        headroom_factor: float,
        spot_percentage: float,
        instance_count: int,
    ) -> float:
        """Calculate feasibility score (0-1)."""
        scores = []

        # Headroom score (1.2-1.5 is ideal)
        if 1.2 <= headroom_factor <= 1.5:
            headroom_score = 1.0
        elif headroom_factor < 1.2:
            headroom_score = headroom_factor / 1.2
        else:
            headroom_score = max(0.5, 1.0 - (headroom_factor - 1.5) * 0.2)
        scores.append(headroom_score)

        # Spot score (0-50% is preferred)
        if spot_percentage <= 0.5:
            spot_score = 1.0 - spot_percentage * 0.2
        else:
            spot_score = 0.9 - (spot_percentage - 0.5) * 0.4
        scores.append(spot_score)

        # Instance count score (fewer is better, within reason)
        if instance_count <= 5:
            count_score = 1.0
        elif instance_count <= 20:
            count_score = 1.0 - (instance_count - 5) * 0.02
        else:
            count_score = 0.7 - (instance_count - 20) * 0.01
        scores.append(max(0.3, count_score))

        return float(np.mean(scores))

    def filter_candidates(
        self,
        candidate_set: CandidateSet,
        max_cost_hourly: float | None = None,
        min_headroom: float | None = None,
        max_spot_percentage: float | None = None,
        preferred_instance_types: list[str] | None = None,
    ) -> CandidateSet:
        """
        Filter candidates based on constraints.

        Args:
            candidate_set: Original candidate set
            max_cost_hourly: Maximum hourly cost
            min_headroom: Minimum headroom factor
            max_spot_percentage: Maximum spot percentage
            preferred_instance_types: Only these instance types

        Returns:
            Filtered candidate set
        """
        filtered = []

        for candidate in candidate_set.candidates:
            # Skip if not feasible
            if not candidate.is_feasible:
                continue

            # Cost filter
            if max_cost_hourly and candidate.cost_estimate.hourly_cost > max_cost_hourly:
                continue

            # Headroom filter
            if min_headroom and candidate.headroom_factor < min_headroom:
                continue

            # Spot percentage filter
            if max_spot_percentage and candidate.config.spot_percentage > max_spot_percentage:
                continue

            # Instance type filter
            if preferred_instance_types and candidate.config.instance_type not in preferred_instance_types:
                continue

            filtered.append(candidate)

        from datetime import datetime, timezone

        return CandidateSet(
            candidates=filtered,
            target_rps=candidate_set.target_rps,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_generated=len(filtered),
            total_feasible=len(filtered),
        )

    def rank_candidates(
        self,
        candidate_set: CandidateSet,
        cost_weight: float = 0.4,
        headroom_weight: float = 0.3,
        preference_weight: float = 0.2,
        feasibility_weight: float = 0.1,
    ) -> list[ScalingCandidate]:
        """
        Rank candidates by weighted scoring.

        Args:
            candidate_set: Candidate set to rank
            cost_weight: Weight for cost (lower is better)
            headroom_weight: Weight for headroom
            preference_weight: Weight for instance type preference
            feasibility_weight: Weight for feasibility score

        Returns:
            Ranked list of feasible candidates
        """
        feasible = candidate_set.feasible_candidates
        if not feasible:
            return []

        # Normalize cost (inverse, lower is better)
        max_cost = max(c.cost_estimate.hourly_cost for c in feasible)
        min_cost = min(c.cost_estimate.hourly_cost for c in feasible)
        cost_range = max_cost - min_cost if max_cost != min_cost else 1.0

        # Score each candidate
        scored = []
        for candidate in feasible:
            # Cost score (inverse normalized)
            cost_score = 1.0 - (
                (candidate.cost_estimate.hourly_cost - min_cost) / cost_range
            )

            # Headroom score (capped at 2.0)
            headroom_score = min(1.0, candidate.headroom_factor / 2.0)

            # Combined score
            total_score = (
                cost_score * cost_weight
                + headroom_score * headroom_weight
                + candidate.preference_score * preference_weight
                + candidate.feasibility_score * feasibility_weight
            )

            scored.append((candidate, total_score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        return [candidate for candidate, _ in scored]

    def get_recommendation(
        self,
        target_rps: float,
        current_config: InfrastructureConfig | None = None,
        optimization_goal: str = "balanced",  # "cost", "performance", "balanced"
    ) -> ScalingCandidate | None:
        """
        Get a single recommended candidate.

        Args:
            target_rps: Target RPS
            current_config: Current configuration (for comparison)
            optimization_goal: What to optimize for

        Returns:
            Recommended candidate or None
        """
        # Generate candidates
        candidate_set = self.generate(
            target_rps=target_rps,
            current_instances=current_config.instance_count if current_config else 0,
            current_instance_type=current_config.instance_type if current_config else None,
        )

        if not candidate_set.feasible_candidates:
            logger.warning("No feasible candidates found", target_rps=target_rps)
            return None

        # Set weights based on optimization goal
        if optimization_goal == "cost":
            weights = {"cost_weight": 0.7, "headroom_weight": 0.2, "preference_weight": 0.05, "feasibility_weight": 0.05}
        elif optimization_goal == "performance":
            weights = {"cost_weight": 0.2, "headroom_weight": 0.5, "preference_weight": 0.15, "feasibility_weight": 0.15}
        else:  # balanced
            weights = {"cost_weight": 0.4, "headroom_weight": 0.3, "preference_weight": 0.2, "feasibility_weight": 0.1}

        # Rank and return top
        ranked = self.rank_candidates(candidate_set, **weights)
        return ranked[0] if ranked else None
