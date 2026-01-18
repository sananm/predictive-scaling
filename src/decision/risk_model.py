"""
Risk Model for scaling decisions.

Responsibilities:
- Estimate spot instance interruption probability
- Calculate risk of under-provisioning based on prediction uncertainty
- Assess stability risk from rapid scaling
- Produce overall risk score (0-1) for scaling decisions
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RiskCategory(str, Enum):
    """Risk categories for scaling decisions."""

    SPOT_INTERRUPTION = "spot_interruption"
    UNDER_PROVISIONING = "under_provisioning"
    OVER_PROVISIONING = "over_provisioning"
    SCALING_STABILITY = "scaling_stability"
    PREDICTION_UNCERTAINTY = "prediction_uncertainty"
    TRANSITION_RISK = "transition_risk"


class RiskLevel(str, Enum):
    """Risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Convert score (0-1) to risk level."""
        if score < 0.25:
            return cls.LOW
        elif score < 0.5:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class RiskFactor:
    """Individual risk factor."""

    category: RiskCategory
    score: float  # 0.0 to 1.0
    weight: float  # Weight in overall score
    description: str
    mitigation: str | None = None

    @property
    def level(self) -> RiskLevel:
        """Get risk level for this factor."""
        return RiskLevel.from_score(self.score)

    @property
    def weighted_score(self) -> float:
        """Get weighted score."""
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "level": self.level.value,
            "description": self.description,
            "mitigation": self.mitigation,
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment for a scaling decision."""

    overall_score: float  # 0.0 to 1.0
    overall_level: RiskLevel
    factors: list[RiskFactor]
    is_acceptable: bool
    requires_approval: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "factors": [f.to_dict() for f in self.factors],
            "is_acceptable": self.is_acceptable,
            "requires_approval": self.requires_approval,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
        }


@dataclass
class RiskConfig:
    """Configuration for risk model."""

    # Risk thresholds
    acceptable_risk_threshold: float = 0.5
    approval_required_threshold: float = 0.7

    # Category weights (must sum to 1.0)
    weights: dict[RiskCategory, float] = field(
        default_factory=lambda: {
            RiskCategory.SPOT_INTERRUPTION: 0.15,
            RiskCategory.UNDER_PROVISIONING: 0.30,
            RiskCategory.OVER_PROVISIONING: 0.10,
            RiskCategory.SCALING_STABILITY: 0.20,
            RiskCategory.PREDICTION_UNCERTAINTY: 0.15,
            RiskCategory.TRANSITION_RISK: 0.10,
        }
    )

    # Spot interruption thresholds
    high_spot_percentage: float = 0.7
    critical_spot_percentage: float = 0.9

    # Scaling stability settings
    min_time_between_scales: timedelta = timedelta(minutes=5)
    max_scale_factor_per_action: float = 3.0  # Max 3x scale up/down

    # Under-provisioning settings
    critical_utilization: float = 0.95
    high_utilization: float = 0.85


class RiskModel:
    """
    Risk assessment model for scaling decisions.

    Evaluates multiple risk factors and produces an overall
    risk score for scaling decisions.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """
        Initialize risk model.

        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()
        self._recent_scaling_actions: list[datetime] = []

    def assess(
        self,
        current_instances: int,
        target_instances: int,
        spot_percentage: float,
        spot_interruption_rate: float,
        prediction_uncertainty: float,
        current_utilization: float,
        target_utilization: float,
        transition_time_seconds: float,
        time_until_demand_spike: float | None = None,
    ) -> RiskAssessment:
        """
        Assess risk for a scaling decision.

        Args:
            current_instances: Current instance count
            target_instances: Target instance count
            spot_percentage: Percentage of spot instances (0.0 to 1.0)
            spot_interruption_rate: Historical spot interruption rate
            prediction_uncertainty: Prediction uncertainty (0.0 to 1.0)
            current_utilization: Current capacity utilization
            target_utilization: Expected utilization after scaling
            transition_time_seconds: Time to complete scaling
            time_until_demand_spike: Time until predicted demand spike (seconds)

        Returns:
            RiskAssessment with overall score and individual factors
        """
        factors = []

        # 1. Spot interruption risk
        spot_risk = self._assess_spot_risk(
            spot_percentage, spot_interruption_rate, target_instances
        )
        factors.append(spot_risk)

        # 2. Under-provisioning risk
        under_risk = self._assess_under_provisioning_risk(
            target_utilization, prediction_uncertainty
        )
        factors.append(under_risk)

        # 3. Over-provisioning risk
        over_risk = self._assess_over_provisioning_risk(
            current_instances, target_instances, target_utilization
        )
        factors.append(over_risk)

        # 4. Scaling stability risk
        stability_risk = self._assess_stability_risk(
            current_instances, target_instances
        )
        factors.append(stability_risk)

        # 5. Prediction uncertainty risk
        uncertainty_risk = self._assess_uncertainty_risk(prediction_uncertainty)
        factors.append(uncertainty_risk)

        # 6. Transition risk
        transition_risk = self._assess_transition_risk(
            transition_time_seconds, time_until_demand_spike
        )
        factors.append(transition_risk)

        # Calculate overall score
        total_weight = sum(f.weight for f in factors)
        overall_score = sum(f.weighted_score for f in factors) / total_weight

        # Determine acceptability
        is_acceptable = overall_score < self.config.acceptable_risk_threshold
        requires_approval = overall_score >= self.config.approval_required_threshold

        # Generate recommendations
        recommendations = self._generate_recommendations(factors)

        return RiskAssessment(
            overall_score=overall_score,
            overall_level=RiskLevel.from_score(overall_score),
            factors=factors,
            is_acceptable=is_acceptable,
            requires_approval=requires_approval,
            recommendations=recommendations,
        )

    def _assess_spot_risk(
        self,
        spot_percentage: float,
        interruption_rate: float,
        instance_count: int,
    ) -> RiskFactor:
        """Assess spot instance interruption risk."""
        if spot_percentage == 0:
            return RiskFactor(
                category=RiskCategory.SPOT_INTERRUPTION,
                score=0.0,
                weight=self.config.weights[RiskCategory.SPOT_INTERRUPTION],
                description="No spot instances - no interruption risk",
            )

        # Calculate expected interruptions
        spot_instances = int(instance_count * spot_percentage)
        expected_interruptions = spot_instances * interruption_rate

        # Score based on spot percentage and interruption rate
        base_score = spot_percentage * 0.5 + interruption_rate * 0.5

        # Increase score if high spot percentage
        if spot_percentage > self.config.critical_spot_percentage:
            base_score = min(1.0, base_score * 1.5)
        elif spot_percentage > self.config.high_spot_percentage:
            base_score = min(1.0, base_score * 1.2)

        description = (
            f"{spot_percentage:.0%} spot instances with {interruption_rate:.1%} "
            f"interruption rate (expected {expected_interruptions:.1f} interruptions)"
        )

        mitigation = None
        if base_score > 0.5:
            mitigation = "Consider reducing spot percentage or using multiple AZs"

        return RiskFactor(
            category=RiskCategory.SPOT_INTERRUPTION,
            score=base_score,
            weight=self.config.weights[RiskCategory.SPOT_INTERRUPTION],
            description=description,
            mitigation=mitigation,
        )

    def _assess_under_provisioning_risk(
        self,
        target_utilization: float,
        prediction_uncertainty: float,
    ) -> RiskFactor:
        """Assess risk of under-provisioning."""
        # Higher utilization = higher risk
        utilization_risk = 0.0
        if target_utilization > self.config.critical_utilization:
            utilization_risk = 1.0
        elif target_utilization > self.config.high_utilization:
            utilization_risk = (
                target_utilization - self.config.high_utilization
            ) / (self.config.critical_utilization - self.config.high_utilization)
        else:
            utilization_risk = target_utilization / self.config.high_utilization * 0.5

        # Adjust for prediction uncertainty
        # High uncertainty means we might underestimate demand
        uncertainty_adjustment = prediction_uncertainty * 0.3
        score = min(1.0, utilization_risk + uncertainty_adjustment)

        description = (
            f"Target utilization {target_utilization:.0%} with "
            f"{prediction_uncertainty:.0%} prediction uncertainty"
        )

        mitigation = None
        if score > 0.5:
            mitigation = "Consider adding more headroom or using p90 predictions"

        return RiskFactor(
            category=RiskCategory.UNDER_PROVISIONING,
            score=score,
            weight=self.config.weights[RiskCategory.UNDER_PROVISIONING],
            description=description,
            mitigation=mitigation,
        )

    def _assess_over_provisioning_risk(
        self,
        current_instances: int,
        target_instances: int,
        target_utilization: float,
    ) -> RiskFactor:
        """Assess risk of over-provisioning (cost waste)."""
        # Low utilization = potential over-provisioning
        if target_utilization > 0.5:
            score = 0.0
            description = f"Target utilization {target_utilization:.0%} is efficient"
        elif target_utilization > 0.3:
            score = (0.5 - target_utilization) / 0.2 * 0.5
            description = f"Target utilization {target_utilization:.0%} may be inefficient"
        else:
            score = 0.5 + (0.3 - target_utilization) / 0.3 * 0.5
            description = f"Target utilization {target_utilization:.0%} indicates over-provisioning"

        # Penalize large scale-ups
        if target_instances > current_instances * 2:
            score = min(1.0, score + 0.2)
            description += f" (scaling from {current_instances} to {target_instances})"

        mitigation = None
        if score > 0.5:
            mitigation = "Consider scaling down or using smaller instances"

        return RiskFactor(
            category=RiskCategory.OVER_PROVISIONING,
            score=score,
            weight=self.config.weights[RiskCategory.OVER_PROVISIONING],
            description=description,
            mitigation=mitigation,
        )

    def _assess_stability_risk(
        self,
        current_instances: int,
        target_instances: int,
    ) -> RiskFactor:
        """Assess risk from scaling stability perspective."""
        # Check recent scaling actions
        now = datetime.now(UTC)
        recent_actions = [
            a for a in self._recent_scaling_actions
            if now - a < self.config.min_time_between_scales
        ]

        # Score based on recent scaling frequency
        frequency_risk = min(1.0, len(recent_actions) * 0.3)

        # Score based on scale factor
        if current_instances == 0:
            scale_factor = target_instances
        else:
            scale_factor = max(
                target_instances / current_instances,
                current_instances / target_instances,
            )

        factor_risk = 0.0
        if scale_factor > self.config.max_scale_factor_per_action:
            factor_risk = 0.8
        elif scale_factor > 2:
            factor_risk = (scale_factor - 1) / (
                self.config.max_scale_factor_per_action - 1
            ) * 0.6
        else:
            factor_risk = (scale_factor - 1) * 0.3

        score = min(1.0, frequency_risk + factor_risk)

        description = (
            f"Scaling {scale_factor:.1f}x "
            f"({current_instances} → {target_instances}), "
            f"{len(recent_actions)} recent actions"
        )

        mitigation = None
        if score > 0.5:
            mitigation = "Consider gradual scaling or increasing cooldown period"

        return RiskFactor(
            category=RiskCategory.SCALING_STABILITY,
            score=score,
            weight=self.config.weights[RiskCategory.SCALING_STABILITY],
            description=description,
            mitigation=mitigation,
        )

    def _assess_uncertainty_risk(
        self,
        prediction_uncertainty: float,
    ) -> RiskFactor:
        """Assess risk from prediction uncertainty."""
        # Direct mapping of uncertainty to risk
        score = prediction_uncertainty

        if score < 0.3:
            description = f"Low prediction uncertainty ({score:.0%})"
        elif score < 0.6:
            description = f"Moderate prediction uncertainty ({score:.0%})"
        else:
            description = f"High prediction uncertainty ({score:.0%})"

        mitigation = None
        if score > 0.5:
            mitigation = "Consider using more conservative predictions or manual review"

        return RiskFactor(
            category=RiskCategory.PREDICTION_UNCERTAINTY,
            score=score,
            weight=self.config.weights[RiskCategory.PREDICTION_UNCERTAINTY],
            description=description,
            mitigation=mitigation,
        )

    def _assess_transition_risk(
        self,
        transition_time_seconds: float,
        time_until_spike: float | None,
    ) -> RiskFactor:
        """Assess risk of not completing transition in time."""
        if time_until_spike is None:
            return RiskFactor(
                category=RiskCategory.TRANSITION_RISK,
                score=0.2,
                weight=self.config.weights[RiskCategory.TRANSITION_RISK],
                description="No predicted demand spike",
            )

        # Calculate if we have enough time
        buffer_time = time_until_spike - transition_time_seconds

        if buffer_time > 300:  # > 5 minutes buffer
            score = 0.1
            description = f"Ample time ({buffer_time/60:.0f}min buffer)"
        elif buffer_time > 60:  # 1-5 minutes buffer
            score = 0.3
            description = f"Adequate time ({buffer_time/60:.1f}min buffer)"
        elif buffer_time > 0:
            score = 0.6
            description = f"Tight timing ({buffer_time:.0f}s buffer)"
        else:
            score = 0.9
            description = f"May not complete in time ({-buffer_time/60:.1f}min late)"

        mitigation = None
        if score > 0.5:
            mitigation = "Consider triggering scaling earlier or using faster scaling"

        return RiskFactor(
            category=RiskCategory.TRANSITION_RISK,
            score=score,
            weight=self.config.weights[RiskCategory.TRANSITION_RISK],
            description=description,
            mitigation=mitigation,
        )

    def _generate_recommendations(
        self,
        factors: list[RiskFactor],
    ) -> list[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []

        # Sort factors by weighted score
        sorted_factors = sorted(
            factors, key=lambda f: f.weighted_score, reverse=True
        )

        # Add mitigations for high-risk factors
        for factor in sorted_factors[:3]:
            if factor.score > 0.5 and factor.mitigation:
                recommendations.append(factor.mitigation)

        return recommendations

    def record_scaling_action(self) -> None:
        """Record a scaling action for stability tracking."""
        now = datetime.now(UTC)
        self._recent_scaling_actions.append(now)

        # Clean up old actions
        cutoff = now - timedelta(hours=1)
        self._recent_scaling_actions = [
            a for a in self._recent_scaling_actions if a > cutoff
        ]

    def calculate_spot_interruption_probability(
        self,
        spot_instances: int,
        interruption_rate: float,
        time_window_hours: float = 1.0,
    ) -> float:
        """
        Calculate probability of at least one spot interruption.

        Args:
            spot_instances: Number of spot instances
            interruption_rate: Per-instance interruption rate (per hour)
            time_window_hours: Time window to consider

        Returns:
            Probability of at least one interruption
        """
        if spot_instances == 0 or interruption_rate == 0:
            return 0.0

        # Probability of NO interruption for a single instance
        p_no_interrupt = (1 - interruption_rate) ** time_window_hours

        # Probability of NO interruption for all instances
        p_all_survive = p_no_interrupt ** spot_instances

        # Probability of at least one interruption
        return 1 - p_all_survive

    def get_stats(self) -> dict[str, Any]:
        """Get risk model statistics."""
        return {
            "recent_scaling_actions": len(self._recent_scaling_actions),
            "config": {
                "acceptable_risk_threshold": self.config.acceptable_risk_threshold,
                "approval_required_threshold": self.config.approval_required_threshold,
                "weights": {k.value: v for k, v in self.config.weights.items()},
            },
        }
