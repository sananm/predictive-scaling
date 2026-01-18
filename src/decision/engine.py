"""
Decision Engine for scaling decisions.

Responsibilities:
- Take current infrastructure state and predictions as input
- Calculate required capacity at different confidence levels
- Generate candidate scaling configurations
- Score candidates using multi-objective optimization
- Select best candidate and determine strategy
- Generate human-readable reasoning
- Create rollback plans and verification criteria
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from src.decision.candidates import (
    CandidateGenerator,
    ScalingCandidate,
)
from src.decision.capacity_model import CapacityModel
from src.decision.cost_model import CloudProvider, CostModel, InfrastructureConfig
from src.decision.risk_model import RiskAssessment, RiskModel
from src.decision.strategies import (
    ScalingPlan,
    StrategySelector,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DecisionStatus(str, Enum):
    """Status of a scaling decision."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class InfrastructureState:
    """Current infrastructure state."""

    instance_type: str
    instance_count: int
    spot_percentage: float
    current_rps: float
    current_utilization: float
    healthy_instances: int
    warming_instances: int = 0

    def to_config(self) -> InfrastructureConfig:
        """Convert to InfrastructureConfig."""
        return InfrastructureConfig(
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            spot_percentage=self.spot_percentage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "spot_percentage": self.spot_percentage,
            "current_rps": self.current_rps,
            "current_utilization": self.current_utilization,
            "healthy_instances": self.healthy_instances,
            "warming_instances": self.warming_instances,
        }


@dataclass
class PredictionInput:
    """Prediction input for decision engine."""

    horizon_minutes: int
    p10: float  # 10th percentile (optimistic)
    p50: float  # 50th percentile (median)
    p90: float  # 90th percentile (pessimistic)
    confidence: float  # Prediction confidence (0-1)
    timestamp: datetime

    @property
    def uncertainty(self) -> float:
        """Calculate uncertainty as relative interval width."""
        if self.p50 == 0:
            return 1.0
        return (self.p90 - self.p10) / self.p50

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizon_minutes": self.horizon_minutes,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CandidateScore:
    """Scores for a scaling candidate."""

    cost_score: float  # Lower cost = higher score
    performance_score: float  # Adequate headroom = higher score
    stability_score: float  # Fewer changes = higher score
    risk_score: float  # Lower risk = higher score
    transition_score: float  # Can complete in time = higher score
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cost_score": self.cost_score,
            "performance_score": self.performance_score,
            "stability_score": self.stability_score,
            "risk_score": self.risk_score,
            "transition_score": self.transition_score,
            "total_score": self.total_score,
        }


@dataclass
class VerificationCriteria:
    """Criteria for verifying scaling success."""

    target_instance_count: int
    max_latency_ms: float
    min_healthy_percentage: float
    max_error_rate: float
    timeout_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_instance_count": self.target_instance_count,
            "max_latency_ms": self.max_latency_ms,
            "min_healthy_percentage": self.min_healthy_percentage,
            "max_error_rate": self.max_error_rate,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ScalingDecision:
    """Complete scaling decision."""

    id: str
    status: DecisionStatus
    current_state: InfrastructureState
    target_config: InfrastructureConfig
    candidate: ScalingCandidate
    score: CandidateScore
    risk_assessment: RiskAssessment
    scaling_plan: ScalingPlan
    verification_criteria: VerificationCriteria
    reasoning: str
    rollback_config: InfrastructureConfig
    created_at: datetime
    predictions: list[PredictionInput] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "current_state": self.current_state.to_dict(),
            "target_config": self.target_config.to_dict(),
            "candidate": self.candidate.to_dict(),
            "score": self.score.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "scaling_plan": self.scaling_plan.to_dict(),
            "verification_criteria": self.verification_criteria.to_dict(),
            "reasoning": self.reasoning,
            "rollback_config": self.rollback_config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "predictions": [p.to_dict() for p in self.predictions],
            "metadata": self.metadata,
        }


@dataclass
class DecisionEngineConfig:
    """Configuration for decision engine."""

    # Scoring weights
    cost_weight: float = 0.25
    performance_weight: float = 0.30
    stability_weight: float = 0.15
    risk_weight: float = 0.20
    transition_weight: float = 0.10

    # Thresholds
    min_utilization_to_scale_up: float = 0.70
    max_utilization_to_scale_down: float = 0.40
    min_headroom_factor: float = 1.2
    max_headroom_factor: float = 2.0

    # Verification settings
    default_latency_threshold_ms: float = 500.0
    default_error_rate_threshold: float = 0.01
    default_healthy_percentage: float = 0.95
    verification_timeout_seconds: float = 300.0

    # Cooldown
    cooldown_seconds: float = 300.0

    # Confidence level for sizing
    sizing_confidence: str = "p90"  # Use p90 predictions for sizing


class DecisionEngine:
    """
    Main decision engine for scaling decisions.

    Orchestrates cost, capacity, and risk models to make
    optimal scaling decisions with full reasoning and rollback support.
    """

    def __init__(
        self,
        config: DecisionEngineConfig | None = None,
        cost_model: CostModel | None = None,
        capacity_model: CapacityModel | None = None,
        risk_model: RiskModel | None = None,
        candidate_generator: CandidateGenerator | None = None,
        strategy_selector: StrategySelector | None = None,
        provider: CloudProvider = CloudProvider.AWS,
    ) -> None:
        """
        Initialize decision engine.

        Args:
            config: Engine configuration
            cost_model: Cost model instance
            capacity_model: Capacity model instance
            risk_model: Risk model instance
            candidate_generator: Candidate generator instance
            strategy_selector: Strategy selector instance
            provider: Cloud provider
        """
        self.config = config or DecisionEngineConfig()
        self.provider = provider

        # Initialize models
        self.cost_model = cost_model or CostModel(provider=provider)
        self.capacity_model = capacity_model or CapacityModel(provider=provider)
        self.risk_model = risk_model or RiskModel()
        self.candidate_generator = candidate_generator or CandidateGenerator(
            cost_model=self.cost_model,
            capacity_model=self.capacity_model,
            provider=provider,
        )
        self.strategy_selector = strategy_selector or StrategySelector()

        # Track recent decisions
        self._recent_decisions: list[ScalingDecision] = []
        self._last_decision_time: datetime | None = None

    def decide(
        self,
        current_state: InfrastructureState,
        predictions: list[PredictionInput],
        force_evaluation: bool = False,
        allowed_instance_types: list[str] | None = None,
    ) -> ScalingDecision | None:
        """
        Make a scaling decision based on current state and predictions.

        Args:
            current_state: Current infrastructure state
            predictions: Predictions for different horizons
            force_evaluation: Skip cooldown check
            allowed_instance_types: If set, only consider these instance types
                                   (for horizontal scaling only, pass [current_type])

        Returns:
            ScalingDecision if action needed, None if maintaining current state
        """
        now = datetime.now(UTC)

        # Check cooldown
        if not force_evaluation and self._in_cooldown():
            logger.debug("In cooldown period, skipping evaluation")
            return None

        # Determine target RPS based on predictions
        target_rps = self._calculate_target_rps(predictions)

        # Check if action is needed
        action_needed, action_reason = self._check_action_needed(
            current_state, target_rps, predictions
        )

        if not action_needed:
            logger.debug("No scaling action needed", reason=action_reason)
            return None

        # Generate candidates
        candidate_set = self.candidate_generator.generate(
            target_rps=target_rps,
            current_instances=current_state.instance_count,
            current_instance_type=current_state.instance_type,
            instance_types=allowed_instance_types,
        )

        if not candidate_set.feasible_candidates:
            logger.warning("No feasible candidates found")
            return None

        # Score candidates
        scored_candidates = self._score_candidates(
            candidate_set.feasible_candidates,
            current_state,
            predictions,
        )

        if not scored_candidates:
            return None

        # Select best candidate
        best_candidate, best_score = scored_candidates[0]

        # Assess risk
        risk_assessment = self._assess_risk(
            current_state, best_candidate, predictions
        )

        # Determine strategy
        time_until_spike = self._get_time_until_spike(predictions)
        strategy_type = self.strategy_selector.select_strategy(
            current_instances=current_state.instance_count,
            target_instances=best_candidate.config.instance_count,
            current_utilization=current_state.current_utilization,
            time_until_spike=time_until_spike,
            is_emergency=current_state.current_utilization > 0.95,
        )

        # Create scaling plan
        scaling_plan = self.strategy_selector.create_plan(
            current_instances=current_state.instance_count,
            target_instances=best_candidate.config.instance_count,
            strategy_type=strategy_type,
            current_utilization=current_state.current_utilization,
            time_until_spike=time_until_spike,
            reason=action_reason,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            current_state,
            best_candidate,
            best_score,
            predictions,
            action_reason,
        )

        # Create verification criteria
        verification = VerificationCriteria(
            target_instance_count=best_candidate.config.instance_count,
            max_latency_ms=self.config.default_latency_threshold_ms,
            min_healthy_percentage=self.config.default_healthy_percentage,
            max_error_rate=self.config.default_error_rate_threshold,
            timeout_seconds=self.config.verification_timeout_seconds,
        )

        # Create decision
        decision = ScalingDecision(
            id=str(uuid4()),
            status=DecisionStatus.PENDING,
            current_state=current_state,
            target_config=best_candidate.config,
            candidate=best_candidate,
            score=best_score,
            risk_assessment=risk_assessment,
            scaling_plan=scaling_plan,
            verification_criteria=verification,
            reasoning=reasoning,
            rollback_config=current_state.to_config(),
            created_at=now,
            predictions=predictions,
        )

        # Track decision
        self._recent_decisions.append(decision)
        self._last_decision_time = now

        logger.info(
            "Scaling decision created",
            decision_id=decision.id,
            strategy=strategy_type.value,
            current_instances=current_state.instance_count,
            target_instances=best_candidate.config.instance_count,
            risk_level=risk_assessment.overall_level.value,
        )

        return decision

    def _calculate_target_rps(
        self, predictions: list[PredictionInput]
    ) -> float:
        """Calculate target RPS from predictions."""
        if not predictions:
            return 0.0

        # Use the configured confidence level
        if self.config.sizing_confidence == "p90":
            # Use p90 for conservative sizing
            return max(p.p90 for p in predictions)
        elif self.config.sizing_confidence == "p50":
            return max(p.p50 for p in predictions)
        else:
            return max(p.p10 for p in predictions)

    def _check_action_needed(
        self,
        current_state: InfrastructureState,
        target_rps: float,
        predictions: list[PredictionInput],
    ) -> tuple[bool, str]:
        """Check if scaling action is needed."""
        # Get current capacity
        capacity = self.capacity_model.get_capacity(current_state.instance_type)
        current_capacity = current_state.instance_count * capacity.base_rps

        # Calculate utilization with target load
        target_utilization = target_rps / current_capacity if current_capacity > 0 else 1.0

        # Check if scale up needed
        if target_utilization > self.config.min_utilization_to_scale_up:
            return True, f"Predicted utilization {target_utilization:.0%} exceeds threshold"

        # Check if scale down possible
        if current_state.current_utilization < self.config.max_utilization_to_scale_down:
            return True, f"Current utilization {current_state.current_utilization:.0%} below threshold"

        # Check headroom
        headroom = current_capacity / target_rps if target_rps > 0 else float("inf")
        if headroom < self.config.min_headroom_factor:
            return True, f"Insufficient headroom ({headroom:.1f}x)"
        if headroom > self.config.max_headroom_factor:
            return True, f"Excessive headroom ({headroom:.1f}x) - cost optimization"

        return False, "Within acceptable utilization range"

    def _score_candidates(
        self,
        candidates: list[ScalingCandidate],
        current_state: InfrastructureState,
        predictions: list[PredictionInput],
    ) -> list[tuple[ScalingCandidate, CandidateScore]]:
        """Score and rank candidates."""
        scored = []

        # Get time until spike for transition scoring
        time_until_spike = self._get_time_until_spike(predictions)

        for candidate in candidates:
            score = self._calculate_candidate_score(
                candidate, current_state, predictions, time_until_spike
            )
            scored.append((candidate, score))

        # Sort by total score (descending)
        scored.sort(key=lambda x: x[1].total_score, reverse=True)

        return scored

    def _calculate_candidate_score(
        self,
        candidate: ScalingCandidate,
        current_state: InfrastructureState,
        predictions: list[PredictionInput],
        time_until_spike: float | None,
    ) -> CandidateScore:
        """Calculate multi-objective score for a candidate."""
        # Cost score (lower cost = higher score)
        current_cost = self.cost_model.calculate_cost(current_state.to_config())
        cost_ratio = candidate.cost_estimate.hourly_cost / current_cost.hourly_cost
        cost_score = 1.0 / (1.0 + cost_ratio)  # Normalize to 0-1

        # Performance score (headroom based)
        if candidate.headroom_factor < self.config.min_headroom_factor:
            performance_score = candidate.headroom_factor / self.config.min_headroom_factor * 0.5
        elif candidate.headroom_factor <= self.config.max_headroom_factor:
            # Ideal range
            performance_score = 1.0
        else:
            # Excessive headroom penalized
            performance_score = max(
                0.5,
                1.0 - (candidate.headroom_factor - self.config.max_headroom_factor) * 0.2,
            )

        # Stability score (smaller changes = higher score)
        scale_factor = max(
            candidate.config.instance_count / current_state.instance_count,
            current_state.instance_count / candidate.config.instance_count,
        ) if current_state.instance_count > 0 else 1.0
        stability_score = 1.0 / (1.0 + (scale_factor - 1) * 2)

        # Risk score (lower risk = higher score)
        avg_uncertainty = np.mean([p.uncertainty for p in predictions]) if predictions else 0.5
        spot_risk = candidate.config.spot_percentage * 0.5
        risk_score = 1.0 - (avg_uncertainty * 0.5 + spot_risk * 0.5)

        # Transition score (can complete in time = higher score)
        if time_until_spike is None:
            transition_score = 0.8
        else:
            transition_time = self.capacity_model.calculate_transition_time(
                candidate.config.instance_type,
                current_state.instance_count,
                candidate.config.instance_count,
            )
            buffer = time_until_spike - transition_time
            if buffer > 300:  # > 5 minutes
                transition_score = 1.0
            elif buffer > 0:
                transition_score = 0.5 + (buffer / 300) * 0.5
            else:
                transition_score = max(0.1, 0.5 + buffer / 600)

        # Calculate weighted total
        total_score = (
            cost_score * self.config.cost_weight
            + performance_score * self.config.performance_weight
            + stability_score * self.config.stability_weight
            + risk_score * self.config.risk_weight
            + transition_score * self.config.transition_weight
        )

        return CandidateScore(
            cost_score=cost_score,
            performance_score=performance_score,
            stability_score=stability_score,
            risk_score=risk_score,
            transition_score=transition_score,
            total_score=total_score,
        )

    def _assess_risk(
        self,
        current_state: InfrastructureState,
        candidate: ScalingCandidate,
        predictions: list[PredictionInput],
    ) -> RiskAssessment:
        """Assess risk for a candidate."""
        avg_uncertainty = np.mean([p.uncertainty for p in predictions]) if predictions else 0.5

        # Get spot interruption rate
        pricing = self.cost_model.get_pricing(candidate.config.instance_type)
        interruption_rate = pricing.spot_interruption_rate if pricing else 0.05

        # Calculate target utilization
        target_rps = self._calculate_target_rps(predictions)
        capacity = self.capacity_model.get_capacity(candidate.config.instance_type)
        target_capacity = candidate.config.instance_count * capacity.base_rps
        target_utilization = target_rps / target_capacity if target_capacity > 0 else 1.0

        # Calculate transition time
        transition_time = self.capacity_model.calculate_transition_time(
            candidate.config.instance_type,
            current_state.instance_count,
            candidate.config.instance_count,
        )

        time_until_spike = self._get_time_until_spike(predictions)

        return self.risk_model.assess(
            current_instances=current_state.instance_count,
            target_instances=candidate.config.instance_count,
            spot_percentage=candidate.config.spot_percentage,
            spot_interruption_rate=interruption_rate,
            prediction_uncertainty=avg_uncertainty,
            current_utilization=current_state.current_utilization,
            target_utilization=target_utilization,
            transition_time_seconds=transition_time,
            time_until_demand_spike=time_until_spike,
        )

    def _get_time_until_spike(
        self, predictions: list[PredictionInput]
    ) -> float | None:
        """Get time until predicted demand spike."""
        if not predictions:
            return None

        # Find the prediction with highest load increase
        datetime.now(UTC)
        for pred in sorted(predictions, key=lambda p: p.horizon_minutes):
            # Check if this prediction shows significant increase
            if pred.p90 > pred.p50 * 1.3:  # 30% increase
                return pred.horizon_minutes * 60  # Convert to seconds

        return None

    def _generate_reasoning(
        self,
        current_state: InfrastructureState,
        candidate: ScalingCandidate,
        score: CandidateScore,
        predictions: list[PredictionInput],
        action_reason: str,
    ) -> str:
        """Generate human-readable reasoning."""
        lines = []

        # Summary
        direction = "up" if candidate.config.instance_count > current_state.instance_count else "down"
        lines.append(f"Scaling {direction} from {current_state.instance_count} to {candidate.config.instance_count} instances.")
        lines.append(f"Reason: {action_reason}")
        lines.append("")

        # Current state
        lines.append("Current State:")
        lines.append(f"  - Instance type: {current_state.instance_type}")
        lines.append(f"  - Current load: {current_state.current_rps:.0f} RPS")
        lines.append(f"  - Utilization: {current_state.current_utilization:.0%}")
        lines.append("")

        # Predictions
        if predictions:
            lines.append("Predictions:")
            for pred in predictions[:3]:
                lines.append(
                    f"  - {pred.horizon_minutes}min: "
                    f"p10={pred.p10:.0f}, p50={pred.p50:.0f}, p90={pred.p90:.0f} RPS"
                )
            lines.append("")

        # Target configuration
        lines.append("Target Configuration:")
        lines.append(f"  - Instance type: {candidate.config.instance_type}")
        lines.append(f"  - Instance count: {candidate.config.instance_count}")
        lines.append(f"  - Spot percentage: {candidate.config.spot_percentage:.0%}")
        lines.append(f"  - Estimated cost: ${candidate.cost_estimate.hourly_cost:.2f}/hr")
        lines.append(f"  - Capacity: {candidate.capacity_rps:.0f} RPS")
        lines.append(f"  - Headroom: {candidate.headroom_factor:.1f}x")
        lines.append("")

        # Scoring
        lines.append("Decision Scores:")
        lines.append(f"  - Cost: {score.cost_score:.2f}")
        lines.append(f"  - Performance: {score.performance_score:.2f}")
        lines.append(f"  - Stability: {score.stability_score:.2f}")
        lines.append(f"  - Risk: {score.risk_score:.2f}")
        lines.append(f"  - Transition: {score.transition_score:.2f}")
        lines.append(f"  - Total: {score.total_score:.2f}")

        return "\n".join(lines)

    def _in_cooldown(self) -> bool:
        """Check if in cooldown period."""
        if self._last_decision_time is None:
            return False

        elapsed = (datetime.now(UTC) - self._last_decision_time).total_seconds()
        return elapsed < self.config.cooldown_seconds

    def approve_decision(self, decision_id: str) -> bool:
        """Approve a pending decision."""
        for decision in self._recent_decisions:
            if decision.id == decision_id:
                if decision.status == DecisionStatus.PENDING:
                    decision.status = DecisionStatus.APPROVED
                    logger.info("Decision approved", decision_id=decision_id)
                    return True
        return False

    def reject_decision(self, decision_id: str, reason: str = "") -> bool:
        """Reject a pending decision."""
        for decision in self._recent_decisions:
            if decision.id == decision_id:
                if decision.status == DecisionStatus.PENDING:
                    decision.status = DecisionStatus.REJECTED
                    decision.metadata["rejection_reason"] = reason
                    logger.info("Decision rejected", decision_id=decision_id, reason=reason)
                    return True
        return False

    def get_decision(self, decision_id: str) -> ScalingDecision | None:
        """Get a decision by ID."""
        for decision in self._recent_decisions:
            if decision.id == decision_id:
                return decision
        return None

    def get_recent_decisions(self, limit: int = 10) -> list[ScalingDecision]:
        """Get recent decisions."""
        return list(reversed(self._recent_decisions[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_decisions": len(self._recent_decisions),
            "pending_decisions": sum(
                1 for d in self._recent_decisions
                if d.status == DecisionStatus.PENDING
            ),
            "approved_decisions": sum(
                1 for d in self._recent_decisions
                if d.status == DecisionStatus.APPROVED
            ),
            "in_cooldown": self._in_cooldown(),
            "last_decision_time": (
                self._last_decision_time.isoformat()
                if self._last_decision_time
                else None
            ),
            "config": {
                "cost_weight": self.config.cost_weight,
                "performance_weight": self.config.performance_weight,
                "stability_weight": self.config.stability_weight,
                "risk_weight": self.config.risk_weight,
                "transition_weight": self.config.transition_weight,
            },
        }
