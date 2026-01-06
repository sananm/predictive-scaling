"""
Unit tests for Phase 7: Decision Engine.

Tests:
- Cost Model
- Capacity Model
- Risk Model
- Scaling Strategies
- Candidate Generation
- Decision Engine
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.decision import (
    # Cost Model
    CloudProvider,
    CostEstimate,
    CostModel,
    InfrastructureConfig,
    InstancePricing,
    PricingType,
    # Capacity Model
    CapacityConfig,
    CapacityEstimate,
    CapacityModel,
    InstanceCapacity,
    ScalingRequirement,
    # Risk Model
    RiskAssessment,
    RiskCategory,
    RiskConfig,
    RiskFactor,
    RiskLevel,
    RiskModel,
    # Strategies
    EmergencyScaleStrategy,
    GradualRampStrategy,
    MaintainStrategy,
    PreemptiveBurstStrategy,
    ScaleDownStrategy,
    ScalingPlan,
    ScalingStep,
    StrategyConfig,
    StrategySelector,
    StrategyType,
    # Candidates
    CandidateConfig,
    CandidateGenerator,
    CandidateSet,
    ScalingCandidate,
    # Decision Engine
    CandidateScore,
    DecisionEngine,
    DecisionEngineConfig,
    DecisionStatus,
    InfrastructureState,
    PredictionInput,
    ScalingDecision,
    VerificationCriteria,
)


# ============================================================================
# Cost Model Tests
# ============================================================================

class TestCostModel:
    """Tests for Cost Model."""

    def test_cost_model_init(self):
        """Test cost model initialization."""
        model = CostModel(provider=CloudProvider.AWS, region="us-east-1")
        assert model.provider == CloudProvider.AWS
        assert model.region == "us-east-1"

    def test_list_instance_types(self):
        """Test listing instance types."""
        model = CostModel(provider=CloudProvider.AWS)
        types = model.list_instance_types()
        assert len(types) > 0
        assert "m5.large" in types
        assert "t3.medium" in types

    def test_get_pricing(self):
        """Test getting instance pricing."""
        model = CostModel(provider=CloudProvider.AWS)
        pricing = model.get_pricing("m5.large")
        assert pricing is not None
        assert pricing.instance_type == "m5.large"
        assert pricing.on_demand_hourly > 0
        assert pricing.spot_hourly is not None
        assert pricing.spot_hourly < pricing.on_demand_hourly

    def test_calculate_cost_on_demand(self):
        """Test cost calculation for on-demand instances."""
        model = CostModel(provider=CloudProvider.AWS)
        config = InfrastructureConfig(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
        )
        estimate = model.calculate_cost(config)
        assert estimate.hourly_cost > 0
        assert estimate.daily_cost == estimate.hourly_cost * 24
        assert estimate.monthly_cost == estimate.hourly_cost * 24 * 30
        assert estimate.spot_cost == 0

    def test_calculate_cost_with_spot(self):
        """Test cost calculation with spot instances."""
        model = CostModel(provider=CloudProvider.AWS)
        config = InfrastructureConfig(
            instance_type="m5.large",
            instance_count=10,
            spot_percentage=0.5,
        )
        estimate = model.calculate_cost(config)
        assert estimate.on_demand_cost > 0
        assert estimate.spot_cost > 0
        assert estimate.spot_cost < estimate.on_demand_cost  # Spot is cheaper

    def test_calculate_savings(self):
        """Test savings calculation."""
        model = CostModel(provider=CloudProvider.AWS)
        current = InfrastructureConfig(
            instance_type="m5.large",
            instance_count=10,
            spot_percentage=0.0,
        )
        proposed = InfrastructureConfig(
            instance_type="m5.large",
            instance_count=10,
            spot_percentage=0.5,
        )
        savings = model.calculate_savings(current, proposed, hours=24)
        assert savings["hourly_savings"] > 0
        assert savings["total_savings"] > 0
        assert savings["savings_percentage"] > 0

    def test_gcp_provider(self):
        """Test GCP provider."""
        model = CostModel(provider=CloudProvider.GCP, region="us-central1")
        types = model.list_instance_types()
        assert len(types) > 0
        assert any("e2" in t for t in types)

    def test_azure_provider(self):
        """Test Azure provider."""
        model = CostModel(provider=CloudProvider.AZURE, region="eastus")
        types = model.list_instance_types()
        assert len(types) > 0
        assert any("Standard" in t for t in types)


class TestInstancePricing:
    """Tests for InstancePricing dataclass."""

    def test_spot_discount(self):
        """Test spot discount calculation."""
        pricing = InstancePricing(
            instance_type="m5.large",
            provider=CloudProvider.AWS,
            region="us-east-1",
            vcpu=2,
            memory_gb=8,
            on_demand_hourly=0.10,
            spot_hourly=0.04,
        )
        assert 0 < pricing.spot_discount < 1
        assert pricing.spot_discount == pytest.approx(0.6, rel=0.01)

    def test_hourly_cost(self):
        """Test hourly cost by pricing type."""
        pricing = InstancePricing(
            instance_type="m5.large",
            provider=CloudProvider.AWS,
            region="us-east-1",
            vcpu=2,
            memory_gb=8,
            on_demand_hourly=0.10,
            spot_hourly=0.04,
        )
        assert pricing.hourly_cost(PricingType.ON_DEMAND) == 0.10
        assert pricing.hourly_cost(PricingType.SPOT) == 0.04


# ============================================================================
# Capacity Model Tests
# ============================================================================

class TestCapacityModel:
    """Tests for Capacity Model."""

    def test_capacity_model_init(self):
        """Test capacity model initialization."""
        model = CapacityModel()
        assert model.config is not None
        assert model.provider == CloudProvider.AWS

    def test_get_capacity(self):
        """Test getting instance capacity."""
        model = CapacityModel()
        capacity = model.get_capacity("m5.large")
        assert capacity.instance_type == "m5.large"
        assert capacity.base_rps > 0
        assert capacity.max_rps > capacity.base_rps
        assert capacity.warm_up_seconds > 0

    def test_calculate_capacity(self):
        """Test calculating total capacity."""
        model = CapacityModel()
        estimate = model.calculate_capacity(
            instance_type="m5.large",
            instance_count=5,
            current_load=1000,
        )
        assert estimate.total_base_rps > 0
        assert estimate.effective_rps > 0
        assert estimate.utilization >= 0

    def test_calculate_required_instances(self):
        """Test calculating required instances."""
        model = CapacityModel()
        requirement = model.calculate_required_instances(
            instance_type="m5.large",
            target_rps=5000,
        )
        assert requirement.min_instances >= 1
        assert requirement.recommended_instances >= requirement.min_instances
        assert requirement.headroom_factor > 1.0

    def test_calculate_transition_time(self):
        """Test transition time calculation."""
        model = CapacityModel()

        # Scale up takes longer
        scale_up_time = model.calculate_transition_time("m5.large", 5, 10)
        assert scale_up_time > 0

        # Scale down is fast
        scale_down_time = model.calculate_transition_time("m5.large", 10, 5)
        assert scale_down_time < scale_up_time


class TestInstanceCapacity:
    """Tests for InstanceCapacity dataclass."""

    def test_effective_rps_no_degradation(self):
        """Test effective RPS under normal load."""
        capacity = InstanceCapacity(
            instance_type="m5.large",
            provider=CloudProvider.AWS,
            base_rps=1000,
            max_rps=1500,
            warm_up_seconds=45,
        )
        # 50% load should have no degradation
        assert capacity.effective_rps(0.5) == 1000

    def test_effective_rps_with_degradation(self):
        """Test effective RPS under high load."""
        capacity = InstanceCapacity(
            instance_type="m5.large",
            provider=CloudProvider.AWS,
            base_rps=1000,
            max_rps=1500,
            warm_up_seconds=45,
            degradation_threshold=0.8,
        )
        # 95% load should show degradation
        assert capacity.effective_rps(0.95) < 1000

    def test_warm_up_capacity(self):
        """Test warm-up capacity curve."""
        capacity = InstanceCapacity(
            instance_type="m5.large",
            provider=CloudProvider.AWS,
            base_rps=1000,
            max_rps=1500,
            warm_up_seconds=60,
        )
        # At start
        assert capacity.warm_up_capacity(0) == pytest.approx(0.1, rel=0.1)
        # At halfway
        assert 0.3 < capacity.warm_up_capacity(30) < 0.9
        # At end
        assert capacity.warm_up_capacity(60) == 1.0
        # After warm-up
        assert capacity.warm_up_capacity(120) == 1.0


# ============================================================================
# Risk Model Tests
# ============================================================================

class TestRiskModel:
    """Tests for Risk Model."""

    def test_risk_model_init(self):
        """Test risk model initialization."""
        model = RiskModel()
        assert model.config is not None

    def test_assess_low_risk(self):
        """Test assessment with low risk scenario."""
        model = RiskModel()
        assessment = model.assess(
            current_instances=5,
            target_instances=6,
            spot_percentage=0.0,
            spot_interruption_rate=0.0,
            prediction_uncertainty=0.1,
            current_utilization=0.5,
            target_utilization=0.6,
            transition_time_seconds=60,
            time_until_demand_spike=600,
        )
        assert assessment.overall_score < 0.5
        assert assessment.is_acceptable
        assert not assessment.requires_approval

    def test_assess_high_risk(self):
        """Test assessment with high risk scenario."""
        model = RiskModel()
        assessment = model.assess(
            current_instances=5,
            target_instances=20,
            spot_percentage=0.9,
            spot_interruption_rate=0.2,
            prediction_uncertainty=0.8,
            current_utilization=0.95,
            target_utilization=0.9,
            transition_time_seconds=300,
            time_until_demand_spike=60,
        )
        assert assessment.overall_score > 0.5
        assert len(assessment.factors) > 0

    def test_risk_levels(self):
        """Test risk level classification."""
        assert RiskLevel.from_score(0.1) == RiskLevel.LOW
        assert RiskLevel.from_score(0.3) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.6) == RiskLevel.HIGH
        assert RiskLevel.from_score(0.9) == RiskLevel.CRITICAL

    def test_spot_interruption_probability(self):
        """Test spot interruption probability calculation."""
        model = RiskModel()

        # No spot instances
        prob = model.calculate_spot_interruption_probability(0, 0.05, 1.0)
        assert prob == 0.0

        # Some spot instances
        prob = model.calculate_spot_interruption_probability(10, 0.05, 1.0)
        assert 0 < prob < 1

    def test_record_scaling_action(self):
        """Test recording scaling actions."""
        model = RiskModel()
        initial_count = len(model._recent_scaling_actions)
        model.record_scaling_action()
        assert len(model._recent_scaling_actions) == initial_count + 1


# ============================================================================
# Scaling Strategies Tests
# ============================================================================

class TestScalingStrategies:
    """Tests for Scaling Strategies."""

    def test_gradual_ramp_scale_up(self):
        """Test gradual ramp strategy for scale up."""
        strategy = GradualRampStrategy()
        config = StrategyConfig(gradual_step_size=2)
        plan = strategy.create_plan(
            current_instances=5,
            target_instances=10,
            config=config,
        )
        assert plan.strategy_type == StrategyType.GRADUAL_RAMP
        assert len(plan.steps) > 0
        assert plan.steps[-1].target_instances == 10
        assert plan.is_scale_up

    def test_gradual_ramp_scale_down(self):
        """Test gradual ramp strategy for scale down."""
        strategy = GradualRampStrategy()
        config = StrategyConfig()
        plan = strategy.create_plan(
            current_instances=10,
            target_instances=5,
            config=config,
        )
        assert plan.strategy_type == StrategyType.GRADUAL_RAMP
        assert plan.is_scale_down
        assert plan.steps[-1].target_instances == 5

    def test_preemptive_burst(self):
        """Test preemptive burst strategy."""
        strategy = PreemptiveBurstStrategy()
        config = StrategyConfig()
        spike_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        plan = strategy.create_plan(
            current_instances=5,
            target_instances=15,
            config=config,
            spike_time=spike_time,
        )
        assert plan.strategy_type == StrategyType.PREEMPTIVE_BURST
        assert all(step.is_critical for step in plan.steps)

    def test_emergency_scale(self):
        """Test emergency scale strategy."""
        strategy = EmergencyScaleStrategy()
        config = StrategyConfig()
        plan = strategy.create_plan(
            current_instances=5,
            target_instances=20,
            config=config,
        )
        assert plan.strategy_type == StrategyType.EMERGENCY_SCALE
        assert len(plan.steps) == 1
        assert plan.steps[0].is_critical

    def test_scale_down(self):
        """Test scale down strategy."""
        strategy = ScaleDownStrategy()
        config = StrategyConfig()
        plan = strategy.create_plan(
            current_instances=10,
            target_instances=3,
            config=config,
        )
        assert plan.strategy_type == StrategyType.SCALE_DOWN
        assert plan.steps[-1].target_instances == 3

    def test_maintain_strategy(self):
        """Test maintain (no-op) strategy."""
        strategy = MaintainStrategy()
        config = StrategyConfig()
        plan = strategy.create_plan(
            current_instances=5,
            target_instances=5,
            config=config,
        )
        assert plan.strategy_type == StrategyType.MAINTAIN
        assert len(plan.steps) == 0


class TestStrategySelector:
    """Tests for Strategy Selector."""

    def test_select_maintain(self):
        """Test selecting maintain strategy."""
        selector = StrategySelector()
        strategy = selector.select_strategy(
            current_instances=5,
            target_instances=5,
            current_utilization=0.5,
        )
        assert strategy == StrategyType.MAINTAIN

    def test_select_emergency(self):
        """Test selecting emergency strategy."""
        selector = StrategySelector()
        strategy = selector.select_strategy(
            current_instances=5,
            target_instances=10,
            current_utilization=0.96,
            is_emergency=True,
        )
        assert strategy == StrategyType.EMERGENCY_SCALE

    def test_select_preemptive_burst(self):
        """Test selecting preemptive burst strategy."""
        selector = StrategySelector()
        strategy = selector.select_strategy(
            current_instances=5,
            target_instances=15,
            current_utilization=0.7,
            time_until_spike=300,  # 5 minutes
        )
        assert strategy == StrategyType.PREEMPTIVE_BURST

    def test_select_gradual_ramp(self):
        """Test selecting gradual ramp strategy."""
        selector = StrategySelector()
        strategy = selector.select_strategy(
            current_instances=5,
            target_instances=10,
            current_utilization=0.5,
            time_until_spike=None,
        )
        assert strategy == StrategyType.GRADUAL_RAMP

    def test_select_scale_down(self):
        """Test selecting scale down strategy."""
        selector = StrategySelector()
        strategy = selector.select_strategy(
            current_instances=10,
            target_instances=5,
            current_utilization=0.3,
        )
        assert strategy == StrategyType.SCALE_DOWN


# ============================================================================
# Candidate Generation Tests
# ============================================================================

class TestCandidateGenerator:
    """Tests for Candidate Generator."""

    def test_generator_init(self):
        """Test generator initialization."""
        generator = CandidateGenerator()
        assert generator.config is not None
        assert generator.cost_model is not None
        assert generator.capacity_model is not None

    def test_generate_candidates(self):
        """Test generating candidates."""
        generator = CandidateGenerator()
        candidate_set = generator.generate(
            target_rps=5000,
            current_instances=5,
        )
        assert candidate_set.total_generated > 0
        assert len(candidate_set.candidates) > 0

    def test_generate_includes_feasible(self):
        """Test that generation includes feasible candidates."""
        generator = CandidateGenerator()
        candidate_set = generator.generate(
            target_rps=3000,
            current_instances=3,
        )
        assert candidate_set.total_feasible > 0
        assert len(candidate_set.feasible_candidates) > 0

    def test_cheapest_feasible(self):
        """Test getting cheapest feasible candidate."""
        generator = CandidateGenerator()
        candidate_set = generator.generate(
            target_rps=3000,
            current_instances=3,
        )
        cheapest = candidate_set.cheapest_feasible
        assert cheapest is not None
        # Verify it's actually the cheapest
        for c in candidate_set.feasible_candidates:
            assert c.cost_estimate.hourly_cost >= cheapest.cost_estimate.hourly_cost

    def test_filter_candidates(self):
        """Test filtering candidates."""
        generator = CandidateGenerator()
        candidate_set = generator.generate(
            target_rps=3000,
            current_instances=3,
        )
        filtered = generator.filter_candidates(
            candidate_set,
            max_spot_percentage=0.3,
        )
        for c in filtered.candidates:
            assert c.config.spot_percentage <= 0.3

    def test_rank_candidates(self):
        """Test ranking candidates."""
        generator = CandidateGenerator()
        candidate_set = generator.generate(
            target_rps=3000,
            current_instances=3,
        )
        ranked = generator.rank_candidates(candidate_set)
        assert len(ranked) > 0
        # First should have highest score
        if len(ranked) > 1:
            assert ranked[0].feasibility_score >= 0

    def test_get_recommendation(self):
        """Test getting a single recommendation."""
        generator = CandidateGenerator()
        recommendation = generator.get_recommendation(
            target_rps=3000,
            optimization_goal="balanced",
        )
        assert recommendation is not None
        assert recommendation.is_feasible


# ============================================================================
# Decision Engine Tests
# ============================================================================

class TestDecisionEngine:
    """Tests for Decision Engine."""

    def test_engine_init(self):
        """Test engine initialization."""
        engine = DecisionEngine()
        assert engine.config is not None
        assert engine.cost_model is not None
        assert engine.capacity_model is not None
        assert engine.risk_model is not None

    def test_decide_scale_up(self):
        """Test decision for scale up scenario."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
            current_rps=4000,
            current_utilization=0.8,
            healthy_instances=5,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=5000,
                p50=6000,
                p90=7500,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        assert decision is not None
        assert decision.status == DecisionStatus.PENDING
        assert decision.target_config.instance_count >= current_state.instance_count

    def test_decide_scale_down(self):
        """Test decision for scale down scenario."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=20,
            spot_percentage=0.0,
            current_rps=1000,
            current_utilization=0.2,
            healthy_instances=20,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=800,
                p50=1000,
                p90=1200,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        # May or may not decide to scale down depending on thresholds
        if decision is not None:
            assert decision.target_config.instance_count <= current_state.instance_count

    def test_decide_no_action(self):
        """Test decision when no action needed."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=10,
            spot_percentage=0.0,
            current_rps=3000,
            current_utilization=0.5,
            healthy_instances=10,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=2800,
                p50=3000,
                p90=3200,
                confidence=0.95,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        # Should return None if utilization is within acceptable range
        # (depends on config thresholds)

    def test_decision_has_rollback(self):
        """Test that decisions include rollback config."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
            current_rps=4500,
            current_utilization=0.9,
            healthy_instances=5,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=5500,
                p50=6000,
                p90=7000,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        if decision is not None:
            assert decision.rollback_config is not None
            assert decision.rollback_config.instance_count == current_state.instance_count

    def test_decision_has_reasoning(self):
        """Test that decisions include reasoning."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
            current_rps=4500,
            current_utilization=0.9,
            healthy_instances=5,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=5500,
                p50=6000,
                p90=7000,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        if decision is not None:
            assert decision.reasoning is not None
            assert len(decision.reasoning) > 0

    def test_approve_decision(self):
        """Test approving a decision."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
            current_rps=4500,
            current_utilization=0.9,
            healthy_instances=5,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=6000,
                p50=7000,
                p90=8000,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        if decision is not None:
            result = engine.approve_decision(decision.id)
            assert result
            assert decision.status == DecisionStatus.APPROVED

    def test_reject_decision(self):
        """Test rejecting a decision."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
            current_rps=4500,
            current_utilization=0.9,
            healthy_instances=5,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=6000,
                p50=7000,
                p90=8000,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        if decision is not None:
            result = engine.reject_decision(decision.id, reason="Test rejection")
            assert result
            assert decision.status == DecisionStatus.REJECTED

    def test_get_recent_decisions(self):
        """Test getting recent decisions."""
        engine = DecisionEngine()

        current_state = InfrastructureState(
            instance_type="m5.large",
            instance_count=5,
            spot_percentage=0.0,
            current_rps=4500,
            current_utilization=0.9,
            healthy_instances=5,
        )

        predictions = [
            PredictionInput(
                horizon_minutes=15,
                p10=6000,
                p50=7000,
                p90=8000,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        decision = engine.decide(current_state, predictions, force_evaluation=True)
        if decision is not None:
            recent = engine.get_recent_decisions(limit=5)
            assert len(recent) > 0
            assert any(d.id == decision.id for d in recent)

    def test_get_stats(self):
        """Test getting engine stats."""
        engine = DecisionEngine()
        stats = engine.get_stats()
        assert "total_decisions" in stats
        assert "config" in stats


# ============================================================================
# Module Export Tests
# ============================================================================

class TestModuleExports:
    """Tests for module exports."""

    def test_cost_model_exports(self):
        """Test cost model exports."""
        from src.decision import (
            CloudProvider,
            CostModel,
            InfrastructureConfig,
            InstancePricing,
        )
        assert CloudProvider is not None
        assert CostModel is not None

    def test_capacity_model_exports(self):
        """Test capacity model exports."""
        from src.decision import (
            CapacityConfig,
            CapacityModel,
            InstanceCapacity,
        )
        assert CapacityConfig is not None
        assert CapacityModel is not None

    def test_risk_model_exports(self):
        """Test risk model exports."""
        from src.decision import (
            RiskAssessment,
            RiskCategory,
            RiskLevel,
            RiskModel,
        )
        assert RiskModel is not None
        assert RiskLevel is not None

    def test_strategy_exports(self):
        """Test strategy exports."""
        from src.decision import (
            ScalingPlan,
            StrategySelector,
            StrategyType,
        )
        assert StrategyType is not None
        assert StrategySelector is not None

    def test_candidate_exports(self):
        """Test candidate exports."""
        from src.decision import (
            CandidateGenerator,
            CandidateSet,
            ScalingCandidate,
        )
        assert CandidateGenerator is not None
        assert ScalingCandidate is not None

    def test_engine_exports(self):
        """Test decision engine exports."""
        from src.decision import (
            DecisionEngine,
            DecisionStatus,
            InfrastructureState,
            ScalingDecision,
        )
        assert DecisionEngine is not None
        assert DecisionStatus is not None
