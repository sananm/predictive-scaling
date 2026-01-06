"""
Decision Engine module for scaling decisions.

This module provides:
- Cost Model: Cloud pricing and cost calculations
- Capacity Model: Instance capacity mapping
- Risk Model: Risk assessment for scaling decisions
- Scaling Strategies: Different scaling approaches
- Candidate Generation: Generate and evaluate scaling candidates
- Decision Engine: Main orchestrator for scaling decisions
"""

from .candidates import (
    CandidateConfig,
    CandidateGenerator,
    CandidateSet,
    ScalingCandidate,
)
from .capacity_model import (
    CapacityConfig,
    CapacityEstimate,
    CapacityModel,
    InstanceCapacity,
    ScalingRequirement,
)
from .cost_model import (
    CloudProvider,
    CostEstimate,
    CostModel,
    InfrastructureConfig,
    InstancePricing,
    PricingType,
)
from .engine import (
    CandidateScore,
    DecisionEngine,
    DecisionEngineConfig,
    DecisionStatus,
    InfrastructureState,
    PredictionInput,
    ScalingDecision,
    VerificationCriteria,
)
from .risk_model import (
    RiskAssessment,
    RiskCategory,
    RiskConfig,
    RiskFactor,
    RiskLevel,
    RiskModel,
)
from .strategies import (
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
)

__all__ = [
    # Cost Model
    "CloudProvider",
    "PricingType",
    "InstancePricing",
    "InfrastructureConfig",
    "CostEstimate",
    "CostModel",
    # Capacity Model
    "InstanceCapacity",
    "CapacityEstimate",
    "ScalingRequirement",
    "CapacityConfig",
    "CapacityModel",
    # Risk Model
    "RiskCategory",
    "RiskLevel",
    "RiskFactor",
    "RiskAssessment",
    "RiskConfig",
    "RiskModel",
    # Strategies
    "StrategyType",
    "ScalingStep",
    "ScalingPlan",
    "StrategyConfig",
    "GradualRampStrategy",
    "PreemptiveBurstStrategy",
    "EmergencyScaleStrategy",
    "ScaleDownStrategy",
    "MaintainStrategy",
    "StrategySelector",
    # Candidates
    "CandidateConfig",
    "ScalingCandidate",
    "CandidateSet",
    "CandidateGenerator",
    # Decision Engine
    "DecisionStatus",
    "InfrastructureState",
    "PredictionInput",
    "CandidateScore",
    "VerificationCriteria",
    "ScalingDecision",
    "DecisionEngineConfig",
    "DecisionEngine",
]
