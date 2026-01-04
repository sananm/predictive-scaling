"""
Prediction models for multi-horizon forecasting.

This package provides models for different prediction horizons:
- Short-term (5-30 min): Transformer-based model
- Medium-term (1-24 hours): Gradient boosting ensemble
- Long-term (1-7 days): Prophet with business events

The EnsembleCombiner intelligently combines predictions from all models.
"""

from .base import (
    BaseModel,
    ModelMetadata,
    PredictionResult,
    calculate_metrics,
    quantile_loss,
)
from .combiner import CombinerConfig, EnsembleCombiner
from .ensemble import EnsembleConfig, GradientBoostingModel, MediumTermModel
from .prophet_model import LongTermModel, ProphetConfig
from .transformer import ShortTermModel, TransformerConfig

__all__ = [
    # Base classes
    "BaseModel",
    "ModelMetadata",
    "PredictionResult",
    "calculate_metrics",
    "quantile_loss",
    # Short-term (Transformer)
    "TransformerConfig",
    "ShortTermModel",
    # Medium-term (Gradient Boosting)
    "EnsembleConfig",
    "GradientBoostingModel",
    "MediumTermModel",
    # Long-term (Prophet)
    "ProphetConfig",
    "LongTermModel",
    # Combiner
    "CombinerConfig",
    "EnsembleCombiner",
]
