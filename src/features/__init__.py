"""
Feature engineering for predictive scaling.

This package transforms raw metrics into ML-ready features:
- Time features: Calendar and cyclical encodings
- Lag features: Historical values at various windows
- Rolling features: Moving statistics (mean, std, percentiles)
- Fourier features: Seasonality decomposition
- Business features: Event indicators and impact multipliers
- Derivative features: Rate-of-change (velocity, acceleration)

Usage:
    from src.features import FeatureEngineer

    engineer = FeatureEngineer()
    features = engineer.compute_features(metrics_df, events=business_events)
"""

from .base import BaseExtractor
from .business_features import BusinessFeatureExtractor
from .config import DEFAULT_CONFIG, FULL_CONFIG, MINIMAL_CONFIG, FeatureConfig
from .derivative_features import DerivativeFeatureExtractor, SavitzkyGolayExtractor
from .engineer import FeatureEngineer, FeatureEngineerFactory
from .fourier_features import FourierFeatureExtractor, SeasonalDecompositionExtractor
from .lag_features import LagFeatureExtractor, MultiColumnLagExtractor
from .rolling_features import ExponentialMovingAverageExtractor, RollingFeatureExtractor
from .time_features import HolidayFeatureExtractor, TimeFeatureExtractor

__all__ = [
    # Main orchestrator
    "FeatureEngineer",
    "FeatureEngineerFactory",
    # Configuration
    "FeatureConfig",
    "DEFAULT_CONFIG",
    "MINIMAL_CONFIG",
    "FULL_CONFIG",
    # Base class
    "BaseExtractor",
    # Extractors
    "TimeFeatureExtractor",
    "HolidayFeatureExtractor",
    "LagFeatureExtractor",
    "MultiColumnLagExtractor",
    "RollingFeatureExtractor",
    "ExponentialMovingAverageExtractor",
    "FourierFeatureExtractor",
    "SeasonalDecompositionExtractor",
    "BusinessFeatureExtractor",
    "DerivativeFeatureExtractor",
    "SavitzkyGolayExtractor",
]
