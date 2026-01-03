"""
Feature engineering orchestrator.

Coordinates all feature extractors and provides:
- Unified interface for feature computation
- Feature caching
- Feature versioning
- Parallel extraction
"""

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

from .base import BaseExtractor
from .business_features import BusinessFeatureExtractor
from .config import DEFAULT_CONFIG, FeatureConfig
from .derivative_features import DerivativeFeatureExtractor
from .fourier_features import FourierFeatureExtractor
from .lag_features import LagFeatureExtractor
from .rolling_features import ExponentialMovingAverageExtractor, RollingFeatureExtractor
from .time_features import HolidayFeatureExtractor, TimeFeatureExtractor

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Main feature engineering orchestrator.

    Coordinates all feature extractors to produce a complete
    feature set for ML models.

    Features are organized into groups:
    - Time: Calendar and cyclical time features
    - Lag: Historical value features
    - Rolling: Moving statistics
    - Fourier: Seasonality features
    - Business: Business event features
    - Derivative: Rate-of-change features
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        target_column: str = "value",
    ) -> None:
        """
        Initialize feature engineer.

        Args:
            config: Feature configuration (default: DEFAULT_CONFIG)
            target_column: Name of the target column in input DataFrames
        """
        self.config = config or DEFAULT_CONFIG
        self.target_column = target_column

        # Initialize extractors
        self._extractors: dict[str, BaseExtractor] = {}
        self._init_extractors()

        # Feature metadata
        self._feature_names: list[str] = []
        self._feature_hash: str = ""
        self._last_computed: datetime | None = None

        # Cache
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_enabled = self.config.cache_features

    def _init_extractors(self) -> None:
        """Initialize all feature extractors based on config."""
        if self.config.enable_time_features:
            self._extractors["time"] = TimeFeatureExtractor(self.config)
            if self.config.time_include_holidays:
                self._extractors["holiday"] = HolidayFeatureExtractor(self.config)

        if self.config.enable_lag_features:
            self._extractors["lag"] = LagFeatureExtractor(self.config)

        if self.config.enable_rolling_features:
            self._extractors["rolling"] = RollingFeatureExtractor(self.config)
            self._extractors["ema"] = ExponentialMovingAverageExtractor(self.config)

        if self.config.enable_fourier_features:
            self._extractors["fourier"] = FourierFeatureExtractor(self.config)

        if self.config.enable_business_features:
            self._extractors["business"] = BusinessFeatureExtractor(self.config)

        if self.config.enable_derivative_features:
            self._extractors["derivative"] = DerivativeFeatureExtractor(self.config)

    def compute_features(
        self,
        df: pd.DataFrame,
        events: list[dict[str, Any]] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all features for input DataFrame.

        Args:
            df: DataFrame with DatetimeIndex and metric columns
            events: Optional list of business events
            use_cache: Whether to use cached results

        Returns:
            DataFrame with all computed features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to feature engineer")
            return pd.DataFrame()

        # Check cache
        cache_key = self._compute_cache_key(df, events)
        if use_cache and self._cache_enabled and cache_key in self._cache:
            logger.debug("Returning cached features", cache_key=cache_key[:8])
            return self._cache[cache_key]

        # Ensure we have a target column
        if self.target_column not in df.columns:
            # Try to use the first numeric column
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                logger.debug(
                    "Using first numeric column as target",
                    column=numeric_cols[0],
                )
                df = df.rename(columns={numeric_cols[0]: self.target_column})
            else:
                raise ValueError(
                    f"No target column '{self.target_column}' and no numeric columns found"
                )

        # Collect features from all extractors
        all_features = []

        for name, extractor in self._extractors.items():
            try:
                if name == "business":
                    features = extractor.extract(df, events=events)
                elif name in ["time", "holiday", "fourier"]:
                    features = extractor.extract(df)
                else:
                    features = extractor.extract(df, target_column=self.target_column)

                all_features.append(features)
                logger.debug(
                    "Extracted features",
                    extractor=name,
                    feature_count=len(features.columns),
                )

            except Exception as e:
                logger.error(
                    "Feature extraction failed",
                    extractor=name,
                    error=str(e),
                )

        if not all_features:
            logger.error("No features extracted")
            return pd.DataFrame(index=df.index)

        # Combine all features
        result = pd.concat(all_features, axis=1)

        # Handle NaN values
        result = self._handle_nan_values(result)

        # Update metadata
        self._feature_names = result.columns.tolist()
        self._feature_hash = self._compute_feature_hash()
        self._last_computed = datetime.now(timezone.utc)

        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = result
            # Limit cache size
            if len(self._cache) > 100:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        logger.info(
            "Feature engineering complete",
            total_features=len(result.columns),
            rows=len(result),
            feature_hash=self._feature_hash[:8],
        )

        return result

    def compute_features_for_prediction(
        self,
        df: pd.DataFrame,
        events: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Compute features for prediction (single row or few rows).

        Optimized for real-time prediction where we need features
        for the most recent data point(s).

        Args:
            df: DataFrame with recent data
            events: Optional business events

        Returns:
            DataFrame with features (last row only if single prediction)
        """
        # Compute all features
        features = self.compute_features(df, events=events)

        # Drop rows with NaN (usually the initial window)
        features = features.dropna()

        return features

    def _handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values in feature DataFrame.

        Strategy:
        1. Forward fill within reasonable limits
        2. Fill remaining with column median
        3. Fill any remaining with 0
        """
        # Forward fill (max 10 periods)
        df = df.ffill(limit=10)

        # Backward fill for initial rows
        df = df.bfill(limit=10)

        # Fill remaining with 0
        df = df.fillna(0)

        return df

    def _compute_cache_key(
        self,
        df: pd.DataFrame,
        events: list[dict[str, Any]] | None,
    ) -> str:
        """Compute cache key for feature DataFrame."""
        # Use hash of index range and shape
        key_parts = [
            str(df.index.min()) if len(df) > 0 else "",
            str(df.index.max()) if len(df) > 0 else "",
            str(df.shape),
            str(len(events)) if events else "0",
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _compute_feature_hash(self) -> str:
        """Compute hash of feature names for versioning."""
        feature_string = ",".join(sorted(self._feature_names))
        return hashlib.md5(feature_string.encode()).hexdigest()

    @property
    def feature_names(self) -> list[str]:
        """Get list of all feature names."""
        return self._feature_names

    @property
    def feature_hash(self) -> str:
        """Get feature set version hash."""
        return self._feature_hash

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self._feature_names)

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Get feature names grouped by extractor."""
        groups = {}
        for name, extractor in self._extractors.items():
            groups[name] = extractor.feature_names
        return groups

    def get_metadata(self) -> dict[str, Any]:
        """Get feature engineering metadata."""
        return {
            "n_features": self.n_features,
            "feature_hash": self._feature_hash,
            "last_computed": self._last_computed.isoformat() if self._last_computed else None,
            "enabled_groups": self.config.get_enabled_groups(),
            "extractors": list(self._extractors.keys()),
            "required_history_minutes": self.config.get_required_history_minutes(),
        }

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")


class FeatureEngineerFactory:
    """
    Factory for creating configured FeatureEngineer instances.
    """

    @staticmethod
    def create_minimal() -> FeatureEngineer:
        """Create feature engineer with minimal features (for testing)."""
        from .config import MINIMAL_CONFIG
        return FeatureEngineer(config=MINIMAL_CONFIG)

    @staticmethod
    def create_full() -> FeatureEngineer:
        """Create feature engineer with full feature set."""
        from .config import FULL_CONFIG
        return FeatureEngineer(config=FULL_CONFIG)

    @staticmethod
    def create_default() -> FeatureEngineer:
        """Create feature engineer with default configuration."""
        return FeatureEngineer()

    @staticmethod
    def create_custom(
        enable_groups: list[str] | None = None,
        lag_windows: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> FeatureEngineer:
        """
        Create feature engineer with custom configuration.

        Args:
            enable_groups: Feature groups to enable
            lag_windows: Custom lag windows
            rolling_windows: Custom rolling windows

        Returns:
            Configured FeatureEngineer
        """
        config = FeatureConfig()

        if enable_groups is not None:
            config.enable_time_features = "time" in enable_groups
            config.enable_lag_features = "lag" in enable_groups
            config.enable_rolling_features = "rolling" in enable_groups
            config.enable_fourier_features = "fourier" in enable_groups
            config.enable_business_features = "business" in enable_groups
            config.enable_derivative_features = "derivative" in enable_groups

        if lag_windows is not None:
            config.lag_windows = lag_windows

        if rolling_windows is not None:
            config.rolling_windows = rolling_windows

        return FeatureEngineer(config=config)
