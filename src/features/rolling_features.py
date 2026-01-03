"""
Rolling window feature extraction.

Extracts statistical features from rolling windows:
- Basic stats (mean, std, min, max)
- Percentiles
- Range and IQR
- Coefficient of variation
- Z-score
- Trend indicators
"""

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseExtractor
from .config import FeatureConfig


class RollingFeatureExtractor(BaseExtractor):
    """
    Extractor for rolling window statistics.

    Creates features by computing statistics over sliding windows:
    - Central tendency (mean, median)
    - Dispersion (std, range, IQR)
    - Shape (skewness, kurtosis)
    - Position (z-score, percentile rank)
    - Trend (slope, direction)
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize rolling feature extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
        """
        Extract rolling features from DataFrame.

        Args:
            df: DataFrame with DatetimeIndex and target column
            target_column: Name of column to create features for

        Returns:
            DataFrame with rolling features
        """
        self._validate_input(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        features = pd.DataFrame(index=df.index)
        values = df[target_column]

        # Infer frequency
        freq_minutes = self._infer_frequency_minutes(df.index)

        for window_minutes in self.config.rolling_windows:
            window_size = max(window_minutes // freq_minutes, 2)  # At least 2 periods
            prefix = f"roll_{window_minutes}m"

            # Create rolling window
            rolling = values.rolling(window=window_size, min_periods=1)

            # Basic statistics
            features[f"{prefix}_mean"] = rolling.mean()
            features[f"{prefix}_std"] = rolling.std()
            features[f"{prefix}_min"] = rolling.min()
            features[f"{prefix}_max"] = rolling.max()

            # Range
            features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]

            # Percentiles
            for pct in self.config.rolling_percentiles:
                pct_name = f"p{int(pct * 100)}"
                features[f"{prefix}_{pct_name}"] = rolling.quantile(pct)

            # IQR (Interquartile Range)
            if 0.25 in self.config.rolling_percentiles and 0.75 in self.config.rolling_percentiles:
                features[f"{prefix}_iqr"] = (
                    features[f"{prefix}_p75"] - features[f"{prefix}_p25"]
                )

            # Coefficient of variation (std / mean)
            features[f"{prefix}_cv"] = (
                features[f"{prefix}_std"] /
                features[f"{prefix}_mean"].replace(0, np.nan)
            )

            # Z-score (how many std deviations from rolling mean)
            if self.config.rolling_include_zscore:
                features[f"{prefix}_zscore"] = (
                    (values - features[f"{prefix}_mean"]) /
                    features[f"{prefix}_std"].replace(0, np.nan)
                )

            # Trend (linear regression slope)
            if self.config.rolling_include_trend:
                features[f"{prefix}_trend"] = self._compute_rolling_trend(
                    values, window_size
                )

            # Position within range (0 = at min, 1 = at max)
            range_vals = features[f"{prefix}_range"].replace(0, np.nan)
            features[f"{prefix}_position"] = (
                (values - features[f"{prefix}_min"]) / range_vals
            )

            # Number of times value crossed the mean
            above_mean = (values > features[f"{prefix}_mean"]).astype(int)
            features[f"{prefix}_mean_crosses"] = (
                above_mean.diff().abs().rolling(window=window_size, min_periods=1).sum()
            )

        # Additional features for specific windows
        if 60 in self.config.rolling_windows:
            window_60 = max(60 // freq_minutes, 2)
            rolling_60 = values.rolling(window=window_60, min_periods=1)

            # Skewness (asymmetry of distribution)
            features["roll_60m_skew"] = rolling_60.skew()

            # Kurtosis (tail heaviness)
            features["roll_60m_kurt"] = rolling_60.kurt()

        # Ratio features comparing different windows
        if 5 in self.config.rolling_windows and 60 in self.config.rolling_windows:
            # Short-term vs long-term mean ratio
            features["roll_ratio_5m_60m_mean"] = (
                features["roll_5m_mean"] /
                features["roll_60m_mean"].replace(0, np.nan)
            )

        if 15 in self.config.rolling_windows and 1440 in self.config.rolling_windows:
            # 15-min vs daily mean ratio
            features["roll_ratio_15m_1440m_mean"] = (
                features["roll_15m_mean"] /
                features["roll_1440m_mean"].replace(0, np.nan)
            )

        # Volatility ratio (short-term std / long-term std)
        if 5 in self.config.rolling_windows and 60 in self.config.rolling_windows:
            features["roll_volatility_ratio"] = (
                features["roll_5m_std"] /
                features["roll_60m_std"].replace(0, np.nan)
            )

        # Add prefix
        features = self._add_prefix(features, "rolling")

        self._feature_names = features.columns.tolist()

        return features

    def _compute_rolling_trend(
        self,
        series: pd.Series,
        window_size: int,
    ) -> pd.Series:
        """
        Compute rolling linear regression slope.

        The slope indicates the trend direction and strength:
        - Positive: upward trend
        - Negative: downward trend
        - Magnitude: strength of trend
        """
        def calc_slope(window):
            if len(window) < 2:
                return 0
            x = np.arange(len(window))
            try:
                slope, _, _, _, _ = stats.linregress(x, window)
                return slope
            except Exception:
                return 0

        return series.rolling(window=window_size, min_periods=2).apply(
            calc_slope, raw=True
        )

    def _infer_frequency_minutes(self, index: pd.DatetimeIndex) -> int:
        """Infer frequency of time series in minutes."""
        if len(index) < 2:
            return self.config.granularity_minutes

        diffs = pd.Series(index).diff().dropna()
        if len(diffs) == 0:
            return self.config.granularity_minutes

        median_diff = diffs.median()
        freq_minutes = int(median_diff.total_seconds() / 60)

        return max(freq_minutes, 1)


class ExponentialMovingAverageExtractor(BaseExtractor):
    """
    Extract exponential moving average features.

    EMA gives more weight to recent values, making it more
    responsive to recent changes than simple moving average.
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize EMA extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
        """
        Extract EMA features from DataFrame.

        Args:
            df: DataFrame with DatetimeIndex and target column
            target_column: Name of column to create features for

        Returns:
            DataFrame with EMA features
        """
        self._validate_input(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        features = pd.DataFrame(index=df.index)
        values = df[target_column]

        # Infer frequency
        freq_minutes = self._infer_frequency_minutes(df.index)

        # EMA with different spans
        for window_minutes in self.config.rolling_windows:
            span = max(window_minutes // freq_minutes, 2)
            prefix = f"ema_{window_minutes}m"

            # EMA value
            features[f"{prefix}"] = values.ewm(span=span, adjust=False).mean()

            # EMA standard deviation
            features[f"{prefix}_std"] = values.ewm(span=span, adjust=False).std()

            # Distance from EMA (current value - EMA)
            features[f"{prefix}_distance"] = values - features[f"{prefix}"]

            # Normalized distance (distance / EMA std)
            features[f"{prefix}_distance_norm"] = (
                features[f"{prefix}_distance"] /
                features[f"{prefix}_std"].replace(0, np.nan)
            )

        # EMA crossover features
        if 5 in self.config.rolling_windows and 15 in self.config.rolling_windows:
            ema_fast = features["ema_5m"]
            ema_slow = features["ema_15m"]

            # Is fast EMA above slow EMA?
            features["ema_crossover_5_15"] = (ema_fast > ema_slow).astype(int)

            # MACD-like feature (fast EMA - slow EMA)
            features["ema_macd_5_15"] = ema_fast - ema_slow

        if 15 in self.config.rolling_windows and 60 in self.config.rolling_windows:
            ema_fast = features["ema_15m"]
            ema_slow = features["ema_60m"]

            features["ema_crossover_15_60"] = (ema_fast > ema_slow).astype(int)
            features["ema_macd_15_60"] = ema_fast - ema_slow

        # Add prefix
        features = self._add_prefix(features, "ema")

        self._feature_names = features.columns.tolist()

        return features

    def _infer_frequency_minutes(self, index: pd.DatetimeIndex) -> int:
        """Infer frequency of time series in minutes."""
        if len(index) < 2:
            return self.config.granularity_minutes

        diffs = pd.Series(index).diff().dropna()
        if len(diffs) == 0:
            return self.config.granularity_minutes

        median_diff = diffs.median()
        freq_minutes = int(median_diff.total_seconds() / 60)

        return max(freq_minutes, 1)
