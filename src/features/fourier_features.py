"""
Fourier feature extraction for seasonality.

Extracts frequency-domain features that capture periodic patterns:
- Daily seasonality (24-hour cycle)
- Weekly seasonality (7-day cycle)
- Custom periods
"""

import numpy as np
import pandas as pd

from .base import BaseExtractor
from .config import FeatureConfig


class FourierFeatureExtractor(BaseExtractor):
    """
    Extractor for Fourier-based seasonality features.

    Creates sine and cosine components at different frequencies
    to capture periodic patterns in the data.

    Why Fourier features?
    - Time series often have periodic patterns (daily, weekly)
    - Sine/cosine pairs can represent any periodic function
    - ML models can learn the amplitude and phase of each component
    - Multiple harmonics capture complex patterns
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize Fourier feature extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Fourier features from DataFrame index.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with Fourier features
        """
        self._validate_input(df)

        features = pd.DataFrame(index=df.index)

        # Convert timestamps to minutes since epoch for consistent calculation
        # Using minutes since start of data for better numerical stability
        start_time = df.index.min()
        minutes_elapsed = (df.index - start_time).total_seconds() / 60

        for period_minutes in self.config.fourier_periods:
            n_harmonics = self.config.fourier_harmonics.get(period_minutes, 2)

            for harmonic in range(1, n_harmonics + 1):
                # Frequency: how many cycles per minute
                frequency = harmonic / period_minutes

                # Sine and cosine components
                sin_name = f"fourier_{period_minutes}m_sin_{harmonic}"
                cos_name = f"fourier_{period_minutes}m_cos_{harmonic}"

                features[sin_name] = np.sin(2 * np.pi * frequency * minutes_elapsed)
                features[cos_name] = np.cos(2 * np.pi * frequency * minutes_elapsed)

        # Add combined amplitude features for primary periods
        if 1440 in self.config.fourier_periods:  # Daily
            # Amplitude of daily seasonality (sqrt of sum of squared sin/cos)
            sin_1 = features.get("fourier_1440m_sin_1", 0)
            cos_1 = features.get("fourier_1440m_cos_1", 0)
            features["fourier_daily_amplitude"] = np.sqrt(sin_1**2 + cos_1**2)

            # Phase of daily seasonality
            features["fourier_daily_phase"] = np.arctan2(sin_1, cos_1)

        if 10080 in self.config.fourier_periods:  # Weekly
            sin_1 = features.get("fourier_10080m_sin_1", 0)
            cos_1 = features.get("fourier_10080m_cos_1", 0)
            features["fourier_weekly_amplitude"] = np.sqrt(sin_1**2 + cos_1**2)
            features["fourier_weekly_phase"] = np.arctan2(sin_1, cos_1)

        # Add prefix
        features = self._add_prefix(features, "fourier")

        self._feature_names = features.columns.tolist()

        return features


class SeasonalDecompositionExtractor(BaseExtractor):
    """
    Extract features from seasonal decomposition.

    Uses STL (Seasonal and Trend decomposition using Loess) or
    simple moving average decomposition to separate:
    - Trend component
    - Seasonal component
    - Residual component
    """

    def __init__(self, config: FeatureConfig, period: int = 1440) -> None:
        """
        Initialize seasonal decomposition extractor.

        Args:
            config: Feature configuration
            period: Seasonality period in minutes (default: 1440 = daily)
        """
        super().__init__(config)
        self.period = period

    def extract(self, df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
        """
        Extract seasonal decomposition features.

        Args:
            df: DataFrame with DatetimeIndex and target column
            target_column: Column to decompose

        Returns:
            DataFrame with decomposition features
        """
        self._validate_input(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        features = pd.DataFrame(index=df.index)
        values = df[target_column]

        # Infer frequency
        freq_minutes = self._infer_frequency_minutes(df.index)
        period_size = max(self.period // freq_minutes, 2)

        # Simple decomposition using rolling mean
        # Trend: long-term rolling mean
        trend = values.rolling(window=period_size, min_periods=1, center=True).mean()
        features["trend"] = trend

        # Detrended: original minus trend
        detrended = values - trend
        features["detrended"] = detrended

        # Seasonal: average of detrended values at same position in cycle
        # This is a simplified version - full STL would be more accurate
        seasonal = self._estimate_seasonal(detrended, period_size)
        features["seasonal"] = seasonal

        # Residual: what's left after removing trend and seasonal
        residual = detrended - seasonal
        features["residual"] = residual

        # Derived features
        # Seasonal strength: ratio of seasonal variance to total variance
        features["seasonal_strength"] = (
            seasonal.rolling(window=period_size, min_periods=1).var() /
            values.rolling(window=period_size, min_periods=1).var().replace(0, np.nan)
        )

        # Trend strength: ratio of trend variance to total variance
        features["trend_strength"] = (
            trend.rolling(window=period_size, min_periods=1).var() /
            values.rolling(window=period_size, min_periods=1).var().replace(0, np.nan)
        )

        # Residual strength (noise level)
        features["residual_strength"] = (
            residual.rolling(window=period_size, min_periods=1).var() /
            values.rolling(window=period_size, min_periods=1).var().replace(0, np.nan)
        )

        # Trend direction
        features["trend_direction"] = np.sign(trend.diff())

        # Seasonal position (where are we in the seasonal cycle?)
        # Normalized to 0-1
        cycle_position = np.arange(len(df)) % period_size
        features["seasonal_position"] = cycle_position / period_size

        # Is residual unusually high? (anomaly indicator)
        residual_std = residual.rolling(window=period_size, min_periods=1).std()
        features["residual_zscore"] = residual / residual_std.replace(0, np.nan)
        features["is_anomaly"] = (np.abs(features["residual_zscore"]) > 3).astype(int)

        # Add prefix
        features = self._add_prefix(features, "seasonal")

        self._feature_names = features.columns.tolist()

        return features

    def _estimate_seasonal(self, detrended: pd.Series, period: int) -> pd.Series:
        """
        Estimate seasonal component by averaging values at same cycle position.
        """
        # Create position in cycle
        positions = np.arange(len(detrended)) % period

        # Calculate mean for each position
        position_means = {}
        for pos in range(period):
            mask = positions == pos
            if mask.sum() > 0:
                position_means[pos] = detrended.iloc[mask].mean()
            else:
                position_means[pos] = 0

        # Map back to full series
        seasonal = pd.Series(
            [position_means.get(pos, 0) for pos in positions],
            index=detrended.index,
        )

        return seasonal

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
