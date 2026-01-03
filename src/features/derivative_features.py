"""
Derivative feature extraction.

Extracts rate-of-change features:
- First derivative (velocity)
- Second derivative (acceleration)
- Smoothed derivatives
- Jerk (rate of change of acceleration)
"""

import numpy as np
import pandas as pd

from .base import BaseExtractor
from .config import FeatureConfig


class DerivativeFeatureExtractor(BaseExtractor):
    """
    Extractor for derivative-based features.

    Creates features that capture the rate of change:
    - Velocity (first derivative): How fast is the value changing?
    - Acceleration (second derivative): Is the change speeding up or slowing down?
    - Jerk (third derivative): How abruptly is acceleration changing?
    - Smoothed derivatives: Noise-reduced versions
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize derivative feature extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
        """
        Extract derivative features from DataFrame.

        Args:
            df: DataFrame with DatetimeIndex and target column
            target_column: Name of column to differentiate

        Returns:
            DataFrame with derivative features
        """
        self._validate_input(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        features = pd.DataFrame(index=df.index)
        values = df[target_column]

        # Infer frequency for scaling
        freq_minutes = self._infer_frequency_minutes(df.index)

        # First derivative (velocity) at different scales
        for window in self.config.derivative_windows:
            periods = max(window // freq_minutes, 1)
            prefix = f"deriv1_{window}m"

            # Raw first derivative
            features[f"{prefix}"] = values.diff(periods=periods) / periods

            # Smoothed first derivative
            smoothing = self.config.derivative_smoothing
            features[f"{prefix}_smooth"] = (
                values.diff(periods=periods)
                .rolling(window=smoothing, min_periods=1)
                .mean() / periods
            )

            # Absolute velocity (magnitude regardless of direction)
            features[f"{prefix}_abs"] = features[f"{prefix}"].abs()

            # Direction only (-1, 0, 1)
            features[f"{prefix}_direction"] = np.sign(features[f"{prefix}"])

        # Second derivative (acceleration) at different scales
        for window in self.config.derivative_windows:
            periods = max(window // freq_minutes, 1)
            prefix = f"deriv2_{window}m"

            # First get velocity
            velocity = values.diff(periods=periods)

            # Then differentiate velocity to get acceleration
            features[f"{prefix}"] = velocity.diff(periods=periods) / (periods ** 2)

            # Smoothed second derivative
            features[f"{prefix}_smooth"] = (
                velocity.diff(periods=periods)
                .rolling(window=self.config.derivative_smoothing, min_periods=1)
                .mean() / (periods ** 2)
            )

            # Absolute acceleration
            features[f"{prefix}_abs"] = features[f"{prefix}"].abs()

        # Third derivative (jerk) - only for primary window
        primary_window = self.config.derivative_windows[0] if self.config.derivative_windows else 5
        periods = max(primary_window // freq_minutes, 1)

        velocity = values.diff(periods=periods)
        acceleration = velocity.diff(periods=periods)
        features["deriv3_jerk"] = acceleration.diff(periods=periods) / (periods ** 3)
        features["deriv3_jerk_smooth"] = (
            features["deriv3_jerk"]
            .rolling(window=self.config.derivative_smoothing, min_periods=1)
            .mean()
        )

        # Normalized derivatives (relative to current value)
        # This makes derivatives comparable across different value scales
        for window in self.config.derivative_windows[:2]:  # Limit to first 2
            periods = max(window // freq_minutes, 1)

            # Relative velocity (% change per period)
            features[f"rel_velocity_{window}m"] = (
                values.diff(periods=periods) / values.shift(periods).replace(0, np.nan)
            )

            # Relative acceleration
            velocity = values.diff(periods=periods)
            features[f"rel_accel_{window}m"] = (
                velocity.diff(periods=periods) / velocity.shift(periods).abs().replace(0, np.nan)
            )

        # Momentum indicators
        # Positive momentum: value increasing and acceleration positive
        for window in self.config.derivative_windows[:2]:
            periods = max(window // freq_minutes, 1)
            velocity = values.diff(periods=periods)
            acceleration = velocity.diff(periods=periods)

            features[f"momentum_{window}m"] = (
                (velocity > 0).astype(int) * 2 +
                (acceleration > 0).astype(int)
            ) - 1.5  # Centers around 0

        # Trend strength (velocity / volatility)
        for window in [5, 15, 60]:
            if window in self.config.derivative_windows or window in [5, 15, 60]:
                periods = max(window // freq_minutes, 1)
                velocity = values.diff(periods=periods)
                volatility = values.rolling(window=periods, min_periods=1).std()

                features[f"trend_strength_{window}m"] = (
                    velocity.abs() / volatility.replace(0, np.nan)
                )

        # Rate of change classification
        # Helps model understand if we're in stable, growing, or declining phase
        primary_velocity = features.get(f"deriv1_{primary_window}m", values.diff())
        velocity_std = primary_velocity.rolling(window=60 // freq_minutes, min_periods=1).std()

        # Classify as: -2 (rapid decline), -1 (decline), 0 (stable), 1 (growth), 2 (rapid growth)
        normalized_velocity = primary_velocity / velocity_std.replace(0, np.nan)
        features["rate_class"] = pd.cut(
            normalized_velocity,
            bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
            labels=[-2, -1, 0, 1, 2],
        ).astype(float)

        # Inflection points (where acceleration changes sign)
        primary_accel = features.get(f"deriv2_{primary_window}m", values.diff().diff())
        accel_sign_change = (np.sign(primary_accel) != np.sign(primary_accel.shift(1)))
        features["is_inflection_point"] = accel_sign_change.astype(int)

        # Cumulative inflection points in last N periods
        features["inflection_count_60m"] = (
            features["is_inflection_point"]
            .rolling(window=60 // freq_minutes, min_periods=1)
            .sum()
        )

        # Add prefix
        features = self._add_prefix(features, "deriv")

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


class SavitzkyGolayExtractor(BaseExtractor):
    """
    Extract smoothed derivatives using Savitzky-Golay filter.

    The Savitzky-Golay filter fits a polynomial to a window of data
    and uses the polynomial coefficients to estimate derivatives.
    This produces smoother derivatives than simple differencing.
    """

    def __init__(
        self,
        config: FeatureConfig,
        window_size: int = 11,
        poly_order: int = 3,
    ) -> None:
        """
        Initialize Savitzky-Golay extractor.

        Args:
            config: Feature configuration
            window_size: Window size for filter (must be odd)
            poly_order: Polynomial order (must be less than window_size)
        """
        super().__init__(config)
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.poly_order = poly_order

    def extract(self, df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
        """
        Extract Savitzky-Golay filtered derivatives.

        Args:
            df: DataFrame with DatetimeIndex and target column
            target_column: Column to process

        Returns:
            DataFrame with smoothed derivative features
        """
        self._validate_input(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        features = pd.DataFrame(index=df.index)
        values = df[target_column].values

        try:
            from scipy.signal import savgol_filter

            # Smoothed value (0th derivative)
            features["sg_smooth"] = savgol_filter(
                values, self.window_size, self.poly_order, deriv=0, mode="nearest"
            )

            # First derivative (velocity)
            features["sg_velocity"] = savgol_filter(
                values, self.window_size, self.poly_order, deriv=1, mode="nearest"
            )

            # Second derivative (acceleration)
            features["sg_acceleration"] = savgol_filter(
                values, self.window_size, self.poly_order, deriv=2, mode="nearest"
            )

            # Residual (original - smoothed)
            features["sg_residual"] = values - features["sg_smooth"]

            # Normalized features
            value_std = pd.Series(values).rolling(
                window=self.window_size, min_periods=1
            ).std().values

            features["sg_velocity_norm"] = (
                features["sg_velocity"] / np.where(value_std > 0, value_std, np.nan)
            )

        except ImportError:
            # scipy not available, use simple rolling mean
            rolling_mean = pd.Series(values).rolling(
                window=self.window_size, min_periods=1, center=True
            ).mean()

            features["sg_smooth"] = rolling_mean
            features["sg_velocity"] = rolling_mean.diff()
            features["sg_acceleration"] = features["sg_velocity"].diff()
            features["sg_residual"] = values - rolling_mean.values

        # Add prefix
        features = self._add_prefix(features, "sg")

        self._feature_names = features.columns.tolist()

        return features
