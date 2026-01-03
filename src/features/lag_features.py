"""
Lag feature extraction.

Extracts historical value features:
- Simple lags at configured windows
- Same-time-yesterday and same-time-last-week
- Absolute and relative changes from lagged values
"""

import numpy as np
import pandas as pd

from .base import BaseExtractor
from .config import FeatureConfig


class LagFeatureExtractor(BaseExtractor):
    """
    Extractor for lag-based features.

    Creates features by looking at historical values:
    - Direct lag values (value N minutes ago)
    - Same-time comparisons (value at same time yesterday/last week)
    - Change features (difference and ratio vs lagged values)
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize lag feature extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
        """
        Extract lag features from DataFrame.

        Args:
            df: DataFrame with DatetimeIndex and target column
            target_column: Name of column to create lag features for

        Returns:
            DataFrame with lag features
        """
        self._validate_input(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        features = pd.DataFrame(index=df.index)
        values = df[target_column]

        # Infer frequency from index
        freq_minutes = self._infer_frequency_minutes(df.index)

        # Simple lag features at configured windows
        for lag_minutes in self.config.lag_windows:
            lag_periods = lag_minutes // freq_minutes
            if lag_periods > 0:
                features[f"lag_{lag_minutes}m"] = values.shift(lag_periods)

        # Same-time-yesterday (1440 minutes = 24 hours)
        yesterday_periods = 1440 // freq_minutes
        if yesterday_periods > 0:
            features["lag_same_time_yesterday"] = values.shift(yesterday_periods)

        # Same-time-last-week (10080 minutes = 7 days)
        last_week_periods = 10080 // freq_minutes
        if last_week_periods > 0:
            features["lag_same_time_last_week"] = values.shift(last_week_periods)

        # Difference from lagged values (absolute change)
        for lag_minutes in self.config.lag_windows[:5]:  # Limit to first 5 windows
            lag_periods = lag_minutes // freq_minutes
            if lag_periods > 0:
                lagged = values.shift(lag_periods)
                features[f"diff_{lag_minutes}m"] = values - lagged

        # Difference from same time yesterday
        if yesterday_periods > 0:
            features["diff_yesterday"] = values - values.shift(yesterday_periods)

        # Difference from same time last week
        if last_week_periods > 0:
            features["diff_last_week"] = values - values.shift(last_week_periods)

        # Ratio to lagged values (relative change)
        for lag_minutes in self.config.lag_windows[:5]:
            lag_periods = lag_minutes // freq_minutes
            if lag_periods > 0:
                lagged = values.shift(lag_periods)
                # Avoid division by zero
                features[f"ratio_{lag_minutes}m"] = values / lagged.replace(0, np.nan)

        # Ratio to same time yesterday
        if yesterday_periods > 0:
            lagged_yesterday = values.shift(yesterday_periods)
            features["ratio_yesterday"] = values / lagged_yesterday.replace(0, np.nan)

        # Ratio to same time last week
        if last_week_periods > 0:
            lagged_week = values.shift(last_week_periods)
            features["ratio_last_week"] = values / lagged_week.replace(0, np.nan)

        # Percent change features
        for lag_minutes in [5, 15, 60]:
            lag_periods = lag_minutes // freq_minutes
            if lag_periods > 0:
                features[f"pct_change_{lag_minutes}m"] = values.pct_change(periods=lag_periods)

        # Momentum features (is the value increasing or decreasing?)
        for lag_minutes in [5, 15, 60]:
            lag_periods = lag_minutes // freq_minutes
            if lag_periods > 0:
                diff = values - values.shift(lag_periods)
                features[f"momentum_{lag_minutes}m"] = np.sign(diff)

        # Acceleration (change in momentum)
        if 5 // freq_minutes > 0:
            lag_5 = 5 // freq_minutes
            momentum = values - values.shift(lag_5)
            features["acceleration_5m"] = momentum - momentum.shift(lag_5)

        # Add prefix
        features = self._add_prefix(features, "lag")

        self._feature_names = features.columns.tolist()

        return features

    def _infer_frequency_minutes(self, index: pd.DatetimeIndex) -> int:
        """
        Infer the frequency of the time series in minutes.

        Args:
            index: DatetimeIndex to analyze

        Returns:
            Frequency in minutes
        """
        if len(index) < 2:
            return self.config.granularity_minutes

        # Calculate median difference between consecutive timestamps
        diffs = pd.Series(index).diff().dropna()
        if len(diffs) == 0:
            return self.config.granularity_minutes

        median_diff = diffs.median()
        freq_minutes = int(median_diff.total_seconds() / 60)

        return max(freq_minutes, 1)  # At least 1 minute


class MultiColumnLagExtractor(BaseExtractor):
    """
    Extract lag features for multiple columns at once.

    Useful when you have multiple metrics (CPU, memory, requests, etc.)
    and want lag features for each.
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize multi-column lag extractor."""
        super().__init__(config)
        self._single_extractor = LagFeatureExtractor(config)

    def extract(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Extract lag features for multiple columns.

        Args:
            df: DataFrame with DatetimeIndex
            columns: Columns to extract features for (default: all numeric)

        Returns:
            DataFrame with lag features for all columns
        """
        self._validate_input(df)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        all_features = []

        for col in columns:
            # Create a temporary DataFrame with just this column
            temp_df = df[[col]].rename(columns={col: "value"})

            # Extract lag features
            col_features = self._single_extractor.extract(temp_df, target_column="value")

            # Rename columns to include original column name
            col_features.columns = [
                c.replace("lag_", f"lag_{col}_") for c in col_features.columns
            ]

            all_features.append(col_features)

        # Combine all features
        result = pd.concat(all_features, axis=1)

        self._feature_names = result.columns.tolist()

        return result
