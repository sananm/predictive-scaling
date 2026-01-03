"""
Feature engineering configuration.

Defines all parameters for feature extraction:
- Data granularity
- Lag window sizes
- Rolling window sizes
- Fourier periods for seasonality
- Feature group toggles
"""

from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.

    Controls which features are computed and their parameters.
    """

    # Data granularity in minutes
    granularity_minutes: int = 1

    # Lag window sizes in minutes
    # Used for: lag features, same-time comparisons
    lag_windows: list[int] = field(
        default_factory=lambda: [5, 10, 15, 30, 60, 120, 360, 720, 1440]
    )

    # Rolling window sizes in minutes
    # Used for: moving averages, standard deviations, percentiles
    rolling_windows: list[int] = field(
        default_factory=lambda: [5, 10, 15, 30, 60, 120, 360, 720, 1440]
    )

    # Fourier periods in minutes for seasonality decomposition
    # 1440 = daily (24 hours), 10080 = weekly (7 days)
    fourier_periods: list[int] = field(
        default_factory=lambda: [1440, 10080]
    )

    # Number of harmonics for each Fourier period
    # More harmonics = capture more complex patterns
    fourier_harmonics: dict[int, int] = field(
        default_factory=lambda: {
            1440: 3,   # 3 harmonics for daily
            10080: 2,  # 2 harmonics for weekly
        }
    )

    # Feature group toggles
    # Enable/disable entire feature categories
    enable_time_features: bool = True
    enable_lag_features: bool = True
    enable_rolling_features: bool = True
    enable_fourier_features: bool = True
    enable_business_features: bool = True
    enable_derivative_features: bool = True

    # Time feature options
    time_cyclical_encoding: bool = True  # Use sin/cos encoding
    time_include_holidays: bool = True   # Include holiday flags

    # Rolling feature options
    rolling_percentiles: list[float] = field(
        default_factory=lambda: [0.25, 0.75]
    )
    rolling_include_zscore: bool = True
    rolling_include_trend: bool = True

    # Derivative feature options
    derivative_windows: list[int] = field(
        default_factory=lambda: [1, 5, 15]  # minutes
    )
    derivative_smoothing: int = 3  # Rolling mean window for smoothing

    # Business feature options
    business_decay_rate: float = 0.1  # Exponential decay rate for event impact
    business_lookahead_days: int = 7  # Days to look ahead for upcoming events

    # Caching options
    cache_features: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

    def get_enabled_groups(self) -> list[str]:
        """Get list of enabled feature groups."""
        groups = []
        if self.enable_time_features:
            groups.append("time")
        if self.enable_lag_features:
            groups.append("lag")
        if self.enable_rolling_features:
            groups.append("rolling")
        if self.enable_fourier_features:
            groups.append("fourier")
        if self.enable_business_features:
            groups.append("business")
        if self.enable_derivative_features:
            groups.append("derivative")
        return groups

    def get_total_lag_minutes(self) -> int:
        """Get the maximum lag window in minutes."""
        return max(self.lag_windows) if self.lag_windows else 0

    def get_required_history_minutes(self) -> int:
        """
        Get minimum history required to compute all features.

        This is the maximum of all window sizes across all feature types.
        """
        max_lag = max(self.lag_windows) if self.lag_windows else 0
        max_rolling = max(self.rolling_windows) if self.rolling_windows else 0
        max_fourier = max(self.fourier_periods) if self.fourier_periods else 0

        # Need at least 2x the max period for Fourier
        return max(max_lag, max_rolling, max_fourier * 2)


# Default configuration instance
DEFAULT_CONFIG = FeatureConfig()


# Minimal configuration for testing
MINIMAL_CONFIG = FeatureConfig(
    lag_windows=[5, 15, 60],
    rolling_windows=[5, 15, 60],
    fourier_periods=[1440],
    fourier_harmonics={1440: 2},
    enable_business_features=False,
    derivative_windows=[5],
)


# Full configuration for production
FULL_CONFIG = FeatureConfig(
    lag_windows=[1, 5, 10, 15, 30, 60, 120, 240, 360, 720, 1440, 2880, 10080],
    rolling_windows=[5, 10, 15, 30, 60, 120, 360, 720, 1440],
    fourier_periods=[1440, 10080],
    fourier_harmonics={1440: 4, 10080: 3},
    rolling_percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    derivative_windows=[1, 5, 15, 60],
)
