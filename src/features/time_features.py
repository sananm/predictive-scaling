"""
Time-based feature extraction.

Extracts calendar and cyclical time features:
- Cyclical encoding (sin/cos) for periodic features
- Boolean flags (weekend, business hours, etc.)
- Time of day categories
- Month/quarter indicators
"""

import numpy as np
import pandas as pd

from .base import BaseExtractor
from .config import FeatureConfig


class TimeFeatureExtractor(BaseExtractor):
    """
    Extractor for time-based features.

    Creates features from the timestamp index that capture:
    - Time of day patterns (cyclical hour encoding)
    - Day of week patterns (cyclical day encoding)
    - Monthly/yearly seasonality
    - Business hour indicators
    - Weekend/holiday flags
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize time feature extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time features from DataFrame index.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with time features
        """
        self._validate_input(df)

        features = pd.DataFrame(index=df.index)

        # Extract basic time components
        features["hour"] = df.index.hour
        features["day_of_week"] = df.index.dayofweek
        features["day_of_month"] = df.index.day
        features["month"] = df.index.month
        features["week_of_year"] = df.index.isocalendar().week.values
        features["quarter"] = df.index.quarter

        # Cyclical encoding for periodic features
        if self.config.time_cyclical_encoding:
            # Hour of day (24-hour cycle)
            features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
            features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

            # Day of week (7-day cycle)
            features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
            features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

            # Day of month (approximate 30-day cycle)
            features["dom_sin"] = np.sin(2 * np.pi * features["day_of_month"] / 31)
            features["dom_cos"] = np.cos(2 * np.pi * features["day_of_month"] / 31)

            # Month of year (12-month cycle)
            features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
            features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

            # Week of year (52-week cycle)
            features["woy_sin"] = np.sin(2 * np.pi * features["week_of_year"] / 52)
            features["woy_cos"] = np.cos(2 * np.pi * features["week_of_year"] / 52)

        # Boolean flags
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
        features["is_monday"] = (features["day_of_week"] == 0).astype(int)
        features["is_friday"] = (features["day_of_week"] == 4).astype(int)

        # Business hours (9 AM - 5 PM, Monday-Friday)
        features["is_business_hours"] = (
            (features["hour"] >= 9) &
            (features["hour"] < 17) &
            (~features["is_weekend"].astype(bool))
        ).astype(int)

        # Extended business hours (7 AM - 9 PM)
        features["is_extended_hours"] = (
            (features["hour"] >= 7) &
            (features["hour"] < 21)
        ).astype(int)

        # Time of day categories
        features["is_morning"] = ((features["hour"] >= 6) & (features["hour"] < 12)).astype(int)
        features["is_afternoon"] = ((features["hour"] >= 12) & (features["hour"] < 18)).astype(int)
        features["is_evening"] = ((features["hour"] >= 18) & (features["hour"] < 22)).astype(int)
        features["is_night"] = ((features["hour"] >= 22) | (features["hour"] < 6)).astype(int)

        # Peak hours (typically 10 AM - 2 PM and 6 PM - 9 PM)
        features["is_peak_hours"] = (
            ((features["hour"] >= 10) & (features["hour"] < 14)) |
            ((features["hour"] >= 18) & (features["hour"] < 21))
        ).astype(int)

        # Month boundary indicators
        features["is_month_start"] = (features["day_of_month"] <= 3).astype(int)
        features["is_month_end"] = (features["day_of_month"] >= 28).astype(int)

        # Quarter indicators
        features["is_quarter_start"] = (
            (features["month"].isin([1, 4, 7, 10])) &
            (features["day_of_month"] <= 7)
        ).astype(int)
        features["is_quarter_end"] = (
            (features["month"].isin([3, 6, 9, 12])) &
            (features["day_of_month"] >= 25)
        ).astype(int)

        # Year boundary
        features["is_year_start"] = (
            (features["month"] == 1) &
            (features["day_of_month"] <= 7)
        ).astype(int)
        features["is_year_end"] = (
            (features["month"] == 12) &
            (features["day_of_month"] >= 25)
        ).astype(int)

        # Minutes since midnight (for finer granularity)
        features["minutes_since_midnight"] = (
            df.index.hour * 60 + df.index.minute
        )

        # Normalized time of day (0-1)
        features["time_of_day_normalized"] = features["minutes_since_midnight"] / 1440

        # Drop raw components if using cyclical encoding
        if self.config.time_cyclical_encoding:
            features = features.drop(columns=[
                "hour", "day_of_week", "day_of_month", "month", "week_of_year"
            ])

        # Add prefix
        features = self._add_prefix(features, "time")

        # Store feature names
        self._feature_names = features.columns.tolist()

        return features


class HolidayFeatureExtractor(BaseExtractor):
    """
    Extractor for holiday-based features.

    Creates features for US holidays and special days.
    """

    # US Federal holidays (month, day) - approximate dates
    US_HOLIDAYS = {
        (1, 1): "new_years_day",
        (1, 15): "mlk_day",  # 3rd Monday, approximate
        (2, 15): "presidents_day",  # 3rd Monday, approximate
        (5, 25): "memorial_day",  # Last Monday, approximate
        (6, 19): "juneteenth",
        (7, 4): "independence_day",
        (9, 1): "labor_day",  # 1st Monday, approximate
        (10, 10): "columbus_day",  # 2nd Monday, approximate
        (11, 11): "veterans_day",
        (11, 25): "thanksgiving",  # 4th Thursday, approximate
        (12, 25): "christmas",
    }

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize holiday feature extractor."""
        super().__init__(config)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract holiday features from DataFrame index.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with holiday features
        """
        self._validate_input(df)

        features = pd.DataFrame(index=df.index)

        # Check each date against holidays
        month_day = list(zip(df.index.month, df.index.day, strict=False))

        # Is it a holiday?
        features["is_holiday"] = pd.Series(
            [1 if (m, d) in self.US_HOLIDAYS else 0 for m, d in month_day],
            index=df.index,
        )

        # Days until next holiday
        features["days_until_holiday"] = self._days_until_holiday(df.index)

        # Days since last holiday
        features["days_since_holiday"] = self._days_since_holiday(df.index)

        # Holiday week (week containing a holiday)
        features["is_holiday_week"] = (
            (features["days_until_holiday"] <= 3) |
            (features["days_since_holiday"] <= 3)
        ).astype(int)

        # Specific holiday proximity
        features["near_christmas"] = self._near_date(df.index, 12, 25, days=7).astype(int)
        features["near_thanksgiving"] = self._near_date(df.index, 11, 25, days=5).astype(int)
        features["near_new_years"] = (
            self._near_date(df.index, 1, 1, days=3) |
            self._near_date(df.index, 12, 31, days=3)
        ).astype(int)

        # Black Friday / Cyber Monday (day after Thanksgiving)
        features["is_black_friday_week"] = (
            (df.index.month == 11) &
            (df.index.day >= 23) &
            (df.index.day <= 30)
        ).astype(int)

        # Add prefix
        features = self._add_prefix(features, "holiday")

        self._feature_names = features.columns.tolist()

        return features

    def _days_until_holiday(self, index: pd.DatetimeIndex) -> pd.Series:
        """Calculate days until next holiday."""
        result = []
        for dt in index:
            min_days = 365
            for (month, day), _ in self.US_HOLIDAYS.items():
                try:
                    # Try this year
                    holiday = dt.replace(month=month, day=day)
                    if holiday < dt:
                        # Holiday passed, try next year
                        holiday = holiday.replace(year=dt.year + 1)
                    days = (holiday - dt).days
                    min_days = min(min_days, days)
                except ValueError:
                    pass  # Invalid date
            result.append(min_days)
        return pd.Series(result, index=index)

    def _days_since_holiday(self, index: pd.DatetimeIndex) -> pd.Series:
        """Calculate days since last holiday."""
        result = []
        for dt in index:
            min_days = 365
            for (month, day), _ in self.US_HOLIDAYS.items():
                try:
                    # Try this year
                    holiday = dt.replace(month=month, day=day)
                    if holiday > dt:
                        # Holiday hasn't happened, try last year
                        holiday = holiday.replace(year=dt.year - 1)
                    days = (dt - holiday).days
                    min_days = min(min_days, days)
                except ValueError:
                    pass
            result.append(min_days)
        return pd.Series(result, index=index)

    def _near_date(
        self,
        index: pd.DatetimeIndex,
        month: int,
        day: int,
        days: int = 7,
    ) -> pd.Series:
        """Check if dates are within N days of a specific date."""
        result = []
        for dt in index:
            try:
                target = dt.replace(month=month, day=day)
                diff = abs((dt - target).days)
                # Also check previous/next year for year boundaries
                if month == 1:
                    target_prev = target.replace(year=dt.year - 1)
                    diff = min(diff, abs((dt - target_prev).days))
                elif month == 12:
                    target_next = target.replace(year=dt.year + 1)
                    diff = min(diff, abs((dt - target_next).days))
                result.append(diff <= days)
            except ValueError:
                result.append(False)
        return pd.Series(result, index=index)
