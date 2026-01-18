"""
Business event feature extraction.

Extracts features from business events that impact traffic:
- Active campaign indicators
- Time until/since events
- Event impact multipliers (with decay)
- Cumulative impact from overlapping events
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseExtractor
from .config import FeatureConfig

# Default event type weights
DEFAULT_EVENT_WEIGHTS = {
    "product_launch": 3.0,
    "major_release": 2.0,
    "marketing_campaign": 1.5,
    "email_blast": 1.3,
    "social_media_push": 1.2,
    "press_release": 1.4,
    "sale_event": 2.5,
    "scheduled_maintenance": 0.1,
    "deployment": 1.0,
}


class BusinessFeatureExtractor(BaseExtractor):
    """
    Extractor for business event features.

    Creates features that capture the impact of business events:
    - Binary indicators for active events
    - Time-based features (hours until, hours since)
    - Impact multipliers with exponential decay
    - Cumulative effects from multiple events
    """

    def __init__(
        self,
        config: FeatureConfig,
        event_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize business feature extractor.

        Args:
            config: Feature configuration
            event_weights: Optional custom event type weights
        """
        super().__init__(config)
        self.event_weights = event_weights or DEFAULT_EVENT_WEIGHTS

    def extract(
        self,
        df: pd.DataFrame,
        events: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Extract business features from DataFrame and events.

        Args:
            df: DataFrame with DatetimeIndex
            events: List of event dictionaries with:
                - event_type: str
                - start_time: datetime
                - end_time: datetime
                - expected_impact_multiplier: float (optional)
                - name: str (optional)

        Returns:
            DataFrame with business features
        """
        self._validate_input(df)

        features = pd.DataFrame(index=df.index)

        if not events:
            # No events - return zero features
            features["has_active_event"] = 0
            features["active_event_count"] = 0
            features["cumulative_impact"] = 1.0
            features["hours_until_next_event"] = -1
            features["hours_since_last_event"] = -1

            # Event type indicators
            for event_type in self.event_weights:
                features[f"is_{event_type}_active"] = 0
                features[f"impact_{event_type}"] = 0.0

            features = self._add_prefix(features, "business")
            self._feature_names = features.columns.tolist()
            return features

        # Parse events
        parsed_events = self._parse_events(events)

        # For each timestamp, compute event features
        active_counts = []
        cumulative_impacts = []
        hours_until_next = []
        hours_since_last = []

        # Event type specific features
        event_type_active = {et: [] for et in self.event_weights}
        event_type_impact = {et: [] for et in self.event_weights}

        for ts in df.index:
            ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

            # Find active events
            active = [e for e in parsed_events if self._is_event_active(e, ts_dt)]
            active_counts.append(len(active))

            # Calculate cumulative impact with decay
            impact = self._calculate_cumulative_impact(parsed_events, ts_dt)
            cumulative_impacts.append(impact)

            # Hours until next event
            next_hours = self._hours_until_next_event(parsed_events, ts_dt)
            hours_until_next.append(next_hours)

            # Hours since last event
            last_hours = self._hours_since_last_event(parsed_events, ts_dt)
            hours_since_last.append(last_hours)

            # Event type specific
            for event_type in self.event_weights:
                type_active = any(
                    e["event_type"] == event_type and self._is_event_active(e, ts_dt)
                    for e in parsed_events
                )
                event_type_active[event_type].append(int(type_active))

                type_impact = self._calculate_type_impact(
                    parsed_events, ts_dt, event_type
                )
                event_type_impact[event_type].append(type_impact)

        # Add computed features
        features["has_active_event"] = (np.array(active_counts) > 0).astype(int)
        features["active_event_count"] = active_counts
        features["cumulative_impact"] = cumulative_impacts
        features["hours_until_next_event"] = hours_until_next
        features["hours_since_last_event"] = hours_since_last

        # Normalized time features
        max_lookahead = self.config.business_lookahead_days * 24
        features["normalized_hours_until"] = np.clip(
            np.array(hours_until_next) / max_lookahead, 0, 1
        )

        # Event proximity (1 when event is now, 0 when far away)
        hours_array = np.array(hours_until_next)
        hours_array[hours_array < 0] = max_lookahead
        features["event_proximity"] = 1 - np.clip(hours_array / max_lookahead, 0, 1)

        # Event type indicators
        for event_type in self.event_weights:
            features[f"is_{event_type}_active"] = event_type_active[event_type]
            features[f"impact_{event_type}"] = event_type_impact[event_type]

        # Impact deviation from baseline
        features["impact_deviation"] = features["cumulative_impact"] - 1.0

        # Log of impact (for multiplicative effects)
        features["log_impact"] = np.log(features["cumulative_impact"].clip(lower=0.01))

        # Is high impact period (impact > 1.5)?
        features["is_high_impact"] = (features["cumulative_impact"] > 1.5).astype(int)

        # Is low impact period (impact < 0.5)?
        features["is_low_impact"] = (features["cumulative_impact"] < 0.5).astype(int)

        # Add prefix
        features = self._add_prefix(features, "business")

        self._feature_names = features.columns.tolist()

        return features

    def _parse_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse and validate events."""
        parsed = []

        for event in events:
            parsed_event = {
                "event_type": event.get("event_type", "unknown"),
                "name": event.get("name", ""),
            }

            # Parse start time
            start = event.get("start_time")
            if isinstance(start, str):
                start = datetime.fromisoformat(start.replace("Z", "+00:00"))
            parsed_event["start_time"] = start

            # Parse end time
            end = event.get("end_time")
            if isinstance(end, str):
                end = datetime.fromisoformat(end.replace("Z", "+00:00"))
            parsed_event["end_time"] = end

            # Get impact multiplier
            event_type = parsed_event["event_type"]
            default_impact = self.event_weights.get(event_type, 1.0)
            parsed_event["impact"] = event.get(
                "expected_impact_multiplier", default_impact
            )

            parsed.append(parsed_event)

        return parsed

    def _is_event_active(self, event: dict[str, Any], ts: datetime) -> bool:
        """Check if event is active at timestamp."""
        start = event.get("start_time")
        end = event.get("end_time")

        if start is None:
            return False

        # Handle timezone-naive comparison
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            if hasattr(start, "tzinfo") and start.tzinfo is None:
                start = start.replace(tzinfo=ts.tzinfo)
            if end and hasattr(end, "tzinfo") and end.tzinfo is None:
                end = end.replace(tzinfo=ts.tzinfo)

        if end is None:
            # Event with no end time - check if within default duration
            end = start + timedelta(hours=24)

        return start <= ts <= end

    def _calculate_cumulative_impact(
        self,
        events: list[dict[str, Any]],
        ts: datetime,
    ) -> float:
        """Calculate cumulative impact with decay."""
        cumulative = 1.0
        decay_rate = self.config.business_decay_rate

        for event in events:
            start = event.get("start_time")
            end = event.get("end_time")
            impact = event.get("impact", 1.0)

            if start is None:
                continue

            # Handle timezone
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                if hasattr(start, "tzinfo") and start.tzinfo is None:
                    start = start.replace(tzinfo=ts.tzinfo)
                if end and hasattr(end, "tzinfo") and end.tzinfo is None:
                    end = end.replace(tzinfo=ts.tzinfo)

            if end is None:
                end = start + timedelta(hours=24)

            if start <= ts <= end:
                # Event is active - full impact
                cumulative *= impact

            elif ts > end:
                # Event has ended - apply decay
                hours_since = (ts - end).total_seconds() / 3600
                decayed_impact = 1.0 + (impact - 1.0) * np.exp(-decay_rate * hours_since)
                cumulative *= decayed_impact

            elif ts < start:
                # Event hasn't started - apply buildup
                hours_until = (start - ts).total_seconds() / 3600
                if hours_until <= 24:  # Only consider events within 24 hours
                    buildup_impact = 1.0 + (impact - 1.0) * (1 - hours_until / 24) * 0.5
                    cumulative *= buildup_impact

        return cumulative

    def _calculate_type_impact(
        self,
        events: list[dict[str, Any]],
        ts: datetime,
        event_type: str,
    ) -> float:
        """Calculate impact for a specific event type."""
        type_events = [e for e in events if e.get("event_type") == event_type]

        if not type_events:
            return 0.0

        total_impact = 0.0
        decay_rate = self.config.business_decay_rate

        for event in type_events:
            start = event.get("start_time")
            end = event.get("end_time")
            impact = event.get("impact", 1.0)

            if start is None:
                continue

            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                if hasattr(start, "tzinfo") and start.tzinfo is None:
                    start = start.replace(tzinfo=ts.tzinfo)
                if end and hasattr(end, "tzinfo") and end.tzinfo is None:
                    end = end.replace(tzinfo=ts.tzinfo)

            if end is None:
                end = start + timedelta(hours=24)

            if start <= ts <= end:
                total_impact += impact
            elif ts > end:
                hours_since = (ts - end).total_seconds() / 3600
                decayed = impact * np.exp(-decay_rate * hours_since)
                total_impact += decayed

        return total_impact

    def _hours_until_next_event(
        self,
        events: list[dict[str, Any]],
        ts: datetime,
    ) -> float:
        """Calculate hours until next event starts."""
        min_hours = -1

        for event in events:
            start = event.get("start_time")
            if start is None:
                continue

            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                if hasattr(start, "tzinfo") and start.tzinfo is None:
                    start = start.replace(tzinfo=ts.tzinfo)

            if start > ts:
                hours = (start - ts).total_seconds() / 3600
                if min_hours < 0 or hours < min_hours:
                    min_hours = hours

        return min_hours

    def _hours_since_last_event(
        self,
        events: list[dict[str, Any]],
        ts: datetime,
    ) -> float:
        """Calculate hours since last event ended."""
        min_hours = -1

        for event in events:
            end = event.get("end_time")
            start = event.get("start_time")

            if end is None and start is not None:
                end = start + timedelta(hours=24)

            if end is None:
                continue

            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                if hasattr(end, "tzinfo") and end.tzinfo is None:
                    end = end.replace(tzinfo=ts.tzinfo)

            if end < ts:
                hours = (ts - end).total_seconds() / 3600
                if min_hours < 0 or hours < min_hours:
                    min_hours = hours

        return min_hours
