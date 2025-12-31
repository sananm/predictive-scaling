"""
Business context collector.

Collects information about business events that may impact traffic:
- Marketing campaigns
- Product launches
- Scheduled deployments
- Sales events
- etc.

Supports integrations with:
- Google Calendar
- Marketing platforms (HubSpot, Mailchimp, etc.)
- CI/CD systems
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.storage.models import BusinessEvent
from src.utils.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)


# Event type to expected impact multiplier mapping
DEFAULT_IMPACT_MULTIPLIERS: dict[str, float] = {
    "product_launch": 3.0,       # 3x normal traffic
    "major_release": 2.0,        # 2x normal traffic
    "marketing_campaign": 1.5,   # 1.5x normal traffic
    "email_blast": 1.3,          # 1.3x normal traffic
    "social_media_push": 1.2,    # 1.2x normal traffic
    "press_release": 1.4,        # 1.4x normal traffic
    "sale_event": 2.5,           # 2.5x normal traffic
    "scheduled_maintenance": 0.1, # Almost no traffic expected
    "deployment": 1.0,           # Normal traffic
}


class BusinessContextCollector(BaseCollector):
    """
    Collector for business events and context.

    Gathers information about scheduled events that may impact
    application traffic patterns.
    """

    def __init__(
        self,
        service_name: str = "default",
        collection_interval: float = 300.0,  # 5 minutes
        google_calendar_id: str | None = None,
        google_api_key: str | None = None,
        event_lookahead_days: int = 7,
    ) -> None:
        """
        Initialize business context collector.

        Args:
            service_name: Name for metrics labeling
            collection_interval: Seconds between collections
            google_calendar_id: Google Calendar ID for events
            google_api_key: Google API key for Calendar access
            event_lookahead_days: Days ahead to look for events
        """
        super().__init__(
            name=f"business-{service_name}",
            collection_interval=collection_interval,
        )

        self.service_name = service_name
        self.google_calendar_id = google_calendar_id
        self.google_api_key = google_api_key
        self.event_lookahead_days = event_lookahead_days

        # Cache for events
        self._events_cache: list[dict[str, Any]] = []
        self._cache_updated: datetime | None = None
        self._cache_ttl = timedelta(minutes=15)

        # HTTP client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def collect(self) -> list[dict[str, Any]]:
        """
        Collect business events from all configured sources.

        Returns:
            List of metric dictionaries representing active/upcoming events
        """
        metrics = []
        timestamp = datetime.now(timezone.utc)

        # Collect from all sources
        events = []

        # Google Calendar events
        if self.google_calendar_id and self.google_api_key:
            calendar_events = await self._collect_google_calendar_events()
            events.extend(calendar_events)

        # Manual/cached events (from database or API)
        cached_events = await self._get_cached_events()
        events.extend(cached_events)

        # Convert events to metrics
        for event in events:
            # Calculate time until event
            event_start = event.get("start_time")
            if isinstance(event_start, str):
                event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))

            if event_start:
                hours_until = (event_start - timestamp).total_seconds() / 3600

                # Only include events within lookahead window
                if hours_until < self.event_lookahead_days * 24:
                    metrics.append({
                        "timestamp": timestamp,
                        "service_name": self.service_name,
                        "metric_name": "business_event",
                        "value": event.get("expected_impact_multiplier", 1.0),
                        "labels": {
                            "event_type": event.get("event_type", "unknown"),
                            "event_name": event.get("name", ""),
                            "hours_until_start": round(hours_until, 2),
                            "source": event.get("source", "unknown"),
                        },
                    })

        # Add aggregate metrics
        active_events = [
            e for e in events
            if self._is_event_active(e, timestamp)
        ]

        upcoming_events = [
            e for e in events
            if self._is_event_upcoming(e, timestamp, hours=24)
        ]

        metrics.append({
            "timestamp": timestamp,
            "service_name": self.service_name,
            "metric_name": "active_events_count",
            "value": float(len(active_events)),
            "labels": {},
        })

        metrics.append({
            "timestamp": timestamp,
            "service_name": self.service_name,
            "metric_name": "upcoming_events_24h_count",
            "value": float(len(upcoming_events)),
            "labels": {},
        })

        # Calculate combined impact multiplier
        combined_impact = 1.0
        for event in active_events:
            combined_impact *= event.get("expected_impact_multiplier", 1.0)

        metrics.append({
            "timestamp": timestamp,
            "service_name": self.service_name,
            "metric_name": "combined_impact_multiplier",
            "value": combined_impact,
            "labels": {},
        })

        logger.info(
            "Business context collection complete",
            total_events=len(events),
            active_events=len(active_events),
            combined_impact=combined_impact,
        )

        return metrics

    async def _collect_google_calendar_events(self) -> list[dict[str, Any]]:
        """
        Fetch events from Google Calendar.

        Returns:
            List of event dictionaries
        """
        if not self.google_calendar_id or not self.google_api_key:
            return []

        events = []
        client = await self._get_client()

        try:
            now = datetime.now(timezone.utc)
            time_min = now.isoformat()
            time_max = (now + timedelta(days=self.event_lookahead_days)).isoformat()

            response = await client.get(
                f"https://www.googleapis.com/calendar/v3/calendars/{self.google_calendar_id}/events",
                params={
                    "key": self.google_api_key,
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                },
            )

            if response.status_code == 200:
                data = response.json()
                for item in data.get("items", []):
                    event = self._parse_google_calendar_event(item)
                    if event:
                        events.append(event)
            else:
                logger.warning(
                    "Failed to fetch Google Calendar events",
                    status_code=response.status_code,
                )

        except Exception as e:
            logger.error("Error fetching Google Calendar events", error=str(e))

        return events

    def _parse_google_calendar_event(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Parse a Google Calendar event into our format."""
        try:
            summary = item.get("summary", "")
            description = item.get("description", "")

            # Extract event type from summary or description
            event_type = "unknown"
            for known_type in DEFAULT_IMPACT_MULTIPLIERS.keys():
                if known_type.replace("_", " ") in summary.lower():
                    event_type = known_type
                    break
                if known_type.replace("_", " ") in description.lower():
                    event_type = known_type
                    break

            # Get start/end times
            start = item.get("start", {})
            end = item.get("end", {})

            start_time = start.get("dateTime") or start.get("date")
            end_time = end.get("dateTime") or end.get("date")

            if not start_time:
                return None

            return {
                "event_type": event_type,
                "name": summary,
                "description": description,
                "start_time": start_time,
                "end_time": end_time,
                "expected_impact_multiplier": DEFAULT_IMPACT_MULTIPLIERS.get(event_type, 1.0),
                "source": "google_calendar",
                "metadata": {
                    "calendar_event_id": item.get("id"),
                },
            }

        except Exception as e:
            logger.warning("Failed to parse calendar event", error=str(e))
            return None

    async def _get_cached_events(self) -> list[dict[str, Any]]:
        """Get events from cache or refresh if stale."""
        now = datetime.now(timezone.utc)

        if (
            self._cache_updated
            and now - self._cache_updated < self._cache_ttl
        ):
            return self._events_cache

        # Cache is stale, would normally refresh from database
        # For now, return empty list
        return self._events_cache

    def add_event(
        self,
        event_type: str,
        name: str,
        start_time: datetime,
        end_time: datetime,
        expected_impact_multiplier: float | None = None,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Manually add an event to the cache.

        Args:
            event_type: Type of event
            name: Event name
            start_time: Event start time
            end_time: Event end time
            expected_impact_multiplier: Expected traffic multiplier
            source: Event source
            metadata: Additional metadata
        """
        if expected_impact_multiplier is None:
            expected_impact_multiplier = DEFAULT_IMPACT_MULTIPLIERS.get(event_type, 1.0)

        event = {
            "event_type": event_type,
            "name": name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "expected_impact_multiplier": expected_impact_multiplier,
            "source": source,
            "metadata": metadata or {},
        }

        self._events_cache.append(event)
        self._cache_updated = datetime.now(timezone.utc)

        logger.info("Added business event", event_type=event_type, name=name)

    def _is_event_active(self, event: dict[str, Any], now: datetime) -> bool:
        """Check if an event is currently active."""
        start = event.get("start_time")
        end = event.get("end_time")

        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))

        if start and end:
            return start <= now <= end
        return False

    def _is_event_upcoming(
        self,
        event: dict[str, Any],
        now: datetime,
        hours: int = 24,
    ) -> bool:
        """Check if an event starts within the specified hours."""
        start = event.get("start_time")

        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))

        if start:
            time_until = start - now
            return timedelta(0) < time_until <= timedelta(hours=hours)
        return False

    def estimate_impact(
        self,
        event_type: str,
        historical_actual: float | None = None,
    ) -> float:
        """
        Estimate the impact multiplier for an event type.

        If historical data is provided, uses that. Otherwise,
        falls back to default multipliers.

        Args:
            event_type: Type of event
            historical_actual: Actual multiplier from past events

        Returns:
            Estimated impact multiplier
        """
        if historical_actual is not None:
            # Blend historical with default (70% historical, 30% default)
            default = DEFAULT_IMPACT_MULTIPLIERS.get(event_type, 1.0)
            return 0.7 * historical_actual + 0.3 * default

        return DEFAULT_IMPACT_MULTIPLIERS.get(event_type, 1.0)

    async def stop(self) -> None:
        """Stop collector and close HTTP client."""
        await super().stop()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
