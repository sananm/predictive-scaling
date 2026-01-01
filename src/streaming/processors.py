"""
Stream processors for real-time data processing.

Processors:
- MetricsProcessor: Validate, normalize, and store raw metrics
- FeatureProcessor: Trigger feature engineering on new metrics
- PredictionTrigger: Trigger predictions when sufficient data arrives
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from src.storage.database import get_session
from src.storage.models import Feature, Metric
from src.storage.repositories import FeaturesRepository, MetricsRepository
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class BaseProcessor(ABC):
    """
    Abstract base class for stream processors.

    Provides:
    - Common processing interface
    - Error handling
    - Statistics tracking
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the processor.

        Args:
            name: Processor name for logging
        """
        self.name = name
        self._processed_count = 0
        self._error_count = 0
        self._last_processed: datetime | None = None

    @abstractmethod
    async def process(self, message: dict[str, Any]) -> None:
        """
        Process a single message.

        Args:
            message: The message to process
        """
        pass

    async def process_batch(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Process a batch of messages.

        Args:
            messages: List of messages to process

        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        for message in messages:
            try:
                await self.process(message)
                successful += 1
                self._processed_count += 1
                self._last_processed = datetime.now(timezone.utc)

            except Exception as e:
                failed += 1
                self._error_count += 1
                logger.error(
                    "Message processing failed",
                    processor=self.name,
                    error=str(e),
                )

        return successful, failed

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        return {
            "name": self.name,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "last_processed": self._last_processed.isoformat() if self._last_processed else None,
        }


class MetricsProcessor(BaseProcessor):
    """
    Processor for raw metrics.

    Responsibilities:
    - Validate incoming metrics
    - Normalize values and labels
    - Store in TimescaleDB
    - Compute basic aggregations
    """

    def __init__(self) -> None:
        """Initialize metrics processor."""
        super().__init__(name="metrics-processor")

        # Validation rules
        self._required_fields = ["timestamp", "service_name", "metric_name", "value"]
        self._valid_metric_names = {
            "requests_per_second",
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "error_rate",
            "cpu_utilization",
            "memory_utilization",
            "active_connections",
            "queue_depth",
            "pod_count_total",
            "pod_count_running",
            "pod_count_pending",
            "deployment_replicas_desired",
            "deployment_replicas_available",
            "hpa_current_replicas",
            "hpa_desired_replicas",
            "business_event",
            "external_signal",
            "external_signals_aggregate",
        }

        # Aggregation buffer
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = 100
        self._last_flush = datetime.now(timezone.utc)
        self._flush_interval = timedelta(seconds=10)

    async def process(self, message: dict[str, Any]) -> None:
        """
        Process a single metric message.

        Args:
            message: Metric dictionary with timestamp, service_name, metric_name, value, labels
        """
        # Validate
        if not self._validate(message):
            logger.warning("Invalid metric message", message=message)
            return

        # Normalize
        normalized = self._normalize(message)

        # Add to buffer
        self._buffer.append(normalized)

        # Flush if buffer is full or enough time has passed
        if len(self._buffer) >= self._buffer_size or self._should_flush():
            await self._flush_buffer()

    def _validate(self, message: dict[str, Any]) -> bool:
        """Validate a metric message."""
        # Check required fields
        for field in self._required_fields:
            if field not in message:
                logger.debug("Missing required field", field=field)
                return False

        # Check value is numeric
        try:
            float(message["value"])
        except (TypeError, ValueError):
            logger.debug("Invalid value", value=message.get("value"))
            return False

        # Warn on unknown metric names (but still accept)
        if message["metric_name"] not in self._valid_metric_names:
            logger.debug(
                "Unknown metric name",
                metric_name=message["metric_name"],
            )

        return True

    def _normalize(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize a metric message."""
        # Parse timestamp
        timestamp = message["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Ensure value is float
        value = float(message["value"])

        # Normalize labels
        labels = message.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}

        return {
            "timestamp": timestamp,
            "service_name": message["service_name"],
            "metric_name": message["metric_name"],
            "value": value,
            "labels": labels,
        }

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed based on time."""
        return datetime.now(timezone.utc) - self._last_flush > self._flush_interval

    async def _flush_buffer(self) -> None:
        """Flush buffered metrics to database."""
        if not self._buffer:
            return

        metrics_to_insert = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = datetime.now(timezone.utc)

        try:
            async with get_session() as session:
                repo = MetricsRepository(session)

                # Create Metric objects
                metric_objects = [
                    Metric(
                        timestamp=m["timestamp"],
                        service_name=m["service_name"],
                        metric_name=m["metric_name"],
                        value=m["value"],
                        labels=m["labels"],
                    )
                    for m in metrics_to_insert
                ]

                # Batch insert
                await repo.insert_batch(metric_objects)

                logger.info(
                    "Flushed metrics to database",
                    count=len(metric_objects),
                )

        except Exception as e:
            logger.error("Failed to flush metrics", error=str(e))
            # Re-add to buffer for retry (up to a limit)
            if len(self._buffer) < self._buffer_size * 2:
                self._buffer.extend(metrics_to_insert)

    async def flush(self) -> None:
        """Force flush the buffer."""
        await self._flush_buffer()


class FeatureProcessor(BaseProcessor):
    """
    Processor that triggers feature engineering.

    Responsibilities:
    - Track when new metrics arrive
    - Trigger feature computation when enough data
    - Cache computed features in Redis
    - Store features in database
    """

    def __init__(
        self,
        feature_interval_minutes: int = 1,
        min_metrics_for_features: int = 5,
    ) -> None:
        """
        Initialize feature processor.

        Args:
            feature_interval_minutes: Minutes between feature computations
            min_metrics_for_features: Minimum metrics needed to compute features
        """
        super().__init__(name="feature-processor")

        self.feature_interval = timedelta(minutes=feature_interval_minutes)
        self.min_metrics = min_metrics_for_features

        # Track metrics by service
        self._metric_counts: dict[str, int] = defaultdict(int)
        self._last_feature_time: dict[str, datetime] = {}

        # Feature computation callback (set by external code)
        self._feature_callback: callable | None = None

    def set_feature_callback(
        self,
        callback: callable,
    ) -> None:
        """
        Set the callback for feature computation.

        The callback should accept (service_name, timestamp) and return features.
        """
        self._feature_callback = callback

    async def process(self, message: dict[str, Any]) -> None:
        """
        Process a metric and potentially trigger feature computation.

        Args:
            message: Metric dictionary
        """
        service_name = message.get("service_name", "unknown")
        timestamp = message.get("timestamp")

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Increment metric count
        self._metric_counts[service_name] += 1

        # Check if we should compute features
        if self._should_compute_features(service_name):
            await self._compute_features(service_name, timestamp)

    def _should_compute_features(self, service_name: str) -> bool:
        """Check if features should be computed for a service."""
        # Not enough metrics
        if self._metric_counts[service_name] < self.min_metrics:
            return False

        # Check time since last computation
        last_time = self._last_feature_time.get(service_name)
        if last_time is None:
            return True

        return datetime.now(timezone.utc) - last_time >= self.feature_interval

    async def _compute_features(
        self,
        service_name: str,
        timestamp: datetime,
    ) -> None:
        """Compute and store features for a service."""
        self._metric_counts[service_name] = 0
        self._last_feature_time[service_name] = datetime.now(timezone.utc)

        if self._feature_callback is None:
            logger.warning("No feature callback set")
            return

        try:
            # Call feature computation
            features = await self._feature_callback(service_name, timestamp)

            if features is not None:
                # Store features
                await self._store_features(service_name, timestamp, features)

                logger.info(
                    "Features computed",
                    service_name=service_name,
                    feature_count=len(features) if isinstance(features, dict) else 0,
                )

        except Exception as e:
            logger.error(
                "Feature computation failed",
                service_name=service_name,
                error=str(e),
            )

    async def _store_features(
        self,
        service_name: str,
        timestamp: datetime,
        features: dict[str, Any],
    ) -> None:
        """Store computed features in database."""
        try:
            async with get_session() as session:
                repo = FeaturesRepository(session)

                # Compute feature hash for versioning
                import hashlib
                feature_names = sorted(features.keys())
                feature_hash = hashlib.md5(
                    ",".join(feature_names).encode()
                ).hexdigest()[:8]

                feature_obj = Feature(
                    timestamp=timestamp,
                    service_name=service_name,
                    feature_set_version=feature_hash,
                    features=features,
                )

                await repo.insert(feature_obj)

        except Exception as e:
            logger.error("Failed to store features", error=str(e))


class PredictionTrigger(BaseProcessor):
    """
    Processor that triggers predictions.

    Responsibilities:
    - Monitor feature availability
    - Trigger predictions when sufficient data
    - Coordinate prediction timing
    - Handle prediction priorities (short-term more frequent)
    """

    def __init__(
        self,
        short_term_interval_minutes: int = 1,
        medium_term_interval_minutes: int = 15,
        long_term_interval_minutes: int = 60,
    ) -> None:
        """
        Initialize prediction trigger.

        Args:
            short_term_interval_minutes: Minutes between short-term predictions
            medium_term_interval_minutes: Minutes between medium-term predictions
            long_term_interval_minutes: Minutes between long-term predictions
        """
        super().__init__(name="prediction-trigger")

        self._intervals = {
            "short_term": timedelta(minutes=short_term_interval_minutes),
            "medium_term": timedelta(minutes=medium_term_interval_minutes),
            "long_term": timedelta(minutes=long_term_interval_minutes),
        }

        # Track last prediction time by service and horizon
        self._last_prediction: dict[str, dict[str, datetime]] = defaultdict(dict)

        # Feature count since last prediction
        self._feature_counts: dict[str, int] = defaultdict(int)

        # Prediction callbacks by horizon
        self._prediction_callbacks: dict[str, callable] = {}

    def set_prediction_callback(
        self,
        horizon: str,
        callback: callable,
    ) -> None:
        """
        Set the callback for a prediction horizon.

        Args:
            horizon: "short_term", "medium_term", or "long_term"
            callback: Async function accepting (service_name, timestamp)
        """
        if horizon not in self._intervals:
            raise ValueError(f"Invalid horizon: {horizon}")
        self._prediction_callbacks[horizon] = callback

    async def process(self, message: dict[str, Any]) -> None:
        """
        Process a feature message and trigger predictions if needed.

        Args:
            message: Feature dictionary
        """
        service_name = message.get("service_name", "unknown")
        timestamp = message.get("timestamp")

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Increment feature count
        self._feature_counts[service_name] += 1

        # Check each horizon
        for horizon in self._intervals:
            if self._should_trigger(service_name, horizon):
                await self._trigger_prediction(service_name, horizon, timestamp)

    def _should_trigger(self, service_name: str, horizon: str) -> bool:
        """Check if prediction should be triggered."""
        last_time = self._last_prediction.get(service_name, {}).get(horizon)

        if last_time is None:
            return True

        interval = self._intervals[horizon]
        return datetime.now(timezone.utc) - last_time >= interval

    async def _trigger_prediction(
        self,
        service_name: str,
        horizon: str,
        timestamp: datetime,
    ) -> None:
        """Trigger a prediction for a service and horizon."""
        # Update last prediction time
        if service_name not in self._last_prediction:
            self._last_prediction[service_name] = {}
        self._last_prediction[service_name][horizon] = datetime.now(timezone.utc)

        # Get callback
        callback = self._prediction_callbacks.get(horizon)
        if callback is None:
            logger.debug("No callback for horizon", horizon=horizon)
            return

        try:
            await callback(service_name, timestamp)

            logger.info(
                "Prediction triggered",
                service_name=service_name,
                horizon=horizon,
            )

        except Exception as e:
            logger.error(
                "Prediction trigger failed",
                service_name=service_name,
                horizon=horizon,
                error=str(e),
            )

    async def force_trigger(
        self,
        service_name: str,
        horizon: str | None = None,
    ) -> None:
        """
        Force trigger predictions.

        Args:
            service_name: Service to predict for
            horizon: Specific horizon or None for all
        """
        timestamp = datetime.now(timezone.utc)
        horizons = [horizon] if horizon else list(self._intervals.keys())

        for h in horizons:
            await self._trigger_prediction(service_name, h, timestamp)


class StreamProcessorManager:
    """
    Manages multiple stream processors.

    Provides:
    - Unified interface for all processors
    - Statistics aggregation
    - Lifecycle management
    """

    def __init__(self) -> None:
        """Initialize the processor manager."""
        self.metrics_processor = MetricsProcessor()
        self.feature_processor = FeatureProcessor()
        self.prediction_trigger = PredictionTrigger()

        self._processors = [
            self.metrics_processor,
            self.feature_processor,
            self.prediction_trigger,
        ]

    async def process_metric(self, message: dict[str, Any]) -> None:
        """Process a metric through the metrics processor."""
        await self.metrics_processor.process(message)

    async def process_feature(self, message: dict[str, Any]) -> None:
        """Process a feature through the feature processor."""
        await self.feature_processor.process(message)

    async def process_for_prediction(self, message: dict[str, Any]) -> None:
        """Process a message for prediction triggering."""
        await self.prediction_trigger.process(message)

    async def flush_all(self) -> None:
        """Flush all processor buffers."""
        await self.metrics_processor.flush()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics from all processors."""
        return {p.name: p.get_stats() for p in self._processors}
