"""
Base collector abstract class.

All collectors inherit from this and implement the collect() method.
The base class handles the background loop, error handling, and publishing.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseCollector(ABC):
    """
    Abstract base class for all metric collectors.

    Provides:
    - Configurable collection interval
    - Background collection loop with error handling
    - Graceful start/stop
    - Publishing interface for sending to Kafka

    Subclasses must implement:
    - collect() -> list[dict]: Gather metrics from the source
    """

    def __init__(
        self,
        name: str,
        collection_interval: float = 15.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        """
        Initialize the collector.

        Args:
            name: Unique name for this collector (used in logs)
            collection_interval: Seconds between collections
            max_retries: Max retries on collection failure
            retry_delay: Seconds to wait between retries
        """
        self.name = name
        self.collection_interval = collection_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._running = False
        self._task: asyncio.Task | None = None
        self._publisher: Any = None  # Will be set to Kafka producer

        # Stats
        self._collections_total = 0
        self._collections_failed = 0
        self._last_collection_time: datetime | None = None
        self._last_error: str | None = None

    @abstractmethod
    async def collect(self) -> list[dict[str, Any]]:
        """
        Collect metrics from the source.

        Returns:
            List of metric dictionaries with keys:
            - timestamp: datetime
            - service_name: str
            - metric_name: str
            - value: float
            - labels: dict (optional)

        Raises:
            Exception: If collection fails
        """
        pass

    def set_publisher(self, publisher: Any) -> None:
        """Set the publisher (Kafka producer) for sending metrics."""
        self._publisher = publisher

    async def publish(self, metrics: list[dict[str, Any]]) -> None:
        """
        Publish collected metrics.

        If a publisher is set, sends to Kafka.
        Otherwise, just logs the metrics.
        """
        if not metrics:
            return

        if self._publisher:
            # Send to Kafka
            for metric in metrics:
                await self._publisher.send(metric)
            logger.debug(
                "Published metrics",
                collector=self.name,
                count=len(metrics),
            )
        else:
            # No publisher, just log
            logger.debug(
                "Metrics collected (no publisher)",
                collector=self.name,
                count=len(metrics),
            )

    async def _collection_loop(self) -> None:
        """Background loop that runs collect() periodically."""
        logger.info(
            "Collector started",
            collector=self.name,
            interval=self.collection_interval,
        )

        while self._running:
            try:
                # Collect with retries
                metrics = await self._collect_with_retry()

                if metrics:
                    await self.publish(metrics)
                    self._collections_total += 1
                    self._last_collection_time = datetime.now(timezone.utc)
                    self._last_error = None

            except Exception as e:
                self._collections_failed += 1
                self._last_error = str(e)
                logger.error(
                    "Collection failed after retries",
                    collector=self.name,
                    error=str(e),
                )

            # Wait for next collection
            await asyncio.sleep(self.collection_interval)

    async def _collect_with_retry(self) -> list[dict[str, Any]]:
        """Run collect() with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await self.collect()
            except Exception as e:
                last_error = e
                logger.warning(
                    "Collection attempt failed",
                    collector=self.name,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        # All retries failed
        if last_error:
            raise last_error
        return []

    async def start(self) -> None:
        """Start the background collection loop."""
        if self._running:
            logger.warning("Collector already running", collector=self.name)
            return

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        logger.info("Collector started", collector=self.name)

    async def stop(self) -> None:
        """Stop the background collection loop gracefully."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Collector stopped", collector=self.name)

    def get_stats(self) -> dict[str, Any]:
        """Get collector statistics."""
        return {
            "name": self.name,
            "running": self._running,
            "collection_interval": self.collection_interval,
            "collections_total": self._collections_total,
            "collections_failed": self._collections_failed,
            "last_collection_time": self._last_collection_time.isoformat()
            if self._last_collection_time
            else None,
            "last_error": self._last_error,
        }

    async def __aenter__(self) -> "BaseCollector":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
