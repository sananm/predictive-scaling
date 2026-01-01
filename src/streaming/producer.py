"""
Async Kafka producer for streaming metrics.

Features:
- Connection management with automatic reconnection
- JSON message serialization with datetime handling
- Batching for efficiency
- Dead letter queue support for failed messages
- Throughput and error metrics
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and UUID objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def serialize_message(message: dict[str, Any]) -> bytes:
    """Serialize a message to JSON bytes."""
    return json.dumps(message, cls=DateTimeEncoder).encode("utf-8")


class MetricsProducer:
    """
    Async Kafka producer for metrics streaming.

    Handles:
    - Connection lifecycle and reconnection
    - Message batching for throughput
    - Error handling with dead letter queue
    - Graceful shutdown with message flushing
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        client_id: str = "predictive-scaler-producer",
        batch_size: int = 16384,  # 16KB
        linger_ms: int = 10,  # Wait up to 10ms to batch
        compression_type: str = "gzip",
        max_request_size: int = 1048576,  # 1MB
        acks: str = "all",  # Wait for all replicas
        retries: int = 5,
        retry_backoff_ms: int = 100,
    ) -> None:
        """
        Initialize the metrics producer.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            client_id: Client identifier
            batch_size: Maximum batch size in bytes
            linger_ms: Time to wait for batching
            compression_type: Compression algorithm (gzip, snappy, lz4)
            max_request_size: Maximum request size
            acks: Acknowledgment level (0, 1, "all")
            retries: Number of retries on failure
            retry_backoff_ms: Backoff between retries
        """
        self.bootstrap_servers = bootstrap_servers or settings.kafka.bootstrap_servers

        self._producer: AIOKafkaProducer | None = None
        self._started = False
        self._lock = asyncio.Lock()

        # Producer configuration
        self._config = {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": client_id,
            "value_serializer": serialize_message,
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "compression_type": compression_type,
            "max_batch_size": batch_size,
            "linger_ms": linger_ms,
            "max_request_size": max_request_size,
            "acks": acks,
            "retries": retries,
            "retry_backoff_ms": retry_backoff_ms,
        }

        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._bytes_sent = 0

        # Dead letter queue
        self._dlq: list[dict[str, Any]] = []
        self._max_dlq_size = 1000

    async def start(self) -> None:
        """Start the producer and connect to Kafka."""
        async with self._lock:
            if self._started:
                return

            try:
                self._producer = AIOKafkaProducer(**self._config)
                await self._producer.start()
                self._started = True

                logger.info(
                    "Kafka producer started",
                    bootstrap_servers=self.bootstrap_servers,
                )

            except KafkaConnectionError as e:
                logger.error("Failed to connect to Kafka", error=str(e))
                raise

    async def stop(self) -> None:
        """Stop the producer and flush pending messages."""
        async with self._lock:
            if not self._started or not self._producer:
                return

            try:
                # Flush any remaining messages
                await self._producer.flush()
                await self._producer.stop()

                logger.info(
                    "Kafka producer stopped",
                    messages_sent=self._messages_sent,
                    messages_failed=self._messages_failed,
                    bytes_sent=self._bytes_sent,
                )

            except Exception as e:
                logger.error("Error stopping producer", error=str(e))

            finally:
                self._producer = None
                self._started = False

    async def send(
        self,
        topic: str,
        message: dict[str, Any],
        key: str | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> bool:
        """
        Send a single message to a Kafka topic.

        Args:
            topic: Target topic
            message: Message payload (will be JSON serialized)
            key: Optional message key for partitioning
            headers: Optional message headers

        Returns:
            True if message was sent successfully
        """
        if not self._started or not self._producer:
            logger.warning("Producer not started, message dropped")
            self._add_to_dlq(topic, message, "producer_not_started")
            return False

        try:
            # Add metadata
            message["_produced_at"] = datetime.now(timezone.utc).isoformat()

            # Send message
            future = await self._producer.send(
                topic=topic,
                value=message,
                key=key,
                headers=headers,
            )

            # Wait for acknowledgment
            record_metadata = await future

            self._messages_sent += 1
            self._bytes_sent += len(serialize_message(message))

            logger.debug(
                "Message sent",
                topic=topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
            )

            return True

        except KafkaError as e:
            self._messages_failed += 1
            self._add_to_dlq(topic, message, str(e))

            logger.error(
                "Failed to send message",
                topic=topic,
                error=str(e),
            )
            return False

    async def send_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
        key_func: callable | None = None,
    ) -> tuple[int, int]:
        """
        Send a batch of messages to a Kafka topic.

        Args:
            topic: Target topic
            messages: List of message payloads
            key_func: Optional function to extract key from message

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not messages:
            return 0, 0

        if not self._started or not self._producer:
            logger.warning("Producer not started, batch dropped")
            for msg in messages:
                self._add_to_dlq(topic, msg, "producer_not_started")
            return 0, len(messages)

        successful = 0
        failed = 0
        futures = []

        # Send all messages (they'll be batched by the producer)
        for message in messages:
            try:
                message["_produced_at"] = datetime.now(timezone.utc).isoformat()

                key = key_func(message) if key_func else None

                future = await self._producer.send(
                    topic=topic,
                    value=message,
                    key=key,
                )
                futures.append((future, message))

            except Exception as e:
                failed += 1
                self._add_to_dlq(topic, message, str(e))

        # Wait for all acknowledgments
        for future, message in futures:
            try:
                await future
                successful += 1
                self._messages_sent += 1
                self._bytes_sent += len(serialize_message(message))

            except Exception as e:
                failed += 1
                self._messages_failed += 1
                self._add_to_dlq(topic, message, str(e))

        logger.info(
            "Batch sent",
            topic=topic,
            successful=successful,
            failed=failed,
        )

        return successful, failed

    async def send_metrics(self, metrics: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Send metrics to the metrics topic.

        Args:
            metrics: List of metric dictionaries

        Returns:
            Tuple of (successful_count, failed_count)
        """
        return await self.send_batch(
            topic=settings.kafka.metrics_topic,
            messages=metrics,
            key_func=lambda m: m.get("service_name", "unknown"),
        )

    async def send_features(self, features: dict[str, Any]) -> bool:
        """
        Send computed features to the features topic.

        Args:
            features: Feature dictionary

        Returns:
            True if sent successfully
        """
        return await self.send(
            topic=settings.kafka.features_topic,
            message=features,
            key=features.get("service_name", "unknown"),
        )

    async def send_prediction(self, prediction: dict[str, Any]) -> bool:
        """
        Send a prediction to the predictions topic.

        Args:
            prediction: Prediction dictionary

        Returns:
            True if sent successfully
        """
        return await self.send(
            topic=settings.kafka.predictions_topic,
            message=prediction,
            key=prediction.get("service_name", "unknown"),
        )

    def _add_to_dlq(
        self,
        topic: str,
        message: dict[str, Any],
        error: str,
    ) -> None:
        """Add a failed message to the dead letter queue."""
        if len(self._dlq) >= self._max_dlq_size:
            # Remove oldest entry
            self._dlq.pop(0)

        self._dlq.append({
            "topic": topic,
            "message": message,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def retry_dlq(self) -> tuple[int, int]:
        """
        Retry sending messages from the dead letter queue.

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not self._dlq:
            return 0, 0

        messages_to_retry = self._dlq.copy()
        self._dlq.clear()

        successful = 0
        failed = 0

        for entry in messages_to_retry:
            success = await self.send(
                topic=entry["topic"],
                message=entry["message"],
            )
            if success:
                successful += 1
            else:
                failed += 1

        logger.info(
            "DLQ retry complete",
            successful=successful,
            failed=failed,
            remaining=len(self._dlq),
        )

        return successful, failed

    def get_stats(self) -> dict[str, Any]:
        """Get producer statistics."""
        return {
            "started": self._started,
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "bytes_sent": self._bytes_sent,
            "dlq_size": len(self._dlq),
        }

    async def __aenter__(self) -> "MetricsProducer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
