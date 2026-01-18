"""
Async Kafka consumer for processing streaming data.

Features:
- Consumer group management for scalability
- Configurable offset management (auto-commit or manual)
- Message deserialization with error handling
- Processing callback registration
- Graceful shutdown with offset commit
"""

import asyncio
import json
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.errors import KafkaConnectionError, KafkaError

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Type alias for message handlers
MessageHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


def deserialize_message(data: bytes) -> dict[str, Any]:
    """Deserialize a JSON message from bytes."""
    return json.loads(data.decode("utf-8"))


class MetricsConsumer:
    """
    Async Kafka consumer for metrics processing.

    Handles:
    - Consumer group coordination
    - Offset management (auto or manual commit)
    - Message deserialization
    - Callback-based message processing
    - Graceful shutdown
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str | None = None,
        bootstrap_servers: str | None = None,
        auto_commit: bool = True,
        auto_commit_interval_ms: int = 5000,
        max_poll_records: int = 500,
        session_timeout_ms: int = 30000,
        heartbeat_interval_ms: int = 10000,
        auto_offset_reset: str = "latest",
    ) -> None:
        """
        Initialize the metrics consumer.

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            bootstrap_servers: Kafka bootstrap servers
            auto_commit: Whether to auto-commit offsets
            auto_commit_interval_ms: Auto-commit interval
            max_poll_records: Maximum records per poll
            session_timeout_ms: Session timeout
            heartbeat_interval_ms: Heartbeat interval
            auto_offset_reset: Where to start if no offset ("earliest" or "latest")
        """
        self.topics = topics
        self.group_id = group_id or settings.kafka.consumer_group
        self.bootstrap_servers = bootstrap_servers or settings.kafka.bootstrap_servers
        self.auto_commit = auto_commit

        self._consumer: AIOKafkaConsumer | None = None
        self._started = False
        self._running = False
        self._lock = asyncio.Lock()
        self._consume_task: asyncio.Task | None = None

        # Consumer configuration
        self._config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": self.group_id,
            "value_deserializer": deserialize_message,
            "key_deserializer": lambda k: k.decode("utf-8") if k else None,
            "enable_auto_commit": auto_commit,
            "auto_commit_interval_ms": auto_commit_interval_ms,
            "max_poll_records": max_poll_records,
            "session_timeout_ms": session_timeout_ms,
            "heartbeat_interval_ms": heartbeat_interval_ms,
            "auto_offset_reset": auto_offset_reset,
        }

        # Message handlers by topic
        self._handlers: dict[str, list[MessageHandler]] = {topic: [] for topic in topics}
        self._default_handler: MessageHandler | None = None

        # Metrics
        self._messages_processed = 0
        self._messages_failed = 0
        self._last_message_time: datetime | None = None

    async def start(self) -> None:
        """Start the consumer and subscribe to topics."""
        async with self._lock:
            if self._started:
                return

            try:
                self._consumer = AIOKafkaConsumer(*self.topics, **self._config)
                await self._consumer.start()
                self._started = True

                logger.info(
                    "Kafka consumer started",
                    topics=self.topics,
                    group_id=self.group_id,
                    bootstrap_servers=self.bootstrap_servers,
                )

            except KafkaConnectionError as e:
                logger.error("Failed to connect to Kafka", error=str(e))
                raise

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        async with self._lock:
            self._running = False

            # Cancel consume task if running
            if self._consume_task and not self._consume_task.done():
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass

            if not self._started or not self._consumer:
                return

            try:
                # Commit final offsets if not auto-committing
                if not self.auto_commit:
                    await self._consumer.commit()

                await self._consumer.stop()

                logger.info(
                    "Kafka consumer stopped",
                    messages_processed=self._messages_processed,
                    messages_failed=self._messages_failed,
                )

            except Exception as e:
                logger.error("Error stopping consumer", error=str(e))

            finally:
                self._consumer = None
                self._started = False

    def register_handler(
        self,
        handler: MessageHandler,
        topic: str | None = None,
    ) -> None:
        """
        Register a message handler.

        Args:
            handler: Async function to process messages
            topic: Specific topic (None for default handler)
        """
        if topic is None:
            self._default_handler = handler
            logger.info("Registered default message handler")
        else:
            if topic not in self._handlers:
                self._handlers[topic] = []
            self._handlers[topic].append(handler)
            logger.info("Registered handler for topic", topic=topic)

    async def consume(self) -> None:
        """
        Start consuming messages in a loop.

        This method runs until stop() is called.
        """
        if not self._started or not self._consumer:
            raise RuntimeError("Consumer not started")

        self._running = True
        logger.info("Starting message consumption loop")

        try:
            async for message in self._consumer:
                if not self._running:
                    break

                await self._process_message(message)

        except asyncio.CancelledError:
            logger.info("Consume loop cancelled")
        except Exception as e:
            logger.error("Error in consume loop", error=str(e))
            raise

    async def consume_batch(
        self,
        timeout_ms: int = 1000,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Consume a batch of messages.

        Args:
            timeout_ms: Maximum time to wait for messages
            max_records: Maximum number of records to return

        Returns:
            List of message values
        """
        if not self._started or not self._consumer:
            raise RuntimeError("Consumer not started")

        messages = []

        try:
            data = await self._consumer.getmany(
                timeout_ms=timeout_ms,
                max_records=max_records,
            )

            for _tp, records in data.items():
                for record in records:
                    messages.append(record.value)
                    self._messages_processed += 1
                    self._last_message_time = datetime.now()

        except Exception as e:
            logger.error("Error consuming batch", error=str(e))

        return messages

    async def _process_message(self, message: Any) -> None:
        """Process a single message through registered handlers."""
        try:
            topic = message.topic
            value = message.value

            self._last_message_time = datetime.now()

            # Get handlers for this topic
            handlers = self._handlers.get(topic, [])

            # Add default handler if no topic-specific handlers
            if not handlers and self._default_handler:
                handlers = [self._default_handler]

            if not handlers:
                logger.warning(
                    "No handlers for message",
                    topic=topic,
                    partition=message.partition,
                    offset=message.offset,
                )
                return

            # Run all handlers
            for handler in handlers:
                try:
                    await handler(value)
                except Exception as e:
                    logger.error(
                        "Handler error",
                        topic=topic,
                        handler=handler.__name__,
                        error=str(e),
                    )
                    self._messages_failed += 1

            self._messages_processed += 1

            logger.debug(
                "Message processed",
                topic=topic,
                partition=message.partition,
                offset=message.offset,
            )

        except Exception as e:
            self._messages_failed += 1
            logger.error(
                "Error processing message",
                topic=message.topic,
                error=str(e),
            )

    async def commit(self) -> None:
        """Manually commit current offsets."""
        if not self._started or not self._consumer:
            return

        try:
            await self._consumer.commit()
            logger.debug("Offsets committed")
        except KafkaError as e:
            logger.error("Failed to commit offsets", error=str(e))

    async def seek_to_beginning(self, topic: str | None = None) -> None:
        """
        Seek to the beginning of topics.

        Args:
            topic: Specific topic (None for all)
        """
        if not self._started or not self._consumer:
            return

        if topic:
            partitions = self._consumer.partitions_for_topic(topic)
            if partitions:
                tps = [TopicPartition(topic, p) for p in partitions]
                await self._consumer.seek_to_beginning(*tps)
        else:
            await self._consumer.seek_to_beginning()

        logger.info("Seeked to beginning", topic=topic or "all")

    async def seek_to_end(self, topic: str | None = None) -> None:
        """
        Seek to the end of topics.

        Args:
            topic: Specific topic (None for all)
        """
        if not self._started or not self._consumer:
            return

        if topic:
            partitions = self._consumer.partitions_for_topic(topic)
            if partitions:
                tps = [TopicPartition(topic, p) for p in partitions]
                await self._consumer.seek_to_end(*tps)
        else:
            await self._consumer.seek_to_end()

        logger.info("Seeked to end", topic=topic or "all")

    def get_stats(self) -> dict[str, Any]:
        """Get consumer statistics."""
        return {
            "started": self._started,
            "running": self._running,
            "messages_processed": self._messages_processed,
            "messages_failed": self._messages_failed,
            "last_message_time": self._last_message_time.isoformat() if self._last_message_time else None,
            "topics": self.topics,
            "group_id": self.group_id,
        }

    async def run_forever(self) -> None:
        """
        Start consuming and run until stopped.

        This is a convenience method that starts the consumer
        and runs the consume loop in the background.
        """
        await self.start()
        self._consume_task = asyncio.create_task(self.consume())

    async def __aenter__(self) -> "MetricsConsumer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


class MultiTopicConsumer:
    """
    Consumer that handles multiple topics with different handlers.

    Useful when you need to consume from multiple topics
    with specialized processing for each.
    """

    def __init__(
        self,
        group_id: str | None = None,
        bootstrap_servers: str | None = None,
    ) -> None:
        """
        Initialize multi-topic consumer.

        Args:
            group_id: Consumer group ID
            bootstrap_servers: Kafka bootstrap servers
        """
        self.group_id = group_id or settings.kafka.consumer_group
        self.bootstrap_servers = bootstrap_servers or settings.kafka.bootstrap_servers

        self._topics: list[str] = []
        self._handlers: dict[str, MessageHandler] = {}
        self._consumer: MetricsConsumer | None = None

    def subscribe(self, topic: str, handler: MessageHandler) -> "MultiTopicConsumer":
        """
        Subscribe to a topic with a handler.

        Args:
            topic: Topic to subscribe to
            handler: Handler for messages from this topic

        Returns:
            Self for chaining
        """
        self._topics.append(topic)
        self._handlers[topic] = handler
        return self

    async def start(self) -> None:
        """Start the consumer with all subscribed topics."""
        if not self._topics:
            raise ValueError("No topics subscribed")

        self._consumer = MetricsConsumer(
            topics=self._topics,
            group_id=self.group_id,
            bootstrap_servers=self.bootstrap_servers,
        )

        # Register handlers
        for topic, handler in self._handlers.items():
            self._consumer.register_handler(handler, topic)

        await self._consumer.start()

    async def stop(self) -> None:
        """Stop the consumer."""
        if self._consumer:
            await self._consumer.stop()

    async def run_forever(self) -> None:
        """Start and run until stopped."""
        await self.start()
        if self._consumer:
            await self._consumer.consume()

    async def __aenter__(self) -> "MultiTopicConsumer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
