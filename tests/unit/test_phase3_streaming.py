"""
Phase 3 Tests: Streaming Pipeline

Tests for:
- Kafka producer
- Kafka consumer
- Stream processors
"""

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest


class TestMetricsProducer:
    """Test Kafka producer."""

    def test_producer_import(self):
        """Test MetricsProducer can be imported."""
        from src.streaming.producer import MetricsProducer

        assert MetricsProducer is not None

    def test_producer_initialization(self):
        """Test MetricsProducer can be initialized."""
        from src.streaming.producer import MetricsProducer

        producer = MetricsProducer(
            bootstrap_servers="localhost:9092",
        )

        assert producer.bootstrap_servers == "localhost:9092"

    def test_producer_config_defaults(self):
        """Test producer configuration defaults."""
        from src.streaming.producer import MetricsProducer

        producer = MetricsProducer()

        # Config should be stored in _config dict
        assert "_config" in dir(producer) or hasattr(producer, "_config")
        # Should have metrics tracking
        assert producer._messages_sent == 0
        assert producer._messages_failed == 0

    def test_datetime_encoder(self):
        """Test custom JSON datetime encoder."""
        from src.streaming.producer import DateTimeEncoder

        encoder = DateTimeEncoder()

        # Test datetime encoding
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = encoder.default(dt)
        assert "2024-01-15" in result

        # Test UUID encoding
        from uuid import uuid4
        test_uuid = uuid4()
        result = encoder.default(test_uuid)
        assert result == str(test_uuid)

    def test_serialize_message_function(self):
        """Test message serialization function."""
        from src.streaming.producer import serialize_message

        message = {
            "metric_name": "cpu_usage",
            "value": 75.5,
            "timestamp": datetime.now(timezone.utc),
        }

        serialized = serialize_message(message)

        assert isinstance(serialized, bytes)
        # Should be valid JSON
        deserialized = json.loads(serialized)
        assert deserialized["metric_name"] == "cpu_usage"
        assert deserialized["value"] == 75.5

    def test_dlq_functionality(self):
        """Test dead letter queue functionality."""
        from src.streaming.producer import MetricsProducer

        producer = MetricsProducer()

        # Add to DLQ
        producer._add_to_dlq("test-topic", {"test": "message"}, "Test error")

        assert len(producer._dlq) == 1
        assert producer._dlq[0]["topic"] == "test-topic"
        assert producer._dlq[0]["error"] == "Test error"

    def test_dlq_max_size(self):
        """Test DLQ max size limit."""
        from src.streaming.producer import MetricsProducer

        producer = MetricsProducer()

        # Fill DLQ beyond max size
        for i in range(producer._max_dlq_size + 10):
            producer._add_to_dlq("test-topic", {"index": i}, f"Error {i}")

        # Should not exceed max size
        assert len(producer._dlq) <= producer._max_dlq_size

    def test_producer_stats(self):
        """Test producer statistics."""
        from src.streaming.producer import MetricsProducer

        producer = MetricsProducer()
        stats = producer.get_stats()

        assert "started" in stats
        assert "messages_sent" in stats
        assert "messages_failed" in stats
        assert "bytes_sent" in stats
        assert "dlq_size" in stats

    @pytest.mark.asyncio
    async def test_producer_context_manager(self):
        """Test producer as async context manager."""
        from src.streaming.producer import MetricsProducer

        producer = MetricsProducer()

        assert hasattr(producer, "__aenter__")
        assert hasattr(producer, "__aexit__")


class TestMetricsConsumer:
    """Test Kafka consumer."""

    def test_consumer_import(self):
        """Test MetricsConsumer can be imported."""
        from src.streaming.consumer import MetricsConsumer

        assert MetricsConsumer is not None

    def test_consumer_initialization(self):
        """Test MetricsConsumer can be initialized."""
        from src.streaming.consumer import MetricsConsumer

        consumer = MetricsConsumer(
            bootstrap_servers="localhost:9092",
            topics=["metrics.raw"],
            group_id="test-group",
        )

        assert consumer.bootstrap_servers == "localhost:9092"
        assert "metrics.raw" in consumer.topics
        assert consumer.group_id == "test-group"

    def test_consumer_config_defaults(self):
        """Test consumer configuration defaults."""
        from src.streaming.consumer import MetricsConsumer

        consumer = MetricsConsumer(topics=["test"])

        # Config should be stored in _config dict
        assert hasattr(consumer, "_config")
        assert consumer._config["auto_offset_reset"] in ["latest", "earliest"]

    def test_register_handler(self):
        """Test handler registration for specific topic."""
        from src.streaming.consumer import MetricsConsumer

        consumer = MetricsConsumer(topics=["test"])

        async def test_handler(message):
            pass

        consumer.register_handler(test_handler, topic="test")

        assert "test" in consumer._handlers
        assert test_handler in consumer._handlers["test"]

    def test_register_default_handler(self):
        """Test default handler registration."""
        from src.streaming.consumer import MetricsConsumer

        consumer = MetricsConsumer(topics=["test"])

        async def default_handler(message):
            pass

        consumer.register_handler(default_handler)  # No topic = default

        assert consumer._default_handler == default_handler

    def test_deserialize_message_function(self):
        """Test message deserialization function."""
        from src.streaming.consumer import deserialize_message

        raw_data = json.dumps({"metric": "test", "value": 42}).encode()

        result = deserialize_message(raw_data)

        assert result["metric"] == "test"
        assert result["value"] == 42

    def test_consumer_stats(self):
        """Test consumer statistics tracking."""
        from src.streaming.consumer import MetricsConsumer

        consumer = MetricsConsumer(topics=["test"])

        stats = consumer.get_stats()

        assert "messages_processed" in stats
        assert "messages_failed" in stats
        assert "started" in stats
        assert "topics" in stats

    @pytest.mark.asyncio
    async def test_consumer_context_manager(self):
        """Test consumer as async context manager."""
        from src.streaming.consumer import MetricsConsumer

        consumer = MetricsConsumer(topics=["test"])

        assert hasattr(consumer, "__aenter__")
        assert hasattr(consumer, "__aexit__")


class TestMultiTopicConsumer:
    """Test multi-topic consumer."""

    def test_multi_topic_consumer_import(self):
        """Test MultiTopicConsumer can be imported."""
        from src.streaming.consumer import MultiTopicConsumer

        assert MultiTopicConsumer is not None

    def test_multi_topic_consumer_subscribe(self):
        """Test subscribing to topics."""
        from src.streaming.consumer import MultiTopicConsumer

        consumer = MultiTopicConsumer()

        async def handler1(msg): pass
        async def handler2(msg): pass

        consumer.subscribe("topic1", handler1).subscribe("topic2", handler2)

        assert "topic1" in consumer._topics
        assert "topic2" in consumer._topics
        assert consumer._handlers["topic1"] == handler1
        assert consumer._handlers["topic2"] == handler2


class TestStreamProcessors:
    """Test stream processors."""

    def test_base_processor_import(self):
        """Test BaseProcessor can be imported."""
        from src.streaming.processors import BaseProcessor

        assert BaseProcessor is not None

    def test_metrics_processor_import(self):
        """Test MetricsProcessor can be imported."""
        from src.streaming.processors import MetricsProcessor

        assert MetricsProcessor is not None

    def test_feature_processor_import(self):
        """Test FeatureProcessor can be imported."""
        from src.streaming.processors import FeatureProcessor

        assert FeatureProcessor is not None

    def test_prediction_trigger_import(self):
        """Test PredictionTrigger can be imported."""
        from src.streaming.processors import PredictionTrigger

        assert PredictionTrigger is not None

    def test_metrics_processor_initialization(self):
        """Test MetricsProcessor initialization."""
        from src.streaming.processors import MetricsProcessor

        processor = MetricsProcessor()

        # Check internal attributes
        assert hasattr(processor, "_buffer")
        assert hasattr(processor, "_buffer_size")
        assert processor._buffer_size > 0

    def test_metrics_processor_validation(self):
        """Test MetricsProcessor message validation."""
        from src.streaming.processors import MetricsProcessor

        processor = MetricsProcessor()

        # Valid message
        valid_msg = {
            "timestamp": "2024-01-15T10:30:00Z",
            "service_name": "test",
            "metric_name": "cpu",
            "value": 75.5,
        }
        assert processor._validate(valid_msg) is True

        # Invalid message (missing required field)
        invalid_msg = {"timestamp": "2024-01-15T10:30:00Z"}
        assert processor._validate(invalid_msg) is False

    def test_metrics_processor_buffering(self):
        """Test MetricsProcessor buffering behavior."""
        from src.streaming.processors import MetricsProcessor

        processor = MetricsProcessor()

        # Add metrics to buffer
        for i in range(3):
            processor._buffer.append({"value": i})

        assert len(processor._buffer) == 3

    def test_metrics_processor_normalize(self):
        """Test MetricsProcessor message normalization."""
        from src.streaming.processors import MetricsProcessor

        processor = MetricsProcessor()

        message = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "service_name": "test",
            "metric_name": "cpu",
            "value": "75.5",  # String value
        }

        normalized = processor._normalize(message)

        assert isinstance(normalized["value"], float)
        assert isinstance(normalized["timestamp"], datetime)
        assert normalized["service_name"] == "test"

    def test_prediction_trigger_intervals(self):
        """Test PredictionTrigger interval configuration."""
        from src.streaming.processors import PredictionTrigger

        trigger = PredictionTrigger()

        # Check intervals are defined
        assert hasattr(trigger, "_intervals")
        assert "short_term" in trigger._intervals
        assert "medium_term" in trigger._intervals
        assert "long_term" in trigger._intervals

    def test_prediction_trigger_set_callback(self):
        """Test setting prediction callbacks."""
        from src.streaming.processors import PredictionTrigger

        trigger = PredictionTrigger()

        async def callback(service_name, timestamp): pass

        trigger.set_prediction_callback("short_term", callback)

        assert trigger._prediction_callbacks["short_term"] == callback

    def test_prediction_trigger_invalid_horizon(self):
        """Test invalid horizon raises error."""
        from src.streaming.processors import PredictionTrigger

        trigger = PredictionTrigger()

        async def callback(service_name, timestamp): pass

        with pytest.raises(ValueError):
            trigger.set_prediction_callback("invalid_horizon", callback)

    def test_processor_stats(self):
        """Test processor statistics."""
        from src.streaming.processors import MetricsProcessor

        processor = MetricsProcessor()
        stats = processor.get_stats()

        assert "name" in stats
        assert "processed_count" in stats
        assert "error_count" in stats


class TestStreamProcessorManager:
    """Test stream processor manager."""

    def test_manager_import(self):
        """Test StreamProcessorManager can be imported."""
        from src.streaming.processors import StreamProcessorManager

        assert StreamProcessorManager is not None

    def test_manager_initialization(self):
        """Test StreamProcessorManager initialization."""
        from src.streaming.processors import StreamProcessorManager

        manager = StreamProcessorManager()

        assert manager.metrics_processor is not None
        assert manager.feature_processor is not None
        assert manager.prediction_trigger is not None

    def test_manager_get_all_stats(self):
        """Test getting stats from all processors."""
        from src.streaming.processors import StreamProcessorManager

        manager = StreamProcessorManager()
        stats = manager.get_all_stats()

        assert "metrics-processor" in stats
        assert "feature-processor" in stats
        assert "prediction-trigger" in stats


class TestStreamingIntegration:
    """Test streaming integration patterns."""

    def test_producer_consumer_topic_alignment(self):
        """Test that producer and consumer can use same topics."""
        from src.streaming.producer import MetricsProducer
        from src.streaming.consumer import MetricsConsumer

        topic = "metrics.raw"

        producer = MetricsProducer()
        consumer = MetricsConsumer(topics=[topic])

        assert topic in consumer.topics

    @pytest.mark.asyncio
    async def test_processor_pipeline(self):
        """Test processor pipeline pattern."""
        from src.streaming.processors import MetricsProcessor, FeatureProcessor

        metrics_processor = MetricsProcessor()
        feature_processor = FeatureProcessor()

        # Simulate message flow
        message = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_name": "test",
            "metric_name": "cpu",
            "value": 75.5,
        }

        # Process through metrics processor
        if metrics_processor._validate(message):
            metrics_processor._buffer.append(message)

        assert len(metrics_processor._buffer) == 1
