"""
Streaming pipeline for real-time data processing.

Components:
- Producer: Async Kafka producer with batching and error handling
- Consumer: Async Kafka consumer with offset management
- Processors: Stream processors for metrics, features, and predictions

Data Flow:
    Collectors → Producer → Kafka Topics → Consumer → Processors → Storage/ML
"""

from .consumer import MetricsConsumer, MultiTopicConsumer
from .processors import (
    BaseProcessor,
    FeatureProcessor,
    MetricsProcessor,
    PredictionTrigger,
    StreamProcessorManager,
)
from .producer import MetricsProducer

__all__ = [
    # Producer
    "MetricsProducer",
    # Consumer
    "MetricsConsumer",
    "MultiTopicConsumer",
    # Processors
    "BaseProcessor",
    "MetricsProcessor",
    "FeatureProcessor",
    "PredictionTrigger",
    "StreamProcessorManager",
]
