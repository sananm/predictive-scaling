"""
Shared test fixtures and configuration.
"""

import asyncio
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# Event Loop Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Time Fixtures
# ============================================================================


@pytest.fixture
def now() -> datetime:
    """Current UTC datetime."""
    return datetime.now(UTC)


@pytest.fixture
def time_index(now: datetime) -> pd.DatetimeIndex:
    """Generate a datetime index for testing (24 hours of minute data)."""
    return pd.date_range(
        start=now - timedelta(hours=24),
        end=now,
        freq="1min",
    )


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_metrics_data(time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate sample metrics data for testing."""
    n = len(time_index)
    np.random.seed(42)

    # Simulate realistic traffic pattern with daily seasonality
    hours = np.array([t.hour for t in time_index])
    base_traffic = 100 + 50 * np.sin(2 * np.pi * hours / 24)  # Daily pattern
    noise = np.random.normal(0, 10, n)

    return pd.DataFrame(
        {
            "requests_per_second": base_traffic + noise,
            "cpu_utilization": np.clip(base_traffic / 2 + np.random.normal(0, 5, n), 0, 100),
            "memory_utilization": np.clip(40 + np.random.normal(0, 10, n), 0, 100),
            "latency_p50": 50 + np.random.exponential(10, n),
            "latency_p99": 100 + np.random.exponential(30, n),
            "error_rate": np.clip(np.random.exponential(0.5, n), 0, 10),
        },
        index=time_index,
    )


@pytest.fixture
def sample_target(sample_metrics_data: pd.DataFrame) -> pd.Series:
    """Target variable for prediction (requests_per_second)."""
    return sample_metrics_data["requests_per_second"]


@pytest.fixture
def sample_features(sample_metrics_data: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for prediction."""
    return sample_metrics_data.drop(columns=["requests_per_second"])


@pytest.fixture
def sample_business_events(now: datetime) -> list[dict[str, Any]]:
    """Sample business events for testing."""
    return [
        {
            "name": "marketing_campaign",
            "event_type": "marketing_campaign",
            "start_time": now - timedelta(hours=2),
            "end_time": now + timedelta(hours=4),
            "expected_impact_multiplier": 1.5,
        },
        {
            "name": "product_launch",
            "event_type": "product_launch",
            "start_time": now + timedelta(days=1),
            "end_time": now + timedelta(days=1, hours=6),
            "expected_impact_multiplier": 3.0,
        },
    ]


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_kafka_producer() -> MagicMock:
    """Mock Kafka producer."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


@pytest.fixture
def mock_kafka_consumer() -> MagicMock:
    """Mock Kafka consumer."""
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    return consumer


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Mock HTTP client for API calls."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_prometheus_response() -> dict[str, Any]:
    """Mock Prometheus API response."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"pod": "app-abc123", "namespace": "production"},
                    "value": [1704067200, "150.5"],
                }
            ],
        },
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Test configuration overrides."""
    return {
        "database_url": "sqlite+aiosqlite:///:memory:",
        "kafka_bootstrap_servers": "localhost:9092",
        "prometheus_url": "http://localhost:9090",
        "log_level": "DEBUG",
    }
