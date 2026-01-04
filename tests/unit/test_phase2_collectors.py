"""
Phase 2 Tests: Data Collectors

Tests for:
- Base collector pattern
- Prometheus collector
- Kubernetes collector
- Business context collector
- External signals collector
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBaseCollector:
    """Test base collector pattern."""

    def test_base_collector_import(self):
        """Test BaseCollector can be imported."""
        from src.collectors.base import BaseCollector

        assert BaseCollector is not None

    def test_base_collector_is_abstract(self):
        """Test that BaseCollector cannot be instantiated directly."""
        from src.collectors.base import BaseCollector

        with pytest.raises(TypeError):
            BaseCollector(name="test")

    def test_base_collector_attributes(self):
        """Test BaseCollector has expected attributes."""
        from src.collectors.base import BaseCollector

        # Check for expected methods
        assert hasattr(BaseCollector, "collect")
        assert hasattr(BaseCollector, "start")
        assert hasattr(BaseCollector, "stop")
        assert hasattr(BaseCollector, "publish")
        assert hasattr(BaseCollector, "get_stats")


class TestPrometheusCollector:
    """Test Prometheus metrics collector."""

    def test_prometheus_collector_import(self):
        """Test PrometheusCollector can be imported."""
        from src.collectors.prometheus import PrometheusCollector

        assert PrometheusCollector is not None

    def test_prometheus_collector_initialization(self):
        """Test PrometheusCollector can be initialized."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(
            service_name="test-service",
            prometheus_url="http://localhost:9090",
        )

        assert collector.service_name == "test-service"
        assert collector.prometheus_url == "http://localhost:9090"

    def test_prometheus_queries_defined(self):
        """Test that Prometheus queries are defined."""
        from src.collectors.prometheus import PrometheusCollector, DEFAULT_QUERIES

        collector = PrometheusCollector(service_name="test")

        # Should have default queries (dict format)
        assert len(collector.queries) > 0

        # Check for essential metrics
        assert "requests_per_second" in collector.queries

    def test_prometheus_default_queries(self):
        """Test that DEFAULT_QUERIES constant exists."""
        from src.collectors.prometheus import DEFAULT_QUERIES

        assert isinstance(DEFAULT_QUERIES, dict)
        assert "requests_per_second" in DEFAULT_QUERIES
        assert "latency_p50" in DEFAULT_QUERIES
        assert "error_rate" in DEFAULT_QUERIES

    def test_prometheus_custom_queries(self):
        """Test PrometheusCollector with custom queries."""
        from src.collectors.prometheus import PrometheusCollector

        custom_queries = {
            "custom_metric": 'sum(rate(my_metric_total[5m]))',
        }
        collector = PrometheusCollector(
            service_name="test",
            queries=custom_queries,
        )

        assert collector.queries == custom_queries

    def test_prometheus_add_query(self):
        """Test adding a query dynamically."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(service_name="test")
        collector.add_query("new_metric", 'sum(new_metric_total)')

        assert "new_metric" in collector.queries

    def test_prometheus_remove_query(self):
        """Test removing a query dynamically."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(service_name="test")
        initial_count = len(collector.queries)

        collector.remove_query("requests_per_second")

        assert len(collector.queries) == initial_count - 1
        assert "requests_per_second" not in collector.queries


class TestKubernetesCollector:
    """Test Kubernetes metrics collector."""

    def test_kubernetes_collector_import(self):
        """Test KubernetesCollector can be imported."""
        from src.collectors.kubernetes import KubernetesCollector

        assert KubernetesCollector is not None

    def test_kubernetes_collector_initialization(self):
        """Test KubernetesCollector can be initialized."""
        from src.collectors.kubernetes import KubernetesCollector

        collector = KubernetesCollector(
            service_name="test-service",
            namespace="default",
            in_cluster=False,
        )

        assert collector.service_name == "test-service"
        assert collector.namespace == "default"
        assert collector.in_cluster is False

    def test_parse_cpu_resources(self):
        """Test CPU resource parsing."""
        from src.collectors.kubernetes import KubernetesCollector

        collector = KubernetesCollector(service_name="test")

        # Test millicores
        assert collector._parse_cpu("100m") == 0.1
        assert collector._parse_cpu("1000m") == 1.0

        # Test whole cores
        assert collector._parse_cpu("2") == 2.0
        assert collector._parse_cpu("0.5") == 0.5

    def test_parse_memory_resources(self):
        """Test memory resource parsing."""
        from src.collectors.kubernetes import KubernetesCollector

        collector = KubernetesCollector(service_name="test")

        # Test various memory formats
        assert collector._parse_memory("128Mi") == 128 * 1024 * 1024
        assert collector._parse_memory("1Gi") == 1024 * 1024 * 1024
        assert collector._parse_memory("512Ki") == 512 * 1024

    def test_parse_cpu_empty(self):
        """Test CPU parsing with empty string."""
        from src.collectors.kubernetes import KubernetesCollector

        collector = KubernetesCollector(service_name="test")
        assert collector._parse_cpu("") == 0.0
        assert collector._parse_cpu("0") == 0.0

    def test_parse_memory_empty(self):
        """Test memory parsing with empty string."""
        from src.collectors.kubernetes import KubernetesCollector

        collector = KubernetesCollector(service_name="test")
        assert collector._parse_memory("") == 0.0
        assert collector._parse_memory("0") == 0.0


class TestBusinessContextCollector:
    """Test business context collector."""

    def test_business_collector_import(self):
        """Test BusinessContextCollector can be imported."""
        from src.collectors.business import BusinessContextCollector

        assert BusinessContextCollector is not None

    def test_business_collector_initialization(self):
        """Test BusinessContextCollector can be initialized."""
        from src.collectors.business import BusinessContextCollector

        collector = BusinessContextCollector(service_name="test-service")
        assert collector.service_name == "test-service"

    def test_impact_multipliers_defined(self):
        """Test that impact multipliers are defined."""
        from src.collectors.business import DEFAULT_IMPACT_MULTIPLIERS

        assert "product_launch" in DEFAULT_IMPACT_MULTIPLIERS
        assert "marketing_campaign" in DEFAULT_IMPACT_MULTIPLIERS
        assert DEFAULT_IMPACT_MULTIPLIERS["product_launch"] > 1.0

    def test_default_impact_multipliers_values(self):
        """Test DEFAULT_IMPACT_MULTIPLIERS has expected values."""
        from src.collectors.business import DEFAULT_IMPACT_MULTIPLIERS

        # High-impact events
        assert DEFAULT_IMPACT_MULTIPLIERS["product_launch"] >= 2.0
        assert DEFAULT_IMPACT_MULTIPLIERS["sale_event"] >= 2.0

        # Medium-impact events
        assert DEFAULT_IMPACT_MULTIPLIERS["marketing_campaign"] >= 1.0
        assert DEFAULT_IMPACT_MULTIPLIERS["email_blast"] >= 1.0

        # Low/special events
        assert DEFAULT_IMPACT_MULTIPLIERS["scheduled_maintenance"] < 1.0

    def test_add_event(self):
        """Test manually adding an event."""
        from datetime import timedelta
        from src.collectors.business import BusinessContextCollector

        collector = BusinessContextCollector(service_name="test")

        now = datetime.now(timezone.utc)
        collector.add_event(
            event_type="marketing_campaign",
            name="Test Campaign",
            start_time=now,
            end_time=now + timedelta(hours=1),
        )

        assert len(collector._events_cache) == 1
        assert collector._events_cache[0]["name"] == "Test Campaign"

    def test_estimate_impact(self):
        """Test impact estimation for event types."""
        from src.collectors.business import BusinessContextCollector, DEFAULT_IMPACT_MULTIPLIERS

        collector = BusinessContextCollector(service_name="test")

        # Without historical data, should return default
        impact = collector.estimate_impact("product_launch")
        assert impact == DEFAULT_IMPACT_MULTIPLIERS["product_launch"]

        # With historical data, should blend
        impact = collector.estimate_impact("product_launch", historical_actual=2.0)
        assert impact != DEFAULT_IMPACT_MULTIPLIERS["product_launch"]


class TestExternalSignalsCollector:
    """Test external signals collector."""

    def test_external_collector_import(self):
        """Test ExternalSignalsCollector can be imported."""
        from src.collectors.external import ExternalSignalsCollector

        assert ExternalSignalsCollector is not None

    def test_external_collector_initialization(self):
        """Test ExternalSignalsCollector can be initialized."""
        from src.collectors.external import ExternalSignalsCollector

        collector = ExternalSignalsCollector(
            service_name="test-service",
            brand_keywords=["myapp", "myproduct"],
        )

        assert collector.service_name == "test-service"
        assert "myapp" in collector.brand_keywords

    def test_source_confidence_defined(self):
        """Test that source confidence scores are defined."""
        from src.collectors.external import ExternalSignalsCollector

        collector = ExternalSignalsCollector(
            service_name="test",
            brand_keywords=["test"],
        )

        # Check confidence scores exist
        assert hasattr(collector, "_source_confidence")
        assert "twitter" in collector._source_confidence
        assert "news_major" in collector._source_confidence
        assert "news_minor" in collector._source_confidence

    def test_source_confidence_values(self):
        """Test source confidence values are in valid range."""
        from src.collectors.external import ExternalSignalsCollector

        collector = ExternalSignalsCollector(
            service_name="test",
            brand_keywords=["test"],
        )

        for source, confidence in collector._source_confidence.items():
            assert 0 <= confidence <= 1.0, f"{source} confidence out of range"

    def test_add_keyword(self):
        """Test adding a keyword to monitor."""
        from src.collectors.external import ExternalSignalsCollector

        collector = ExternalSignalsCollector(
            service_name="test",
            brand_keywords=["test"],
        )

        collector.add_keyword("newkeyword")
        assert "newkeyword" in collector.brand_keywords

    def test_remove_keyword(self):
        """Test removing a keyword from monitoring."""
        from src.collectors.external import ExternalSignalsCollector

        collector = ExternalSignalsCollector(
            service_name="test",
            brand_keywords=["test", "remove_me"],
        )

        collector.remove_keyword("remove_me")
        assert "remove_me" not in collector.brand_keywords
        assert "test" in collector.brand_keywords


class TestCollectorIntegration:
    """Test collector integration patterns."""

    @pytest.mark.asyncio
    async def test_collector_start_stop_pattern(self):
        """Test that collectors follow start/stop pattern."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(service_name="test")

        # Should have start and stop methods
        assert hasattr(collector, "start")
        assert hasattr(collector, "stop")
        assert callable(collector.start)
        assert callable(collector.stop)

    @pytest.mark.asyncio
    async def test_collector_context_manager(self):
        """Test collector as async context manager."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(service_name="test")

        # Should support async context manager
        assert hasattr(collector, "__aenter__")
        assert hasattr(collector, "__aexit__")

    def test_collector_stats(self):
        """Test collector statistics."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(service_name="test")
        stats = collector.get_stats()

        assert "name" in stats
        assert "running" in stats
        assert "collection_interval" in stats
        assert "collections_total" in stats
        assert "collections_failed" in stats

    def test_collector_publisher_interface(self):
        """Test collector publisher interface."""
        from src.collectors.prometheus import PrometheusCollector

        collector = PrometheusCollector(service_name="test")

        # Should have set_publisher method
        assert hasattr(collector, "set_publisher")
        assert callable(collector.set_publisher)

        # Test setting publisher
        mock_publisher = MagicMock()
        collector.set_publisher(mock_publisher)
        assert collector._publisher == mock_publisher
