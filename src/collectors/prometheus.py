"""
Prometheus metrics collector.

Queries Prometheus for application metrics like request rate,
latency, CPU/memory utilization, etc.
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from config.settings import get_settings
from src.utils.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)
settings = get_settings()


# Default PromQL queries for common metrics
DEFAULT_QUERIES: dict[str, str] = {
    # Request rate (requests per second)
    "requests_per_second": 'sum(rate(http_requests_total{job="app"}[1m]))',
    # Latency percentiles
    "latency_p50": 'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job="app"}[5m])) by (le))',
    "latency_p95": 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="app"}[5m])) by (le))',
    "latency_p99": 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job="app"}[5m])) by (le))',
    # Error rate
    "error_rate": 'sum(rate(http_requests_total{job="app",status=~"5.."}[1m])) / sum(rate(http_requests_total{job="app"}[1m]))',
    # Resource utilization
    "cpu_utilization": 'avg(rate(container_cpu_usage_seconds_total{container!=""}[1m])) by (pod)',
    "memory_utilization": 'avg(container_memory_usage_bytes{container!=""}) by (pod) / avg(container_spec_memory_limit_bytes{container!=""}) by (pod)',
    # Connections
    "active_connections": 'sum(nginx_connections_active) or vector(0)',
    # Queue depth (if using a message queue)
    "queue_depth": 'sum(rabbitmq_queue_messages) or sum(kafka_consumer_lag) or vector(0)',
}


class PrometheusCollector(BaseCollector):
    """
    Collector that queries Prometheus for application metrics.

    Supports:
    - Instant queries (current value)
    - Range queries (values over time)
    - Custom queries per service
    - Automatic label extraction
    """

    def __init__(
        self,
        service_name: str = "default",
        prometheus_url: str | None = None,
        queries: dict[str, str] | None = None,
        collection_interval: float = 15.0,
        query_timeout: float = 30.0,
    ) -> None:
        """
        Initialize Prometheus collector.

        Args:
            service_name: Name of the service being monitored
            prometheus_url: Prometheus server URL (defaults to settings)
            queries: Custom PromQL queries (defaults to DEFAULT_QUERIES)
            collection_interval: Seconds between collections
            query_timeout: Timeout for Prometheus queries
        """
        super().__init__(
            name=f"prometheus-{service_name}",
            collection_interval=collection_interval,
        )

        self.service_name = service_name
        self.prometheus_url = prometheus_url or settings.prometheus.url
        self.queries = queries or DEFAULT_QUERIES.copy()
        self.query_timeout = query_timeout

        # HTTP client for Prometheus API
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.prometheus_url,
                timeout=self.query_timeout,
            )
        return self._client

    async def _query_instant(self, query: str) -> dict[str, Any]:
        """
        Execute an instant query against Prometheus.

        Args:
            query: PromQL query string

        Returns:
            Prometheus response data

        Raises:
            httpx.HTTPError: On HTTP errors
            ValueError: On Prometheus query errors
        """
        client = await self._get_client()

        response = await client.get(
            "/api/v1/query",
            params={"query": query},
        )
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            error = data.get("error", "Unknown error")
            raise ValueError(f"Prometheus query failed: {error}")

        return data.get("data", {})

    async def _query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s",
    ) -> dict[str, Any]:
        """
        Execute a range query against Prometheus.

        Args:
            query: PromQL query string
            start: Start time
            end: End time
            step: Query resolution

        Returns:
            Prometheus response data
        """
        client = await self._get_client()

        response = await client.get(
            "/api/v1/query_range",
            params={
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step,
            },
        )
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            error = data.get("error", "Unknown error")
            raise ValueError(f"Prometheus query failed: {error}")

        return data.get("data", {})

    def _parse_instant_result(
        self,
        result: dict[str, Any],
        metric_name: str,
    ) -> list[dict[str, Any]]:
        """
        Parse Prometheus instant query result into metric dictionaries.

        Prometheus returns results in this format:
        {
            "resultType": "vector",
            "result": [
                {"metric": {"label1": "value1"}, "value": [timestamp, "value"]}
            ]
        }
        """
        metrics = []
        result_type = result.get("resultType", "")
        results = result.get("result", [])

        if result_type == "vector":
            for item in results:
                labels = item.get("metric", {})
                value_data = item.get("value", [])

                if len(value_data) >= 2:
                    timestamp = datetime.fromtimestamp(
                        float(value_data[0]), tz=timezone.utc
                    )
                    try:
                        value = float(value_data[1])
                    except (ValueError, TypeError):
                        # Handle NaN or invalid values
                        continue

                    metrics.append({
                        "timestamp": timestamp,
                        "service_name": self.service_name,
                        "metric_name": metric_name,
                        "value": value,
                        "labels": labels,
                    })

        elif result_type == "scalar":
            # Scalar result: single value
            if len(results) >= 2:
                timestamp = datetime.fromtimestamp(float(results[0]), tz=timezone.utc)
                try:
                    value = float(results[1])
                except (ValueError, TypeError):
                    return metrics

                metrics.append({
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": metric_name,
                    "value": value,
                    "labels": {},
                })

        return metrics

    async def collect(self) -> list[dict[str, Any]]:
        """
        Collect all configured metrics from Prometheus.

        Returns:
            List of metric dictionaries
        """
        all_metrics = []

        for metric_name, query in self.queries.items():
            try:
                result = await self._query_instant(query)
                metrics = self._parse_instant_result(result, metric_name)
                all_metrics.extend(metrics)

                logger.debug(
                    "Collected metric",
                    metric=metric_name,
                    count=len(metrics),
                )

            except httpx.HTTPError as e:
                logger.warning(
                    "HTTP error querying Prometheus",
                    metric=metric_name,
                    error=str(e),
                )
            except ValueError as e:
                logger.warning(
                    "Invalid Prometheus response",
                    metric=metric_name,
                    error=str(e),
                )
            except Exception as e:
                logger.error(
                    "Unexpected error collecting metric",
                    metric=metric_name,
                    error=str(e),
                )

        logger.info(
            "Prometheus collection complete",
            service=self.service_name,
            total_metrics=len(all_metrics),
        )

        return all_metrics

    async def collect_metric(self, metric_name: str) -> list[dict[str, Any]]:
        """Collect a single metric by name."""
        if metric_name not in self.queries:
            raise ValueError(f"Unknown metric: {metric_name}")

        query = self.queries[metric_name]
        result = await self._query_instant(query)
        return self._parse_instant_result(result, metric_name)

    def add_query(self, metric_name: str, query: str) -> None:
        """Add or update a custom query."""
        self.queries[metric_name] = query
        logger.info("Added Prometheus query", metric=metric_name)

    def remove_query(self, metric_name: str) -> None:
        """Remove a query."""
        if metric_name in self.queries:
            del self.queries[metric_name]
            logger.info("Removed Prometheus query", metric=metric_name)

    async def stop(self) -> None:
        """Stop collector and close HTTP client."""
        await super().stop()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Prometheus is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("/-/healthy")
            return response.status_code == 200
        except Exception:
            return False
