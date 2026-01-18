"""
Locust load tests for the predictive scaling API.

Tests cover:
- Realistic traffic patterns
- API response latency
- Prediction throughput
- Scaling endpoint stress

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000
"""

import random
import time
from datetime import UTC, datetime, timedelta

from locust import HttpUser, between, events, tag, task

# =============================================================================
# Custom Metrics Tracking
# =============================================================================


class MetricsCollector:
    """Collect custom metrics during load testing."""

    def __init__(self):
        self.prediction_latencies = []
        self.scaling_latencies = []
        self.error_counts = {"prediction": 0, "scaling": 0, "other": 0}

    def record_prediction_latency(self, latency_ms: float):
        """Record prediction latency."""
        self.prediction_latencies.append(latency_ms)

    def record_scaling_latency(self, latency_ms: float):
        """Record scaling latency."""
        self.scaling_latencies.append(latency_ms)

    def record_error(self, category: str):
        """Record an error."""
        if category in self.error_counts:
            self.error_counts[category] += 1
        else:
            self.error_counts["other"] += 1

    def get_summary(self) -> dict:
        """Get metrics summary."""
        def percentile(data: list, p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "prediction": {
                "count": len(self.prediction_latencies),
                "avg_ms": sum(self.prediction_latencies) / len(self.prediction_latencies)
                if self.prediction_latencies else 0,
                "p50_ms": percentile(self.prediction_latencies, 0.5),
                "p95_ms": percentile(self.prediction_latencies, 0.95),
                "p99_ms": percentile(self.prediction_latencies, 0.99),
            },
            "scaling": {
                "count": len(self.scaling_latencies),
                "avg_ms": sum(self.scaling_latencies) / len(self.scaling_latencies)
                if self.scaling_latencies else 0,
                "p50_ms": percentile(self.scaling_latencies, 0.5),
                "p95_ms": percentile(self.scaling_latencies, 0.95),
                "p99_ms": percentile(self.scaling_latencies, 0.99),
            },
            "errors": self.error_counts,
        }


# Global metrics collector
metrics = MetricsCollector()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print metrics summary when test stops."""
    summary = metrics.get_summary()
    print("\n" + "=" * 60)
    print("CUSTOM METRICS SUMMARY")
    print("=" * 60)
    print("\nPrediction Endpoints:")
    print(f"  Count: {summary['prediction']['count']}")
    print(f"  Avg Latency: {summary['prediction']['avg_ms']:.2f}ms")
    print(f"  P50 Latency: {summary['prediction']['p50_ms']:.2f}ms")
    print(f"  P95 Latency: {summary['prediction']['p95_ms']:.2f}ms")
    print(f"  P99 Latency: {summary['prediction']['p99_ms']:.2f}ms")
    print("\nScaling Endpoints:")
    print(f"  Count: {summary['scaling']['count']}")
    print(f"  Avg Latency: {summary['scaling']['avg_ms']:.2f}ms")
    print(f"  P50 Latency: {summary['scaling']['p50_ms']:.2f}ms")
    print(f"  P95 Latency: {summary['scaling']['p95_ms']:.2f}ms")
    print(f"  P99 Latency: {summary['scaling']['p99_ms']:.2f}ms")
    print("\nErrors:")
    print(f"  Prediction: {summary['errors']['prediction']}")
    print(f"  Scaling: {summary['errors']['scaling']}")
    print(f"  Other: {summary['errors']['other']}")
    print("=" * 60 + "\n")


# =============================================================================
# Base User Class
# =============================================================================


class PredictiveScalingUser(HttpUser):
    """Base user for predictive scaling load tests."""

    wait_time = between(0.5, 2.0)
    abstract = True

    def on_start(self):
        """Initialize user session."""
        # Check health on start
        self.client.get("/health")

    def _get_service_name(self) -> str:
        """Get a random service name."""
        return random.choice(["api", "worker", "cache", "web", "backend"])

    def _get_horizon(self) -> str:
        """Get a random prediction horizon."""
        return random.choice(["short", "medium", "long"])


# =============================================================================
# Health Check User
# =============================================================================


class HealthCheckUser(PredictiveScalingUser):
    """User that primarily checks health endpoints."""

    weight = 5  # Lower weight

    @task(10)
    @tag("health")
    def check_health(self):
        """Check health endpoint."""
        self.client.get("/health")

    @task(5)
    @tag("health")
    def check_ready(self):
        """Check readiness endpoint."""
        self.client.get("/ready")

    @task(3)
    @tag("metrics")
    def get_metrics(self):
        """Get Prometheus metrics."""
        self.client.get("/metrics")


# =============================================================================
# Prediction API User
# =============================================================================


class PredictionUser(PredictiveScalingUser):
    """User focused on prediction API endpoints."""

    weight = 30

    @task(10)
    @tag("prediction", "read")
    def get_predictions(self):
        """Get current predictions."""
        start = time.time()
        with self.client.get(
            "/api/v1/predictions",
            catch_response=True,
        ) as response:
            latency = (time.time() - start) * 1000
            metrics.record_prediction_latency(latency)

            if response.status_code != 200:
                metrics.record_error("prediction")
                response.failure(f"Got status {response.status_code}")

    @task(8)
    @tag("prediction", "read")
    def get_predictions_by_service(self):
        """Get predictions for a specific service."""
        service = self._get_service_name()
        start = time.time()
        with self.client.get(
            f"/api/v1/predictions?service={service}",
            name="/api/v1/predictions?service=[service]",
            catch_response=True,
        ) as response:
            latency = (time.time() - start) * 1000
            metrics.record_prediction_latency(latency)

            if response.status_code != 200:
                metrics.record_error("prediction")
                response.failure(f"Got status {response.status_code}")

    @task(5)
    @tag("prediction", "write")
    def request_prediction(self):
        """Request a new prediction."""
        service = self._get_service_name()
        horizon = self._get_horizon()

        target_time = datetime.now(UTC) + timedelta(minutes=30)

        start = time.time()
        with self.client.post(
            "/api/v1/predictions",
            json={
                "service_name": service,
                "horizon": horizon,
                "target_time": target_time.isoformat(),
                "features": {
                    "cpu_usage": random.uniform(0.3, 0.9),
                    "memory_usage": random.uniform(0.4, 0.8),
                    "requests_per_second": random.randint(100, 1000),
                },
            },
            catch_response=True,
        ) as response:
            latency = (time.time() - start) * 1000
            metrics.record_prediction_latency(latency)

            if response.status_code not in [200, 201, 202]:
                metrics.record_error("prediction")
                response.failure(f"Got status {response.status_code}")

    @task(3)
    @tag("prediction", "read")
    def get_prediction_accuracy(self):
        """Get prediction accuracy metrics."""
        service = self._get_service_name()
        with self.client.get(
            f"/api/v1/predictions/accuracy?service={service}",
            name="/api/v1/predictions/accuracy?service=[service]",
            catch_response=True,
        ) as response:
            if response.status_code not in [200, 404]:
                response.failure(f"Got status {response.status_code}")


# =============================================================================
# Scaling API User
# =============================================================================


class ScalingUser(PredictiveScalingUser):
    """User focused on scaling API endpoints."""

    weight = 25

    @task(10)
    @tag("scaling", "read")
    def get_scaling_decisions(self):
        """Get scaling decisions."""
        start = time.time()
        with self.client.get(
            "/api/v1/scaling/decisions",
            catch_response=True,
        ) as response:
            latency = (time.time() - start) * 1000
            metrics.record_scaling_latency(latency)

            if response.status_code != 200:
                metrics.record_error("scaling")
                response.failure(f"Got status {response.status_code}")

    @task(8)
    @tag("scaling", "read")
    def get_pending_actions(self):
        """Get pending scaling actions."""
        start = time.time()
        with self.client.get(
            "/api/v1/scaling/pending",
            catch_response=True,
        ) as response:
            latency = (time.time() - start) * 1000
            metrics.record_scaling_latency(latency)

            if response.status_code not in [200, 404]:
                metrics.record_error("scaling")
                response.failure(f"Got status {response.status_code}")

    @task(5)
    @tag("scaling", "read")
    def get_scaling_history(self):
        """Get scaling history."""
        service = self._get_service_name()
        with self.client.get(
            f"/api/v1/scaling/history?service={service}",
            name="/api/v1/scaling/history?service=[service]",
            catch_response=True,
        ) as response:
            if response.status_code not in [200, 404]:
                response.failure(f"Got status {response.status_code}")

    @task(3)
    @tag("scaling", "write")
    def evaluate_scaling(self):
        """Request scaling evaluation."""
        service = self._get_service_name()

        start = time.time()
        with self.client.post(
            "/api/v1/scaling/evaluate",
            json={
                "service_name": service,
                "current_instances": random.randint(2, 10),
                "current_utilization": random.uniform(0.4, 0.95),
            },
            catch_response=True,
        ) as response:
            latency = (time.time() - start) * 1000
            metrics.record_scaling_latency(latency)

            if response.status_code not in [200, 201, 202, 404]:
                metrics.record_error("scaling")
                response.failure(f"Got status {response.status_code}")


# =============================================================================
# Config API User
# =============================================================================


class ConfigUser(PredictiveScalingUser):
    """User focused on config API endpoints."""

    weight = 10

    @task(10)
    @tag("config", "read")
    def get_config(self):
        """Get current config."""
        self.client.get("/api/v1/config")

    @task(5)
    @tag("config", "read")
    def get_scaling_config(self):
        """Get scaling config."""
        self.client.get("/api/v1/config/scaling")

    @task(5)
    @tag("config", "read")
    def get_model_config(self):
        """Get model config."""
        self.client.get("/api/v1/config/models")

    @task(2)
    @tag("config", "write")
    def update_scaling_config(self):
        """Update scaling config."""
        with self.client.put(
            "/api/v1/config/scaling",
            json={
                "target_utilization": random.uniform(0.6, 0.8),
            },
            catch_response=True,
        ) as response:
            if response.status_code not in [200, 400]:
                response.failure(f"Got status {response.status_code}")


# =============================================================================
# Events API User
# =============================================================================


class EventsUser(PredictiveScalingUser):
    """User focused on events API endpoints."""

    weight = 15

    @task(10)
    @tag("events", "read")
    def get_events(self):
        """Get recent events."""
        self.client.get("/api/v1/events")

    @task(5)
    @tag("events", "read")
    def get_events_by_type(self):
        """Get events by type."""
        event_type = random.choice(["prediction", "scaling", "alert", "system"])
        self.client.get(
            f"/api/v1/events?type={event_type}",
            name="/api/v1/events?type=[type]",
        )

    @task(3)
    @tag("events", "read")
    def get_events_by_service(self):
        """Get events by service."""
        service = self._get_service_name()
        self.client.get(
            f"/api/v1/events?service={service}",
            name="/api/v1/events?service=[service]",
        )


# =============================================================================
# Cost Tracking User
# =============================================================================


class CostTrackingUser(PredictiveScalingUser):
    """User focused on cost tracking endpoints."""

    weight = 15

    @task(10)
    @tag("cost", "read")
    def get_cost_summary(self):
        """Get cost summary."""
        service = self._get_service_name()
        self.client.get(
            f"/api/v1/costs/summary?service={service}",
            name="/api/v1/costs/summary?service=[service]",
        )

    @task(5)
    @tag("cost", "read")
    def get_savings_report(self):
        """Get savings report."""
        self.client.get("/api/v1/costs/savings")

    @task(3)
    @tag("cost", "read")
    def get_cost_report(self):
        """Get full cost report."""
        period = random.choice(["1h", "24h", "7d"])
        self.client.get(
            f"/api/v1/costs/report?period={period}",
            name="/api/v1/costs/report?period=[period]",
        )


# =============================================================================
# Spike Traffic Pattern
# =============================================================================


class SpikeUser(PredictiveScalingUser):
    """User that simulates traffic spikes."""

    weight = 5
    wait_time = between(0.1, 0.5)  # Faster requests during spike

    @task
    @tag("spike")
    def burst_predictions(self):
        """Burst of prediction requests."""
        # Send multiple requests quickly
        for _ in range(5):
            service = self._get_service_name()
            self.client.get(
                f"/api/v1/predictions?service={service}",
                name="/api/v1/predictions?service=[service]",
            )

    @task
    @tag("spike")
    def burst_scaling(self):
        """Burst of scaling requests."""
        for _ in range(3):
            self.client.get("/api/v1/scaling/decisions")


# =============================================================================
# Mixed Realistic User
# =============================================================================


class RealisticUser(PredictiveScalingUser):
    """User with realistic mixed behavior."""

    weight = 40
    wait_time = between(1.0, 5.0)

    @task(20)
    @tag("realistic")
    def typical_monitoring_flow(self):
        """Typical monitoring workflow."""
        # Check health
        self.client.get("/health")

        # Get predictions
        service = self._get_service_name()
        self.client.get(
            f"/api/v1/predictions?service={service}",
            name="/api/v1/predictions?service=[service]",
        )

        # Check scaling status
        self.client.get("/api/v1/scaling/decisions")

    @task(10)
    @tag("realistic")
    def dashboard_refresh(self):
        """Simulate dashboard refresh - multiple endpoints."""
        # Parallel requests like a dashboard would make
        self.client.get("/health")
        self.client.get("/api/v1/predictions")
        self.client.get("/api/v1/scaling/decisions")
        self.client.get("/api/v1/events")
        self.client.get("/api/v1/costs/summary?service=api")

    @task(5)
    @tag("realistic")
    def operator_workflow(self):
        """Simulate operator checking system."""
        # Check config
        self.client.get("/api/v1/config")

        # Check pending actions
        self.client.get("/api/v1/scaling/pending")

        # Check events
        self.client.get("/api/v1/events?type=alert")

        # Get metrics
        self.client.get("/metrics")
