"""
Unit tests for Phase 10: Accuracy and Cost Trackers.

Tests cover:
- Accuracy tracker for prediction monitoring
- Cost tracker for infrastructure costs
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.monitoring.accuracy import (
    AccuracyMetrics,
    AccuracyTracker,
    PredictionRecord,
    init_accuracy_tracker,
)
from src.monitoring.cost import (
    CostPeriod,
    CostRecord,
    CostTracker,
    InstanceCost,
    SavingsRecord,
    init_cost_tracker,
)

# =============================================================================
# Prediction Record Tests
# =============================================================================


class TestPredictionRecord:
    """Tests for PredictionRecord."""

    def test_create_record(self):
        """Test creating a prediction record."""
        now = datetime.now(UTC)
        record = PredictionRecord(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=now + timedelta(minutes=30),
            p10=80.0,
            p50=100.0,
            p90=120.0,
        )

        assert record.prediction_id == "pred_123"
        assert record.model_name == "transformer"
        assert record.p50 == 100.0

    def test_error_without_actual(self):
        """Test error calculation without actual value."""
        now = datetime.now(UTC)
        record = PredictionRecord(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=now + timedelta(minutes=30),
            p10=80.0,
            p50=100.0,
            p90=120.0,
        )

        assert record.error is None
        assert record.absolute_error is None
        assert record.percentage_error is None

    def test_error_with_actual(self):
        """Test error calculation with actual value."""
        now = datetime.now(UTC)
        record = PredictionRecord(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=now + timedelta(minutes=30),
            p10=80.0,
            p50=100.0,
            p90=120.0,
            actual=110.0,
        )

        assert record.error == 10.0  # actual - predicted
        assert record.absolute_error == 10.0
        assert record.percentage_error == pytest.approx(10.0 / 110.0)

    def test_within_confidence_interval(self):
        """Test confidence interval checks."""
        now = datetime.now(UTC)
        record = PredictionRecord(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=now + timedelta(minutes=30),
            p10=80.0,
            p50=100.0,
            p90=120.0,
            actual=100.0,
        )

        assert record.within_80_ci is True  # 100 is within [80, 120]

    def test_outside_confidence_interval(self):
        """Test when actual is outside CI."""
        now = datetime.now(UTC)
        record = PredictionRecord(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=now + timedelta(minutes=30),
            p10=80.0,
            p50=100.0,
            p90=120.0,
            actual=150.0,
        )

        assert record.within_80_ci is False


# =============================================================================
# Accuracy Tracker Tests
# =============================================================================


class TestAccuracyTracker:
    """Tests for AccuracyTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return AccuracyTracker(
            retention_hours=24,
            min_samples_for_metrics=5,
        )

    def test_create_tracker(self, tracker):
        """Test creating an accuracy tracker."""
        assert tracker is not None

    def test_record_prediction(self, tracker):
        """Test recording a prediction."""
        now = datetime.now(UTC)
        record = tracker.record_prediction(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=now + timedelta(minutes=30),
            p10=80.0,
            p50=100.0,
            p90=120.0,
        )

        assert record is not None
        assert record.prediction_id == "pred_123"

    def test_record_actual(self, tracker):
        """Test recording actual values."""
        now = datetime.now(UTC)
        target_time = now + timedelta(minutes=30)

        # Record prediction
        tracker.record_prediction(
            prediction_id="pred_123",
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            predicted_at=now,
            target_time=target_time,
            p10=80.0,
            p50=100.0,
            p90=120.0,
        )

        # Record actual
        matched = tracker.record_actual(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            target_time=target_time,
            actual_value=105.0,
        )

        assert matched == 1

    def test_calculate_metrics_insufficient_data(self, tracker):
        """Test metrics calculation with insufficient data."""
        metrics = tracker.calculate_metrics(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
        )

        assert metrics is None  # Not enough samples

    def test_calculate_metrics_with_data(self, tracker):
        """Test metrics calculation with sufficient data."""
        now = datetime.now(UTC)

        # Create enough samples
        for i in range(10):
            target_time = now + timedelta(minutes=30 + i)
            tracker.record_prediction(
                prediction_id=f"pred_{i}",
                model_name="transformer",
                horizon_minutes=30,
                service_name="api",
                predicted_at=now,
                target_time=target_time,
                p10=80.0,
                p50=100.0,
                p90=120.0,
            )
            # Record actual
            tracker.record_actual(
                model_name="transformer",
                horizon_minutes=30,
                service_name="api",
                target_time=target_time,
                actual_value=100.0 + i,  # Slight variation
            )

        metrics = tracker.calculate_metrics(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
        )

        assert metrics is not None
        assert metrics.sample_count == 10
        assert metrics.mape >= 0
        assert 0 <= metrics.coverage_80 <= 1

    def test_get_stats(self, tracker):
        """Test getting tracker stats."""
        stats = tracker.get_stats()

        assert "total_predictions" in stats
        assert "tracked_models" in stats

    def test_add_callback(self, tracker):
        """Test adding a callback."""
        callback = MagicMock()
        tracker.add_callback(callback)

        # Callback should be added
        assert callback in tracker._callbacks


class TestAccuracyMetrics:
    """Tests for AccuracyMetrics dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now(UTC)
        metrics = AccuracyMetrics(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            period_start=now - timedelta(hours=24),
            period_end=now,
            sample_count=100,
            mae=5.0,
            mape=0.05,
            rmse=6.0,
            mse=36.0,
            median_ae=4.0,
            coverage_80=0.82,
            coverage_90=0.91,
            coverage_95=0.96,
            mean_error=-1.0,
            bias_direction="over",
            max_error=20.0,
            min_error=0.5,
            std_error=3.0,
        )

        d = metrics.to_dict()

        assert d["model_name"] == "transformer"
        assert d["mape"] == 0.05
        assert d["coverage_80"] == 0.82

    def test_is_acceptable(self):
        """Test acceptability check."""
        now = datetime.now(UTC)

        # Acceptable metrics
        good_metrics = AccuracyMetrics(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            period_start=now - timedelta(hours=24),
            period_end=now,
            sample_count=100,
            mae=5.0,
            mape=0.10,  # Below 15% threshold
            rmse=6.0,
            mse=36.0,
            median_ae=4.0,
            coverage_80=0.80,  # Above 75% threshold
            coverage_90=0.91,
            coverage_95=0.96,
            mean_error=-1.0,
            bias_direction="neutral",
            max_error=20.0,
            min_error=0.5,
            std_error=3.0,
        )

        assert good_metrics.is_acceptable is True

        # Unacceptable metrics
        bad_metrics = AccuracyMetrics(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
            period_start=now - timedelta(hours=24),
            period_end=now,
            sample_count=100,
            mae=15.0,
            mape=0.25,  # Above threshold
            rmse=18.0,
            mse=324.0,
            median_ae=12.0,
            coverage_80=0.60,  # Below threshold
            coverage_90=0.70,
            coverage_95=0.80,
            mean_error=-5.0,
            bias_direction="over",
            max_error=50.0,
            min_error=1.0,
            std_error=10.0,
        )

        assert bad_metrics.is_acceptable is False


# =============================================================================
# Cost Tracker Tests
# =============================================================================


class TestInstanceCost:
    """Tests for InstanceCost."""

    def test_create_instance_cost(self):
        """Test creating instance cost."""
        cost = InstanceCost(
            instance_type="m5.large",
            hourly_cost=0.096,
            vcpus=2,
            memory_gb=8,
            capacity_rps=1000,
        )

        assert cost.instance_type == "m5.large"
        assert cost.hourly_cost == 0.096


class TestCostRecord:
    """Tests for CostRecord."""

    def test_create_cost_record(self):
        """Test creating cost record."""
        now = datetime.now(UTC)
        record = CostRecord(
            timestamp=now,
            service_name="api",
            instance_type="m5.large",
            instance_count=5,
            hourly_cost=0.096,
        )

        assert record.instance_count == 5
        assert record.total_hourly_cost == 0.096 * 5

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now(UTC)
        record = CostRecord(
            timestamp=now,
            service_name="api",
            instance_type="m5.large",
            instance_count=5,
            hourly_cost=0.096,
            utilization=0.7,
        )

        d = record.to_dict()

        assert d["service_name"] == "api"
        assert d["instance_count"] == 5
        assert d["utilization"] == 0.7


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return CostTracker(retention_days=7)

    def test_create_tracker(self, tracker):
        """Test creating a cost tracker."""
        assert tracker is not None

    def test_default_instance_costs(self, tracker):
        """Test default instance costs are loaded."""
        m5_large = tracker.get_instance_cost("m5.large")

        assert m5_large is not None
        assert m5_large.hourly_cost == 0.096

    def test_add_instance_type(self, tracker):
        """Test adding a custom instance type."""
        custom = InstanceCost(
            instance_type="custom.large",
            hourly_cost=0.5,
            vcpus=4,
            memory_gb=16,
            capacity_rps=2000,
        )

        tracker.add_instance_type(custom)

        result = tracker.get_instance_cost("custom.large")
        assert result is not None
        assert result.hourly_cost == 0.5

    def test_record_cost(self, tracker):
        """Test recording a cost."""
        record = tracker.record_cost(
            service_name="api",
            instance_type="m5.large",
            instance_count=5,
            utilization=0.7,
        )

        assert record is not None
        assert record.instance_count == 5

    def test_get_recent_costs(self, tracker):
        """Test getting recent costs."""
        # Record some costs
        for i in range(5):
            tracker.record_cost(
                service_name="api",
                instance_type="m5.large",
                instance_count=5 + i,
                utilization=0.7,
            )

        recent = tracker.get_recent_costs("api", hours=1)

        assert len(recent) == 5

    def test_calculate_savings(self, tracker):
        """Test calculating savings."""
        # Record costs
        for i in range(10):
            tracker.record_cost(
                service_name="api",
                instance_type="m5.large",
                instance_count=5,
                utilization=0.7,
            )

        savings = tracker.calculate_savings("api", CostPeriod.HOURLY)

        assert savings is not None
        assert savings.actual_cost >= 0

    def test_get_cost_summary(self, tracker):
        """Test getting cost summary."""
        # Record costs
        for i in range(10):
            tracker.record_cost(
                service_name="api",
                instance_type="m5.large",
                instance_count=5,
                utilization=0.7,
            )

        summary = tracker.get_cost_summary("api", period_hours=1)

        assert summary is not None
        assert summary.avg_instances == 5

    def test_generate_report(self, tracker):
        """Test generating cost report."""
        # Record costs
        for i in range(5):
            tracker.record_cost(
                service_name="api",
                instance_type="m5.large",
                instance_count=5,
                utilization=0.7,
            )

        report = tracker.generate_report(period_hours=1)

        assert "generated_at" in report
        assert "services" in report
        assert "totals" in report

    def test_get_stats(self, tracker):
        """Test getting tracker stats."""
        stats = tracker.get_stats()

        assert "tracked_services" in stats
        assert "instance_types_cataloged" in stats


class TestSavingsRecord:
    """Tests for SavingsRecord."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now(UTC)
        record = SavingsRecord(
            timestamp=now,
            service_name="api",
            period=CostPeriod.DAILY,
            actual_cost=100.0,
            reactive_cost=120.0,
            savings=20.0,
            savings_percent=16.67,
            over_provision_cost=5.0,
            sla_violation_cost=0.0,
        )

        d = record.to_dict()

        assert d["service_name"] == "api"
        assert d["savings"] == 20.0
        assert d["savings_percent"] == 16.67


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Test module imports work correctly."""

    def test_import_accuracy_tracker(self):
        """Test importing accuracy tracker."""
        from src.monitoring import get_accuracy_tracker

        tracker = get_accuracy_tracker()
        assert tracker is not None

    def test_import_cost_tracker(self):
        """Test importing cost tracker."""
        from src.monitoring import get_cost_tracker

        tracker = get_cost_tracker()
        assert tracker is not None

    def test_init_accuracy_tracker(self):
        """Test initializing accuracy tracker."""
        tracker = init_accuracy_tracker(retention_hours=48)
        assert tracker is not None

    def test_init_cost_tracker(self):
        """Test initializing cost tracker."""
        tracker = init_cost_tracker(retention_days=30)
        assert tracker is not None
