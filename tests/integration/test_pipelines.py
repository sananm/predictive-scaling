"""
Integration tests for prediction and scaling pipelines.

Tests cover:
- Full prediction pipeline (metrics → features → prediction)
- Scaling pipeline (prediction → decision → execution)
- API endpoints
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.monitoring.accuracy import AccuracyTracker
from src.monitoring.cost import CostTracker
from src.services.prediction import PredictionHorizon, PredictionService
from src.services.scaling import ScalingActionType, ScalingService

# =============================================================================
# Prediction Pipeline Integration Tests
# =============================================================================


class TestPredictionPipeline:
    """Integration tests for the prediction pipeline."""

    @pytest.fixture
    def prediction_service(self):
        """Create prediction service with mock model."""
        service = PredictionService()

        # Create mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={
            "p10": 80.0,
            "p50": 100.0,
            "p90": 120.0,
            "confidence": 0.85,
        })

        service.set_model(PredictionHorizon.SHORT, mock_model)
        service.set_model(PredictionHorizon.MEDIUM, mock_model)
        service.set_model(PredictionHorizon.LONG, mock_model)

        return service

    @pytest.fixture
    def accuracy_tracker(self):
        """Create accuracy tracker."""
        return AccuracyTracker(min_samples_for_metrics=3)

    @pytest.mark.asyncio
    async def test_prediction_to_accuracy_tracking(
        self,
        prediction_service,
        accuracy_tracker,
    ):
        """Test prediction flows to accuracy tracking."""
        # Set up accuracy tracker
        prediction_service._accuracy_tracker = accuracy_tracker

        await prediction_service.start()

        # Run predictions
        results = await prediction_service.run_predictions_for_service(
            service_name="api",
            horizons=[PredictionHorizon.SHORT],
        )

        await prediction_service.stop()

        # Verify predictions were made
        assert len(results) > 0

        # Verify accuracy tracker received predictions
        stats = accuracy_tracker.get_stats()
        assert stats["total_predictions"] > 0

    @pytest.mark.asyncio
    async def test_prediction_with_features(self, prediction_service):
        """Test prediction with pre-computed features."""
        await prediction_service.start()

        # Request prediction with features
        request_id = await prediction_service.request_prediction(
            service_name="api",
            horizon=PredictionHorizon.SHORT,
            target_time=datetime.now(UTC) + timedelta(minutes=30),
            features={
                "cpu_usage": 0.7,
                "memory_usage": 0.6,
                "requests_per_second": 500,
            },
        )

        # Wait for processing
        await asyncio.sleep(0.5)

        await prediction_service.stop()

        assert request_id is not None

    @pytest.mark.asyncio
    async def test_multiple_horizon_predictions(self, prediction_service):
        """Test predictions across all horizons."""
        await prediction_service.start()

        results = await prediction_service.run_predictions_for_service(
            service_name="api",
            horizons=None,  # All horizons
        )

        await prediction_service.stop()

        # Should have predictions for short, medium, long
        horizons = set(r.horizon for r in results)
        assert PredictionHorizon.SHORT in horizons
        assert PredictionHorizon.MEDIUM in horizons
        assert PredictionHorizon.LONG in horizons


# =============================================================================
# Scaling Pipeline Integration Tests
# =============================================================================


class TestScalingPipeline:
    """Integration tests for the scaling pipeline."""

    @pytest.fixture
    def prediction_service(self):
        """Create prediction service with mock data."""
        service = PredictionService()

        # Set up mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={
            "p10": 150.0,
            "p50": 180.0,
            "p90": 220.0,
            "confidence": 0.9,
        })
        service.set_model(PredictionHorizon.MEDIUM, mock_model)

        return service

    @pytest.fixture
    def scaling_service(self, prediction_service):
        """Create scaling service."""
        return ScalingService(
            prediction_service=prediction_service,
            auto_approve_threshold=0.95,
            cooldown_seconds=1,  # Short cooldown for testing
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"success": True})
        executor.verify = AsyncMock(return_value={"verified": True})
        executor.rollback = AsyncMock()
        return executor

    @pytest.mark.asyncio
    async def test_scaling_decision_flow(self, scaling_service):
        """Test full scaling decision flow."""
        await scaling_service.start()

        # Evaluate scaling (should generate action due to high utilization)
        action = await scaling_service.evaluate_scaling(
            service_name="api",
            current_instances=2,
            current_utilization=0.9,  # High utilization
        )

        await scaling_service.stop()

        # Action may be generated based on simple decision logic
        stats = scaling_service.get_stats()
        assert stats["total_evaluations"] == 1

    @pytest.mark.asyncio
    async def test_scaling_with_executor(self, scaling_service, mock_executor):
        """Test scaling with executor."""
        scaling_service._executor = mock_executor
        scaling_service._auto_approve_threshold = 0.5  # Lower threshold

        await scaling_service.start()

        # Evaluate scaling
        action = await scaling_service.evaluate_scaling(
            service_name="api",
            current_instances=2,
            current_utilization=0.95,
        )

        # Wait for execution
        await asyncio.sleep(1)

        await scaling_service.stop()

        stats = scaling_service.get_stats()
        assert stats["total_evaluations"] >= 1

    @pytest.mark.asyncio
    async def test_scaling_approval_flow(self, scaling_service):
        """Test manual approval flow."""
        scaling_service._auto_approve_threshold = 1.0  # Never auto-approve

        await scaling_service.start()

        # Create a pending action manually
        from src.services.scaling import ScalingAction

        action = ScalingAction(
            action_id="test_action",
            service_name="api",
            action_type=ScalingActionType.SCALE_UP,
            current_count=2,
            target_count=4,
            reason="Test",
            confidence=0.8,
            risk_score=0.2,
            estimated_cost_change=0.2,
            requires_approval=True,
        )
        scaling_service._pending_actions[action.action_id] = action

        # Approve action
        result = await scaling_service.approve_action(
            action_id="test_action",
            approver="test_user",
        )

        await scaling_service.stop()

        assert result is True
        assert "test_action" not in scaling_service._pending_actions

    @pytest.mark.asyncio
    async def test_scaling_rejection_flow(self, scaling_service):
        """Test action rejection flow."""
        await scaling_service.start()

        # Create a pending action manually
        from src.services.scaling import ScalingAction

        action = ScalingAction(
            action_id="test_reject",
            service_name="api",
            action_type=ScalingActionType.SCALE_DOWN,
            current_count=5,
            target_count=3,
            reason="Low utilization",
            confidence=0.7,
            risk_score=0.3,
            estimated_cost_change=-0.2,
            requires_approval=True,
        )
        scaling_service._pending_actions[action.action_id] = action

        # Reject action
        result = await scaling_service.reject_action(
            action_id="test_reject",
            reason="Not safe to scale down right now",
        )

        await scaling_service.stop()

        assert result is True


# =============================================================================
# Accuracy to Scaling Integration Tests
# =============================================================================


class TestAccuracyToScalingIntegration:
    """Test integration between accuracy tracking and scaling decisions."""

    @pytest.fixture
    def accuracy_tracker(self):
        """Create accuracy tracker."""
        return AccuracyTracker(
            min_samples_for_metrics=5,
            mape_warning_threshold=0.10,
            mape_error_threshold=0.20,
        )

    @pytest.fixture
    def cost_tracker(self):
        """Create cost tracker."""
        return CostTracker()

    @pytest.mark.asyncio
    async def test_accuracy_triggers_retrain(self, accuracy_tracker):
        """Test that accuracy degradation can trigger retraining."""
        now = datetime.now(UTC)
        alerts_triggered = []

        def capture_alert(alert):
            alerts_triggered.append(alert)

        accuracy_tracker.add_callback(capture_alert)

        # Add predictions with poor accuracy
        for i in range(10):
            target_time = now + timedelta(minutes=30 + i)
            accuracy_tracker.record_prediction(
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
            # Record actual with significant error
            accuracy_tracker.record_actual(
                model_name="transformer",
                horizon_minutes=30,
                service_name="api",
                target_time=target_time,
                actual_value=150.0,  # 50% error
            )

        # Calculate metrics (should trigger alert)
        metrics = accuracy_tracker.calculate_metrics(
            model_name="transformer",
            horizon_minutes=30,
            service_name="api",
        )

        assert metrics is not None
        assert metrics.mape > 0.20  # Should be high
        assert len(alerts_triggered) > 0  # Should have triggered alert

    @pytest.mark.asyncio
    async def test_cost_tracking_with_scaling(self, cost_tracker):
        """Test cost tracking during scaling operations."""
        # Record costs before scaling
        for i in range(5):
            cost_tracker.record_cost(
                service_name="api",
                instance_type="m5.large",
                instance_count=5,
                utilization=0.7,
            )

        # Simulate scale-up
        cost_tracker.record_cost(
            service_name="api",
            instance_type="m5.large",
            instance_count=8,
            utilization=0.5,
        )

        # Get summary
        summary = cost_tracker.get_cost_summary("api", period_hours=1)

        assert summary is not None
        assert summary.max_instances == 8
        assert summary.min_instances == 5


# =============================================================================
# API Integration Tests
# =============================================================================


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_ready_endpoint(self, client):
        """Test readiness endpoint."""
        response = client.get("/health/ready")

        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert b"python_" in response.content or b"process_" in response.content

    def test_config_endpoint(self, client):
        """Test config endpoint."""
        response = client.get("/api/v1/config")

        assert response.status_code == 200
        data = response.json()
        assert "scaling" in data
        assert "model" in data

    def test_scaling_config_update(self, client):
        """Test updating scaling config."""
        response = client.put(
            "/api/v1/config/scaling",
            json={
                "target_utilization": 0.75,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "target_utilization" in data["updated_fields"]

    @pytest.mark.skip(reason="TestClient async loop conflicts with asyncpg - works in real app")
    def test_predictions_endpoint(self, client):
        """Test predictions endpoint."""
        response = client.get("/api/v1/predictions/current")

        assert response.status_code == 200

    @pytest.mark.skip(reason="TestClient async loop conflicts with asyncpg - works in real app")
    def test_scaling_decisions_endpoint(self, client):
        """Test scaling decisions endpoint."""
        response = client.get("/api/v1/scaling/decisions")

        assert response.status_code == 200

    @pytest.mark.skip(reason="TestClient async loop conflicts with asyncpg - works in real app")
    def test_events_endpoint(self, client):
        """Test events endpoint."""
        response = client.get("/api/v1/events")

        assert response.status_code == 200


# =============================================================================
# Service Coordination Tests
# =============================================================================


class TestServiceCoordination:
    """Test coordination between services."""

    @pytest.mark.asyncio
    async def test_prediction_to_scaling_flow(self):
        """Test flow from prediction service to scaling service."""
        # Create services
        prediction_service = PredictionService()
        scaling_service = ScalingService(
            prediction_service=prediction_service,
            auto_approve_threshold=1.0,  # Don't auto-approve
        )

        # Mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={
            "p10": 200.0,
            "p50": 250.0,
            "p90": 300.0,
            "confidence": 0.9,
        })
        prediction_service.set_model(PredictionHorizon.MEDIUM, mock_model)

        await prediction_service.start()
        await scaling_service.start()

        # Run predictions
        await prediction_service.run_predictions_for_service(
            service_name="api",
            horizons=[PredictionHorizon.MEDIUM],
        )

        # Evaluate scaling based on predictions
        action = await scaling_service.evaluate_scaling(
            service_name="api",
            current_instances=3,
            current_utilization=0.8,
        )

        await prediction_service.stop()
        await scaling_service.stop()

        # Verify flow worked
        pred_stats = prediction_service.get_stats()
        scale_stats = scaling_service.get_stats()

        assert pred_stats["total_predictions"] > 0
        assert scale_stats["total_evaluations"] > 0
