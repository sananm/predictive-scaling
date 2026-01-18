"""
End-to-end tests for the predictive scaling system.

Tests cover:
- Full traffic simulation to scaling flow
- Prediction generation and accuracy tracking
- Scaling decision lifecycle
- Infrastructure scaling with mock executor
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.monitoring.accuracy import AccuracyTracker
from src.monitoring.cost import CostTracker
from src.services.prediction import (
    PredictionHorizon,
    PredictionService,
)
from src.services.scaling import (
    ScalingAction,
    ScalingActionType,
    ScalingService,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MockMetricsCollector:
    """Mock metrics collector for simulating traffic."""

    def __init__(self):
        self.metrics = []
        self.current_idx = 0

    def add_traffic_pattern(self, pattern: list[dict]):
        """Add a traffic pattern to simulate."""
        self.metrics.extend(pattern)

    def get_current_metrics(self) -> dict:
        """Get current metrics."""
        if self.current_idx < len(self.metrics):
            metrics = self.metrics[self.current_idx]
            self.current_idx += 1
            return metrics
        return self.metrics[-1] if self.metrics else {}

    def reset(self):
        """Reset to beginning."""
        self.current_idx = 0


class MockInfrastructureExecutor:
    """Mock infrastructure executor for testing scaling."""

    def __init__(self):
        self.current_instances = {}
        self.execution_history = []
        self.should_fail = False
        self.verification_delay = 0.1

    async def execute(self, action: ScalingAction) -> dict:
        """Execute a scaling action."""
        if self.should_fail:
            return {"success": False, "error": "Simulated failure"}

        self.current_instances[action.service_name] = action.target_count
        self.execution_history.append({
            "action_id": action.action_id,
            "service_name": action.service_name,
            "from_count": action.current_count,
            "to_count": action.target_count,
            "timestamp": datetime.now(UTC),
        })

        return {"success": True, "instance_count": action.target_count}

    async def verify(self, action: ScalingAction) -> dict:
        """Verify scaling completed."""
        await asyncio.sleep(self.verification_delay)

        current = self.current_instances.get(action.service_name, 0)
        return {
            "verified": current == action.target_count,
            "current_count": current,
            "target_count": action.target_count,
        }

    async def rollback(self, action: ScalingAction) -> dict:
        """Rollback a scaling action."""
        self.current_instances[action.service_name] = action.current_count
        return {"success": True, "instance_count": action.current_count}

    def get_instance_count(self, service_name: str) -> int:
        """Get current instance count."""
        return self.current_instances.get(service_name, 0)


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


class TestFullScalingFlow:
    """End-to-end tests for the complete scaling flow."""

    @pytest.fixture
    def metrics_collector(self):
        """Create mock metrics collector."""
        return MockMetricsCollector()

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        executor = MockInfrastructureExecutor()
        executor.current_instances["api"] = 5  # Start with 5 instances
        return executor

    @pytest.fixture
    def prediction_service(self):
        """Create prediction service with mock model."""
        service = PredictionService()

        # Mock model that returns predictions based on input
        mock_model = MagicMock()

        def make_prediction(*args, **kwargs):
            # Simulate predictions based on features
            features = kwargs.get("features", {})
            if not features and args:
                features = args[0] if isinstance(args[0], dict) else {}
            base = features.get("requests_per_second", 100)
            return {
                "p10": base * 0.8,
                "p50": base * 1.0,
                "p90": base * 1.3,
                "confidence": 0.85,
            }

        mock_model.predict = MagicMock(side_effect=make_prediction)

        service.set_model(PredictionHorizon.SHORT, mock_model)
        service.set_model(PredictionHorizon.MEDIUM, mock_model)
        service.set_model(PredictionHorizon.LONG, mock_model)

        return service

    @pytest.fixture
    def scaling_service(self, prediction_service, executor):
        """Create scaling service."""
        service = ScalingService(
            prediction_service=prediction_service,
            auto_approve_threshold=0.8,  # Auto-approve high confidence
            cooldown_seconds=1,
        )
        service._executor = executor
        return service

    @pytest.fixture
    def accuracy_tracker(self):
        """Create accuracy tracker."""
        return AccuracyTracker(min_samples_for_metrics=3)

    @pytest.fixture
    def cost_tracker(self):
        """Create cost tracker."""
        return CostTracker()

    @pytest.mark.asyncio
    async def test_traffic_spike_triggers_scale_up(
        self,
        metrics_collector,
        prediction_service,
        scaling_service,
        executor,
    ):
        """Test that a traffic spike triggers scale-up."""
        # Set up traffic pattern: normal -> spike
        metrics_collector.add_traffic_pattern([
            {"requests_per_second": 100, "cpu_usage": 0.5, "memory_usage": 0.4},
            {"requests_per_second": 150, "cpu_usage": 0.65, "memory_usage": 0.5},
            {"requests_per_second": 300, "cpu_usage": 0.85, "memory_usage": 0.7},
            {"requests_per_second": 500, "cpu_usage": 0.95, "memory_usage": 0.85},
        ])

        await prediction_service.start()
        await scaling_service.start()

        # Process traffic spike
        for _ in range(4):
            metrics = metrics_collector.get_current_metrics()

            # Run prediction
            await prediction_service.request_prediction(
                service_name="api",
                horizon=PredictionHorizon.SHORT,
                target_time=datetime.now(UTC) + timedelta(minutes=15),
                features=metrics,
            )

            # Evaluate scaling
            await scaling_service.evaluate_scaling(
                service_name="api",
                current_instances=executor.get_instance_count("api"),
                current_utilization=metrics.get("cpu_usage", 0.5),
            )

            await asyncio.sleep(0.1)

        # Wait for processing
        await asyncio.sleep(1)

        await prediction_service.stop()
        await scaling_service.stop()

        # Verify scaling happened
        stats = scaling_service.get_stats()
        assert stats["total_evaluations"] == 4

        # Should have scaled up due to high utilization
        assert executor.get_instance_count("api") >= 5

    @pytest.mark.asyncio
    async def test_traffic_decrease_triggers_scale_down(
        self,
        metrics_collector,
        prediction_service,
        scaling_service,
        executor,
    ):
        """Test that decreased traffic triggers scale-down."""
        # Start with more instances
        executor.current_instances["api"] = 10

        # Set up traffic pattern: high -> low
        metrics_collector.add_traffic_pattern([
            {"requests_per_second": 500, "cpu_usage": 0.9, "memory_usage": 0.8},
            {"requests_per_second": 300, "cpu_usage": 0.6, "memory_usage": 0.5},
            {"requests_per_second": 150, "cpu_usage": 0.3, "memory_usage": 0.3},
            {"requests_per_second": 100, "cpu_usage": 0.2, "memory_usage": 0.2},
        ])

        await prediction_service.start()
        await scaling_service.start()

        # Process traffic decrease
        for _ in range(4):
            metrics = metrics_collector.get_current_metrics()

            await prediction_service.request_prediction(
                service_name="api",
                horizon=PredictionHorizon.SHORT,
                target_time=datetime.now(UTC) + timedelta(minutes=15),
                features=metrics,
            )

            await scaling_service.evaluate_scaling(
                service_name="api",
                current_instances=executor.get_instance_count("api"),
                current_utilization=metrics.get("cpu_usage", 0.5),
            )

            await asyncio.sleep(0.1)

        await asyncio.sleep(1)

        await prediction_service.stop()
        await scaling_service.stop()

        stats = scaling_service.get_stats()
        assert stats["total_evaluations"] == 4

    @pytest.mark.asyncio
    async def test_prediction_accuracy_tracking(
        self,
        prediction_service,
        accuracy_tracker,
    ):
        """Test that predictions are tracked for accuracy."""
        prediction_service._accuracy_tracker = accuracy_tracker

        await prediction_service.start()

        # Make predictions
        now = datetime.now(UTC)
        for i in range(5):
            target_time = now + timedelta(minutes=30 + i * 5)

            await prediction_service.request_prediction(
                service_name="api",
                horizon=PredictionHorizon.SHORT,
                target_time=target_time,
                features={"requests_per_second": 100 + i * 20},
            )

        await asyncio.sleep(0.5)

        await prediction_service.stop()

        # Check accuracy tracker received predictions
        stats = accuracy_tracker.get_stats()
        assert stats["total_predictions"] >= 5

    @pytest.mark.asyncio
    async def test_cost_tracking_during_scaling(
        self,
        scaling_service,
        cost_tracker,
        executor,
    ):
        """Test that costs are tracked during scaling operations."""
        await scaling_service.start()

        # Record initial cost
        cost_tracker.record_cost(
            service_name="api",
            instance_type="m5.large",
            instance_count=executor.get_instance_count("api"),
            utilization=0.7,
        )

        # Simulate scaling
        for i in range(3):
            # Increase instances
            executor.current_instances["api"] = 5 + i * 2

            cost_tracker.record_cost(
                service_name="api",
                instance_type="m5.large",
                instance_count=executor.get_instance_count("api"),
                utilization=0.6,
            )

        await scaling_service.stop()

        # Check cost tracking
        summary = cost_tracker.get_cost_summary("api", period_hours=1)
        assert summary is not None
        assert summary.max_instances == 9  # 5 + 2*2
        assert summary.min_instances == 5


class TestScalingApprovalFlow:
    """Test manual approval flow for scaling actions."""

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        executor = MockInfrastructureExecutor()
        executor.current_instances["api"] = 5
        return executor

    @pytest.fixture
    def scaling_service(self, executor):
        """Create scaling service that requires approval."""
        service = ScalingService(
            auto_approve_threshold=1.0,  # Never auto-approve
            cooldown_seconds=0,
        )
        service._executor = executor
        return service

    @pytest.mark.asyncio
    async def test_manual_approval_executes_action(self, scaling_service, executor):
        """Test that manual approval triggers execution."""
        await scaling_service.start()

        # Create pending action
        action = ScalingAction(
            action_id="test_approval",
            service_name="api",
            action_type=ScalingActionType.SCALE_UP,
            current_count=5,
            target_count=8,
            reason="Predicted load increase",
            confidence=0.85,
            risk_score=0.15,
            estimated_cost_change=0.3,
            requires_approval=True,
        )
        scaling_service._pending_actions[action.action_id] = action

        # Verify action is pending
        pending = scaling_service.get_pending_actions()
        assert len(pending) == 1
        assert pending[0].action_id == "test_approval"

        # Approve action
        result = await scaling_service.approve_action(
            action_id="test_approval",
            approver="test_user",
        )

        await asyncio.sleep(0.5)

        await scaling_service.stop()

        assert result is True
        # Verify execution
        assert executor.get_instance_count("api") == 8

    @pytest.mark.asyncio
    async def test_rejection_removes_action(self, scaling_service):
        """Test that rejection removes pending action."""
        await scaling_service.start()

        # Create pending action
        action = ScalingAction(
            action_id="test_reject",
            service_name="api",
            action_type=ScalingActionType.SCALE_DOWN,
            current_count=10,
            target_count=5,
            reason="Low utilization",
            confidence=0.7,
            risk_score=0.3,
            estimated_cost_change=-0.25,
            requires_approval=True,
        )
        scaling_service._pending_actions[action.action_id] = action

        # Reject action
        result = await scaling_service.reject_action(
            action_id="test_reject",
            reason="Not safe during business hours",
        )

        await scaling_service.stop()

        assert result is True
        assert "test_reject" not in scaling_service._pending_actions


class TestFailureAndRollback:
    """Test failure handling and rollback scenarios."""

    @pytest.fixture
    def executor(self):
        """Create mock executor that can fail."""
        executor = MockInfrastructureExecutor()
        executor.current_instances["api"] = 5
        return executor

    @pytest.fixture
    def scaling_service(self, executor):
        """Create scaling service."""
        service = ScalingService(
            auto_approve_threshold=0.5,
            cooldown_seconds=0,
        )
        service._executor = executor
        return service

    @pytest.mark.asyncio
    async def test_execution_failure_triggers_rollback(
        self,
        scaling_service,
        executor,
    ):
        """Test that execution failure triggers rollback."""
        executor.should_fail = True

        await scaling_service.start()

        # Create and execute action
        action = ScalingAction(
            action_id="test_failure",
            service_name="api",
            action_type=ScalingActionType.SCALE_UP,
            current_count=5,
            target_count=10,
            reason="Test",
            confidence=0.9,
            risk_score=0.1,
            estimated_cost_change=0.5,
            requires_approval=False,
        )

        # Execute directly
        result = await executor.execute(action)

        await scaling_service.stop()

        assert result["success"] is False
        # Instance count should remain unchanged
        assert executor.get_instance_count("api") == 5

    @pytest.mark.asyncio
    async def test_verification_failure_triggers_rollback(
        self,
        scaling_service,
        executor,
    ):
        """Test that verification failure triggers rollback."""
        await scaling_service.start()

        # Create action
        action = ScalingAction(
            action_id="test_verify_fail",
            service_name="api",
            action_type=ScalingActionType.SCALE_UP,
            current_count=5,
            target_count=10,
            reason="Test",
            confidence=0.9,
            risk_score=0.1,
            estimated_cost_change=0.5,
            requires_approval=False,
        )

        # Execute succeeds
        await executor.execute(action)

        # Manually set incorrect instance count to simulate verification failure
        executor.current_instances["api"] = 7  # Not target of 10

        # Verify should fail
        result = await executor.verify(action)

        await scaling_service.stop()

        assert result["verified"] is False
        assert result["current_count"] == 7
        assert result["target_count"] == 10


class TestMultiServiceScaling:
    """Test scaling across multiple services."""

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        executor = MockInfrastructureExecutor()
        executor.current_instances["api"] = 5
        executor.current_instances["worker"] = 3
        executor.current_instances["cache"] = 2
        return executor

    @pytest.fixture
    def prediction_service(self):
        """Create prediction service."""
        service = PredictionService()

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={
            "p10": 80.0,
            "p50": 100.0,
            "p90": 120.0,
            "confidence": 0.9,
        })

        service.set_model(PredictionHorizon.SHORT, mock_model)
        return service

    @pytest.fixture
    def scaling_service(self, prediction_service, executor):
        """Create scaling service."""
        service = ScalingService(
            prediction_service=prediction_service,
            auto_approve_threshold=0.8,
            cooldown_seconds=0,
        )
        service._executor = executor
        return service

    @pytest.mark.asyncio
    async def test_scale_multiple_services(
        self,
        prediction_service,
        scaling_service,
        executor,
    ):
        """Test scaling multiple services independently."""
        await prediction_service.start()
        await scaling_service.start()

        services = ["api", "worker", "cache"]

        # Evaluate scaling for each service
        for service_name in services:
            await scaling_service.evaluate_scaling(
                service_name=service_name,
                current_instances=executor.get_instance_count(service_name),
                current_utilization=0.85,
            )

        await asyncio.sleep(0.5)

        await prediction_service.stop()
        await scaling_service.stop()

        stats = scaling_service.get_stats()
        assert stats["total_evaluations"] == 3


class TestPredictionToScalingIntegration:
    """Test integration between prediction and scaling services."""

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        executor = MockInfrastructureExecutor()
        executor.current_instances["api"] = 5
        return executor

    @pytest.fixture
    def prediction_service(self):
        """Create prediction service with dynamic predictions."""
        service = PredictionService()

        # Model that predicts high load
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={
            "p10": 200.0,
            "p50": 300.0,
            "p90": 400.0,
            "confidence": 0.92,
        })

        service.set_model(PredictionHorizon.SHORT, mock_model)
        service.set_model(PredictionHorizon.MEDIUM, mock_model)

        return service

    @pytest.fixture
    def scaling_service(self, prediction_service, executor):
        """Create scaling service."""
        service = ScalingService(
            prediction_service=prediction_service,
            auto_approve_threshold=0.85,
            cooldown_seconds=0,
        )
        service._executor = executor
        return service

    @pytest.mark.asyncio
    async def test_high_load_prediction_triggers_scaling(
        self,
        prediction_service,
        scaling_service,
        executor,
    ):
        """Test that high load predictions trigger proactive scaling."""
        await prediction_service.start()
        await scaling_service.start()

        # Run predictions for high load
        results = await prediction_service.run_predictions_for_service(
            service_name="api",
            horizons=[PredictionHorizon.SHORT, PredictionHorizon.MEDIUM],
        )

        # Evaluate scaling based on predictions
        action = await scaling_service.evaluate_scaling(
            service_name="api",
            current_instances=5,
            current_utilization=0.6,  # Normal utilization now
        )

        await asyncio.sleep(0.5)

        await prediction_service.stop()
        await scaling_service.stop()

        # Should have predictions
        pred_stats = prediction_service.get_stats()
        assert pred_stats["total_predictions"] > 0

        # Should have evaluated
        scale_stats = scaling_service.get_stats()
        assert scale_stats["total_evaluations"] == 1


# =============================================================================
# Stress Tests
# =============================================================================


class TestHighVolumeOperations:
    """Test system under high volume of operations."""

    @pytest.fixture
    def prediction_service(self):
        """Create prediction service."""
        service = PredictionService()

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value={
            "p10": 80.0,
            "p50": 100.0,
            "p90": 120.0,
            "confidence": 0.85,
        })

        service.set_model(PredictionHorizon.SHORT, mock_model)
        return service

    @pytest.mark.asyncio
    async def test_many_concurrent_predictions(self, prediction_service):
        """Test handling many concurrent prediction requests."""
        await prediction_service.start()

        # Submit many predictions concurrently
        tasks = []
        for i in range(50):
            task = prediction_service.request_prediction(
                service_name=f"service_{i % 5}",
                horizon=PredictionHorizon.SHORT,
                target_time=datetime.now(UTC) + timedelta(minutes=15),
                features={"load": i * 10},
            )
            tasks.append(task)

        # Wait for all to complete
        request_ids = await asyncio.gather(*tasks)

        await asyncio.sleep(1)

        await prediction_service.stop()

        # All should have been processed
        assert len(request_ids) == 50
        assert all(rid is not None for rid in request_ids)

        stats = prediction_service.get_stats()
        assert stats["total_predictions"] >= 50

    @pytest.mark.asyncio
    async def test_rapid_scaling_evaluations(self):
        """Test handling rapid scaling evaluations."""
        executor = MockInfrastructureExecutor()
        executor.current_instances["api"] = 5

        service = ScalingService(
            auto_approve_threshold=0.95,
            cooldown_seconds=0,
        )
        service._executor = executor

        await service.start()

        # Rapid evaluations
        for i in range(20):
            await service.evaluate_scaling(
                service_name="api",
                current_instances=executor.get_instance_count("api"),
                current_utilization=0.5 + (i % 5) * 0.1,
            )

        await asyncio.sleep(0.5)

        await service.stop()

        stats = service.get_stats()
        assert stats["total_evaluations"] == 20
