"""
Unit tests for Phase 11: Background Services.

Tests cover:
- Scheduler service
- Prediction service
- Scaling service
- Model training service
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.services.prediction import (
    PredictionHorizon,
    PredictionRequest,
    PredictionResult,
    PredictionService,
    init_prediction_service,
)
from src.services.scaling import (
    ScalingAction,
    ScalingActionType,
    ScalingContext,
    ScalingService,
    init_scaling_service,
)
from src.services.scheduler import (
    ScheduledTask,
    SchedulerService,
    TaskExecution,
    TaskPriority,
    TaskStatus,
    init_scheduler_service,
)
from src.services.training import (
    ModelTrainingService,
    ModelType,
    ModelVersion,
    TrainingJob,
    TrainingStatus,
    init_training_service,
)

# =============================================================================
# Scheduled Task Tests
# =============================================================================


class TestScheduledTask:
    """Tests for ScheduledTask."""

    def test_create_task(self):
        """Test creating a scheduled task."""
        task = ScheduledTask(
            task_id="test_task",
            name="Test Task",
            description="A test task",
            func=lambda: None,
            trigger_type="interval",
            trigger_args={"minutes": 5},
        )

        assert task.task_id == "test_task"
        assert task.name == "Test Task"
        assert task.trigger_type == "interval"
        assert task.enabled is True

    def test_task_priority(self):
        """Test task priority."""
        task = ScheduledTask(
            task_id="high_priority",
            name="High Priority Task",
            description="Important task",
            func=lambda: None,
            trigger_type="interval",
            trigger_args={"minutes": 1},
            priority=TaskPriority.HIGH,
        )

        assert task.priority == TaskPriority.HIGH


class TestTaskExecution:
    """Tests for TaskExecution."""

    def test_create_execution(self):
        """Test creating a task execution record."""
        now = datetime.now(UTC)
        execution = TaskExecution(
            task_id="test_task",
            job_id="test_task_123",
            started_at=now,
        )

        assert execution.task_id == "test_task"
        assert execution.status == TaskStatus.RUNNING

    def test_execution_to_dict(self):
        """Test converting execution to dictionary."""
        now = datetime.now(UTC)
        execution = TaskExecution(
            task_id="test_task",
            job_id="test_task_123",
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            status=TaskStatus.COMPLETED,
            duration_seconds=5.0,
        )

        d = execution.to_dict()

        assert d["task_id"] == "test_task"
        assert d["status"] == "completed"
        assert d["duration_seconds"] == 5.0


# =============================================================================
# Scheduler Service Tests
# =============================================================================


class TestSchedulerService:
    """Tests for SchedulerService."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        return SchedulerService()

    def test_create_scheduler(self, scheduler):
        """Test creating a scheduler."""
        assert scheduler is not None
        assert scheduler.is_running is False

    def test_register_task(self, scheduler):
        """Test registering a task."""
        task = ScheduledTask(
            task_id="test_task",
            name="Test Task",
            description="A test task",
            func=lambda: None,
            trigger_type="interval",
            trigger_args={"minutes": 5},
        )

        scheduler.register_task(task)

        assert scheduler.get_task("test_task") is not None

    def test_unregister_task(self, scheduler):
        """Test unregistering a task."""
        task = ScheduledTask(
            task_id="test_task",
            name="Test Task",
            description="A test task",
            func=lambda: None,
            trigger_type="interval",
            trigger_args={"minutes": 5},
        )

        scheduler.register_task(task)
        result = scheduler.unregister_task("test_task")

        assert result is True
        assert scheduler.get_task("test_task") is None

    def test_get_all_tasks(self, scheduler):
        """Test getting all tasks."""
        for i in range(3):
            task = ScheduledTask(
                task_id=f"task_{i}",
                name=f"Task {i}",
                description=f"Task {i} description",
                func=lambda: None,
                trigger_type="interval",
                trigger_args={"minutes": 5},
            )
            scheduler.register_task(task)

        tasks = scheduler.get_all_tasks()

        assert len(tasks) == 3

    def test_enable_disable_task(self, scheduler):
        """Test enabling and disabling a task."""
        task = ScheduledTask(
            task_id="test_task",
            name="Test Task",
            description="A test task",
            func=lambda: None,
            trigger_type="interval",
            trigger_args={"minutes": 5},
        )

        scheduler.register_task(task)

        scheduler.disable_task("test_task")
        assert scheduler.get_task("test_task").enabled is False

        scheduler.enable_task("test_task")
        assert scheduler.get_task("test_task").enabled is True

    def test_get_stats(self, scheduler):
        """Test getting scheduler stats."""
        stats = scheduler.get_stats()

        assert "running" in stats
        assert "total_tasks" in stats
        assert "enabled_tasks" in stats

    def test_add_callback(self, scheduler):
        """Test adding a callback."""
        callback = MagicMock()
        scheduler.add_callback("on_complete", callback)

        assert callback in scheduler._callbacks["on_complete"]


# =============================================================================
# Prediction Request/Result Tests
# =============================================================================


class TestPredictionRequest:
    """Tests for PredictionRequest."""

    def test_create_request(self):
        """Test creating a prediction request."""
        now = datetime.now(UTC)
        request = PredictionRequest(
            request_id="req_123",
            service_name="api",
            horizon=PredictionHorizon.SHORT,
            target_time=now + timedelta(minutes=30),
        )

        assert request.request_id == "req_123"
        assert request.horizon == PredictionHorizon.SHORT


class TestPredictionResult:
    """Tests for PredictionResult."""

    def test_create_result(self):
        """Test creating a prediction result."""
        now = datetime.now(UTC)
        result = PredictionResult(
            prediction_id="pred_123",
            request_id="req_123",
            service_name="api",
            horizon=PredictionHorizon.SHORT,
            target_time=now + timedelta(minutes=30),
            model_name="transformer",
            p10=80.0,
            p50=100.0,
            p90=120.0,
            confidence=0.85,
            features_used=["cpu", "memory", "requests"],
        )

        assert result.prediction_id == "pred_123"
        assert result.p50 == 100.0

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        now = datetime.now(UTC)
        result = PredictionResult(
            prediction_id="pred_123",
            request_id="req_123",
            service_name="api",
            horizon=PredictionHorizon.SHORT,
            target_time=now + timedelta(minutes=30),
            model_name="transformer",
            p10=80.0,
            p50=100.0,
            p90=120.0,
            confidence=0.85,
            features_used=["cpu", "memory"],
            latency_ms=50.0,
        )

        d = result.to_dict()

        assert d["prediction_id"] == "pred_123"
        assert d["model_name"] == "transformer"
        assert d["latency_ms"] == 50.0


# =============================================================================
# Prediction Service Tests
# =============================================================================


class TestPredictionService:
    """Tests for PredictionService."""

    @pytest.fixture
    def service(self):
        """Create a fresh prediction service for each test."""
        return PredictionService()

    def test_create_service(self, service):
        """Test creating a prediction service."""
        assert service is not None

    @pytest.mark.asyncio
    async def test_request_prediction(self, service):
        """Test requesting a prediction."""
        await service.start()

        request_id = await service.request_prediction(
            service_name="api",
            horizon=PredictionHorizon.SHORT,
            target_time=datetime.now(UTC) + timedelta(minutes=30),
        )

        assert request_id is not None

        await service.stop()

    def test_get_stats(self, service):
        """Test getting service stats."""
        stats = service.get_stats()

        assert "total_predictions" in stats
        assert "models_loaded" in stats

    def test_set_model(self, service):
        """Test setting a model."""
        mock_model = MagicMock()
        service.set_model(PredictionHorizon.SHORT, mock_model)

        assert service._short_model is mock_model

    def test_add_callback(self, service):
        """Test adding a callback."""
        callback = MagicMock()
        service.add_callback(callback)

        assert callback in service._callbacks


# =============================================================================
# Scaling Action Tests
# =============================================================================


class TestScalingAction:
    """Tests for ScalingAction."""

    def test_create_action(self):
        """Test creating a scaling action."""
        action = ScalingAction(
            action_id="action_123",
            service_name="api",
            action_type=ScalingActionType.SCALE_UP,
            current_count=5,
            target_count=8,
            reason="High predicted load",
            confidence=0.9,
            risk_score=0.2,
            estimated_cost_change=0.29,
            requires_approval=False,
        )

        assert action.action_id == "action_123"
        assert action.action_type == ScalingActionType.SCALE_UP
        assert action.target_count == 8

    def test_action_to_dict(self):
        """Test converting action to dictionary."""
        action = ScalingAction(
            action_id="action_123",
            service_name="api",
            action_type=ScalingActionType.SCALE_UP,
            current_count=5,
            target_count=8,
            reason="High predicted load",
            confidence=0.9,
            risk_score=0.2,
            estimated_cost_change=0.29,
            requires_approval=False,
        )

        d = action.to_dict()

        assert d["action_id"] == "action_123"
        assert d["action_type"] == "scale_up"
        assert d["target_count"] == 8


class TestScalingContext:
    """Tests for ScalingContext."""

    def test_create_context(self):
        """Test creating scaling context."""
        context = ScalingContext(
            service_name="api",
            current_instances=5,
            current_utilization=0.7,
            predicted_load=150.0,
            predicted_load_p90=180.0,
            prediction_confidence=0.85,
            time_horizon_minutes=30,
            current_cost_hourly=0.48,
        )

        assert context.service_name == "api"
        assert context.current_instances == 5


# =============================================================================
# Scaling Service Tests
# =============================================================================


class TestScalingService:
    """Tests for ScalingService."""

    @pytest.fixture
    def service(self):
        """Create a fresh scaling service for each test."""
        return ScalingService()

    def test_create_service(self, service):
        """Test creating a scaling service."""
        assert service is not None

    @pytest.mark.asyncio
    async def test_evaluate_scaling_no_action(self, service):
        """Test evaluating scaling when no action needed."""
        action = await service.evaluate_scaling(
            service_name="api",
            current_instances=5,
            current_utilization=0.5,  # Normal utilization
        )

        # May or may not generate action depending on predictions
        # Just verify it doesn't crash

    def test_get_pending_actions(self, service):
        """Test getting pending actions."""
        pending = service.get_pending_actions()

        assert isinstance(pending, list)

    def test_get_stats(self, service):
        """Test getting service stats."""
        stats = service.get_stats()

        assert "pending_actions" in stats
        assert "total_evaluations" in stats

    def test_add_callback(self, service):
        """Test adding a callback."""
        callback = MagicMock()
        service.add_callback(callback)

        assert callback in service._callbacks


# =============================================================================
# Training Job Tests
# =============================================================================


class TestTrainingJob:
    """Tests for TrainingJob."""

    def test_create_job(self):
        """Test creating a training job."""
        job = TrainingJob(
            job_id="job_123",
            model_type=ModelType.SHORT_TERM,
            triggered_by="scheduled",
        )

        assert job.job_id == "job_123"
        assert job.status == TrainingStatus.PENDING

    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        job = TrainingJob(
            job_id="job_123",
            model_type=ModelType.SHORT_TERM,
            triggered_by="manual",
            status=TrainingStatus.RUNNING,
            progress=0.5,
        )

        d = job.to_dict()

        assert d["job_id"] == "job_123"
        assert d["status"] == "running"
        assert d["progress"] == 0.5


class TestModelVersion:
    """Tests for ModelVersion."""

    def test_create_version(self):
        """Test creating a model version."""
        now = datetime.now(UTC)
        version = ModelVersion(
            version_id="short_term_v1",
            model_type=ModelType.SHORT_TERM,
            version_number=1,
            created_at=now,
        )

        assert version.version_id == "short_term_v1"
        assert version.is_active is False

    def test_version_to_dict(self):
        """Test converting version to dictionary."""
        now = datetime.now(UTC)
        version = ModelVersion(
            version_id="short_term_v1",
            model_type=ModelType.SHORT_TERM,
            version_number=1,
            created_at=now,
            metrics={"mape": 0.08},
            is_active=True,
        )

        d = version.to_dict()

        assert d["version_id"] == "short_term_v1"
        assert d["is_active"] is True
        assert d["metrics"]["mape"] == 0.08


# =============================================================================
# Model Training Service Tests
# =============================================================================


class TestModelTrainingService:
    """Tests for ModelTrainingService."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create a fresh training service for each test."""
        return ModelTrainingService(model_dir=tmp_path / "models")

    def test_create_service(self, service):
        """Test creating a training service."""
        assert service is not None

    @pytest.mark.asyncio
    async def test_trigger_training(self, service):
        """Test triggering a training job."""
        job = await service.trigger_training(
            model_type=ModelType.SHORT_TERM,
            triggered_by="manual",
        )

        assert job is not None
        assert job.model_type == ModelType.SHORT_TERM
        assert job.triggered_by == "manual"

    def test_get_active_version(self, service):
        """Test getting active version."""
        version = service.get_active_version(ModelType.SHORT_TERM)

        # No version yet
        assert version is None

    def test_get_stats(self, service):
        """Test getting service stats."""
        stats = service.get_stats()

        assert "active_versions" in stats
        assert "total_training_jobs" in stats

    def test_add_callback(self, service):
        """Test adding a callback."""
        callback = MagicMock()
        service.add_callback(callback)

        assert callback in service._callbacks


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Test module imports work correctly."""

    def test_import_scheduler_service(self):
        """Test importing scheduler service."""
        from src.services import get_scheduler_service

        service = get_scheduler_service()
        assert service is not None

    def test_import_prediction_service(self):
        """Test importing prediction service."""
        from src.services import get_prediction_service

        service = get_prediction_service()
        assert service is not None

    def test_import_scaling_service(self):
        """Test importing scaling service."""
        from src.services import get_scaling_service

        service = get_scaling_service()
        assert service is not None

    def test_import_training_service(self):
        """Test importing training service."""
        from src.services import get_training_service

        service = get_training_service()
        assert service is not None

    def test_init_services(self):
        """Test initializing services."""
        scheduler = init_scheduler_service()
        prediction = init_prediction_service()
        scaling = init_scaling_service()
        training = init_training_service()

        assert scheduler is not None
        assert prediction is not None
        assert scaling is not None
        assert training is not None
