"""
Model Training Service for maintaining ML models.

Responsibilities:
- Monitor model performance
- Trigger retraining when accuracy degrades
- Handle model versioning
- Blue-green model deployment (train new, verify, swap)
"""

import asyncio
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Types of models."""

    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class TrainingStatus(str, Enum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStatus(str, Enum):
    """Status of model deployment."""

    STAGED = "staged"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """A version of a model."""

    version_id: str
    model_type: ModelType
    version_number: int
    created_at: datetime
    trained_at: datetime | None = None
    deployed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    is_active: bool = False
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_type": self.model_type.value,
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "metrics": self.metrics,
            "is_active": self.is_active,
        }


@dataclass
class TrainingJob:
    """A model training job."""

    job_id: str
    model_type: ModelType
    triggered_by: str  # "scheduled", "manual", "accuracy_degradation"
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    training_config: dict[str, Any] = field(default_factory=dict)
    result_metrics: dict[str, float] = field(default_factory=dict)
    result_version_id: str | None = None
    error: str | None = None
    progress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "model_type": self.model_type.value,
            "triggered_by": self.triggered_by,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_metrics": self.result_metrics,
            "error": self.error,
            "progress": self.progress,
        }


@dataclass
class DeploymentRecord:
    """Record of a model deployment."""

    deployment_id: str
    model_type: ModelType
    old_version_id: str | None
    new_version_id: str
    status: DeploymentStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    validation_results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ModelTrainingService:
    """
    Background service for model training and maintenance.

    Handles:
    - Scheduled model retraining
    - Performance-triggered retraining
    - Model versioning
    - Blue-green deployment
    """

    def __init__(
        self,
        model_dir: Path | str = "./models",
        accuracy_tracker: Any = None,
        prediction_service: Any = None,
        metrics: Any = None,
        mape_threshold: float = 0.15,
        coverage_threshold: float = 0.75,
        min_training_interval_hours: int = 12,
        validation_samples: int = 100,
    ) -> None:
        """
        Initialize model training service.

        Args:
            model_dir: Directory for model storage
            accuracy_tracker: Accuracy tracker for monitoring
            prediction_service: Prediction service to update models
            metrics: Metrics exporter
            mape_threshold: MAPE threshold for retraining trigger
            coverage_threshold: Coverage threshold for retraining
            min_training_interval_hours: Minimum hours between retraining
            validation_samples: Samples for model validation
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._accuracy_tracker = accuracy_tracker
        self._prediction_service = prediction_service
        self._metrics = metrics
        self._mape_threshold = mape_threshold
        self._coverage_threshold = coverage_threshold
        self._min_interval = timedelta(hours=min_training_interval_hours)
        self._validation_samples = validation_samples

        self._running = False
        self._versions: dict[ModelType, list[ModelVersion]] = {
            t: [] for t in ModelType
        }
        self._active_versions: dict[ModelType, ModelVersion | None] = dict.fromkeys(ModelType)
        self._training_jobs: dict[str, TrainingJob] = {}
        self._job_history: list[TrainingJob] = []
        self._deployments: list[DeploymentRecord] = []
        self._last_training: dict[ModelType, datetime] = {}
        self._callbacks: list[callable] = []

        # Statistics
        self._stats = {
            "total_training_jobs": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "total_deployments": 0,
            "successful_deployments": 0,
            "rollbacks": 0,
        }

        logger.info(
            "Model training service initialized",
            model_dir=str(self._model_dir),
        )

    def add_callback(self, callback: callable) -> None:
        """Add callback for training events."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the training service."""
        if self._running:
            return

        self._running = True

        # Load existing model versions
        self._load_existing_versions()

        # Start background monitoring
        asyncio.create_task(self._monitor_accuracy())

        logger.info("Model training service started")

    async def stop(self) -> None:
        """Stop the training service."""
        self._running = False
        logger.info("Model training service stopped")

    def _load_existing_versions(self) -> None:
        """Load existing model versions from disk."""
        for model_type in ModelType:
            model_path = self._model_dir / model_type.value
            if model_path.exists():
                # Look for version directories
                for version_dir in sorted(model_path.iterdir()):
                    if version_dir.is_dir() and version_dir.name.startswith("v"):
                        try:
                            version_num = int(version_dir.name[1:])
                            version = ModelVersion(
                                version_id=f"{model_type.value}_{version_dir.name}",
                                model_type=model_type,
                                version_number=version_num,
                                created_at=datetime.fromtimestamp(
                                    version_dir.stat().st_ctime,
                                    tz=UTC,
                                ),
                                path=str(version_dir),
                            )
                            self._versions[model_type].append(version)
                        except ValueError:
                            continue

                # Set latest as active
                if self._versions[model_type]:
                    latest = max(
                        self._versions[model_type],
                        key=lambda v: v.version_number,
                    )
                    latest.is_active = True
                    self._active_versions[model_type] = latest

    async def _monitor_accuracy(self) -> None:
        """Monitor model accuracy and trigger retraining if needed."""
        while self._running:
            try:
                if self._accuracy_tracker:
                    for model_type in ModelType:
                        await self._check_model_accuracy(model_type)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error("Accuracy monitoring error", error=str(e))
                await asyncio.sleep(60)

    async def _check_model_accuracy(self, model_type: ModelType) -> None:
        """Check accuracy for a model and trigger retraining if needed."""
        metrics = self._accuracy_tracker.get_latest_metrics(
            model_name=self._model_type_to_name(model_type),
        )

        if not metrics:
            return

        for metric in metrics:
            if (
                metric.mape > self._mape_threshold
                or metric.coverage_80 < self._coverage_threshold
            ):
                # Check if we can retrain
                if self._can_retrain(model_type):
                    logger.warning(
                        "Accuracy degradation detected, triggering retraining",
                        model=model_type.value,
                        mape=metric.mape,
                        coverage=metric.coverage_80,
                    )
                    await self.trigger_training(
                        model_type,
                        triggered_by="accuracy_degradation",
                    )
                    break

    def _model_type_to_name(self, model_type: ModelType) -> str:
        """Convert model type to model name."""
        if model_type == ModelType.SHORT_TERM:
            return "transformer"
        elif model_type == ModelType.MEDIUM_TERM:
            return "gbm_ensemble"
        else:
            return "prophet"

    def _can_retrain(self, model_type: ModelType) -> bool:
        """Check if model can be retrained (not in cooldown)."""
        last = self._last_training.get(model_type)
        if last is None:
            return True
        return datetime.now(UTC) - last >= self._min_interval

    async def trigger_training(
        self,
        model_type: ModelType,
        triggered_by: str = "manual",
        config: dict[str, Any] | None = None,
    ) -> TrainingJob:
        """
        Trigger a training job.

        Args:
            model_type: Type of model to train
            triggered_by: What triggered the training
            config: Optional training configuration

        Returns:
            TrainingJob
        """
        job = TrainingJob(
            job_id=str(uuid4()),
            model_type=model_type,
            triggered_by=triggered_by,
            training_config=config or {},
        )

        self._training_jobs[job.job_id] = job
        self._stats["total_training_jobs"] += 1

        logger.info(
            "Training job created",
            job_id=job.job_id,
            model=model_type.value,
            triggered_by=triggered_by,
        )

        # Start training in background
        asyncio.create_task(self._run_training(job))

        return job

    async def _run_training(self, job: TrainingJob) -> None:
        """Run a training job."""
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now(UTC)

        try:
            # Create new version
            version_number = self._get_next_version_number(job.model_type)
            version_id = f"{job.model_type.value}_v{version_number}"
            version_path = self._model_dir / job.model_type.value / f"v{version_number}"
            version_path.mkdir(parents=True, exist_ok=True)

            # Simulate training (in production, would train actual model)
            await self._train_model(job, version_path)

            # Validate the model
            job.status = TrainingStatus.VALIDATING
            validation_result = await self._validate_model(job, version_path)

            if validation_result["passed"]:
                # Create version record
                version = ModelVersion(
                    version_id=version_id,
                    model_type=job.model_type,
                    version_number=version_number,
                    created_at=datetime.now(UTC),
                    trained_at=datetime.now(UTC),
                    metrics=validation_result["metrics"],
                    path=str(version_path),
                )

                self._versions[job.model_type].append(version)
                job.result_version_id = version_id
                job.result_metrics = validation_result["metrics"]
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now(UTC)
                job.progress = 1.0

                self._stats["successful_trainings"] += 1
                self._last_training[job.model_type] = datetime.now(UTC)

                logger.info(
                    "Training completed",
                    job_id=job.job_id,
                    version_id=version_id,
                    metrics=validation_result["metrics"],
                )

                # Auto-deploy if validation passes well
                if validation_result["metrics"].get("mape", 1.0) < self._mape_threshold:
                    await self.deploy_version(version_id)

            else:
                job.status = TrainingStatus.FAILED
                job.error = "Validation failed"
                job.completed_at = datetime.now(UTC)
                self._stats["failed_trainings"] += 1

                # Cleanup failed version
                shutil.rmtree(version_path, ignore_errors=True)

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(UTC)
            self._stats["failed_trainings"] += 1
            logger.error("Training failed", job_id=job.job_id, error=str(e))

        finally:
            # Move to history
            self._training_jobs.pop(job.job_id, None)
            self._job_history.append(job)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(job)
                    else:
                        callback(job)
                except Exception as e:
                    logger.error("Training callback error", error=str(e))

    async def _train_model(
        self,
        job: TrainingJob,
        output_path: Path,
    ) -> None:
        """Train a model (simulated for now)."""
        # In production, this would:
        # 1. Load training data
        # 2. Run feature engineering
        # 3. Train the model
        # 4. Save model artifacts

        # Simulate training progress
        for progress in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            job.progress = progress
            await asyncio.sleep(1)

        # Save placeholder model file
        model_file = output_path / "model.pkl"
        model_file.write_text(f"placeholder_model_{job.model_type.value}")

        logger.debug("Model trained", job_id=job.job_id)

    async def _validate_model(
        self,
        job: TrainingJob,
        model_path: Path,
    ) -> dict[str, Any]:
        """Validate a trained model."""
        # In production, this would:
        # 1. Load the model
        # 2. Run predictions on validation set
        # 3. Calculate accuracy metrics
        # 4. Compare to current model

        await asyncio.sleep(1)  # Simulate validation

        # Mock validation results
        return {
            "passed": True,
            "metrics": {
                "mape": 0.08,
                "mae": 5.2,
                "coverage_80": 0.82,
            },
            "samples_validated": self._validation_samples,
        }

    def _get_next_version_number(self, model_type: ModelType) -> int:
        """Get next version number for a model type."""
        versions = self._versions[model_type]
        if not versions:
            return 1
        return max(v.version_number for v in versions) + 1

    async def deploy_version(
        self,
        version_id: str,
        validate: bool = True,
    ) -> DeploymentRecord:
        """
        Deploy a model version.

        Args:
            version_id: Version to deploy
            validate: Whether to validate before deploying

        Returns:
            DeploymentRecord
        """
        # Find the version
        version = None
        for _model_type, versions in self._versions.items():
            for v in versions:
                if v.version_id == version_id:
                    version = v
                    break
            if version:
                break

        if version is None:
            raise ValueError(f"Version not found: {version_id}")

        old_version = self._active_versions[version.model_type]

        deployment = DeploymentRecord(
            deployment_id=str(uuid4()),
            model_type=version.model_type,
            old_version_id=old_version.version_id if old_version else None,
            new_version_id=version_id,
            status=DeploymentStatus.STAGED,
        )

        self._deployments.append(deployment)
        self._stats["total_deployments"] += 1

        logger.info(
            "Deploying model version",
            deployment_id=deployment.deployment_id,
            version_id=version_id,
        )

        try:
            if validate:
                deployment.status = DeploymentStatus.VALIDATING
                validation = await self._validate_deployment(version)

                if not validation["passed"]:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.error = "Deployment validation failed"
                    deployment.completed_at = datetime.now(UTC)
                    return deployment

                deployment.validation_results = validation

            # Swap active version
            if old_version:
                old_version.is_active = False

            version.is_active = True
            version.deployed_at = datetime.now(UTC)
            self._active_versions[version.model_type] = version

            # Update prediction service
            if self._prediction_service:
                # In production, would load and set the model
                pass

            deployment.status = DeploymentStatus.DEPLOYED
            deployment.completed_at = datetime.now(UTC)
            self._stats["successful_deployments"] += 1

            logger.info(
                "Model deployed",
                deployment_id=deployment.deployment_id,
                version_id=version_id,
            )

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error = str(e)
            deployment.completed_at = datetime.now(UTC)

            # Rollback if needed
            if old_version:
                old_version.is_active = True
                version.is_active = False
                self._active_versions[version.model_type] = old_version

            logger.error("Deployment failed", error=str(e))

        return deployment

    async def _validate_deployment(
        self,
        version: ModelVersion,
    ) -> dict[str, Any]:
        """Validate a model deployment."""
        await asyncio.sleep(1)  # Simulate validation

        return {
            "passed": True,
            "latency_p50_ms": 50,
            "latency_p99_ms": 150,
            "memory_mb": 256,
        }

    async def rollback(self, model_type: ModelType) -> bool:
        """
        Rollback to previous model version.

        Args:
            model_type: Type of model to rollback

        Returns:
            True if rollback successful
        """
        versions = self._versions[model_type]
        if len(versions) < 2:
            return False

        current = self._active_versions[model_type]
        if not current:
            return False

        # Find previous version
        sorted_versions = sorted(versions, key=lambda v: v.version_number, reverse=True)
        previous = None
        for v in sorted_versions:
            if v.version_id != current.version_id:
                previous = v
                break

        if not previous:
            return False

        # Swap
        current.is_active = False
        previous.is_active = True
        previous.deployed_at = datetime.now(UTC)
        self._active_versions[model_type] = previous

        self._stats["rollbacks"] += 1

        logger.info(
            "Model rolled back",
            model=model_type.value,
            from_version=current.version_id,
            to_version=previous.version_id,
        )

        return True

    def get_active_version(self, model_type: ModelType) -> ModelVersion | None:
        """Get the active version for a model type."""
        return self._active_versions.get(model_type)

    def get_all_versions(self, model_type: ModelType) -> list[ModelVersion]:
        """Get all versions for a model type."""
        return self._versions.get(model_type, [])

    def get_training_job(self, job_id: str) -> TrainingJob | None:
        """Get a training job by ID."""
        if job_id in self._training_jobs:
            return self._training_jobs[job_id]
        for job in self._job_history:
            if job.job_id == job_id:
                return job
        return None

    def get_running_jobs(self) -> list[TrainingJob]:
        """Get all running training jobs."""
        return [j for j in self._training_jobs.values() if j.status == TrainingStatus.RUNNING]

    def get_job_history(self, limit: int = 100) -> list[TrainingJob]:
        """Get training job history."""
        return list(reversed(self._job_history[-limit:]))

    def get_deployment_history(self, limit: int = 100) -> list[DeploymentRecord]:
        """Get deployment history."""
        return list(reversed(self._deployments[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "running": self._running,
            "active_versions": {
                t.value: v.version_id if v else None
                for t, v in self._active_versions.items()
            },
            "total_versions": {
                t.value: len(v) for t, v in self._versions.items()
            },
            "running_jobs": len(self.get_running_jobs()),
            "total_training_jobs": self._stats["total_training_jobs"],
            "successful_trainings": self._stats["successful_trainings"],
            "failed_trainings": self._stats["failed_trainings"],
            "total_deployments": self._stats["total_deployments"],
            "successful_deployments": self._stats["successful_deployments"],
            "rollbacks": self._stats["rollbacks"],
        }


# Global instance
_training_service: ModelTrainingService | None = None


def get_training_service() -> ModelTrainingService:
    """Get the global training service."""
    global _training_service
    if _training_service is None:
        _training_service = ModelTrainingService()
    return _training_service


def init_training_service(**kwargs) -> ModelTrainingService:
    """Initialize the global training service."""
    global _training_service
    _training_service = ModelTrainingService(**kwargs)
    return _training_service
