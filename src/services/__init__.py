"""
Background Services for the predictive scaling system.

This module provides:
- Scheduler service for coordinating background tasks
- Prediction service for running ML predictions
- Scaling service for executing scaling operations
- Model training service for maintaining ML models
"""

from src.services.prediction import (
    PredictionHorizon,
    PredictionRequest,
    PredictionResult,
    PredictionService,
    get_prediction_service,
    init_prediction_service,
)
from src.services.scaling import (
    ScalingAction,
    ScalingActionStatus,
    ScalingActionType,
    ScalingContext,
    ScalingService,
    get_scaling_service,
    init_scaling_service,
)
from src.services.scheduler import (
    ScheduledTask,
    SchedulerService,
    TaskExecution,
    TaskPriority,
    TaskStatus,
    create_scheduler_with_default_tasks,
    get_scheduler_service,
    init_scheduler_service,
)
from src.services.training import (
    DeploymentRecord,
    DeploymentStatus,
    ModelTrainingService,
    ModelType,
    ModelVersion,
    TrainingJob,
    TrainingStatus,
    get_training_service,
    init_training_service,
)

__all__ = [
    # Scheduler
    "ScheduledTask",
    "SchedulerService",
    "TaskExecution",
    "TaskPriority",
    "TaskStatus",
    "create_scheduler_with_default_tasks",
    "get_scheduler_service",
    "init_scheduler_service",
    # Prediction
    "PredictionHorizon",
    "PredictionRequest",
    "PredictionResult",
    "PredictionService",
    "get_prediction_service",
    "init_prediction_service",
    # Scaling
    "ScalingAction",
    "ScalingActionStatus",
    "ScalingActionType",
    "ScalingContext",
    "ScalingService",
    "get_scaling_service",
    "init_scaling_service",
    # Training
    "DeploymentRecord",
    "DeploymentStatus",
    "ModelTrainingService",
    "ModelType",
    "ModelVersion",
    "TrainingJob",
    "TrainingStatus",
    "get_training_service",
    "init_training_service",
]
