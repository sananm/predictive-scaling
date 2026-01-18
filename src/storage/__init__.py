"""
Storage module for the predictive scaling system.

Provides database connectivity, models, and repositories.
"""

from src.storage.database import (
    Base,
    async_session_factory,
    close_db,
    engine,
    get_db,
    get_session,
    init_db,
)
from src.storage.models import (
    AlertLog,
    BusinessEvent,
    CostTracking,
    Feature,
    Metric,
    ModelPerformance,
    Prediction,
    ScalingDecision,
)
from src.storage.repositories import (
    AlertLogsRepository,
    BaseRepository,
    BusinessEventsRepository,
    CostTrackingRepository,
    FeaturesRepository,
    MetricsRepository,
    ModelPerformanceRepository,
    PredictionsRepository,
    ScalingDecisionsRepository,
)

__all__ = [
    # Database
    "Base",
    "async_session_factory",
    "close_db",
    "engine",
    "get_db",
    "get_session",
    "init_db",
    # Models
    "AlertLog",
    "BusinessEvent",
    "CostTracking",
    "Feature",
    "Metric",
    "ModelPerformance",
    "Prediction",
    "ScalingDecision",
    # Repositories
    "AlertLogsRepository",
    "BaseRepository",
    "BusinessEventsRepository",
    "CostTrackingRepository",
    "FeaturesRepository",
    "MetricsRepository",
    "ModelPerformanceRepository",
    "PredictionsRepository",
    "ScalingDecisionsRepository",
]
