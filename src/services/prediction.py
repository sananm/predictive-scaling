"""
Prediction Service for running ML predictions in the background.

Responsibilities:
- Listen for new feature data
- Run predictions when enough new data arrives
- Publish predictions to Kafka
- Store predictions in database
- Track prediction accuracy
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionHorizon(str, Enum):
    """Prediction time horizons."""

    SHORT = "short"  # 15-60 minutes
    MEDIUM = "medium"  # 1-24 hours
    LONG = "long"  # 1-7 days


@dataclass
class PredictionRequest:
    """Request for a prediction."""

    request_id: str
    service_name: str
    horizon: PredictionHorizon
    target_time: datetime
    features: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PredictionResult:
    """Result of a prediction."""

    prediction_id: str
    request_id: str
    service_name: str
    horizon: PredictionHorizon
    target_time: datetime
    model_name: str

    # Prediction values
    p10: float  # 10th percentile
    p50: float  # Median (point prediction)
    p90: float  # 90th percentile

    # Metadata
    confidence: float
    features_used: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "request_id": self.request_id,
            "service_name": self.service_name,
            "horizon": self.horizon.value,
            "target_time": self.target_time.isoformat(),
            "model_name": self.model_name,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "confidence": self.confidence,
            "features_used": self.features_used,
            "created_at": self.created_at.isoformat(),
            "latency_ms": self.latency_ms,
        }


class PredictionService:
    """
    Background service for running predictions.

    Coordinates between feature data, models, and prediction storage.
    """

    def __init__(
        self,
        short_term_model: Any = None,
        medium_term_model: Any = None,
        long_term_model: Any = None,
        kafka_producer: Any = None,
        prediction_repository: Any = None,
        accuracy_tracker: Any = None,
        min_data_points: int = 10,
        prediction_topic: str = "predictions",
    ) -> None:
        """
        Initialize prediction service.

        Args:
            short_term_model: Model for short-term predictions
            medium_term_model: Model for medium-term predictions
            long_term_model: Model for long-term predictions
            kafka_producer: Kafka producer for publishing predictions
            prediction_repository: Repository for storing predictions
            accuracy_tracker: Accuracy tracker for monitoring
            min_data_points: Minimum data points needed for prediction
            prediction_topic: Kafka topic for predictions
        """
        self._short_model = short_term_model
        self._medium_model = medium_term_model
        self._long_model = long_term_model
        self._producer = kafka_producer
        self._repository = prediction_repository
        self._accuracy_tracker = accuracy_tracker
        self._min_data_points = min_data_points
        self._prediction_topic = prediction_topic

        self._running = False
        self._pending_requests: asyncio.Queue[PredictionRequest] = asyncio.Queue()
        self._recent_predictions: dict[str, list[PredictionResult]] = {}
        self._callbacks: list[callable] = []

        # Statistics
        self._stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_latency_ms": 0.0,
        }

        logger.info("Prediction service initialized")

    def set_model(self, horizon: PredictionHorizon, model: Any) -> None:
        """Set a model for a specific horizon."""
        if horizon == PredictionHorizon.SHORT:
            self._short_model = model
        elif horizon == PredictionHorizon.MEDIUM:
            self._medium_model = model
        elif horizon == PredictionHorizon.LONG:
            self._long_model = model

    def add_callback(self, callback: callable) -> None:
        """Add callback for prediction completion."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the prediction service."""
        if self._running:
            return

        self._running = True
        logger.info("Prediction service started")

        # Start processing loop
        asyncio.create_task(self._process_requests())

    async def stop(self) -> None:
        """Stop the prediction service."""
        self._running = False
        logger.info("Prediction service stopped")

    async def _process_requests(self) -> None:
        """Process pending prediction requests."""
        while self._running:
            try:
                # Wait for request with timeout
                try:
                    request = await asyncio.wait_for(
                        self._pending_requests.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                # Process the request
                try:
                    result = await self._run_prediction(request)
                    if result:
                        await self._handle_prediction_result(result)
                except Exception as e:
                    logger.error(
                        "Prediction failed",
                        request_id=request.request_id,
                        error=str(e),
                    )
                    self._stats["failed_predictions"] += 1

            except Exception as e:
                logger.error("Request processing error", error=str(e))
                await asyncio.sleep(1)

    async def request_prediction(
        self,
        service_name: str,
        horizon: PredictionHorizon,
        target_time: datetime,
        features: dict[str, Any] | None = None,
    ) -> str:
        """
        Request a prediction.

        Args:
            service_name: Service to predict for
            horizon: Prediction horizon
            target_time: Target time for prediction
            features: Pre-computed features (optional)

        Returns:
            Request ID
        """
        request = PredictionRequest(
            request_id=str(uuid4()),
            service_name=service_name,
            horizon=horizon,
            target_time=target_time,
            features=features,
        )

        await self._pending_requests.put(request)
        logger.debug(
            "Prediction requested",
            request_id=request.request_id,
            service=service_name,
            horizon=horizon.value,
        )

        return request.request_id

    async def run_predictions_for_service(
        self,
        service_name: str,
        horizons: list[PredictionHorizon] | None = None,
    ) -> list[PredictionResult]:
        """
        Run predictions for a service across multiple horizons.

        Args:
            service_name: Service to predict for
            horizons: List of horizons (defaults to all)

        Returns:
            List of prediction results
        """
        if horizons is None:
            horizons = list(PredictionHorizon)

        results = []
        now = datetime.now(UTC)

        for horizon in horizons:
            # Calculate target times based on horizon
            target_times = self._get_target_times(horizon, now)

            for target_time in target_times:
                request = PredictionRequest(
                    request_id=str(uuid4()),
                    service_name=service_name,
                    horizon=horizon,
                    target_time=target_time,
                )

                try:
                    result = await self._run_prediction(request)
                    if result:
                        results.append(result)
                        await self._handle_prediction_result(result)
                except Exception as e:
                    logger.error(
                        "Prediction failed",
                        service=service_name,
                        horizon=horizon.value,
                        target_time=target_time.isoformat(),
                        error=str(e),
                    )

        return results

    def _get_target_times(
        self,
        horizon: PredictionHorizon,
        now: datetime,
    ) -> list[datetime]:
        """Get target times for a prediction horizon."""
        if horizon == PredictionHorizon.SHORT:
            # 15, 30, 45, 60 minutes ahead
            return [now + timedelta(minutes=m) for m in [15, 30, 45, 60]]
        elif horizon == PredictionHorizon.MEDIUM:
            # 1, 2, 4, 8, 12, 24 hours ahead
            return [now + timedelta(hours=h) for h in [1, 2, 4, 8, 12, 24]]
        else:  # LONG
            # 1, 2, 3, 5, 7 days ahead
            return [now + timedelta(days=d) for d in [1, 2, 3, 5, 7]]

    async def _run_prediction(
        self,
        request: PredictionRequest,
    ) -> PredictionResult | None:
        """Run a single prediction."""
        start_time = datetime.now(UTC)

        # Get the appropriate model
        model = self._get_model(request.horizon)

        # Get or compute features
        features = request.features
        if features is None:
            features = await self._get_features(request.service_name, request.horizon)

        if not features:
            features = {"service_name": request.service_name}

        # Run prediction
        try:
            if model is not None and hasattr(model, "predict"):
                # Use real model
                prediction = await self._invoke_model(model, features)
            else:
                # Use mock prediction when no model available (for demo/testing)
                logger.debug(
                    "Using mock prediction (no model loaded)",
                    horizon=request.horizon.value,
                )
                prediction = self._mock_prediction(request)

            latency_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            # Get event multiplier for business events (applies to all predictions)
            event_multiplier = self._get_active_event_multiplier()

            # Extract base predictions
            p10 = prediction.get("p10", prediction.get("value", 0) * 0.8)
            p50 = prediction.get("p50", prediction.get("value", 0))
            p90 = prediction.get("p90", prediction.get("value", 0) * 1.2)

            # Apply event multiplier to scale predictions based on active events
            if event_multiplier != 1.0:
                if event_multiplier > 1.0:
                    # Scale UP: For demo, ensure minimum prediction during events
                    # Events like flash sales should have meaningful traffic
                    min_event_traffic = 100.0 * event_multiplier

                    p10 = max(p10 * event_multiplier, min_event_traffic * 0.8)
                    p50 = max(p50 * event_multiplier, min_event_traffic)
                    p90 = max(p90 * event_multiplier, min_event_traffic * 1.2)

                    logger.info(
                        "Applied scale UP multiplier to prediction",
                        multiplier=event_multiplier,
                        adjusted_p50=p50,
                        min_traffic=min_event_traffic,
                    )
                else:
                    # Scale DOWN: Reduce predictions for maintenance/low traffic events
                    # This simulates traffic decreasing (multiplier < 1.0)
                    p10 = p10 * event_multiplier
                    p50 = p50 * event_multiplier
                    p90 = p90 * event_multiplier

                    logger.info(
                        "Applied scale DOWN multiplier to prediction",
                        multiplier=event_multiplier,
                        adjusted_p50=p50,
                    )

            result = PredictionResult(
                prediction_id=str(uuid4()),
                request_id=request.request_id,
                service_name=request.service_name,
                horizon=request.horizon,
                target_time=request.target_time,
                model_name=self._get_model_name(request.horizon),
                p10=p10,
                p50=p50,
                p90=p90,
                confidence=prediction.get("confidence", 0.8),
                features_used=list(features.keys()) if features else [],
                latency_ms=latency_ms,
            )

            self._stats["total_predictions"] += 1
            self._stats["successful_predictions"] += 1
            self._stats["total_latency_ms"] += latency_ms

            logger.debug(
                "Prediction completed",
                prediction_id=result.prediction_id,
                service=request.service_name,
                horizon=request.horizon.value,
                p50=result.p50,
                latency_ms=f"{latency_ms:.2f}",
            )

            return result

        except Exception:
            self._stats["total_predictions"] += 1
            self._stats["failed_predictions"] += 1
            raise

    def _get_model(self, horizon: PredictionHorizon) -> Any:
        """Get model for horizon."""
        if horizon == PredictionHorizon.SHORT:
            return self._short_model
        elif horizon == PredictionHorizon.MEDIUM:
            return self._medium_model
        else:
            return self._long_model

    def _get_model_name(self, horizon: PredictionHorizon) -> str:
        """Get model name for horizon."""
        if horizon == PredictionHorizon.SHORT:
            return "transformer"
        elif horizon == PredictionHorizon.MEDIUM:
            return "gbm_ensemble"
        else:
            return "prophet"

    async def _get_features(
        self,
        service_name: str,
        horizon: PredictionHorizon,
    ) -> dict[str, Any] | None:
        """Get features for prediction."""
        # This would typically fetch from feature store or compute
        # For now, return placeholder
        return {
            "service_name": service_name,
            "horizon": horizon.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _invoke_model(
        self,
        model: Any,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke a model for prediction."""
        if asyncio.iscoroutinefunction(model.predict):
            return await model.predict(features)
        else:
            # Run sync prediction in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, model.predict, features)

    def _mock_prediction(self, request: PredictionRequest) -> dict[str, Any]:
        """Generate mock prediction for testing based on actual traffic.

        Note: Event multiplier is applied separately at the prediction result level
        to work with both mock and real model predictions.
        """
        import random

        # Get actual current RPS from the middleware tracker
        try:
            from src.api.middleware import get_current_rps

            current_rps = get_current_rps()
            if current_rps < 1.0:
                current_rps = 100.0  # Default baseline (tuned for demo capacity of ~3 instances)
        except Exception:
            current_rps = 100.0

        # Add some growth factor based on horizon (longer horizon = more uncertainty)
        horizon_multiplier = {
            PredictionHorizon.SHORT: 1.1,   # Expect slight growth
            PredictionHorizon.MEDIUM: 1.2,  # More potential growth
            PredictionHorizon.LONG: 1.3,    # Longer term projection
        }.get(request.horizon, 1.1)

        # Add random variation
        variation = random.uniform(0.9, 1.1)

        # Base prediction without event multiplier (applied at result level)
        base_value = current_rps * horizon_multiplier * variation

        logger.debug(
            "Mock prediction generated",
            current_rps=round(current_rps, 2),
            predicted_rps=round(base_value, 2),
        )

        return {
            "p10": base_value * 0.8,
            "p50": base_value,
            "p90": base_value * 1.2,
            "confidence": 0.85,
        }

    def _get_active_event_multiplier(self) -> float:
        """Get the maximum impact multiplier from active business events."""
        try:
            from datetime import UTC, datetime

            from sqlalchemy import create_engine, select
            from sqlalchemy.orm import Session

            from config.settings import get_settings
            from src.storage.models import BusinessEvent

            # Use sync engine to query directly (avoid deadlock from HTTP call)
            settings = get_settings()
            # Convert async URL to sync
            sync_url = settings.database.url.replace("+asyncpg", "").replace("+aiosqlite", "")
            engine = create_engine(sync_url)

            now = datetime.now(UTC)
            with Session(engine) as session:
                result = session.execute(
                    select(BusinessEvent)
                    .where(
                        BusinessEvent.is_active == True,
                        BusinessEvent.start_time <= now,
                        BusinessEvent.end_time >= now,
                    )
                )
                events = result.scalars().all()

                if events:
                    max_multiplier = max(e.expected_impact_multiplier for e in events)
                    logger.info(
                        "Active business event detected",
                        events_count=len(events),
                        max_multiplier=max_multiplier,
                    )
                    return max_multiplier
        except Exception as e:
            logger.debug("Could not fetch active events", error=str(e))

        return 1.0  # No active events or error

    async def _handle_prediction_result(self, result: PredictionResult) -> None:
        """Handle a completed prediction."""
        # Store in recent predictions
        if result.service_name not in self._recent_predictions:
            self._recent_predictions[result.service_name] = []
        self._recent_predictions[result.service_name].append(result)

        # Trim to last 100 predictions per service
        if len(self._recent_predictions[result.service_name]) > 100:
            self._recent_predictions[result.service_name] = \
                self._recent_predictions[result.service_name][-100:]

        # Store in database
        if self._repository:
            try:
                await self._repository.save_prediction(result.to_dict())
            except Exception as e:
                logger.error("Failed to save prediction", error=str(e))

        # Publish to Kafka
        if self._producer:
            try:
                await self._producer.send(
                    self._prediction_topic,
                    result.to_dict(),
                )
            except Exception as e:
                logger.error("Failed to publish prediction", error=str(e))

        # Track for accuracy
        if self._accuracy_tracker:
            try:
                self._accuracy_tracker.record_prediction(
                    prediction_id=result.prediction_id,
                    model_name=result.model_name,
                    horizon_minutes=self._horizon_to_minutes(result.horizon),
                    service_name=result.service_name,
                    predicted_at=result.created_at,
                    target_time=result.target_time,
                    p10=result.p10,
                    p50=result.p50,
                    p90=result.p90,
                )
            except Exception as e:
                logger.error("Failed to track prediction", error=str(e))

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error("Prediction callback error", error=str(e))

    def _horizon_to_minutes(self, horizon: PredictionHorizon) -> int:
        """Convert horizon to minutes."""
        if horizon == PredictionHorizon.SHORT:
            return 30  # Average of 15-60
        elif horizon == PredictionHorizon.MEDIUM:
            return 360  # 6 hours average
        else:
            return 2880  # 2 days average

    def get_recent_predictions(
        self,
        service_name: str,
        horizon: PredictionHorizon | None = None,
        limit: int = 10,
    ) -> list[PredictionResult]:
        """Get recent predictions for a service."""
        predictions = self._recent_predictions.get(service_name, [])
        if horizon:
            predictions = [p for p in predictions if p.horizon == horizon]
        return list(reversed(predictions[-limit:]))

    def get_latest_prediction(
        self,
        service_name: str,
        horizon: PredictionHorizon,
    ) -> PredictionResult | None:
        """Get the most recent prediction."""
        predictions = self.get_recent_predictions(service_name, horizon, limit=1)
        return predictions[0] if predictions else None

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        avg_latency = (
            self._stats["total_latency_ms"] / self._stats["total_predictions"]
            if self._stats["total_predictions"] > 0
            else 0
        )

        return {
            "running": self._running,
            "pending_requests": self._pending_requests.qsize(),
            "total_predictions": self._stats["total_predictions"],
            "successful_predictions": self._stats["successful_predictions"],
            "failed_predictions": self._stats["failed_predictions"],
            "success_rate": (
                self._stats["successful_predictions"] / self._stats["total_predictions"]
                if self._stats["total_predictions"] > 0
                else 0
            ),
            "avg_latency_ms": avg_latency,
            "services_tracked": len(self._recent_predictions),
            "models_loaded": {
                "short_term": self._short_model is not None,
                "medium_term": self._medium_model is not None,
                "long_term": self._long_model is not None,
            },
        }


# Global instance
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get the global prediction service."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def init_prediction_service(**kwargs) -> PredictionService:
    """Initialize the global prediction service."""
    global _prediction_service
    _prediction_service = PredictionService(**kwargs)
    return _prediction_service
