"""
Predictor Orchestrator for running predictions across all models.

Responsibilities:
- Load trained models at startup
- Fetch recent metrics and compute features
- Run predictions across all model types
- Combine results into unified prediction output
- Store predictions in database
- Publish to Kafka for downstream consumers
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_settings
from src.features.engineer import FeatureEngineer, FeatureEngineerFactory
from src.models.base import PredictionResult
from src.models.combiner import EnsembleCombiner
from src.models.ensemble import MediumTermModel
from src.models.prophet_model import LongTermModel
from src.models.transformer import ShortTermModel
from src.storage.repositories import (
    FeaturesRepository,
    MetricsRepository,
    PredictionsRepository,
)
from src.streaming.producer import MetricsProducer
from src.utils.logging import get_logger

from .calibration import PredictionCalibrator
from .uncertainty import UncertaintyQuantifier, UncertaintyResult

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class PredictionOutput:
    """Combined prediction output from all models."""

    timestamp: datetime
    service_name: str
    horizon_minutes: int

    # Combined prediction (from ensemble combiner)
    p10: float
    p50: float
    p90: float

    # Individual model predictions
    short_term: PredictionResult | None = None
    medium_term: PredictionResult | None = None
    long_term: PredictionResult | None = None

    # Uncertainty assessment
    uncertainty: UncertaintyResult | None = None

    # Calibration info
    calibrated: bool = False
    calibration_factor: float = 1.0

    # Metadata
    model_agreement: float = 0.0
    confidence_score: float = 0.0
    feature_hash: str = ""
    prediction_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "service_name": self.service_name,
            "horizon_minutes": self.horizon_minutes,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "model_agreement": self.model_agreement,
            "confidence_score": self.confidence_score,
            "calibrated": self.calibrated,
            "calibration_factor": self.calibration_factor,
            "feature_hash": self.feature_hash,
            "metadata": self.prediction_metadata,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the predictor orchestrator."""

    # Model paths
    model_dir: Path = Path("models")

    # Prediction horizons (minutes)
    short_term_horizons: list[int] = field(default_factory=lambda: [5, 10, 15])
    medium_term_horizons: list[int] = field(
        default_factory=lambda: [60, 120, 240, 360, 720, 1440]
    )
    long_term_horizons: list[int] = field(
        default_factory=lambda: [1440, 2880, 4320, 7200, 10080]
    )

    # Data requirements
    min_history_minutes: int = 1440  # 24 hours minimum
    max_history_minutes: int = 10080  # 7 days maximum

    # Feature engineering
    feature_config_name: str = "default"  # "minimal", "default", "full"

    # Prediction settings
    run_calibration: bool = True
    publish_to_kafka: bool = True
    store_in_database: bool = True

    # Concurrency
    max_concurrent_predictions: int = 3


class PredictorOrchestrator:
    """
    Main prediction service that orchestrates all model predictions.

    This service:
    1. Loads trained models from disk
    2. Fetches recent metrics from the database
    3. Runs feature engineering
    4. Executes predictions across all model types
    5. Combines results with uncertainty quantification
    6. Calibrates prediction intervals
    7. Stores results and publishes to Kafka
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        metrics_repo: MetricsRepository | None = None,
        features_repo: FeaturesRepository | None = None,
        predictions_repo: PredictionsRepository | None = None,
        producer: MetricsProducer | None = None,
    ) -> None:
        """
        Initialize the predictor orchestrator.

        Args:
            config: Orchestrator configuration
            metrics_repo: Repository for fetching metrics
            features_repo: Repository for caching features
            predictions_repo: Repository for storing predictions
            producer: Kafka producer for publishing predictions
        """
        self.config = config or OrchestratorConfig()

        # Repositories (will be initialized on start)
        self._metrics_repo = metrics_repo
        self._features_repo = features_repo
        self._predictions_repo = predictions_repo

        # Kafka producer
        self._producer = producer

        # Models (loaded on start)
        self._short_term_model: ShortTermModel | None = None
        self._medium_term_model: MediumTermModel | None = None
        self._long_term_model: LongTermModel | None = None
        self._combiner: EnsembleCombiner | None = None

        # Feature engineer
        self._feature_engineer: FeatureEngineer | None = None

        # Uncertainty and calibration
        self._uncertainty_quantifier = UncertaintyQuantifier()
        self._calibrator = PredictionCalibrator()

        # State
        self._initialized = False
        self._models_loaded = False

        # Statistics
        self._predictions_generated = 0
        self._predictions_failed = 0
        self._last_prediction_time: datetime | None = None

    async def initialize(self) -> None:
        """Initialize the orchestrator (load models, connect to services)."""
        if self._initialized:
            return

        logger.info("Initializing predictor orchestrator")

        # Initialize feature engineer
        if self.config.feature_config_name == "minimal":
            self._feature_engineer = FeatureEngineerFactory.create_minimal()
        elif self.config.feature_config_name == "full":
            self._feature_engineer = FeatureEngineerFactory.create_full()
        else:
            self._feature_engineer = FeatureEngineerFactory.create_default()

        # Load models
        await self._load_models()

        # Initialize combiner
        self._combiner = EnsembleCombiner()

        self._initialized = True
        logger.info("Predictor orchestrator initialized")

    async def _load_models(self) -> None:
        """Load trained models from disk."""
        model_dir = self.config.model_dir

        # Load short-term model
        short_term_path = model_dir / "short_term"
        if short_term_path.exists():
            try:
                self._short_term_model = ShortTermModel()
                self._short_term_model.load(short_term_path)
                logger.info("Loaded short-term model", path=str(short_term_path))
            except Exception as e:
                logger.warning("Failed to load short-term model", error=str(e))

        # Load medium-term model
        medium_term_path = model_dir / "medium_term"
        if medium_term_path.exists():
            try:
                self._medium_term_model = MediumTermModel()
                self._medium_term_model.load(medium_term_path)
                logger.info("Loaded medium-term model", path=str(medium_term_path))
            except Exception as e:
                logger.warning("Failed to load medium-term model", error=str(e))

        # Load long-term model
        long_term_path = model_dir / "long_term"
        if long_term_path.exists():
            try:
                self._long_term_model = LongTermModel()
                self._long_term_model.load(long_term_path)
                logger.info("Loaded long-term model", path=str(long_term_path))
            except Exception as e:
                logger.warning("Failed to load long-term model", error=str(e))

        # Load calibrator state
        calibrator_path = model_dir / "calibrator"
        if calibrator_path.exists():
            try:
                self._calibrator.load(calibrator_path)
                logger.info("Loaded calibrator state")
            except Exception as e:
                logger.warning("Failed to load calibrator", error=str(e))

        self._models_loaded = any([
            self._short_term_model and self._short_term_model.is_trained,
            self._medium_term_model and self._medium_term_model.is_trained,
            self._long_term_model and self._long_term_model.is_trained,
        ])

        if not self._models_loaded:
            logger.warning("No trained models available")

    async def predict(
        self,
        service_name: str,
        horizon_minutes: int,
        features: pd.DataFrame | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> PredictionOutput:
        """
        Generate predictions for a service at a specific horizon.

        Args:
            service_name: Name of the service to predict for
            horizon_minutes: Prediction horizon in minutes
            features: Pre-computed features (optional, will fetch if not provided)
            events: Business events to consider

        Returns:
            PredictionOutput with combined predictions
        """
        if not self._initialized:
            await self.initialize()

        now = datetime.now(timezone.utc)

        try:
            # Get or compute features
            if features is None:
                features = await self._get_features(service_name, events)

            if features is None or len(features) == 0:
                raise ValueError(f"No features available for {service_name}")

            # Select appropriate model based on horizon
            predictions: dict[str, PredictionResult] = {}

            # Run predictions concurrently
            tasks = []

            if horizon_minutes <= 30 and self._short_term_model and self._short_term_model.is_trained:
                tasks.append(("transformer", self._predict_short_term(features, horizon_minutes)))

            if 30 < horizon_minutes <= 1440 and self._medium_term_model and self._medium_term_model.is_trained:
                tasks.append(("gradient_boosting", self._predict_medium_term(features, horizon_minutes)))

            if horizon_minutes > 720 and self._long_term_model and self._long_term_model.is_trained:
                tasks.append(("prophet", self._predict_long_term(features, horizon_minutes, events)))

            # Execute predictions
            if tasks:
                results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True,
                )

                for (model_name, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.error(f"Prediction failed for {model_name}", error=str(result))
                    else:
                        predictions[model_name] = result

            # Combine predictions
            if predictions and self._combiner:
                combined = self._combiner.combine(predictions, horizon_minutes)
            elif predictions:
                # Use first available prediction if combiner not available
                combined = list(predictions.values())[0]
            else:
                raise ValueError("No predictions generated")

            # Quantify uncertainty
            uncertainty = self._uncertainty_quantifier.quantify(
                predictions,
                horizon_minutes,
            )

            # Calibrate if enabled
            calibration_factor = 1.0
            if self.config.run_calibration:
                calibration_factor = self._calibrator.get_calibration_factor(
                    horizon_minutes,
                    uncertainty.confidence_score,
                )

            # Apply calibration to intervals
            interval_width = combined.p90[0] - combined.p10[0]
            calibrated_half_width = (interval_width / 2) * calibration_factor
            center = combined.p50[0]

            calibrated_p10 = center - calibrated_half_width
            calibrated_p90 = center + calibrated_half_width

            # Build output
            output = PredictionOutput(
                timestamp=now,
                service_name=service_name,
                horizon_minutes=horizon_minutes,
                p10=calibrated_p10,
                p50=combined.p50[0],
                p90=calibrated_p90,
                short_term=predictions.get("transformer"),
                medium_term=predictions.get("gradient_boosting"),
                long_term=predictions.get("prophet"),
                uncertainty=uncertainty,
                calibrated=self.config.run_calibration,
                calibration_factor=calibration_factor,
                model_agreement=uncertainty.model_agreement if uncertainty else 0.0,
                confidence_score=uncertainty.confidence_score if uncertainty else 0.0,
                feature_hash=self._feature_engineer.get_metadata().get("feature_hash", "")
                if self._feature_engineer else "",
            )

            # Store and publish
            if self.config.store_in_database and self._predictions_repo:
                await self._store_prediction(output)

            if self.config.publish_to_kafka and self._producer:
                await self._publish_prediction(output)

            self._predictions_generated += 1
            self._last_prediction_time = now

            logger.info(
                "Prediction generated",
                service=service_name,
                horizon=horizon_minutes,
                p50=output.p50,
                confidence=output.confidence_score,
            )

            return output

        except Exception as e:
            self._predictions_failed += 1
            logger.error(
                "Prediction failed",
                service=service_name,
                horizon=horizon_minutes,
                error=str(e),
            )
            raise

    async def predict_all_horizons(
        self,
        service_name: str,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[int, PredictionOutput]:
        """
        Generate predictions for all configured horizons.

        Args:
            service_name: Name of the service to predict for
            events: Business events to consider

        Returns:
            Dictionary mapping horizon_minutes to PredictionOutput
        """
        features = await self._get_features(service_name, events)

        all_horizons = (
            self.config.short_term_horizons
            + self.config.medium_term_horizons
            + self.config.long_term_horizons
        )
        # Remove duplicates and sort
        all_horizons = sorted(set(all_horizons))

        results = {}

        for horizon in all_horizons:
            try:
                result = await self.predict(
                    service_name=service_name,
                    horizon_minutes=horizon,
                    features=features,
                    events=events,
                )
                results[horizon] = result
            except Exception as e:
                logger.warning(
                    f"Failed to predict for horizon {horizon}",
                    error=str(e),
                )

        return results

    async def _get_features(
        self,
        service_name: str,
        events: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Fetch metrics and compute features."""
        if not self._metrics_repo:
            raise RuntimeError("Metrics repository not configured")

        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=self.config.max_history_minutes)

        # Fetch metrics from database
        metrics_df = await self._metrics_repo.to_dataframe(
            service_name=service_name,
            start_time=start_time,
            end_time=now,
        )

        if metrics_df is None or len(metrics_df) == 0:
            return pd.DataFrame()

        # Compute features
        if self._feature_engineer:
            features = self._feature_engineer.compute_features(
                metrics_df,
                events=events,
            )
            return features

        return metrics_df

    async def _predict_short_term(
        self,
        features: pd.DataFrame,
        horizon_minutes: int,
    ) -> PredictionResult:
        """Run short-term (transformer) prediction."""
        return self._short_term_model.predict(features)

    async def _predict_medium_term(
        self,
        features: pd.DataFrame,
        horizon_minutes: int,
    ) -> PredictionResult:
        """Run medium-term (gradient boosting) prediction."""
        horizon_hours = horizon_minutes // 60
        return self._medium_term_model.predict(features, horizon_hours=horizon_hours)

    async def _predict_long_term(
        self,
        features: pd.DataFrame,
        horizon_minutes: int,
        events: list[dict[str, Any]] | None = None,
    ) -> PredictionResult:
        """Run long-term (prophet) prediction."""
        return self._long_term_model.predict(features, events=events)

    async def _store_prediction(self, output: PredictionOutput) -> None:
        """Store prediction in database."""
        if not self._predictions_repo:
            return

        await self._predictions_repo.insert(
            service_name=output.service_name,
            model_name="ensemble",
            model_version="1.0.0",
            horizon_minutes=output.horizon_minutes,
            target_timestamp=output.timestamp + timedelta(minutes=output.horizon_minutes),
            prediction_p10=output.p10,
            prediction_p50=output.p50,
            prediction_p90=output.p90,
            prediction_metadata=output.prediction_metadata,
        )

    async def _publish_prediction(self, output: PredictionOutput) -> None:
        """Publish prediction to Kafka."""
        if not self._producer:
            return

        await self._producer.send_prediction(output.to_dict())

    def update_with_actual(
        self,
        service_name: str,
        horizon_minutes: int,
        predicted_p10: float,
        predicted_p50: float,
        predicted_p90: float,
        actual_value: float,
    ) -> None:
        """
        Update calibration with actual observed value.

        This should be called when the actual value for a past prediction
        becomes available.
        """
        self._calibrator.update(
            horizon_minutes=horizon_minutes,
            predicted_p10=predicted_p10,
            predicted_p50=predicted_p50,
            predicted_p90=predicted_p90,
            actual_value=actual_value,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "initialized": self._initialized,
            "models_loaded": self._models_loaded,
            "short_term_available": self._short_term_model is not None
            and self._short_term_model.is_trained,
            "medium_term_available": self._medium_term_model is not None
            and self._medium_term_model.is_trained,
            "long_term_available": self._long_term_model is not None
            and self._long_term_model.is_trained,
            "predictions_generated": self._predictions_generated,
            "predictions_failed": self._predictions_failed,
            "last_prediction_time": self._last_prediction_time.isoformat()
            if self._last_prediction_time else None,
            "calibration_stats": self._calibrator.get_stats(),
        }

    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down predictor orchestrator")

        # Save calibrator state
        if self.config.model_dir:
            calibrator_path = self.config.model_dir / "calibrator"
            calibrator_path.mkdir(parents=True, exist_ok=True)
            self._calibrator.save(calibrator_path)

        self._initialized = False
