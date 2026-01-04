"""
Ensemble combiner for multi-horizon predictions.

Combines predictions from:
- Short-term (Transformer): 5-15 minutes
- Medium-term (Gradient Boosting): 1-24 hours
- Long-term (Prophet): 1-7 days

Uses horizon-specific weights and handles overlapping predictions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

from .base import BaseModel, PredictionResult
from .ensemble import MediumTermModel
from .prophet_model import LongTermModel
from .transformer import ShortTermModel

logger = get_logger(__name__)


@dataclass
class CombinerConfig:
    """Configuration for ensemble combiner."""

    # Horizon ranges for each model (in minutes)
    short_term_max_minutes: int = 30
    medium_term_max_minutes: int = 1440  # 24 hours
    long_term_max_minutes: int = 10080  # 7 days

    # Model weights by horizon (dynamically adjusted)
    # Weights are learned based on recent performance
    initial_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "short": {"transformer": 0.8, "gradient_boosting": 0.2, "prophet": 0.0},
            "medium": {"transformer": 0.2, "gradient_boosting": 0.6, "prophet": 0.2},
            "long": {"transformer": 0.0, "gradient_boosting": 0.3, "prophet": 0.7},
        }
    )

    # Blending zones (minutes where models overlap)
    short_medium_blend_start: int = 15
    short_medium_blend_end: int = 60
    medium_long_blend_start: int = 720  # 12 hours
    medium_long_blend_end: int = 2880  # 48 hours

    # Uncertainty combination
    uncertainty_method: str = "conservative"  # "conservative", "average", "model_weighted"

    # Performance tracking
    track_performance: bool = True
    performance_window_hours: int = 24
    min_samples_for_weight_update: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "short_term_max_minutes": self.short_term_max_minutes,
            "medium_term_max_minutes": self.medium_term_max_minutes,
            "long_term_max_minutes": self.long_term_max_minutes,
            "initial_weights": self.initial_weights,
            "short_medium_blend_start": self.short_medium_blend_start,
            "short_medium_blend_end": self.short_medium_blend_end,
            "medium_long_blend_start": self.medium_long_blend_start,
            "medium_long_blend_end": self.medium_long_blend_end,
            "uncertainty_method": self.uncertainty_method,
        }


class EnsembleCombiner:
    """
    Combines predictions from multiple models.

    Features:
    - Horizon-specific model selection
    - Smooth blending in transition zones
    - Uncertainty propagation
    - Dynamic weight adjustment based on recent accuracy
    """

    def __init__(
        self,
        short_term_model: ShortTermModel | None = None,
        medium_term_model: MediumTermModel | None = None,
        long_term_model: LongTermModel | None = None,
        config: CombinerConfig | None = None,
    ) -> None:
        """
        Initialize ensemble combiner.

        Args:
            short_term_model: Transformer model
            medium_term_model: Gradient boosting ensemble
            long_term_model: Prophet model
            config: Combiner configuration
        """
        self.short_term_model = short_term_model
        self.medium_term_model = medium_term_model
        self.long_term_model = long_term_model
        self.config = config or CombinerConfig()

        # Current weights (may be updated based on performance)
        self._weights = self.config.initial_weights.copy()

        # Performance tracking
        self._performance_history: list[dict[str, Any]] = []

    def predict(
        self,
        X: pd.DataFrame,
        horizon_minutes: int,
        events: list[dict[str, Any]] | None = None,
    ) -> PredictionResult:
        """
        Generate combined predictions for a specific horizon.

        Args:
            X: Features DataFrame
            horizon_minutes: Prediction horizon in minutes
            events: Business events (for long-term model)

        Returns:
            Combined PredictionResult
        """
        # Determine which models to use and their weights
        model_weights = self._get_model_weights(horizon_minutes)

        predictions = {}
        weights = {}

        # Get predictions from each applicable model
        if model_weights.get("transformer", 0) > 0 and self.short_term_model is not None:
            try:
                pred = self.short_term_model.predict(X)
                predictions["transformer"] = pred
                weights["transformer"] = model_weights["transformer"]
            except Exception as e:
                logger.warning(f"Short-term prediction failed: {e}")

        if model_weights.get("gradient_boosting", 0) > 0 and self.medium_term_model is not None:
            try:
                horizon_hours = max(1, horizon_minutes // 60)
                pred = self.medium_term_model.predict(X, horizon_hours=horizon_hours)
                predictions["gradient_boosting"] = pred
                weights["gradient_boosting"] = model_weights["gradient_boosting"]
            except Exception as e:
                logger.warning(f"Medium-term prediction failed: {e}")

        if model_weights.get("prophet", 0) > 0 and self.long_term_model is not None:
            try:
                horizon_days = max(1, horizon_minutes // 1440)
                pred = self.long_term_model.predict(X, horizon_days=horizon_days, events=events)
                predictions["prophet"] = pred
                weights["prophet"] = model_weights["prophet"]
            except Exception as e:
                logger.warning(f"Long-term prediction failed: {e}")

        if not predictions:
            raise RuntimeError("No model predictions available")

        # Combine predictions
        combined = self._combine_predictions(predictions, weights, horizon_minutes)

        return combined

    def predict_all_horizons(
        self,
        X: pd.DataFrame,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[int, PredictionResult]:
        """
        Generate predictions for all supported horizons.

        Args:
            X: Features DataFrame
            events: Business events

        Returns:
            Dictionary mapping horizon_minutes to PredictionResult
        """
        horizons = [5, 15, 30, 60, 120, 240, 360, 720, 1440, 2880, 4320, 7200, 10080]
        results = {}

        for horizon in horizons:
            try:
                results[horizon] = self.predict(X, horizon, events)
            except Exception as e:
                logger.warning(f"Failed to predict for horizon {horizon}: {e}")

        return results

    def _get_model_weights(self, horizon_minutes: int) -> dict[str, float]:
        """
        Get model weights for a specific horizon.

        Weights are determined by horizon range and blending zones.
        """
        config = self.config

        # Pure short-term zone
        if horizon_minutes <= config.short_medium_blend_start:
            return self._weights["short"]

        # Short-medium blending zone
        if horizon_minutes <= config.short_medium_blend_end:
            blend = self._calculate_blend(
                horizon_minutes,
                config.short_medium_blend_start,
                config.short_medium_blend_end,
            )
            return self._blend_weights(
                self._weights["short"],
                self._weights["medium"],
                blend,
            )

        # Pure medium-term zone
        if horizon_minutes <= config.medium_long_blend_start:
            return self._weights["medium"]

        # Medium-long blending zone
        if horizon_minutes <= config.medium_long_blend_end:
            blend = self._calculate_blend(
                horizon_minutes,
                config.medium_long_blend_start,
                config.medium_long_blend_end,
            )
            return self._blend_weights(
                self._weights["medium"],
                self._weights["long"],
                blend,
            )

        # Pure long-term zone
        return self._weights["long"]

    def _calculate_blend(
        self,
        value: float,
        start: float,
        end: float,
    ) -> float:
        """Calculate blend factor (0 = start, 1 = end)."""
        if end <= start:
            return 0.5
        return (value - start) / (end - start)

    def _blend_weights(
        self,
        weights1: dict[str, float],
        weights2: dict[str, float],
        blend: float,
    ) -> dict[str, float]:
        """Blend two weight dictionaries."""
        result = {}
        all_keys = set(weights1.keys()) | set(weights2.keys())

        for key in all_keys:
            w1 = weights1.get(key, 0)
            w2 = weights2.get(key, 0)
            result[key] = w1 * (1 - blend) + w2 * blend

        return result

    def _combine_predictions(
        self,
        predictions: dict[str, PredictionResult],
        weights: dict[str, float],
        horizon_minutes: int,
    ) -> PredictionResult:
        """
        Combine predictions from multiple models.

        Args:
            predictions: Dict of model name to PredictionResult
            weights: Dict of model name to weight
            horizon_minutes: Target horizon

        Returns:
            Combined PredictionResult
        """
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # Find common length
        min_len = min(len(p.p50) for p in predictions.values())

        # Combine point predictions (p50)
        combined_p50 = np.zeros(min_len)
        for model_name, pred in predictions.items():
            w = norm_weights.get(model_name, 0)
            combined_p50 += pred.p50[:min_len] * w

        # Combine uncertainty bounds
        if self.config.uncertainty_method == "conservative":
            # Use widest interval
            combined_p10 = np.min(
                [p.p10[:min_len] for p in predictions.values()],
                axis=0,
            )
            combined_p90 = np.max(
                [p.p90[:min_len] for p in predictions.values()],
                axis=0,
            )

        elif self.config.uncertainty_method == "average":
            # Average intervals
            combined_p10 = np.zeros(min_len)
            combined_p90 = np.zeros(min_len)
            for model_name, pred in predictions.items():
                w = norm_weights.get(model_name, 0)
                combined_p10 += pred.p10[:min_len] * w
                combined_p90 += pred.p90[:min_len] * w

        else:  # model_weighted
            # Weight by model performance
            combined_p10 = np.zeros(min_len)
            combined_p90 = np.zeros(min_len)
            for model_name, pred in predictions.items():
                w = norm_weights.get(model_name, 0)
                combined_p10 += pred.p10[:min_len] * w
                combined_p90 += pred.p90[:min_len] * w

        # Use timestamps from first prediction
        first_pred = next(iter(predictions.values()))
        timestamps = first_pred.timestamps[:min_len]

        return PredictionResult(
            timestamps=timestamps,
            p10=combined_p10,
            p50=combined_p50,
            p90=combined_p90,
            model_name="ensemble",
            model_version="1.0.0",
            horizon_minutes=horizon_minutes,
            metadata={
                "models_used": list(predictions.keys()),
                "weights": norm_weights,
                "uncertainty_method": self.config.uncertainty_method,
            },
        )

    def update_weights_from_performance(
        self,
        actuals: pd.Series,
        predictions: dict[str, PredictionResult],
    ) -> None:
        """
        Update model weights based on recent prediction accuracy.

        Args:
            actuals: Actual values
            predictions: Dict of model name to predictions
        """
        if not self.config.track_performance:
            return

        # Calculate error for each model
        errors = {}
        for model_name, pred in predictions.items():
            # Align predictions with actuals
            n = min(len(pred.p50), len(actuals))
            if n < 10:
                continue

            mae = np.mean(np.abs(actuals.values[:n] - pred.p50[:n]))
            errors[model_name] = mae

        if not errors:
            return

        # Convert errors to weights (inverse error weighting)
        inv_errors = {k: 1 / (v + 1e-8) for k, v in errors.items()}
        total = sum(inv_errors.values())

        new_weights = {k: v / total for k, v in inv_errors.items()}

        # Smooth weight update (exponential moving average)
        alpha = 0.1
        for horizon_type in self._weights:
            for model_name in self._weights[horizon_type]:
                if model_name in new_weights:
                    old = self._weights[horizon_type][model_name]
                    new = new_weights[model_name]
                    self._weights[horizon_type][model_name] = (
                        alpha * new + (1 - alpha) * old
                    )

        # Record performance
        self._performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "errors": errors,
            "updated_weights": self._weights.copy(),
        })

        # Trim history
        max_history = 1000
        if len(self._performance_history) > max_history:
            self._performance_history = self._performance_history[-max_history:]

        logger.info("Updated model weights", new_weights=self._weights)

    def get_model_agreement(
        self,
        predictions: dict[str, PredictionResult],
    ) -> float:
        """
        Calculate agreement between models (0 = disagree, 1 = agree).

        High agreement indicates higher confidence.
        """
        if len(predictions) < 2:
            return 1.0

        # Get median predictions from each model
        p50_list = [p.p50 for p in predictions.values()]

        # Find minimum length
        min_len = min(len(p) for p in p50_list)
        p50_array = np.array([p[:min_len] for p in p50_list])

        # Calculate coefficient of variation at each timestep
        cv = np.std(p50_array, axis=0) / (np.mean(p50_array, axis=0) + 1e-8)

        # Convert to agreement (low CV = high agreement)
        mean_cv = np.mean(cv)
        agreement = np.exp(-mean_cv)  # Maps CV to 0-1 range

        return float(agreement)

    def save(self, path: str | Path) -> None:
        """Save combiner state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save current weights
        with open(path / "weights.json", "w") as f:
            json.dump(self._weights, f, indent=2)

        # Save performance history
        with open(path / "performance_history.json", "w") as f:
            json.dump(self._performance_history, f, indent=2)

        # Save individual models
        if self.short_term_model is not None:
            self.short_term_model.save(path / "short_term")

        if self.medium_term_model is not None:
            self.medium_term_model.save(path / "medium_term")

        if self.long_term_model is not None:
            self.long_term_model.save(path / "long_term")

        logger.info("Ensemble combiner saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load combiner state."""
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                self.config = CombinerConfig(**{
                    k: v for k, v in config_dict.items()
                    if k in CombinerConfig.__dataclass_fields__
                })

        # Load weights
        weights_path = path / "weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                self._weights = json.load(f)

        # Load performance history
        history_path = path / "performance_history.json"
        if history_path.exists():
            with open(history_path) as f:
                self._performance_history = json.load(f)

        # Load individual models
        short_term_path = path / "short_term"
        if short_term_path.exists():
            from .transformer import ShortTermModel
            self.short_term_model = ShortTermModel()
            self.short_term_model.load(short_term_path)

        medium_term_path = path / "medium_term"
        if medium_term_path.exists():
            from .ensemble import MediumTermModel
            self.medium_term_model = MediumTermModel()
            self.medium_term_model.load(medium_term_path)

        long_term_path = path / "long_term"
        if long_term_path.exists():
            from .prophet_model import LongTermModel
            self.long_term_model = LongTermModel()
            self.long_term_model.load(long_term_path)

        logger.info("Ensemble combiner loaded", path=str(path))
