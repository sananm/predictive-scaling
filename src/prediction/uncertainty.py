"""
Uncertainty Quantification for prediction intervals.

Responsibilities:
- Combine prediction intervals from multiple models
- Compute model agreement as confidence indicator
- Flag predictions with high uncertainty for human review
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.base import PredictionResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification."""

    # Combined intervals
    combined_p10: float
    combined_p50: float
    combined_p90: float

    # Uncertainty metrics
    model_agreement: float  # 0-1, higher = more agreement
    confidence_score: float  # 0-1, higher = more confident
    interval_width: float  # Width of prediction interval
    relative_uncertainty: float  # interval_width / p50

    # Individual model contributions
    model_weights: dict[str, float] = field(default_factory=dict)
    model_intervals: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    # Flags
    high_uncertainty: bool = False
    requires_review: bool = False

    # Metadata
    n_models: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "combined_p10": self.combined_p10,
            "combined_p50": self.combined_p50,
            "combined_p90": self.combined_p90,
            "model_agreement": self.model_agreement,
            "confidence_score": self.confidence_score,
            "interval_width": self.interval_width,
            "relative_uncertainty": self.relative_uncertainty,
            "model_weights": self.model_weights,
            "high_uncertainty": self.high_uncertainty,
            "requires_review": self.requires_review,
            "n_models": self.n_models,
        }


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    # Model weights by horizon range (minutes)
    # Format: (min_horizon, max_horizon) -> {model_name: weight}
    horizon_weights: dict[tuple[int, int], dict[str, float]] = field(
        default_factory=lambda: {
            (0, 30): {"transformer": 0.7, "gradient_boosting": 0.2, "prophet": 0.1},
            (30, 240): {"transformer": 0.2, "gradient_boosting": 0.6, "prophet": 0.2},
            (240, 1440): {"transformer": 0.1, "gradient_boosting": 0.5, "prophet": 0.4},
            (1440, 10080): {"transformer": 0.05, "gradient_boosting": 0.25, "prophet": 0.7},
        }
    )

    # Thresholds
    high_uncertainty_threshold: float = 0.3  # relative_uncertainty > this = high uncertainty
    low_agreement_threshold: float = 0.5  # model_agreement < this = requires review
    min_confidence_threshold: float = 0.3  # confidence < this = requires review

    # Interval combination method
    combination_method: str = "weighted_average"  # "weighted_average", "conservative", "optimistic"

    # Agreement calculation
    agreement_method: str = "cv"  # "cv" (coefficient of variation), "range", "std"


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in predictions from multiple models.

    This class:
    1. Combines prediction intervals from different models
    2. Calculates model agreement (how much models agree)
    3. Computes confidence scores
    4. Flags high-uncertainty predictions for review
    """

    def __init__(self, config: UncertaintyConfig | None = None) -> None:
        """
        Initialize uncertainty quantifier.

        Args:
            config: Configuration for uncertainty quantification
        """
        self.config = config or UncertaintyConfig()

    def quantify(
        self,
        predictions: dict[str, PredictionResult],
        horizon_minutes: int,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty from multiple model predictions.

        Args:
            predictions: Dictionary of model_name -> PredictionResult
            horizon_minutes: Prediction horizon in minutes

        Returns:
            UncertaintyResult with combined intervals and metrics
        """
        if not predictions:
            return self._empty_result()

        # Get weights for this horizon
        weights = self._get_weights(horizon_minutes, list(predictions.keys()))

        # Extract p10, p50, p90 from each model
        model_intervals = {}
        for model_name, pred in predictions.items():
            if pred is not None and len(pred.p50) > 0:
                model_intervals[model_name] = (
                    float(pred.p10[0]),
                    float(pred.p50[0]),
                    float(pred.p90[0]),
                )

        if not model_intervals:
            return self._empty_result()

        # Combine intervals
        combined_p10, combined_p50, combined_p90 = self._combine_intervals(
            model_intervals, weights
        )

        # Calculate model agreement
        model_agreement = self._calculate_agreement(model_intervals)

        # Calculate interval width and relative uncertainty
        interval_width = combined_p90 - combined_p10
        relative_uncertainty = interval_width / combined_p50 if combined_p50 != 0 else float("inf")

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            model_agreement,
            relative_uncertainty,
            len(model_intervals),
        )

        # Determine flags
        high_uncertainty = relative_uncertainty > self.config.high_uncertainty_threshold
        requires_review = (
            model_agreement < self.config.low_agreement_threshold
            or confidence_score < self.config.min_confidence_threshold
            or high_uncertainty
        )

        return UncertaintyResult(
            combined_p10=combined_p10,
            combined_p50=combined_p50,
            combined_p90=combined_p90,
            model_agreement=model_agreement,
            confidence_score=confidence_score,
            interval_width=interval_width,
            relative_uncertainty=relative_uncertainty,
            model_weights=weights,
            model_intervals=model_intervals,
            high_uncertainty=high_uncertainty,
            requires_review=requires_review,
            n_models=len(model_intervals),
        )

    def _get_weights(
        self,
        horizon_minutes: int,
        available_models: list[str],
    ) -> dict[str, float]:
        """Get model weights for the given horizon."""
        # Find matching horizon range
        weights = {}
        for (min_h, max_h), w in self.config.horizon_weights.items():
            if min_h <= horizon_minutes < max_h:
                weights = w.copy()
                break

        # Default weights if no match
        if not weights:
            weights = {model: 1.0 / len(available_models) for model in available_models}

        # Filter to available models and renormalize
        filtered_weights = {
            model: weight
            for model, weight in weights.items()
            if model in available_models
        }

        if not filtered_weights:
            # Equal weights for available models
            filtered_weights = {
                model: 1.0 / len(available_models) for model in available_models
            }

        # Normalize weights to sum to 1
        total = sum(filtered_weights.values())
        if total > 0:
            filtered_weights = {k: v / total for k, v in filtered_weights.items()}

        return filtered_weights

    def _combine_intervals(
        self,
        model_intervals: dict[str, tuple[float, float, float]],
        weights: dict[str, float],
    ) -> tuple[float, float, float]:
        """Combine prediction intervals from multiple models."""
        method = self.config.combination_method

        if method == "weighted_average":
            return self._weighted_average_combine(model_intervals, weights)
        elif method == "conservative":
            return self._conservative_combine(model_intervals)
        elif method == "optimistic":
            return self._optimistic_combine(model_intervals)
        else:
            return self._weighted_average_combine(model_intervals, weights)

    def _weighted_average_combine(
        self,
        model_intervals: dict[str, tuple[float, float, float]],
        weights: dict[str, float],
    ) -> tuple[float, float, float]:
        """Combine using weighted average."""
        p10_sum = 0.0
        p50_sum = 0.0
        p90_sum = 0.0
        total_weight = 0.0

        for model_name, (p10, p50, p90) in model_intervals.items():
            weight = weights.get(model_name, 0)
            p10_sum += p10 * weight
            p50_sum += p50 * weight
            p90_sum += p90 * weight
            total_weight += weight

        if total_weight > 0:
            return (
                p10_sum / total_weight,
                p50_sum / total_weight,
                p90_sum / total_weight,
            )

        # Fallback to simple average
        p10s = [v[0] for v in model_intervals.values()]
        p50s = [v[1] for v in model_intervals.values()]
        p90s = [v[2] for v in model_intervals.values()]
        return (np.mean(p10s), np.mean(p50s), np.mean(p90s))

    def _conservative_combine(
        self,
        model_intervals: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Combine using conservative approach (widest interval)."""
        p10s = [v[0] for v in model_intervals.values()]
        p50s = [v[1] for v in model_intervals.values()]
        p90s = [v[2] for v in model_intervals.values()]

        return (min(p10s), np.mean(p50s), max(p90s))

    def _optimistic_combine(
        self,
        model_intervals: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Combine using optimistic approach (narrowest interval)."""
        p10s = [v[0] for v in model_intervals.values()]
        p50s = [v[1] for v in model_intervals.values()]
        p90s = [v[2] for v in model_intervals.values()]

        return (max(p10s), np.mean(p50s), min(p90s))

    def _calculate_agreement(
        self,
        model_intervals: dict[str, tuple[float, float, float]],
    ) -> float:
        """
        Calculate model agreement score (0-1).

        Higher values indicate more agreement between models.
        """
        if len(model_intervals) < 2:
            return 1.0  # Single model = perfect agreement with itself

        # Get p50 values from all models
        p50s = np.array([v[1] for v in model_intervals.values()])

        method = self.config.agreement_method

        if method == "cv":
            # Coefficient of variation (lower = more agreement)
            cv = np.std(p50s) / np.mean(p50s) if np.mean(p50s) != 0 else float("inf")
            # Convert to agreement score (0-1)
            agreement = 1.0 / (1.0 + cv)

        elif method == "range":
            # Range-based agreement
            range_val = np.max(p50s) - np.min(p50s)
            mean_val = np.mean(p50s)
            if mean_val != 0:
                relative_range = range_val / mean_val
                agreement = 1.0 / (1.0 + relative_range)
            else:
                agreement = 0.0

        elif method == "std":
            # Standard deviation based
            std = np.std(p50s)
            mean_val = np.mean(p50s)
            if mean_val != 0:
                relative_std = std / mean_val
                agreement = max(0, 1.0 - relative_std)
            else:
                agreement = 0.0

        else:
            agreement = 0.5  # Default

        return float(np.clip(agreement, 0, 1))

    def _calculate_confidence(
        self,
        model_agreement: float,
        relative_uncertainty: float,
        n_models: int,
    ) -> float:
        """
        Calculate overall confidence score (0-1).

        Considers:
        - Model agreement
        - Relative uncertainty
        - Number of contributing models
        """
        # Base confidence from agreement
        confidence = model_agreement * 0.5

        # Adjust for uncertainty (lower uncertainty = higher confidence)
        uncertainty_factor = 1.0 / (1.0 + relative_uncertainty)
        confidence += uncertainty_factor * 0.3

        # Adjust for number of models (more models = more confidence, up to a point)
        model_factor = min(n_models / 3.0, 1.0)
        confidence += model_factor * 0.2

        return float(np.clip(confidence, 0, 1))

    def _empty_result(self) -> UncertaintyResult:
        """Return empty result when no predictions available."""
        return UncertaintyResult(
            combined_p10=0.0,
            combined_p50=0.0,
            combined_p90=0.0,
            model_agreement=0.0,
            confidence_score=0.0,
            interval_width=0.0,
            relative_uncertainty=float("inf"),
            high_uncertainty=True,
            requires_review=True,
            n_models=0,
        )

    def get_model_contribution(
        self,
        predictions: dict[str, PredictionResult],
        horizon_minutes: int,
    ) -> dict[str, dict[str, float]]:
        """
        Get detailed contribution of each model.

        Returns:
            Dictionary with model contributions and diagnostics
        """
        weights = self._get_weights(horizon_minutes, list(predictions.keys()))

        contributions = {}
        for model_name, pred in predictions.items():
            if pred is not None and len(pred.p50) > 0:
                contributions[model_name] = {
                    "weight": weights.get(model_name, 0),
                    "p10": float(pred.p10[0]),
                    "p50": float(pred.p50[0]),
                    "p90": float(pred.p90[0]),
                    "interval_width": float(pred.p90[0] - pred.p10[0]),
                }

        return contributions
