"""
Prediction Calibration for adjusting prediction intervals.

Responsibilities:
- Track empirical coverage (% of actuals within predicted intervals)
- Adjust interval widths to achieve target coverage (e.g., 80%, 90%)
- Separate calibration by horizon and model
- Periodic recalibration based on recent performance
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""

    horizon_minutes: int
    target_coverage: float
    empirical_coverage: float
    calibration_factor: float
    n_samples: int
    last_updated: datetime
    is_calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizon_minutes": self.horizon_minutes,
            "target_coverage": self.target_coverage,
            "empirical_coverage": self.empirical_coverage,
            "calibration_factor": self.calibration_factor,
            "n_samples": self.n_samples,
            "last_updated": self.last_updated.isoformat(),
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class CalibrationConfig:
    """Configuration for prediction calibration."""

    # Target coverage levels
    target_coverage_80: float = 0.80
    target_coverage_90: float = 0.90

    # Minimum samples required for calibration
    min_samples: int = 50

    # Maximum samples to keep in history
    max_samples: int = 1000

    # Recalibration settings
    recalibration_interval_hours: int = 24
    recalibration_min_new_samples: int = 20

    # Calibration factor bounds
    min_calibration_factor: float = 0.5
    max_calibration_factor: float = 2.0

    # Learning rate for calibration updates
    learning_rate: float = 0.1

    # Horizon grouping (minutes)
    horizon_groups: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (0, 30),      # Short-term
            (30, 240),    # Medium-short
            (240, 1440),  # Medium-long
            (1440, 10080),  # Long-term
        ]
    )


@dataclass
class CalibrationSample:
    """Single calibration sample."""

    timestamp: datetime
    horizon_minutes: int
    predicted_p10: float
    predicted_p50: float
    predicted_p90: float
    actual_value: float

    @property
    def in_80_interval(self) -> bool:
        """Check if actual is within 80% prediction interval."""
        # 80% interval is approximately p10 to p90 with some adjustment
        margin = (self.predicted_p90 - self.predicted_p10) * 0.1
        return (self.predicted_p10 + margin) <= self.actual_value <= (self.predicted_p90 - margin)

    @property
    def in_90_interval(self) -> bool:
        """Check if actual is within 90% prediction interval (p10-p90)."""
        return self.predicted_p10 <= self.actual_value <= self.predicted_p90

    @property
    def prediction_error(self) -> float:
        """Absolute error of p50 prediction."""
        return abs(self.actual_value - self.predicted_p50)

    @property
    def relative_error(self) -> float:
        """Relative error of p50 prediction."""
        if self.actual_value != 0:
            return abs(self.actual_value - self.predicted_p50) / abs(self.actual_value)
        return float("inf")


class PredictionCalibrator:
    """
    Calibrates prediction intervals to achieve target coverage.

    This class:
    1. Tracks historical prediction vs actual pairs
    2. Calculates empirical coverage for different horizon groups
    3. Computes calibration factors to adjust interval widths
    4. Supports periodic recalibration
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        """
        Initialize prediction calibrator.

        Args:
            config: Configuration for calibration
        """
        self.config = config or CalibrationConfig()

        # Storage for calibration samples by horizon group
        self._samples: dict[tuple[int, int], list[CalibrationSample]] = defaultdict(list)

        # Current calibration factors by horizon group
        self._calibration_factors: dict[tuple[int, int], float] = {}

        # Last calibration time by horizon group
        self._last_calibration: dict[tuple[int, int], datetime] = {}

        # Statistics
        self._total_samples = 0
        self._last_update: datetime | None = None

    def update(
        self,
        horizon_minutes: int,
        predicted_p10: float,
        predicted_p50: float,
        predicted_p90: float,
        actual_value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add a new calibration sample.

        Args:
            horizon_minutes: Prediction horizon
            predicted_p10: Predicted 10th percentile
            predicted_p50: Predicted 50th percentile (median)
            predicted_p90: Predicted 90th percentile
            actual_value: Actual observed value
            timestamp: Timestamp of the observation (default: now)
        """
        timestamp = timestamp or datetime.now(UTC)

        sample = CalibrationSample(
            timestamp=timestamp,
            horizon_minutes=horizon_minutes,
            predicted_p10=predicted_p10,
            predicted_p50=predicted_p50,
            predicted_p90=predicted_p90,
            actual_value=actual_value,
        )

        # Find horizon group
        horizon_group = self._get_horizon_group(horizon_minutes)

        # Add sample
        self._samples[horizon_group].append(sample)
        self._total_samples += 1
        self._last_update = timestamp

        # Trim old samples if needed
        if len(self._samples[horizon_group]) > self.config.max_samples:
            self._samples[horizon_group] = self._samples[horizon_group][-self.config.max_samples:]

        # Check if recalibration needed
        self._maybe_recalibrate(horizon_group)

        logger.debug(
            "Calibration sample added",
            horizon=horizon_minutes,
            horizon_group=horizon_group,
            total_samples=len(self._samples[horizon_group]),
        )

    def get_calibration_factor(
        self,
        horizon_minutes: int,
        confidence_score: float | None = None,
    ) -> float:
        """
        Get calibration factor for a horizon.

        Args:
            horizon_minutes: Prediction horizon
            confidence_score: Optional confidence score to adjust factor

        Returns:
            Calibration factor (multiply interval width by this)
        """
        horizon_group = self._get_horizon_group(horizon_minutes)

        factor = self._calibration_factors.get(horizon_group, 1.0)

        # Optionally adjust based on confidence
        if confidence_score is not None:
            # Lower confidence = wider intervals
            confidence_adjustment = 1.0 + (1.0 - confidence_score) * 0.2
            factor *= confidence_adjustment

        # Clip to bounds
        factor = np.clip(
            factor,
            self.config.min_calibration_factor,
            self.config.max_calibration_factor,
        )

        return float(factor)

    def get_calibration_result(self, horizon_minutes: int) -> CalibrationResult:
        """
        Get detailed calibration result for a horizon.

        Args:
            horizon_minutes: Prediction horizon

        Returns:
            CalibrationResult with coverage and calibration info
        """
        horizon_group = self._get_horizon_group(horizon_minutes)
        samples = self._samples.get(horizon_group, [])

        if len(samples) < self.config.min_samples:
            return CalibrationResult(
                horizon_minutes=horizon_minutes,
                target_coverage=self.config.target_coverage_90,
                empirical_coverage=0.0,
                calibration_factor=1.0,
                n_samples=len(samples),
                last_updated=self._last_calibration.get(horizon_group, datetime.now(UTC)),
                is_calibrated=False,
            )

        empirical_coverage = self._calculate_coverage(samples)
        calibration_factor = self._calibration_factors.get(horizon_group, 1.0)

        return CalibrationResult(
            horizon_minutes=horizon_minutes,
            target_coverage=self.config.target_coverage_90,
            empirical_coverage=empirical_coverage,
            calibration_factor=calibration_factor,
            n_samples=len(samples),
            last_updated=self._last_calibration.get(horizon_group, datetime.now(UTC)),
            is_calibrated=True,
        )

    def recalibrate(self, horizon_group: tuple[int, int] | None = None) -> dict[str, float]:
        """
        Force recalibration for a horizon group or all groups.

        Args:
            horizon_group: Specific group to recalibrate, or None for all

        Returns:
            Dictionary of new calibration factors
        """
        if horizon_group:
            groups = [horizon_group]
        else:
            groups = list(self._samples.keys())

        results = {}

        for group in groups:
            new_factor = self._recalibrate_group(group)
            results[f"{group[0]}-{group[1]}"] = new_factor

        return results

    def _get_horizon_group(self, horizon_minutes: int) -> tuple[int, int]:
        """Get horizon group for a given horizon."""
        for min_h, max_h in self.config.horizon_groups:
            if min_h <= horizon_minutes < max_h:
                return (min_h, max_h)

        # Default to last group
        return self.config.horizon_groups[-1]

    def _maybe_recalibrate(self, horizon_group: tuple[int, int]) -> None:
        """Check if recalibration is needed and perform if so."""
        samples = self._samples.get(horizon_group, [])

        if len(samples) < self.config.min_samples:
            return

        last_cal = self._last_calibration.get(horizon_group)
        now = datetime.now(UTC)

        # Check time-based trigger
        if last_cal:
            hours_since = (now - last_cal).total_seconds() / 3600
            if hours_since < self.config.recalibration_interval_hours:
                return

        # Check sample-based trigger (if last calibration exists)
        if last_cal:
            new_samples = sum(
                1 for s in samples
                if s.timestamp > last_cal
            )
            if new_samples < self.config.recalibration_min_new_samples:
                return

        # Perform recalibration
        self._recalibrate_group(horizon_group)

    def _recalibrate_group(self, horizon_group: tuple[int, int]) -> float:
        """Recalibrate a specific horizon group."""
        samples = self._samples.get(horizon_group, [])

        if len(samples) < self.config.min_samples:
            return 1.0

        # Calculate current empirical coverage
        empirical_coverage = self._calculate_coverage(samples)
        target_coverage = self.config.target_coverage_90

        # Calculate required adjustment
        current_factor = self._calibration_factors.get(horizon_group, 1.0)

        if empirical_coverage < target_coverage:
            # Coverage too low -> need wider intervals
            adjustment = 1.0 + (target_coverage - empirical_coverage)
        else:
            # Coverage too high -> can narrow intervals
            adjustment = 1.0 - (empirical_coverage - target_coverage) * 0.5

        # Apply learning rate for gradual updates
        new_factor = current_factor * (1 - self.config.learning_rate) + \
                     (current_factor * adjustment) * self.config.learning_rate

        # Clip to bounds
        new_factor = np.clip(
            new_factor,
            self.config.min_calibration_factor,
            self.config.max_calibration_factor,
        )

        self._calibration_factors[horizon_group] = float(new_factor)
        self._last_calibration[horizon_group] = datetime.now(UTC)

        logger.info(
            "Recalibrated",
            horizon_group=horizon_group,
            empirical_coverage=f"{empirical_coverage:.2%}",
            target_coverage=f"{target_coverage:.2%}",
            new_factor=f"{new_factor:.3f}",
        )

        return float(new_factor)

    def _calculate_coverage(self, samples: list[CalibrationSample]) -> float:
        """Calculate empirical coverage for samples."""
        if not samples:
            return 0.0

        in_interval = sum(1 for s in samples if s.in_90_interval)
        return in_interval / len(samples)

    def get_stats(self) -> dict[str, Any]:
        """Get calibrator statistics."""
        stats = {
            "total_samples": self._total_samples,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "horizon_groups": {},
        }

        for group in self.config.horizon_groups:
            samples = self._samples.get(group, [])
            stats["horizon_groups"][f"{group[0]}-{group[1]}"] = {
                "n_samples": len(samples),
                "calibration_factor": self._calibration_factors.get(group, 1.0),
                "last_calibration": self._last_calibration.get(group, None),
                "empirical_coverage": self._calculate_coverage(samples)
                if len(samples) >= 10 else None,
            }

        return stats

    def get_accuracy_metrics(self, horizon_minutes: int | None = None) -> dict[str, float]:
        """
        Get accuracy metrics for predictions.

        Args:
            horizon_minutes: Specific horizon or None for all

        Returns:
            Dictionary with MAE, MAPE, RMSE, coverage metrics
        """
        if horizon_minutes:
            horizon_group = self._get_horizon_group(horizon_minutes)
            samples = self._samples.get(horizon_group, [])
        else:
            samples = []
            for s in self._samples.values():
                samples.extend(s)

        if not samples:
            return {}

        errors = [s.prediction_error for s in samples]
        relative_errors = [s.relative_error for s in samples if s.relative_error != float("inf")]

        metrics = {
            "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
            "coverage_90": self._calculate_coverage(samples),
            "n_samples": len(samples),
        }

        if relative_errors:
            metrics["mape"] = float(np.mean(relative_errors) * 100)

        return metrics

    def save(self, path: Path) -> None:
        """Save calibrator state to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save calibration factors
        factors_data = {
            f"{k[0]}-{k[1]}": v
            for k, v in self._calibration_factors.items()
        }
        with open(path / "factors.json", "w") as f:
            json.dump(factors_data, f, indent=2)

        # Save last calibration times
        times_data = {
            f"{k[0]}-{k[1]}": v.isoformat()
            for k, v in self._last_calibration.items()
        }
        with open(path / "calibration_times.json", "w") as f:
            json.dump(times_data, f, indent=2)

        # Save recent samples (last 100 per group for bootstrap)
        samples_data = {}
        for group, samples in self._samples.items():
            group_key = f"{group[0]}-{group[1]}"
            samples_data[group_key] = [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "horizon_minutes": s.horizon_minutes,
                    "predicted_p10": s.predicted_p10,
                    "predicted_p50": s.predicted_p50,
                    "predicted_p90": s.predicted_p90,
                    "actual_value": s.actual_value,
                }
                for s in samples[-100:]  # Keep last 100
            ]
        with open(path / "samples.json", "w") as f:
            json.dump(samples_data, f, indent=2)

        logger.info("Calibrator state saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load calibrator state from disk."""
        # Load calibration factors
        factors_path = path / "factors.json"
        if factors_path.exists():
            with open(factors_path) as f:
                factors_data = json.load(f)
            self._calibration_factors = {
                tuple(map(int, k.split("-"))): v
                for k, v in factors_data.items()
            }

        # Load last calibration times
        times_path = path / "calibration_times.json"
        if times_path.exists():
            with open(times_path) as f:
                times_data = json.load(f)
            self._last_calibration = {
                tuple(map(int, k.split("-"))): datetime.fromisoformat(v)
                for k, v in times_data.items()
            }

        # Load samples
        samples_path = path / "samples.json"
        if samples_path.exists():
            with open(samples_path) as f:
                samples_data = json.load(f)
            for group_key, samples in samples_data.items():
                group = tuple(map(int, group_key.split("-")))
                self._samples[group] = [
                    CalibrationSample(
                        timestamp=datetime.fromisoformat(s["timestamp"]),
                        horizon_minutes=s["horizon_minutes"],
                        predicted_p10=s["predicted_p10"],
                        predicted_p50=s["predicted_p50"],
                        predicted_p90=s["predicted_p90"],
                        actual_value=s["actual_value"],
                    )
                    for s in samples
                ]
                self._total_samples += len(samples)

        logger.info("Calibrator state loaded", path=str(path))
