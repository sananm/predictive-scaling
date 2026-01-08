"""
Accuracy Tracker for prediction model performance monitoring.

Responsibilities:
- Compare predictions to actual values
- Calculate accuracy metrics (MAE, MAPE, RMSE)
- Track prediction interval coverage
- Detect accuracy degradation
- Generate accuracy reports
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from collections import defaultdict

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a model/horizon combination."""

    model_name: str
    horizon_minutes: int
    service_name: str
    period_start: datetime
    period_end: datetime
    sample_count: int

    # Error metrics
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mse: float  # Mean Square Error
    median_ae: float  # Median Absolute Error

    # Coverage metrics (for prediction intervals)
    coverage_80: float  # % of actuals within 80% CI
    coverage_90: float  # % of actuals within 90% CI
    coverage_95: float  # % of actuals within 95% CI

    # Bias metrics
    mean_error: float  # Average signed error (bias)
    bias_direction: str  # "over", "under", or "neutral"

    # Additional stats
    max_error: float
    min_error: float
    std_error: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "horizon_minutes": self.horizon_minutes,
            "service_name": self.service_name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "sample_count": self.sample_count,
            "mae": self.mae,
            "mape": self.mape,
            "rmse": self.rmse,
            "mse": self.mse,
            "median_ae": self.median_ae,
            "coverage_80": self.coverage_80,
            "coverage_90": self.coverage_90,
            "coverage_95": self.coverage_95,
            "mean_error": self.mean_error,
            "bias_direction": self.bias_direction,
            "max_error": self.max_error,
            "min_error": self.min_error,
            "std_error": self.std_error,
        }

    @property
    def is_acceptable(self) -> bool:
        """Check if accuracy is within acceptable thresholds."""
        # Default thresholds - can be customized
        return self.mape < 0.15 and self.coverage_80 > 0.75


@dataclass
class PredictionRecord:
    """Record of a prediction and its actual value."""

    prediction_id: str
    model_name: str
    horizon_minutes: int
    service_name: str
    predicted_at: datetime
    target_time: datetime

    # Predicted values
    p10: float  # 10th percentile
    p50: float  # Median prediction
    p90: float  # 90th percentile

    # Actual value (filled in later)
    actual: float | None = None
    actual_recorded_at: datetime | None = None

    @property
    def error(self) -> float | None:
        """Signed error (actual - predicted)."""
        if self.actual is None:
            return None
        return self.actual - self.p50

    @property
    def absolute_error(self) -> float | None:
        """Absolute error."""
        if self.error is None:
            return None
        return abs(self.error)

    @property
    def percentage_error(self) -> float | None:
        """Percentage error."""
        if self.actual is None or self.actual == 0:
            return None
        return abs(self.error) / self.actual

    @property
    def within_80_ci(self) -> bool | None:
        """Check if actual is within 80% confidence interval (p10-p90)."""
        if self.actual is None:
            return None
        return self.p10 <= self.actual <= self.p90

    @property
    def within_90_ci(self) -> bool | None:
        """Check if actual is within 90% confidence interval."""
        if self.actual is None:
            return None
        # Approximate 90% CI from p10/p90
        range_80 = self.p90 - self.p10
        margin = range_80 * 0.125  # Expand by 12.5% on each side
        return (self.p10 - margin) <= self.actual <= (self.p90 + margin)

    @property
    def within_95_ci(self) -> bool | None:
        """Check if actual is within 95% confidence interval."""
        if self.actual is None:
            return None
        # Approximate 95% CI from p10/p90
        range_80 = self.p90 - self.p10
        margin = range_80 * 0.25  # Expand by 25% on each side
        return (self.p10 - margin) <= self.actual <= (self.p90 + margin)


@dataclass
class AccuracyAlert:
    """Alert for accuracy degradation."""

    alert_id: str
    model_name: str
    horizon_minutes: int
    service_name: str
    metric_name: str  # "mape", "coverage", etc.
    current_value: float
    threshold: float
    severity: str  # "warning", "error", "critical"
    triggered_at: datetime
    message: str


class AccuracyTracker:
    """
    Tracks and analyzes prediction accuracy.

    Compares predictions to actual values, calculates accuracy metrics,
    and detects when model performance degrades.
    """

    def __init__(
        self,
        retention_hours: int = 168,  # 1 week
        min_samples_for_metrics: int = 10,
        mape_warning_threshold: float = 0.15,
        mape_error_threshold: float = 0.25,
        coverage_warning_threshold: float = 0.70,
        coverage_error_threshold: float = 0.60,
    ) -> None:
        """
        Initialize accuracy tracker.

        Args:
            retention_hours: How long to keep prediction records
            min_samples_for_metrics: Minimum samples needed to calculate metrics
            mape_warning_threshold: MAPE threshold for warning alerts
            mape_error_threshold: MAPE threshold for error alerts
            coverage_warning_threshold: Coverage threshold for warning alerts
            coverage_error_threshold: Coverage threshold for error alerts
        """
        self._retention_hours = retention_hours
        self._min_samples = min_samples_for_metrics
        self._mape_warning = mape_warning_threshold
        self._mape_error = mape_error_threshold
        self._coverage_warning = coverage_warning_threshold
        self._coverage_error = coverage_error_threshold

        # Storage: key = (model_name, horizon, service)
        self._predictions: dict[tuple[str, int, str], list[PredictionRecord]] = (
            defaultdict(list)
        )

        # Latest metrics by key
        self._latest_metrics: dict[tuple[str, int, str], AccuracyMetrics] = {}

        # Active alerts
        self._alerts: list[AccuracyAlert] = []

        # Callbacks
        self._callbacks: list[callable] = []

        logger.info(
            "Accuracy tracker initialized",
            retention_hours=retention_hours,
            min_samples=min_samples_for_metrics,
        )

    def add_callback(self, callback: callable) -> None:
        """Add callback for accuracy alerts."""
        self._callbacks.append(callback)

    def record_prediction(
        self,
        prediction_id: str,
        model_name: str,
        horizon_minutes: int,
        service_name: str,
        predicted_at: datetime,
        target_time: datetime,
        p10: float,
        p50: float,
        p90: float,
    ) -> PredictionRecord:
        """
        Record a new prediction.

        Args:
            prediction_id: Unique prediction identifier
            model_name: Name of the model
            horizon_minutes: Prediction horizon in minutes
            service_name: Service being predicted
            predicted_at: When prediction was made
            target_time: Time the prediction is for
            p10: 10th percentile prediction
            p50: Median prediction
            p90: 90th percentile prediction

        Returns:
            The created PredictionRecord
        """
        record = PredictionRecord(
            prediction_id=prediction_id,
            model_name=model_name,
            horizon_minutes=horizon_minutes,
            service_name=service_name,
            predicted_at=predicted_at,
            target_time=target_time,
            p10=p10,
            p50=p50,
            p90=p90,
        )

        key = (model_name, horizon_minutes, service_name)
        self._predictions[key].append(record)

        # Cleanup old records
        self._cleanup_old_records(key)

        logger.debug(
            "Prediction recorded",
            prediction_id=prediction_id,
            model=model_name,
            horizon=horizon_minutes,
        )

        return record

    def record_actual(
        self,
        model_name: str,
        horizon_minutes: int,
        service_name: str,
        target_time: datetime,
        actual_value: float,
        tolerance_minutes: int = 5,
    ) -> int:
        """
        Record actual value and match to predictions.

        Args:
            model_name: Model name
            horizon_minutes: Horizon in minutes
            service_name: Service name
            target_time: The time this actual value is for
            actual_value: The actual observed value
            tolerance_minutes: Time tolerance for matching predictions

        Returns:
            Number of predictions matched
        """
        key = (model_name, horizon_minutes, service_name)
        records = self._predictions.get(key, [])

        matched = 0
        now = datetime.now(timezone.utc)
        tolerance = timedelta(minutes=tolerance_minutes)

        for record in records:
            if record.actual is not None:
                continue  # Already has actual

            # Check if target time matches within tolerance
            time_diff = abs((record.target_time - target_time).total_seconds())
            if time_diff <= tolerance.total_seconds():
                record.actual = actual_value
                record.actual_recorded_at = now
                matched += 1

        if matched > 0:
            logger.debug(
                "Actual values recorded",
                model=model_name,
                horizon=horizon_minutes,
                matched=matched,
            )

        return matched

    def calculate_metrics(
        self,
        model_name: str,
        horizon_minutes: int,
        service_name: str,
        period_hours: int = 24,
    ) -> AccuracyMetrics | None:
        """
        Calculate accuracy metrics for a model/horizon.

        Args:
            model_name: Model name
            horizon_minutes: Horizon in minutes
            service_name: Service name
            period_hours: Period to analyze

        Returns:
            AccuracyMetrics or None if insufficient data
        """
        key = (model_name, horizon_minutes, service_name)
        records = self._predictions.get(key, [])

        # Filter to records with actuals in the period
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(hours=period_hours)

        valid_records = [
            r
            for r in records
            if r.actual is not None
            and r.actual_recorded_at
            and r.actual_recorded_at >= period_start
        ]

        if len(valid_records) < self._min_samples:
            logger.debug(
                "Insufficient samples for metrics",
                model=model_name,
                horizon=horizon_minutes,
                samples=len(valid_records),
                required=self._min_samples,
            )
            return None

        # Calculate errors
        errors = [r.error for r in valid_records]
        abs_errors = [r.absolute_error for r in valid_records]
        pct_errors = [r.percentage_error for r in valid_records if r.percentage_error is not None]

        # Calculate metrics
        mae = float(np.mean(abs_errors))
        mape = float(np.mean(pct_errors)) if pct_errors else 0.0
        mse = float(np.mean([e**2 for e in errors]))
        rmse = math.sqrt(mse)
        median_ae = float(np.median(abs_errors))

        # Coverage
        coverage_80 = sum(1 for r in valid_records if r.within_80_ci) / len(valid_records)
        coverage_90 = sum(1 for r in valid_records if r.within_90_ci) / len(valid_records)
        coverage_95 = sum(1 for r in valid_records if r.within_95_ci) / len(valid_records)

        # Bias
        mean_error = float(np.mean(errors))
        if mean_error > mae * 0.1:
            bias_direction = "under"  # Model underpredicts
        elif mean_error < -mae * 0.1:
            bias_direction = "over"  # Model overpredicts
        else:
            bias_direction = "neutral"

        metrics = AccuracyMetrics(
            model_name=model_name,
            horizon_minutes=horizon_minutes,
            service_name=service_name,
            period_start=period_start,
            period_end=now,
            sample_count=len(valid_records),
            mae=mae,
            mape=mape,
            rmse=rmse,
            mse=mse,
            median_ae=median_ae,
            coverage_80=coverage_80,
            coverage_90=coverage_90,
            coverage_95=coverage_95,
            mean_error=mean_error,
            bias_direction=bias_direction,
            max_error=float(max(abs_errors)),
            min_error=float(min(abs_errors)),
            std_error=float(np.std(errors)),
        )

        # Store latest metrics
        self._latest_metrics[key] = metrics

        # Check for alerts
        self._check_accuracy_alerts(metrics)

        logger.info(
            "Accuracy metrics calculated",
            model=model_name,
            horizon=horizon_minutes,
            mae=f"{mae:.4f}",
            mape=f"{mape:.2%}",
            coverage_80=f"{coverage_80:.2%}",
        )

        return metrics

    def _check_accuracy_alerts(self, metrics: AccuracyMetrics) -> None:
        """Check if metrics trigger any alerts."""
        alerts = []
        now = datetime.now(timezone.utc)

        # MAPE alerts
        if metrics.mape >= self._mape_error:
            alerts.append(
                AccuracyAlert(
                    alert_id=f"mape_error_{metrics.model_name}_{metrics.horizon_minutes}",
                    model_name=metrics.model_name,
                    horizon_minutes=metrics.horizon_minutes,
                    service_name=metrics.service_name,
                    metric_name="mape",
                    current_value=metrics.mape,
                    threshold=self._mape_error,
                    severity="error",
                    triggered_at=now,
                    message=f"MAPE ({metrics.mape:.1%}) exceeds error threshold ({self._mape_error:.1%})",
                )
            )
        elif metrics.mape >= self._mape_warning:
            alerts.append(
                AccuracyAlert(
                    alert_id=f"mape_warning_{metrics.model_name}_{metrics.horizon_minutes}",
                    model_name=metrics.model_name,
                    horizon_minutes=metrics.horizon_minutes,
                    service_name=metrics.service_name,
                    metric_name="mape",
                    current_value=metrics.mape,
                    threshold=self._mape_warning,
                    severity="warning",
                    triggered_at=now,
                    message=f"MAPE ({metrics.mape:.1%}) exceeds warning threshold ({self._mape_warning:.1%})",
                )
            )

        # Coverage alerts
        if metrics.coverage_80 < self._coverage_error:
            alerts.append(
                AccuracyAlert(
                    alert_id=f"coverage_error_{metrics.model_name}_{metrics.horizon_minutes}",
                    model_name=metrics.model_name,
                    horizon_minutes=metrics.horizon_minutes,
                    service_name=metrics.service_name,
                    metric_name="coverage_80",
                    current_value=metrics.coverage_80,
                    threshold=self._coverage_error,
                    severity="error",
                    triggered_at=now,
                    message=f"Coverage ({metrics.coverage_80:.1%}) below error threshold ({self._coverage_error:.1%})",
                )
            )
        elif metrics.coverage_80 < self._coverage_warning:
            alerts.append(
                AccuracyAlert(
                    alert_id=f"coverage_warning_{metrics.model_name}_{metrics.horizon_minutes}",
                    model_name=metrics.model_name,
                    horizon_minutes=metrics.horizon_minutes,
                    service_name=metrics.service_name,
                    metric_name="coverage_80",
                    current_value=metrics.coverage_80,
                    threshold=self._coverage_warning,
                    severity="warning",
                    triggered_at=now,
                    message=f"Coverage ({metrics.coverage_80:.1%}) below warning threshold ({self._coverage_warning:.1%})",
                )
            )

        # Store and notify
        for alert in alerts:
            self._alerts.append(alert)
            logger.warning(
                "Accuracy alert triggered",
                alert_id=alert.alert_id,
                severity=alert.severity,
                message=alert.message,
            )

            for callback in self._callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Accuracy alert callback error", error=str(e))

    def _cleanup_old_records(self, key: tuple[str, int, str]) -> None:
        """Remove records older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._retention_hours)
        self._predictions[key] = [
            r for r in self._predictions[key] if r.predicted_at >= cutoff
        ]

    def get_latest_metrics(
        self,
        model_name: str | None = None,
        horizon_minutes: int | None = None,
        service_name: str | None = None,
    ) -> list[AccuracyMetrics]:
        """Get latest metrics, optionally filtered."""
        results = []
        for key, metrics in self._latest_metrics.items():
            m_name, h_min, s_name = key
            if model_name and m_name != model_name:
                continue
            if horizon_minutes and h_min != horizon_minutes:
                continue
            if service_name and s_name != service_name:
                continue
            results.append(metrics)
        return results

    def get_active_alerts(self) -> list[AccuracyAlert]:
        """Get active accuracy alerts."""
        # Return alerts from the last hour
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        return [a for a in self._alerts if a.triggered_at >= cutoff]

    def generate_report(
        self,
        period_hours: int = 24,
        service_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate accuracy report.

        Args:
            period_hours: Period to analyze
            service_name: Optional service filter

        Returns:
            Report dictionary
        """
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_hours": period_hours,
            "service_filter": service_name,
            "models": [],
            "summary": {
                "total_predictions": 0,
                "avg_mape": 0.0,
                "avg_coverage_80": 0.0,
                "models_below_threshold": 0,
            },
        }

        all_metrics = []

        for key in self._predictions.keys():
            model_name, horizon, svc = key
            if service_name and svc != service_name:
                continue

            metrics = self.calculate_metrics(model_name, horizon, svc, period_hours)
            if metrics:
                all_metrics.append(metrics)
                report["models"].append(metrics.to_dict())

        if all_metrics:
            report["summary"]["total_predictions"] = sum(
                m.sample_count for m in all_metrics
            )
            report["summary"]["avg_mape"] = sum(m.mape for m in all_metrics) / len(
                all_metrics
            )
            report["summary"]["avg_coverage_80"] = sum(
                m.coverage_80 for m in all_metrics
            ) / len(all_metrics)
            report["summary"]["models_below_threshold"] = sum(
                1 for m in all_metrics if not m.is_acceptable
            )

        return report

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        total_predictions = sum(len(p) for p in self._predictions.values())
        with_actuals = sum(
            sum(1 for r in p if r.actual is not None)
            for p in self._predictions.values()
        )

        return {
            "total_predictions": total_predictions,
            "predictions_with_actuals": with_actuals,
            "match_rate": with_actuals / total_predictions if total_predictions else 0,
            "tracked_models": len(self._predictions),
            "latest_metrics_count": len(self._latest_metrics),
            "active_alerts": len(self.get_active_alerts()),
            "retention_hours": self._retention_hours,
        }


# Global instance
_accuracy_tracker: AccuracyTracker | None = None


def get_accuracy_tracker() -> AccuracyTracker:
    """Get the global accuracy tracker."""
    global _accuracy_tracker
    if _accuracy_tracker is None:
        _accuracy_tracker = AccuracyTracker()
    return _accuracy_tracker


def init_accuracy_tracker(**kwargs) -> AccuracyTracker:
    """Initialize the global accuracy tracker."""
    global _accuracy_tracker
    _accuracy_tracker = AccuracyTracker(**kwargs)
    return _accuracy_tracker
