"""
Prophet model for long-term predictions (1-7 days).

Uses Facebook Prophet for seasonal decomposition with:
- Daily, weekly, yearly seasonality
- Business event regressors
- Holiday effects
- Residual adjustment using LightGBM
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

from .base import BaseModel, PredictionResult, calculate_metrics

logger = get_logger(__name__)


@dataclass
class ProphetConfig:
    """Configuration for Prophet model."""

    # Prediction horizons (days)
    horizons_days: list[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 7]
    )

    # Seasonality configuration
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    seasonality_mode: str = "multiplicative"  # or "additive"

    # Custom seasonality
    add_monthly_seasonality: bool = True
    monthly_seasonality_order: int = 5

    # Growth model
    growth: str = "linear"  # or "logistic"

    # Changepoint configuration
    changepoint_prior_scale: float = 0.05
    n_changepoints: int = 25

    # Uncertainty
    interval_width: float = 0.8
    mcmc_samples: int = 0  # 0 for MAP estimation

    # Business events
    event_types: list[str] = field(
        default_factory=lambda: [
            "product_launch",
            "marketing_campaign",
            "sale_event",
            "deployment",
        ]
    )
    event_decay_rate: float = 0.1

    # US holidays
    include_holidays: bool = True

    # Residual model
    train_residual_model: bool = True
    residual_model_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizons_days": self.horizons_days,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "add_monthly_seasonality": self.add_monthly_seasonality,
            "monthly_seasonality_order": self.monthly_seasonality_order,
            "growth": self.growth,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_changepoints": self.n_changepoints,
            "interval_width": self.interval_width,
            "event_types": self.event_types,
            "event_decay_rate": self.event_decay_rate,
            "include_holidays": self.include_holidays,
            "train_residual_model": self.train_residual_model,
        }


class LongTermModel(BaseModel):
    """
    Prophet-based model for long-term predictions (1-7 days).

    Uses Prophet for:
    - Trend modeling
    - Multiple seasonalities (daily, weekly, yearly)
    - Holiday effects
    - Business event regressors

    Plus a residual model (LightGBM) to capture patterns
    Prophet misses, especially during business events.
    """

    def __init__(
        self,
        config: ProphetConfig | None = None,
        horizon_minutes: int = 1440,  # 1 day
    ) -> None:
        """
        Initialize long-term model.

        Args:
            config: Prophet configuration
            horizon_minutes: Default prediction horizon
        """
        super().__init__(
            name="prophet",
            horizon_minutes=horizon_minutes,
            version="1.0.0",
        )

        self.config = config or ProphetConfig()

        # Prophet model
        self._prophet = None

        # Residual model for adjustments
        self._residual_model = None

        # Event regressors
        self._event_columns: list[str] = []

        # Training data statistics
        self._y_min: float = 0
        self._y_max: float = 1

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """
        Train Prophet model.

        Args:
            X: Features DataFrame (used for residual model)
            y: Target Series with DatetimeIndex
            X_val: Validation features
            y_val: Validation targets
            events: Business events for regressors

        Returns:
            Training metrics
        """
        from prophet import Prophet

        self._feature_names = X.columns.tolist()

        # Store target statistics
        self._y_min = float(y.min())
        self._y_max = float(y.max())

        # Prepare Prophet DataFrame
        prophet_df = self._prepare_prophet_df(y, events)

        # Create Prophet model
        self._prophet = Prophet(
            growth=self.config.growth,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            n_changepoints=self.config.n_changepoints,
            interval_width=self.config.interval_width,
            mcmc_samples=self.config.mcmc_samples,
        )

        # Add monthly seasonality
        if self.config.add_monthly_seasonality:
            self._prophet.add_seasonality(
                name="monthly",
                period=30.5,
                fourier_order=self.config.monthly_seasonality_order,
            )

        # Add US holidays
        if self.config.include_holidays:
            self._prophet.add_country_holidays(country_name="US")

        # Add event regressors
        for col in self._event_columns:
            self._prophet.add_regressor(col)

        # Fit Prophet
        logger.info("Fitting Prophet model")
        self._prophet.fit(prophet_df)

        # Train residual model if configured
        if self.config.train_residual_model:
            self._train_residual_model(prophet_df, X, y)

        self._is_trained = True

        # Evaluate
        if X_val is not None and y_val is not None:
            metrics = self._evaluate(X_val, y_val, events)
        else:
            # Cross-validation on training data
            metrics = self._cross_validate(prophet_df)

        # Update metadata
        self._update_metadata(
            validation_metrics=metrics,
            hyperparameters=self.config.to_dict(),
        )
        self._metadata.training_samples = len(prophet_df)
        self._metadata.n_features = len(self._feature_names)
        self._metadata.input_features = self._feature_names

        logger.info(
            "Prophet training complete",
            metrics=metrics,
        )

        return metrics

    def predict(
        self,
        X: pd.DataFrame,
        horizon_days: int | None = None,
        events: list[dict[str, Any]] | None = None,
        return_quantiles: bool = True,
    ) -> PredictionResult:
        """
        Generate predictions.

        Args:
            X: Features DataFrame
            horizon_days: Prediction horizon in days
            events: Future business events
            return_quantiles: Whether to return quantile predictions

        Returns:
            PredictionResult with predictions
        """
        if not self._is_trained or self._prophet is None:
            raise RuntimeError("Model must be trained before prediction")

        if horizon_days is None:
            horizon_days = self.horizon_minutes // 1440

        # Create future dataframe
        X.index.max()
        future = self._prophet.make_future_dataframe(
            periods=horizon_days * 24 * 60,  # Minutes
            freq="T",  # Minute frequency
            include_history=False,
        )

        # Add event regressors to future
        future = self._add_event_regressors(future, events)

        # Prophet predictions
        forecast = self._prophet.predict(future)

        # Get predictions
        p50 = forecast["yhat"].values
        p10 = forecast["yhat_lower"].values
        p90 = forecast["yhat_upper"].values

        # Apply residual model adjustments
        if self._residual_model is not None and len(X) > 0:
            X_aligned = self._validate_features(X.copy())
            adjustments = self._predict_residuals(X_aligned)

            # Apply adjustments (limited to avoid overcorrection)
            if len(adjustments) == len(p50):
                p50 = p50 + np.clip(adjustments, -p50 * 0.2, p50 * 0.2)
                p10 = p10 + np.clip(adjustments * 0.8, -p10 * 0.2, p10 * 0.2)
                p90 = p90 + np.clip(adjustments * 1.2, -p90 * 0.2, p90 * 0.2)

        # Ensure predictions are non-negative and ordered
        p10 = np.maximum(p10, 0)
        p50 = np.maximum(p50, p10)
        p90 = np.maximum(p90, p50)

        return PredictionResult(
            timestamps=forecast["ds"].tolist(),
            p10=p10,
            p50=p50,
            p90=p90,
            model_name=self.name,
            model_version=self.version,
            horizon_minutes=horizon_days * 1440,
            metadata={
                "horizon_days": horizon_days,
                "has_residual_adjustment": self._residual_model is not None,
            },
        )

    def _prepare_prophet_df(
        self,
        y: pd.Series,
        events: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Prepare DataFrame for Prophet training."""
        # Prophet requires 'ds' and 'y' columns
        df = pd.DataFrame({
            "ds": y.index,
            "y": y.values,
        })

        # Add event regressors
        df = self._add_event_regressors(df, events)

        return df

    def _add_event_regressors(
        self,
        df: pd.DataFrame,
        events: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Add event regressor columns to DataFrame."""
        # Initialize all event columns with zeros
        for event_type in self.config.event_types:
            col_name = f"event_{event_type}"
            df[col_name] = 0.0
            if col_name not in self._event_columns:
                self._event_columns.append(col_name)

        if events is None:
            return df

        # Process each event
        for event in events:
            event_type = event.get("event_type", "unknown")
            if event_type not in self.config.event_types:
                continue

            start_time = event.get("start_time")
            end_time = event.get("end_time")
            impact = event.get("expected_impact_multiplier", 1.0)

            if start_time is None:
                continue

            # Parse times
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if end_time and isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            if end_time is None:
                end_time = start_time + timedelta(hours=24)

            col_name = f"event_{event_type}"

            # Mark event period with impact value (with decay)
            for idx, row in df.iterrows():
                ts = row["ds"]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)

                if start_time <= ts <= end_time:
                    # During event: full impact
                    df.at[idx, col_name] = impact - 1.0
                elif ts > end_time:
                    # After event: decay
                    hours_since = (ts - end_time).total_seconds() / 3600
                    decayed = (impact - 1.0) * np.exp(
                        -self.config.event_decay_rate * hours_since
                    )
                    if decayed > 0.01:  # Only if significant
                        df.at[idx, col_name] = max(df.at[idx, col_name], decayed)

        return df

    def _train_residual_model(
        self,
        prophet_df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train residual model to capture patterns Prophet misses."""
        try:
            import lightgbm as lgb

            # Get Prophet predictions on training data
            prophet_pred = self._prophet.predict(prophet_df)

            # Calculate residuals
            residuals = y.values - prophet_pred["yhat"].values[: len(y)]

            # Align features with residuals
            X_aligned = X.iloc[: len(residuals)]

            # Train LightGBM on residuals
            self._residual_model = lgb.LGBMRegressor(
                **self.config.residual_model_params
            )
            self._residual_model.fit(X_aligned.values, residuals)

            logger.info("Residual model trained")

        except Exception as e:
            logger.warning(f"Failed to train residual model: {e}")
            self._residual_model = None

    def _predict_residuals(self, X: pd.DataFrame) -> np.ndarray:
        """Predict residual adjustments."""
        if self._residual_model is None:
            return np.zeros(len(X))

        try:
            return self._residual_model.predict(X.values)
        except Exception:
            return np.zeros(len(X))

    def _evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Evaluate model on validation data."""
        metrics = {}

        for horizon_days in self.config.horizons_days:
            try:
                result = self.predict(X, horizon_days=horizon_days, events=events)

                # Align predictions with actuals
                # This is approximate - in practice would need proper alignment
                n_samples = min(len(result.p50), len(y))
                if n_samples < 10:
                    continue

                horizon_metrics = calculate_metrics(
                    y.values[:n_samples],
                    result.p50[:n_samples],
                    result.p10[:n_samples],
                    result.p90[:n_samples],
                )

                for key, value in horizon_metrics.items():
                    metrics[f"{horizon_days}d_{key}"] = value

            except Exception as e:
                logger.warning(f"Evaluation failed for {horizon_days}d: {e}")

        return metrics

    def _cross_validate(
        self,
        df: pd.DataFrame,
        horizon: str = "30 days",
        period: str = "15 days",
        initial: str = "90 days",
    ) -> dict[str, float]:
        """Perform Prophet cross-validation."""
        try:
            from prophet.diagnostics import cross_validation, performance_metrics

            df_cv = cross_validation(
                self._prophet,
                horizon=horizon,
                period=period,
                initial=initial,
            )

            df_p = performance_metrics(df_cv)

            return {
                "cv_mae": float(df_p["mae"].mean()),
                "cv_rmse": float(df_p["rmse"].mean()),
                "cv_mape": float(df_p["mape"].mean() * 100),
                "cv_coverage": float(df_p["coverage"].mean() * 100),
            }

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {}

    def get_components(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Get Prophet component decomposition."""
        if self._prophet is None:
            raise RuntimeError("Model not trained")

        future = pd.DataFrame({"ds": dates})
        for col in self._event_columns:
            future[col] = 0.0

        forecast = self._prophet.predict(future)

        components = ["trend", "weekly", "yearly"]
        if self.config.daily_seasonality:
            components.append("daily")
        if self.config.add_monthly_seasonality:
            components.append("monthly")

        result = forecast[["ds"] + [c for c in components if c in forecast.columns]]
        return result

    def _save_model_artifacts(self, path: Path) -> None:
        """Save Prophet model and residual model."""
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save Prophet model
        with open(path / "prophet.pkl", "wb") as f:
            pickle.dump(self._prophet, f)

        # Save residual model
        if self._residual_model is not None:
            with open(path / "residual_model.pkl", "wb") as f:
                pickle.dump(self._residual_model, f)

        # Save event columns
        with open(path / "event_columns.json", "w") as f:
            json.dump(self._event_columns, f)

        # Save statistics
        with open(path / "stats.json", "w") as f:
            json.dump({
                "y_min": self._y_min,
                "y_max": self._y_max,
            }, f)

    def _load_model_artifacts(self, path: Path) -> None:
        """Load Prophet model and residual model."""
        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
            self.config = ProphetConfig(**{
                k: v for k, v in config_dict.items()
                if k in ProphetConfig.__dataclass_fields__
            })

        # Load Prophet model
        with open(path / "prophet.pkl", "rb") as f:
            self._prophet = pickle.load(f)

        # Load residual model
        residual_path = path / "residual_model.pkl"
        if residual_path.exists():
            with open(residual_path, "rb") as f:
                self._residual_model = pickle.load(f)

        # Load event columns
        event_path = path / "event_columns.json"
        if event_path.exists():
            with open(event_path) as f:
                self._event_columns = json.load(f)

        # Load statistics
        stats_path = path / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                self._y_min = stats.get("y_min", 0)
                self._y_max = stats.get("y_max", 1)
