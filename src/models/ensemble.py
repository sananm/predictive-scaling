"""
Gradient Boosting ensemble for medium-term predictions (1-24 hours).

Combines XGBoost, LightGBM, and CatBoost for robust predictions.
Each model is trained for specific quantiles, and predictions
are combined using optimized weights.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

from .base import BaseModel, PredictionResult, calculate_metrics

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for gradient boosting ensemble."""

    # Prediction horizons (hours)
    horizons_hours: list[int] = field(
        default_factory=lambda: [1, 2, 4, 6, 12, 24]
    )

    # Quantiles for prediction
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    # Model weights (learned during training)
    model_weights: dict[str, float] = field(
        default_factory=lambda: {
            "xgboost": 0.4,
            "lightgbm": 0.35,
            "catboost": 0.25,
        }
    )

    # XGBoost parameters
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
        }
    )

    # LightGBM parameters
    lgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
    )

    # CatBoost parameters
    cat_params: dict[str, Any] = field(
        default_factory=lambda: {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": False,
        }
    )

    # Cross-validation
    n_cv_folds: int = 5
    cv_strategy: str = "time_series"  # "time_series" or "kfold"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizons_hours": self.horizons_hours,
            "quantiles": list(self.quantiles),
            "model_weights": self.model_weights,
            "xgb_params": self.xgb_params,
            "lgb_params": self.lgb_params,
            "cat_params": self.cat_params,
            "n_cv_folds": self.n_cv_folds,
            "cv_strategy": self.cv_strategy,
        }


class GradientBoostingModel:
    """
    Wrapper for a single gradient boosting model with quantile regression.
    """

    def __init__(
        self,
        model_type: str,
        quantile: float,
        params: dict[str, Any],
    ):
        """
        Initialize gradient boosting model.

        Args:
            model_type: "xgboost", "lightgbm", or "catboost"
            quantile: Target quantile (0-1)
            params: Model parameters
        """
        self.model_type = model_type
        self.quantile = quantile
        self.params = params.copy()
        self.model = None

        self._configure_quantile()

    def _configure_quantile(self) -> None:
        """Configure model for quantile regression."""
        if self.model_type == "xgboost":
            self.params["objective"] = f"reg:quantileerror"
            self.params["quantile_alpha"] = self.quantile
        elif self.model_type == "lightgbm":
            self.params["objective"] = "quantile"
            self.params["alpha"] = self.quantile
        elif self.model_type == "catboost":
            self.params["loss_function"] = f"Quantile:alpha={self.quantile}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingModel":
        """Fit the model."""
        if self.model_type == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y)

        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y)

        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(**self.params)
            self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            return np.array([])

        if self.model_type == "xgboost":
            return self.model.feature_importances_
        elif self.model_type == "lightgbm":
            return self.model.feature_importances_
        elif self.model_type == "catboost":
            return self.model.feature_importances_

        return np.array([])


class MediumTermModel(BaseModel):
    """
    Ensemble model for medium-term predictions (1-24 hours).

    Combines XGBoost, LightGBM, and CatBoost models trained
    for different quantiles and horizons.
    """

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        horizon_minutes: int = 60,
    ) -> None:
        """
        Initialize medium-term model.

        Args:
            config: Ensemble configuration
            horizon_minutes: Default prediction horizon
        """
        super().__init__(
            name="gradient_boosting_ensemble",
            horizon_minutes=horizon_minutes,
            version="1.0.0",
        )

        self.config = config or EnsembleConfig()

        # Models organized by: horizon -> quantile -> model_type -> model
        self._models: dict[int, dict[float, dict[str, GradientBoostingModel]]] = {}

        # Feature importance (aggregated)
        self._feature_importance: dict[str, float] = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train ensemble models for all horizons and quantiles.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training metrics
        """
        self._feature_names = X.columns.tolist()
        all_metrics = {}

        # Train for each horizon
        for horizon_hours in self.config.horizons_hours:
            horizon_minutes = horizon_hours * 60
            logger.info(f"Training models for {horizon_hours}h horizon")

            # Prepare target shifted by horizon
            y_shifted = y.shift(-horizon_minutes // self._infer_frequency(X))
            valid_mask = ~y_shifted.isna()

            X_train = X[valid_mask].values
            y_train = y_shifted[valid_mask].values

            if len(X_train) < 100:
                logger.warning(f"Insufficient data for {horizon_hours}h horizon")
                continue

            self._models[horizon_hours] = {}

            # Train for each quantile
            for quantile in self.config.quantiles:
                self._models[horizon_hours][quantile] = {}

                # Train each model type
                for model_type, params in [
                    ("xgboost", self.config.xgb_params),
                    ("lightgbm", self.config.lgb_params),
                    ("catboost", self.config.cat_params),
                ]:
                    try:
                        model = GradientBoostingModel(model_type, quantile, params)
                        model.fit(X_train, y_train)
                        self._models[horizon_hours][quantile][model_type] = model

                        logger.debug(
                            f"Trained {model_type} for q{int(quantile*100)} @ {horizon_hours}h"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to train {model_type}",
                            error=str(e),
                        )

        # Validate and compute metrics
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val)
            all_metrics.update(val_metrics)
        else:
            # Use last 20% of training data for validation
            split_idx = int(len(X) * 0.8)
            val_metrics = self._evaluate(X.iloc[split_idx:], y.iloc[split_idx:])
            all_metrics.update(val_metrics)

        # Aggregate feature importance
        self._aggregate_feature_importance()

        self._is_trained = True

        # Update metadata
        self._update_metadata(
            validation_metrics=all_metrics,
            hyperparameters=self.config.to_dict(),
            feature_importance=self._feature_importance,
        )
        self._metadata.training_samples = len(X)
        self._metadata.n_features = len(self._feature_names)
        self._metadata.input_features = self._feature_names

        logger.info(
            "Ensemble training complete",
            horizons=len(self._models),
            metrics=all_metrics,
        )

        return all_metrics

    def predict(
        self,
        X: pd.DataFrame,
        horizon_hours: int | None = None,
        return_quantiles: bool = True,
    ) -> PredictionResult:
        """
        Generate predictions.

        Args:
            X: Features DataFrame
            horizon_hours: Specific horizon (uses default if None)
            return_quantiles: Whether to return quantile predictions

        Returns:
            PredictionResult with predictions
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X = self._validate_features(X.copy())

        # Use default horizon if not specified
        if horizon_hours is None:
            horizon_hours = self.horizon_minutes // 60
            if horizon_hours not in self._models:
                # Find closest available horizon
                available = list(self._models.keys())
                if not available:
                    raise RuntimeError("No trained models available")
                horizon_hours = min(available, key=lambda h: abs(h - horizon_hours))

        if horizon_hours not in self._models:
            raise ValueError(f"No model for horizon {horizon_hours}h")

        # Get predictions from all models
        predictions = {q: [] for q in self.config.quantiles}

        for quantile in self.config.quantiles:
            if quantile not in self._models[horizon_hours]:
                continue

            for model_type, weight in self.config.model_weights.items():
                if model_type in self._models[horizon_hours][quantile]:
                    model = self._models[horizon_hours][quantile][model_type]
                    pred = model.predict(X.values)
                    predictions[quantile].append((pred, weight))

        # Weighted average of predictions
        p10 = self._weighted_average(predictions.get(0.1, []))
        p50 = self._weighted_average(predictions.get(0.5, []))
        p90 = self._weighted_average(predictions.get(0.9, []))

        # Create timestamps
        if hasattr(X.index, "to_pydatetime"):
            from datetime import timedelta
            timestamps = [
                X.index[i] + timedelta(hours=horizon_hours)
                for i in range(len(X))
            ]
        else:
            timestamps = list(range(len(X)))

        return PredictionResult(
            timestamps=timestamps,
            p10=p10,
            p50=p50,
            p90=p90,
            model_name=self.name,
            model_version=self.version,
            horizon_minutes=horizon_hours * 60,
            metadata={"horizon_hours": horizon_hours},
        )

    def predict_all_horizons(
        self,
        X: pd.DataFrame,
    ) -> dict[int, PredictionResult]:
        """
        Generate predictions for all trained horizons.

        Args:
            X: Features DataFrame

        Returns:
            Dictionary mapping horizon_hours to PredictionResult
        """
        results = {}
        for horizon_hours in self._models.keys():
            results[horizon_hours] = self.predict(X, horizon_hours=horizon_hours)
        return results

    def _weighted_average(
        self,
        predictions: list[tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """Compute weighted average of predictions."""
        if not predictions:
            return np.array([])

        total_weight = sum(w for _, w in predictions)
        if total_weight == 0:
            return predictions[0][0]

        result = np.zeros_like(predictions[0][0])
        for pred, weight in predictions:
            result += pred * (weight / total_weight)

        return result

    def _evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model on validation data."""
        metrics = {}

        for horizon_hours in self._models.keys():
            # Shift target
            freq = self._infer_frequency(X)
            y_shifted = y.shift(-horizon_hours * 60 // freq)
            valid_mask = ~y_shifted.isna()

            if valid_mask.sum() < 10:
                continue

            X_val = X[valid_mask]
            y_val = y_shifted[valid_mask].values

            # Get predictions
            result = self.predict(X_val, horizon_hours=horizon_hours)

            # Calculate metrics
            horizon_metrics = calculate_metrics(
                y_val, result.p50, result.p10, result.p90
            )

            for key, value in horizon_metrics.items():
                metrics[f"{horizon_hours}h_{key}"] = value

        return metrics

    def _aggregate_feature_importance(self) -> None:
        """Aggregate feature importance across all models."""
        importance_sums = {name: 0.0 for name in self._feature_names}
        n_models = 0

        for horizon_models in self._models.values():
            for quantile_models in horizon_models.values():
                for model in quantile_models.values():
                    imp = model.get_feature_importance()
                    if len(imp) == len(self._feature_names):
                        for i, name in enumerate(self._feature_names):
                            importance_sums[name] += imp[i]
                        n_models += 1

        if n_models > 0:
            self._feature_importance = {
                name: val / n_models
                for name, val in importance_sums.items()
            }

    def _infer_frequency(self, X: pd.DataFrame) -> int:
        """Infer data frequency in minutes."""
        if len(X) < 2:
            return 1
        diff = pd.Series(X.index).diff().median()
        return max(int(diff.total_seconds() / 60), 1)

    def _save_model_artifacts(self, path: Path) -> None:
        """Save all model artifacts."""
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save models
        models_path = path / "models"
        models_path.mkdir(exist_ok=True)

        for horizon, horizon_models in self._models.items():
            for quantile, quantile_models in horizon_models.items():
                for model_type, model in quantile_models.items():
                    filename = f"{horizon}h_q{int(quantile*100)}_{model_type}.pkl"
                    with open(models_path / filename, "wb") as f:
                        pickle.dump(model, f)

        # Save feature importance
        with open(path / "feature_importance.json", "w") as f:
            json.dump(self._feature_importance, f, indent=2)

    def _load_model_artifacts(self, path: Path) -> None:
        """Load all model artifacts."""
        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
            # Reconstruct config (simplified)
            self.config = EnsembleConfig()
            self.config.horizons_hours = config_dict.get("horizons_hours", [])
            self.config.model_weights = config_dict.get("model_weights", {})

        # Load models
        models_path = path / "models"
        self._models = {}

        for model_file in models_path.glob("*.pkl"):
            # Parse filename: {horizon}h_q{quantile}_{model_type}.pkl
            parts = model_file.stem.split("_")
            horizon = int(parts[0].replace("h", ""))
            quantile = int(parts[1].replace("q", "")) / 100
            model_type = parts[2]

            if horizon not in self._models:
                self._models[horizon] = {}
            if quantile not in self._models[horizon]:
                self._models[horizon][quantile] = {}

            with open(model_file, "rb") as f:
                self._models[horizon][quantile][model_type] = pickle.load(f)

        # Load feature importance
        importance_path = path / "feature_importance.json"
        if importance_path.exists():
            with open(importance_path) as f:
                self._feature_importance = json.load(f)
