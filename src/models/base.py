"""
Base classes for prediction models.

Provides:
- Abstract base model with train/predict/save/load
- Model versioning and metadata
- Quantile prediction support
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Container for model predictions with uncertainty.

    Attributes:
        timestamps: Prediction timestamps
        p10: 10th percentile (lower bound)
        p50: 50th percentile (median/point estimate)
        p90: 90th percentile (upper bound)
        model_name: Name of the model that generated predictions
        model_version: Version of the model
        horizon_minutes: Prediction horizon in minutes
        metadata: Additional prediction metadata
    """

    timestamps: np.ndarray | list[datetime]
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    model_name: str
    model_version: str
    horizon_minutes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizon_minutes": self.horizon_minutes,
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamps": [
                t.isoformat() if isinstance(t, datetime) else str(t)
                for t in self.timestamps
            ],
            "p10": self.p10.tolist() if isinstance(self.p10, np.ndarray) else self.p10,
            "p50": self.p50.tolist() if isinstance(self.p50, np.ndarray) else self.p50,
            "p90": self.p90.tolist() if isinstance(self.p90, np.ndarray) else self.p90,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizon_minutes": self.horizon_minutes,
            "metadata": self.metadata,
        }

    @property
    def confidence_interval_width(self) -> np.ndarray:
        """Get width of 80% confidence interval (p90 - p10)."""
        return self.p90 - self.p10

    @property
    def mean_prediction(self) -> float:
        """Get mean of median predictions."""
        return float(np.mean(self.p50))


@dataclass
class ModelMetadata:
    """
    Metadata for a trained model.

    Tracks training information, performance metrics,
    and configuration for reproducibility.
    """

    model_id: str
    model_name: str
    model_version: str
    created_at: datetime
    trained_at: datetime | None = None
    horizon_minutes: int = 15
    input_features: list[str] = field(default_factory=list)
    n_features: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    training_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "horizon_minutes": self.horizon_minutes,
            "input_features": self.input_features,
            "n_features": self.n_features,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "hyperparameters": self.hyperparameters,
            "feature_importance": self.feature_importance,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            model_version=data["model_version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            trained_at=datetime.fromisoformat(data["trained_at"]) if data.get("trained_at") else None,
            horizon_minutes=data.get("horizon_minutes", 15),
            input_features=data.get("input_features", []),
            n_features=data.get("n_features", 0),
            training_samples=data.get("training_samples", 0),
            validation_samples=data.get("validation_samples", 0),
            training_metrics=data.get("training_metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            hyperparameters=data.get("hyperparameters", {}),
            feature_importance=data.get("feature_importance", {}),
            tags=data.get("tags", []),
        )


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    Provides:
    - Standard interface for train/predict
    - Model persistence (save/load)
    - Metadata tracking
    - Quantile prediction support
    """

    # Quantiles for prediction intervals
    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(
        self,
        name: str,
        horizon_minutes: int = 15,
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize base model.

        Args:
            name: Model name (e.g., "transformer", "xgboost")
            horizon_minutes: Prediction horizon in minutes
            version: Model version string
        """
        self.name = name
        self.horizon_minutes = horizon_minutes
        self.version = version

        # Generate unique model ID
        self._model_id = str(uuid4())[:8]

        # Metadata
        self._metadata = ModelMetadata(
            model_id=self._model_id,
            model_name=name,
            model_version=version,
            created_at=datetime.now(UTC),
            horizon_minutes=horizon_minutes,
        )

        # Training state
        self._is_trained = False
        self._feature_names: list[str] = []

    @property
    def model_id(self) -> str:
        """Get unique model ID."""
        return self._model_id

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @property
    def metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names used by model."""
        return self._feature_names

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train the model.

        Args:
            X: Training features DataFrame
            y: Training target Series
            X_val: Optional validation features
            y_val: Optional validation target

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        return_quantiles: bool = True,
    ) -> PredictionResult:
        """
        Generate predictions.

        Args:
            X: Features DataFrame
            return_quantiles: Whether to return quantile predictions

        Returns:
            PredictionResult with predictions and uncertainty
        """
        pass

    @abstractmethod
    def _save_model_artifacts(self, path: Path) -> None:
        """Save model-specific artifacts (weights, trees, etc.)."""
        pass

    @abstractmethod
    def _load_model_artifacts(self, path: Path) -> None:
        """Load model-specific artifacts."""
        pass

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata.to_dict(), f, indent=2)

        # Save feature names
        features_path = path / "features.json"
        with open(features_path, "w") as f:
            json.dump(self._feature_names, f)

        # Save model-specific artifacts
        self._save_model_artifacts(path)

        logger.info(
            "Model saved",
            model_name=self.name,
            model_id=self._model_id,
            path=str(path),
        )

    def load(self, path: str | Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path containing saved model
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = ModelMetadata.from_dict(json.load(f))
                self._model_id = self._metadata.model_id

        # Load feature names
        features_path = path / "features.json"
        if features_path.exists():
            with open(features_path) as f:
                self._feature_names = json.load(f)

        # Load model-specific artifacts
        self._load_model_artifacts(path)
        self._is_trained = True

        logger.info(
            "Model loaded",
            model_name=self.name,
            model_id=self._model_id,
            path=str(path),
        )

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and align features with training features.

        Args:
            X: Input features

        Returns:
            Aligned features DataFrame
        """
        if not self._feature_names:
            return X

        # Check for missing features
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            logger.warning(
                "Missing features, filling with zeros",
                missing_count=len(missing),
            )
            for col in missing:
                X[col] = 0

        # Ensure column order matches training
        return X[self._feature_names]

    def _update_metadata(
        self,
        training_metrics: dict[str, float] | None = None,
        validation_metrics: dict[str, float] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> None:
        """Update model metadata after training."""
        self._metadata.trained_at = datetime.now(UTC)

        if training_metrics:
            self._metadata.training_metrics.update(training_metrics)

        if validation_metrics:
            self._metadata.validation_metrics.update(validation_metrics)

        if hyperparameters:
            self._metadata.hyperparameters.update(hyperparameters)

        if feature_importance:
            self._metadata.feature_importance.update(feature_importance)


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate quantile loss (pinball loss).

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile (0-1)

    Returns:
        Quantile loss value
    """
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_lower: np.ndarray | None = None,
    y_pred_upper: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate standard prediction metrics.

    Args:
        y_true: True values
        y_pred: Point predictions (median)
        y_pred_lower: Lower bound predictions (p10)
        y_pred_upper: Upper bound predictions (p90)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Point prediction metrics
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        metrics["mape"] = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        metrics["mape"] = 0.0

    # Quantile losses
    metrics["quantile_loss_50"] = quantile_loss(y_true, y_pred, 0.5)

    if y_pred_lower is not None:
        metrics["quantile_loss_10"] = quantile_loss(y_true, y_pred_lower, 0.1)

    if y_pred_upper is not None:
        metrics["quantile_loss_90"] = quantile_loss(y_true, y_pred_upper, 0.9)

    # Coverage (percentage of actuals within prediction interval)
    if y_pred_lower is not None and y_pred_upper is not None:
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        metrics["coverage_80"] = float(np.mean(in_interval) * 100)

        # Average interval width
        metrics["avg_interval_width"] = float(np.mean(y_pred_upper - y_pred_lower))

    return metrics
