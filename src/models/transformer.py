"""
Transformer model for short-term predictions (5-15 minutes).

Architecture:
- Input projection layer
- Positional encoding (sinusoidal)
- Transformer encoder (multi-head attention)
- Separate output heads for quantile predictions
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.logging import get_logger

from .base import BaseModel, PredictionResult, calculate_metrics

logger = get_logger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""

    # Model architecture
    input_dim: int = 100  # Number of input features
    d_model: int = 128  # Transformer hidden dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dim_feedforward: int = 512  # Feedforward network dimension
    dropout: float = 0.1  # Dropout rate

    # Sequence parameters
    max_seq_length: int = 60  # Maximum sequence length (minutes of history)
    prediction_horizon: int = 15  # Prediction horizon (minutes)

    # Output
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)  # Prediction quantiles

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_seq_length": self.max_seq_length,
            "prediction_horizon": self.prediction_horizon,
            "quantiles": list(self.quantiles),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip": self.gradient_clip,
        }


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.

    Adds position information to the input embeddings so the model
    knows the order of the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class QuantileHead(nn.Module):
    """Output head for a single quantile prediction."""

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerPredictor(nn.Module):
    """
    Transformer model for time series prediction.

    Architecture:
    1. Input projection: features → d_model
    2. Positional encoding
    3. Transformer encoder (self-attention)
    4. Global pooling
    5. Separate heads for each quantile
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.d_model,
            max_len=config.max_seq_length,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(config.d_model)

        # Quantile prediction heads
        self.quantile_heads = nn.ModuleDict({
            f"q{int(q * 100)}": QuantileHead(config.d_model)
            for q in config.quantiles
        })

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Dictionary of quantile predictions
        """
        # Project input to d_model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, mask=mask)

        # Layer normalization
        x = self.norm(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)

        # Quantile predictions
        outputs = {}
        for name, head in self.quantile_heads.items():
            outputs[name] = head(x).squeeze(-1)

        return outputs


class QuantileLoss(nn.Module):
    """
    Quantile loss (pinball loss) for probabilistic predictions.

    For quantile q:
    - If y > y_pred: loss = q * (y - y_pred)
    - If y < y_pred: loss = (1-q) * (y_pred - y)
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate combined quantile loss.

        Args:
            predictions: Dict of quantile predictions
            target: True values

        Returns:
            Combined loss
        """
        total_loss = 0.0

        for q in self.quantiles:
            key = f"q{int(q * 100)}"
            pred = predictions[key]
            errors = target - pred

            loss = torch.where(
                errors >= 0,
                q * errors,
                (q - 1) * errors,
            )
            total_loss = total_loss + loss.mean()

        return total_loss / len(self.quantiles)


class ShortTermModel(BaseModel):
    """
    Transformer-based model for short-term predictions (5-15 minutes).

    Uses attention mechanism to capture patterns in recent history
    and predict multiple quantiles for uncertainty estimation.
    """

    def __init__(
        self,
        config: TransformerConfig | None = None,
        horizon_minutes: int = 15,
        device: str | None = None,
    ) -> None:
        """
        Initialize short-term model.

        Args:
            config: Model configuration
            horizon_minutes: Prediction horizon
            device: Device to use (cpu/cuda)
        """
        super().__init__(
            name="transformer",
            horizon_minutes=horizon_minutes,
            version="1.0.0",
        )

        self.config = config or TransformerConfig(prediction_horizon=horizon_minutes)

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model will be created after we know input_dim
        self._model: TransformerPredictor | None = None
        self._loss_fn = QuantileLoss(self.config.quantiles)

        # Normalization parameters
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    def _create_model(self, input_dim: int) -> None:
        """Create model with correct input dimension."""
        self.config.input_dim = input_dim
        self._model = TransformerPredictor(self.config).to(self.device)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train the transformer model.

        Args:
            X: Training features (should be 2D: samples x features)
            y: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training metrics
        """
        # Store feature names
        self._feature_names = X.columns.tolist()

        # Create model
        self._create_model(len(self._feature_names))

        # Compute normalization parameters
        self._feature_mean = X.values.mean(axis=0)
        self._feature_std = X.values.std(axis=0) + 1e-8
        self._target_mean = float(y.mean())
        self._target_std = float(y.std()) + 1e-8

        # Normalize data
        X_norm = (X.values - self._feature_mean) / self._feature_std
        y_norm = (y.values - self._target_mean) / self._target_std

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)

        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Validation data
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val.values - self._feature_mean) / self._feature_std
            y_val_norm = (y_val.values - self._target_mean) / self._target_std
            X_val_seq, y_val_seq = self._create_sequences(X_val_norm, y_val_norm)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        else:
            # Split training data
            split_idx = int(len(X_tensor) * 0.8)
            X_val_tensor = X_tensor[split_idx:]
            y_val_tensor = y_tensor[split_idx:]
            X_tensor = X_tensor[:split_idx]
            y_tensor = y_tensor[:split_idx]

        # Optimizer
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        training_history = []

        for epoch in range(self.config.max_epochs):
            # Training
            self._model.train()
            train_loss = self._train_epoch(
                X_tensor, y_tensor, optimizer, self.config.batch_size
            )

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_predictions = self._model(X_val_tensor)
                val_loss = self._loss_fn(val_predictions, y_val_tensor).item()

            training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    "Training progress",
                    epoch=epoch,
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping", epoch=epoch)
                break

        # Restore best model
        if best_state:
            self._model.load_state_dict(best_state)

        self._is_trained = True

        # Calculate final metrics
        self._model.eval()
        with torch.no_grad():
            val_pred = self._model(X_val_tensor)
            y_val_pred = val_pred["q50"].cpu().numpy() * self._target_std + self._target_mean
            y_val_actual = y_val_tensor.cpu().numpy() * self._target_std + self._target_mean

            y_pred_lower = val_pred["q10"].cpu().numpy() * self._target_std + self._target_mean
            y_pred_upper = val_pred["q90"].cpu().numpy() * self._target_std + self._target_mean

        metrics = calculate_metrics(y_val_actual, y_val_pred, y_pred_lower, y_pred_upper)
        metrics["final_train_loss"] = train_loss
        metrics["final_val_loss"] = best_val_loss

        # Update metadata
        self._update_metadata(
            training_metrics={"train_loss": train_loss},
            validation_metrics=metrics,
            hyperparameters=self.config.to_dict(),
        )
        self._metadata.training_samples = len(X_tensor)
        self._metadata.validation_samples = len(X_val_tensor)
        self._metadata.n_features = len(self._feature_names)
        self._metadata.input_features = self._feature_names

        logger.info(
            "Training complete",
            mae=f"{metrics['mae']:.4f}",
            rmse=f"{metrics['rmse']:.4f}",
            coverage=f"{metrics.get('coverage_80', 0):.1f}%",
        )

        return metrics

    def _train_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
    ) -> float:
        """Train for one epoch."""
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        # Shuffle indices
        indices = torch.randperm(len(X))

        for i in range(0, len(X), batch_size):
            batch_indices = indices[i : i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            optimizer.zero_grad()
            predictions = self._model(X_batch)
            loss = self._loss_fn(predictions, y_batch)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self.config.gradient_clip,
            )

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for transformer input.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Tuple of (sequences, targets)
        """
        seq_len = self.config.max_seq_length
        horizon = self.config.prediction_horizon

        sequences = []
        targets = []

        for i in range(len(X) - seq_len - horizon + 1):
            seq = X[i : i + seq_len]
            target = y[i + seq_len + horizon - 1]

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

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
            PredictionResult with predictions
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Validate and align features
        X = self._validate_features(X.copy())

        # Normalize
        X_norm = (X.values - self._feature_mean) / self._feature_std

        # Create sequences
        X_seq, _ = self._create_sequences(X_norm, np.zeros(len(X_norm)))

        if len(X_seq) == 0:
            # Not enough data for sequence, use last available
            X_seq = X_norm[-self.config.max_seq_length :].reshape(1, -1, X_norm.shape[1])

        # Predict
        self._model.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            predictions = self._model(X_tensor)

        # Denormalize
        p10 = predictions["q10"].cpu().numpy() * self._target_std + self._target_mean
        p50 = predictions["q50"].cpu().numpy() * self._target_std + self._target_mean
        p90 = predictions["q90"].cpu().numpy() * self._target_std + self._target_mean

        # Create timestamps for predictions
        if hasattr(X.index, "to_pydatetime"):
            last_timestamp = X.index[-1]
            from datetime import timedelta
            timestamps = [
                last_timestamp + timedelta(minutes=self.horizon_minutes)
                for _ in range(len(p50))
            ]
        else:
            timestamps = list(range(len(p50)))

        return PredictionResult(
            timestamps=timestamps,
            p10=p10,
            p50=p50,
            p90=p90,
            model_name=self.name,
            model_version=self.version,
            horizon_minutes=self.horizon_minutes,
            metadata={"sequence_length": self.config.max_seq_length},
        )

    def _save_model_artifacts(self, path: Path) -> None:
        """Save PyTorch model and normalization parameters."""
        if self._model is not None:
            torch.save(self._model.state_dict(), path / "model.pt")

        # Save config
        import json
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save normalization parameters
        np.savez(
            path / "normalization.npz",
            feature_mean=self._feature_mean,
            feature_std=self._feature_std,
            target_mean=np.array([self._target_mean]),
            target_std=np.array([self._target_std]),
        )

    def _load_model_artifacts(self, path: Path) -> None:
        """Load PyTorch model and normalization parameters."""
        import json

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
            self.config = TransformerConfig(**config_dict)

        # Create model
        self._create_model(self.config.input_dim)

        # Load weights
        state_dict = torch.load(path / "model.pt", map_location=self.device)
        self._model.load_state_dict(state_dict)

        # Load normalization
        norm_data = np.load(path / "normalization.npz")
        self._feature_mean = norm_data["feature_mean"]
        self._feature_std = norm_data["feature_std"]
        self._target_mean = float(norm_data["target_mean"][0])
        self._target_std = float(norm_data["target_std"][0])
