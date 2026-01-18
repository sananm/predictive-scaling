#!/usr/bin/env python3
"""
Model training script for the Predictive Scaling system.

This script:
- Loads historical data
- Runs feature engineering
- Trains all models (short, medium, long term)
- Evaluates on holdout set
- Saves models with versioning
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings


def generate_synthetic_training_data(
    days: int = 30,
    services: list[str] | None = None,
) -> pd.DataFrame:
    """Generate synthetic training data if no historical data available."""
    if services is None:
        services = ["api", "worker", "web"]

    print(f"Generating {days} days of synthetic training data...")

    np.random.seed(42)
    records = []

    start_time = datetime.now(timezone.utc) - timedelta(days=days)

    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 5):  # 5-minute intervals
                timestamp = start_time + timedelta(days=day, hours=hour, minutes=minute)

                for service in services:
                    # Base load with daily pattern
                    hour_factor = 1.0 + 0.5 * np.sin((hour - 6) * np.pi / 12)

                    # Weekly pattern (lower on weekends)
                    weekday = timestamp.weekday()
                    weekend_factor = 0.6 if weekday >= 5 else 1.0

                    # Base metrics
                    base_rps = 100 * hour_factor * weekend_factor

                    # Add noise
                    rps = base_rps * (1 + np.random.normal(0, 0.1))

                    # Occasional spikes
                    if np.random.random() < 0.02:
                        rps *= np.random.uniform(1.5, 3.0)

                    # CPU correlates with RPS
                    cpu = min(0.95, 0.3 + (rps / 300) * 0.5 + np.random.normal(0, 0.05))

                    # Memory is more stable
                    memory = 0.5 + np.random.normal(0, 0.1)
                    memory = max(0.2, min(0.9, memory))

                    # Latency increases with load
                    latency = 50 + (rps / 10) + np.random.exponential(10)

                    records.append({
                        "timestamp": timestamp,
                        "service_name": service,
                        "requests_per_second": max(0, rps),
                        "cpu_utilization": max(0, min(1, cpu)),
                        "memory_utilization": max(0, min(1, memory)),
                        "latency_p50": max(1, latency),
                        "latency_p99": max(1, latency * 2.5),
                        "error_rate": max(0, np.random.exponential(0.01)),
                        "active_connections": int(rps * 0.5 + np.random.normal(0, 10)),
                    })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} training samples")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw metrics."""
    print("Engineering features...")

    df = df.sort_values(["service_name", "timestamp"]).copy()

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Rolling features per service
    for service in df["service_name"].unique():
        mask = df["service_name"] == service

        # Rolling means
        for col in ["requests_per_second", "cpu_utilization", "latency_p50"]:
            df.loc[mask, f"{col}_rolling_mean_12"] = (
                df.loc[mask, col].rolling(12, min_periods=1).mean()
            )
            df.loc[mask, f"{col}_rolling_std_12"] = (
                df.loc[mask, col].rolling(12, min_periods=1).std().fillna(0)
            )

        # Lag features
        for col in ["requests_per_second", "cpu_utilization"]:
            df.loc[mask, f"{col}_lag_1"] = df.loc[mask, col].shift(1)
            df.loc[mask, f"{col}_lag_12"] = df.loc[mask, col].shift(12)

    # Fill NaN from lag features
    df = df.fillna(method="bfill").fillna(0)

    print(f"Engineered {len(df.columns)} features")
    return df


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "requests_per_second",
    horizon: int = 12,  # 12 * 5min = 1 hour ahead
) -> tuple:
    """Prepare data for model training."""
    print(f"Preparing training data for {horizon}-step ahead prediction...")

    feature_cols = [
        "hour_sin", "hour_cos", "is_weekend",
        "cpu_utilization", "memory_utilization",
        "requests_per_second_rolling_mean_12",
        "requests_per_second_rolling_std_12",
        "cpu_utilization_rolling_mean_12",
        "requests_per_second_lag_1",
        "requests_per_second_lag_12",
    ]

    # Create target (future value)
    df["target"] = df.groupby("service_name")[target_col].shift(-horizon)

    # Remove rows without target
    df = df.dropna(subset=["target"])

    X = df[feature_cols].values
    y = df["target"].values

    # Train/test split (80/20, respecting time order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols


def train_gradient_boosting(X_train, y_train):
    """Train a gradient boosting model."""
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a random forest model."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model performance."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def save_model(model, name: str, metrics: dict, feature_cols: list, model_dir: Path):
    """Save model with metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_v{version}.joblib"
    filepath = model_dir / filename

    # Save model with metadata
    model_data = {
        "model": model,
        "version": version,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(model_data, filepath)
    print(f"Saved model: {filepath}")

    # Also save as "latest"
    latest_path = model_dir / f"{name}_model.joblib"
    joblib.dump(model_data, latest_path)
    print(f"Saved as latest: {latest_path}")

    return filepath


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument("--days", type=int, default=30, help="Days of training data")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Predictive Scaling - Model Training")
    print("=" * 60)

    model_dir = Path(args.model_dir)

    # Step 1: Generate/load training data
    print("\nStep 1: Loading training data...")
    df = generate_synthetic_training_data(days=args.days)

    # Step 2: Feature engineering
    print("\nStep 2: Engineering features...")
    df = engineer_features(df)

    # Step 3: Train models for different horizons
    horizons = {
        "short_term": 3,   # 15 minutes (3 * 5min)
        "medium_term": 12,  # 1 hour (12 * 5min)
        "long_term": 288,   # 24 hours (288 * 5min)
    }

    for model_name, horizon in horizons.items():
        print(f"\n{'=' * 60}")
        print(f"Training {model_name} model (horizon: {horizon * 5} minutes)")
        print("=" * 60)

        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(
            df.copy(), horizon=horizon
        )

        # Train model
        print("\nTraining Gradient Boosting model...")
        model = train_gradient_boosting(X_train, y_train)

        # Evaluate
        print("\nEvaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  R2:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        # Save model
        print("\nSaving model...")
        save_model(model, model_name, metrics, feature_cols, model_dir)

    print("\n" + "=" * 60)
    print("Model training complete!")
    print(f"Models saved to: {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
