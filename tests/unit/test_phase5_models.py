"""
Phase 5 Tests: Prediction Models

Tests for:
- Base model classes
- Transformer model (short-term)
- Gradient boosting ensemble (medium-term)
- Prophet model (long-term)
- Ensemble combiner
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest


# Helper function for conditional tests
def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


class TestBaseModel:
    """Test base model classes."""

    def test_base_model_import(self):
        """Test BaseModel can be imported."""
        from src.models.base import BaseModel

        assert BaseModel is not None

    def test_prediction_result_import(self):
        """Test PredictionResult can be imported."""
        from src.models.base import PredictionResult

        assert PredictionResult is not None

    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        from src.models.base import PredictionResult

        timestamps = [datetime.now(UTC) + timedelta(minutes=i) for i in range(10)]
        p10 = np.random.rand(10) * 100
        p50 = p10 + np.random.rand(10) * 20
        p90 = p50 + np.random.rand(10) * 20

        result = PredictionResult(
            timestamps=timestamps,
            p10=p10,
            p50=p50,
            p90=p90,
            model_name="test",
            model_version="1.0.0",
            horizon_minutes=15,
        )

        assert len(result.timestamps) == 10
        assert len(result.p50) == 10
        assert result.model_name == "test"
        assert result.horizon_minutes == 15

    def test_prediction_result_confidence_interval(self):
        """Test confidence interval width calculation."""
        from src.models.base import PredictionResult

        p10 = np.array([80, 85, 90])
        p90 = np.array([120, 125, 130])

        result = PredictionResult(
            timestamps=[datetime.now(UTC)] * 3,
            p10=p10,
            p50=np.array([100, 105, 110]),
            p90=p90,
            model_name="test",
            model_version="1.0.0",
            horizon_minutes=15,
        )

        width = result.confidence_interval_width
        np.testing.assert_array_equal(width, p90 - p10)

    def test_prediction_result_to_dataframe(self):
        """Test conversion to DataFrame."""
        from src.models.base import PredictionResult

        result = PredictionResult(
            timestamps=[datetime.now(UTC)] * 5,
            p10=np.array([80, 85, 90, 95, 100]),
            p50=np.array([100, 105, 110, 115, 120]),
            p90=np.array([120, 125, 130, 135, 140]),
            model_name="test",
            model_version="1.0.0",
            horizon_minutes=15,
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "p10" in df.columns
        assert "p50" in df.columns
        assert "p90" in df.columns
        assert len(df) == 5

    def test_model_metadata_import(self):
        """Test ModelMetadata can be imported."""
        from src.models.base import ModelMetadata

        assert ModelMetadata is not None

    def test_model_metadata_creation(self):
        """Test ModelMetadata creation."""
        from src.models.base import ModelMetadata

        metadata = ModelMetadata(
            model_id="test-123",
            model_name="transformer",
            model_version="1.0.0",
            created_at=datetime.now(UTC),
        )

        assert metadata.model_id == "test-123"
        assert metadata.model_name == "transformer"

    def test_quantile_loss_function(self):
        """Test quantile loss calculation."""
        from src.models.base import quantile_loss

        y_true = np.array([100, 110, 120])
        y_pred = np.array([95, 115, 115])

        # Test median (0.5) quantile loss
        loss_50 = quantile_loss(y_true, y_pred, 0.5)
        assert loss_50 > 0

        # Test asymmetric loss for different quantiles
        loss_10 = quantile_loss(y_true, y_pred, 0.1)
        loss_90 = quantile_loss(y_true, y_pred, 0.9)

        # Different quantiles should give different losses
        assert loss_10 != loss_90

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from src.models.base import calculate_metrics

        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([102, 108, 122, 128, 142])
        y_lower = np.array([90, 98, 110, 118, 130])
        y_upper = np.array([110, 118, 130, 138, 150])

        metrics = calculate_metrics(y_true, y_pred, y_lower, y_upper)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE always

    def test_coverage_metric(self):
        """Test coverage metric calculation."""
        from src.models.base import calculate_metrics

        y_true = np.array([100, 110, 120, 130, 140])
        y_lower = np.array([95, 105, 115, 125, 135])
        y_upper = np.array([105, 115, 125, 135, 145])

        metrics = calculate_metrics(y_true, y_true, y_lower, y_upper)

        assert "coverage_80" in metrics
        assert metrics["coverage_80"] == 100.0  # All within bounds


class TestTransformerModel:
    """Test Transformer model for short-term predictions."""

    def test_transformer_config_import(self):
        """Test TransformerConfig can be imported."""
        from src.models.transformer import TransformerConfig

        assert TransformerConfig is not None

    def test_transformer_config_defaults(self):
        """Test TransformerConfig default values."""
        from src.models.transformer import TransformerConfig

        config = TransformerConfig()

        assert config.d_model > 0
        assert config.nhead > 0
        assert config.num_layers > 0
        assert config.max_seq_length > 0
        assert config.prediction_horizon > 0

    def test_transformer_config_validation(self):
        """Test that d_model is divisible by nhead."""
        from src.models.transformer import TransformerConfig

        config = TransformerConfig()
        assert config.d_model % config.nhead == 0

    def test_short_term_model_import(self):
        """Test ShortTermModel can be imported."""
        from src.models.transformer import ShortTermModel

        assert ShortTermModel is not None

    def test_short_term_model_initialization(self):
        """Test ShortTermModel initialization."""
        from src.models.transformer import ShortTermModel, TransformerConfig

        config = TransformerConfig(
            input_dim=10,
            d_model=32,
            nhead=4,
            num_layers=2,
        )
        model = ShortTermModel(config=config)

        assert model.name == "transformer"
        assert model.is_trained is False

    def test_transformer_predictor_import(self):
        """Test TransformerPredictor can be imported."""
        from src.models.transformer import TransformerPredictor

        assert TransformerPredictor is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available",
    )
    def test_transformer_predictor_forward(self):
        """Test TransformerPredictor forward pass."""
        import torch

        from src.models.transformer import TransformerConfig, TransformerPredictor

        config = TransformerConfig(
            input_dim=10,
            d_model=32,
            nhead=4,
            num_layers=2,
            max_seq_length=30,
            prediction_horizon=10,
        )

        model = TransformerPredictor(config)

        # Create dummy input: (batch, seq_len, features)
        x = torch.randn(4, 30, 10)

        output = model(x)

        # Output keys are "q10", "q50", "q90" (not "p10", "p50", "p90")
        assert "q10" in output
        assert "q50" in output
        assert "q90" in output
        assert output["q50"].shape == (4,)  # Shape is (batch,) not (batch, prediction_length)


class TestGradientBoostingEnsemble:
    """Test Gradient Boosting ensemble for medium-term predictions."""

    def test_ensemble_config_import(self):
        """Test EnsembleConfig can be imported."""
        from src.models.ensemble import EnsembleConfig

        assert EnsembleConfig is not None

    def test_ensemble_config_defaults(self):
        """Test EnsembleConfig default values."""
        from src.models.ensemble import EnsembleConfig

        config = EnsembleConfig()

        assert len(config.horizons_hours) > 0
        assert len(config.quantiles) == 3
        assert 0.1 in config.quantiles
        assert 0.5 in config.quantiles
        assert 0.9 in config.quantiles

    def test_model_weights_sum_to_one(self):
        """Test that model weights sum to approximately 1."""
        from src.models.ensemble import EnsembleConfig

        config = EnsembleConfig()
        total = sum(config.model_weights.values())

        assert abs(total - 1.0) < 0.01

    def test_medium_term_model_import(self):
        """Test MediumTermModel can be imported."""
        from src.models.ensemble import MediumTermModel

        assert MediumTermModel is not None

    def test_medium_term_model_initialization(self):
        """Test MediumTermModel initialization."""
        from src.models.ensemble import MediumTermModel

        model = MediumTermModel()

        assert model.name == "gradient_boosting_ensemble"
        assert model.is_trained is False

    def test_gradient_boosting_model_wrapper(self):
        """Test GradientBoostingModel wrapper."""
        from src.models.ensemble import GradientBoostingModel

        model = GradientBoostingModel(
            model_type="xgboost",
            quantile=0.5,
            params={"n_estimators": 10},
        )

        assert model.model_type == "xgboost"
        assert model.quantile == 0.5

    def test_quantile_configuration(self):
        """Test quantile loss configuration for each model type."""
        from src.models.ensemble import GradientBoostingModel

        # XGBoost
        xgb_model = GradientBoostingModel("xgboost", 0.9, {})
        assert "quantile_alpha" in xgb_model.params

        # LightGBM
        lgb_model = GradientBoostingModel("lightgbm", 0.9, {})
        assert "alpha" in lgb_model.params

        # CatBoost
        cat_model = GradientBoostingModel("catboost", 0.9, {})
        assert "Quantile" in cat_model.params.get("loss_function", "")


class TestProphetModel:
    """Test Prophet model for long-term predictions."""

    def test_prophet_config_import(self):
        """Test ProphetConfig can be imported."""
        from src.models.prophet_model import ProphetConfig

        assert ProphetConfig is not None

    def test_prophet_config_defaults(self):
        """Test ProphetConfig default values."""
        from src.models.prophet_model import ProphetConfig

        config = ProphetConfig()

        assert config.yearly_seasonality is True or config.yearly_seasonality is False
        assert config.weekly_seasonality is True or config.weekly_seasonality is False
        assert config.daily_seasonality is True or config.daily_seasonality is False
        assert config.interval_width > 0
        assert config.event_decay_rate >= 0

    def test_long_term_model_import(self):
        """Test LongTermModel can be imported."""
        from src.models.prophet_model import LongTermModel

        assert LongTermModel is not None

    def test_long_term_model_initialization(self):
        """Test LongTermModel initialization."""
        from src.models.prophet_model import LongTermModel

        model = LongTermModel()

        assert model.name == "prophet"
        assert model.is_trained is False

    def test_prophet_config_to_dict(self):
        """Test ProphetConfig to_dict method."""
        from src.models.prophet_model import ProphetConfig

        config = ProphetConfig()
        config_dict = config.to_dict()

        assert "yearly_seasonality" in config_dict
        assert "weekly_seasonality" in config_dict
        assert "daily_seasonality" in config_dict
        assert "event_decay_rate" in config_dict


class TestEnsembleCombiner:
    """Test ensemble combiner."""

    def test_combiner_config_import(self):
        """Test CombinerConfig can be imported."""
        from src.models.combiner import CombinerConfig

        assert CombinerConfig is not None

    def test_combiner_config_defaults(self):
        """Test CombinerConfig default values."""
        from src.models.combiner import CombinerConfig

        config = CombinerConfig()

        assert config.short_term_max_minutes > 0
        assert config.medium_term_max_minutes > config.short_term_max_minutes
        assert config.long_term_max_minutes > config.medium_term_max_minutes

    def test_blending_zones(self):
        """Test blending zone configuration."""
        from src.models.combiner import CombinerConfig

        config = CombinerConfig()

        # Short-medium blend should be between short and medium max
        assert config.short_medium_blend_start < config.short_medium_blend_end
        assert config.short_medium_blend_end <= config.medium_term_max_minutes

        # Medium-long blend should be between medium and long max
        assert config.medium_long_blend_start < config.medium_long_blend_end

    def test_ensemble_combiner_import(self):
        """Test EnsembleCombiner can be imported."""
        from src.models.combiner import EnsembleCombiner

        assert EnsembleCombiner is not None

    def test_ensemble_combiner_initialization(self):
        """Test EnsembleCombiner initialization."""
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        assert combiner.config is not None
        assert combiner._weights is not None

    def test_get_model_weights_short_term(self):
        """Test weight calculation for short-term horizon."""
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        # Very short horizon should favor transformer
        weights = combiner._get_model_weights(5)

        assert weights["transformer"] > weights["gradient_boosting"]
        assert weights["transformer"] > weights["prophet"]

    def test_get_model_weights_medium_term(self):
        """Test weight calculation for medium-term horizon."""
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        # Medium horizon (2 hours) should favor gradient boosting
        weights = combiner._get_model_weights(120)

        assert weights["gradient_boosting"] >= weights["transformer"]

    def test_get_model_weights_long_term(self):
        """Test weight calculation for long-term horizon."""
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        # Long horizon (3 days) should favor prophet
        weights = combiner._get_model_weights(4320)

        assert weights["prophet"] > weights["transformer"]

    def test_blend_weights(self):
        """Test weight blending function."""
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        weights1 = {"a": 0.8, "b": 0.2}
        weights2 = {"a": 0.2, "b": 0.8}

        # blend=0 should give weights1
        result = combiner._blend_weights(weights1, weights2, 0.0)
        assert result["a"] == 0.8
        assert result["b"] == 0.2

        # blend=1 should give weights2
        result = combiner._blend_weights(weights1, weights2, 1.0)
        assert result["a"] == 0.2
        assert result["b"] == 0.8

        # blend=0.5 should give average
        result = combiner._blend_weights(weights1, weights2, 0.5)
        assert result["a"] == 0.5
        assert result["b"] == 0.5

    def test_model_agreement_metric(self):
        """Test model agreement calculation."""
        from src.models.base import PredictionResult
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        # Create predictions that agree
        pred1 = PredictionResult(
            timestamps=[datetime.now(UTC)] * 5,
            p10=np.array([90, 95, 100, 105, 110]),
            p50=np.array([100, 105, 110, 115, 120]),
            p90=np.array([110, 115, 120, 125, 130]),
            model_name="model1",
            model_version="1.0",
            horizon_minutes=15,
        )

        pred2 = PredictionResult(
            timestamps=[datetime.now(UTC)] * 5,
            p10=np.array([92, 97, 102, 107, 112]),
            p50=np.array([102, 107, 112, 117, 122]),
            p90=np.array([112, 117, 122, 127, 132]),
            model_name="model2",
            model_version="1.0",
            horizon_minutes=15,
        )

        agreement = combiner.get_model_agreement({"model1": pred1, "model2": pred2})

        assert 0 <= agreement <= 1
        assert agreement > 0.5  # Should have high agreement

    def test_combine_predictions(self):
        """Test prediction combination."""
        from src.models.base import PredictionResult
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()

        pred1 = PredictionResult(
            timestamps=[datetime.now(UTC)] * 3,
            p10=np.array([80, 85, 90]),
            p50=np.array([100, 105, 110]),
            p90=np.array([120, 125, 130]),
            model_name="model1",
            model_version="1.0",
            horizon_minutes=15,
        )

        pred2 = PredictionResult(
            timestamps=[datetime.now(UTC)] * 3,
            p10=np.array([85, 90, 95]),
            p50=np.array([105, 110, 115]),
            p90=np.array([125, 130, 135]),
            model_name="model2",
            model_version="1.0",
            horizon_minutes=15,
        )

        predictions = {"model1": pred1, "model2": pred2}
        weights = {"model1": 0.6, "model2": 0.4}

        combined = combiner._combine_predictions(predictions, weights, 15)

        assert combined.model_name == "ensemble"
        assert len(combined.p50) == 3
        # Combined p50 should be between the two
        assert all(combined.p50 >= pred1.p50 - 10)
        assert all(combined.p50 <= pred2.p50 + 10)


class TestModelPersistence:
    """Test model save/load functionality."""

    def test_base_model_save_load_interface(self):
        """Test that base model has save/load methods."""
        from src.models.base import BaseModel

        assert hasattr(BaseModel, "save")
        assert hasattr(BaseModel, "load")

    def test_combiner_save_structure(self, tmp_path):
        """Test combiner save creates expected structure."""
        from src.models.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()
        save_path = tmp_path / "combiner"

        combiner.save(save_path)

        assert (save_path / "config.json").exists()
        assert (save_path / "weights.json").exists()

    def test_combiner_load(self, tmp_path):
        """Test combiner can load saved state."""
        from src.models.combiner import EnsembleCombiner

        # Save
        combiner1 = EnsembleCombiner()
        save_path = tmp_path / "combiner"
        combiner1.save(save_path)

        # Load
        combiner2 = EnsembleCombiner()
        combiner2.load(save_path)

        # Verify weights match
        assert combiner1._weights == combiner2._weights
