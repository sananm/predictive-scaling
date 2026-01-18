"""
Phase 6 Tests: Prediction Service

Tests for:
- Predictor Orchestrator
- Uncertainty Quantification
- Prediction Calibration
"""

from datetime import UTC, datetime

import numpy as np


class TestPredictorOrchestrator:
    """Test predictor orchestrator."""

    def test_orchestrator_import(self):
        """Test PredictorOrchestrator can be imported."""
        from src.prediction.orchestrator import PredictorOrchestrator

        assert PredictorOrchestrator is not None

    def test_orchestrator_config_import(self):
        """Test OrchestratorConfig can be imported."""
        from src.prediction.orchestrator import OrchestratorConfig

        assert OrchestratorConfig is not None

    def test_orchestrator_config_defaults(self):
        """Test OrchestratorConfig default values."""
        from src.prediction.orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert len(config.short_term_horizons) > 0
        assert len(config.medium_term_horizons) > 0
        assert len(config.long_term_horizons) > 0
        assert config.min_history_minutes > 0
        assert config.max_history_minutes > config.min_history_minutes

    def test_orchestrator_initialization(self):
        """Test PredictorOrchestrator initialization."""
        from src.prediction.orchestrator import OrchestratorConfig, PredictorOrchestrator

        config = OrchestratorConfig()
        orchestrator = PredictorOrchestrator(config=config)

        assert orchestrator.config is not None
        assert orchestrator._initialized is False

    def test_prediction_output_import(self):
        """Test PredictionOutput can be imported."""
        from src.prediction.orchestrator import PredictionOutput

        assert PredictionOutput is not None

    def test_prediction_output_creation(self):
        """Test PredictionOutput creation."""
        from src.prediction.orchestrator import PredictionOutput

        now = datetime.now(UTC)
        output = PredictionOutput(
            timestamp=now,
            service_name="test-service",
            horizon_minutes=15,
            p10=90.0,
            p50=100.0,
            p90=110.0,
        )

        assert output.p50 == 100.0
        assert output.service_name == "test-service"
        assert output.horizon_minutes == 15

    def test_prediction_output_to_dict(self):
        """Test PredictionOutput to_dict method."""
        from src.prediction.orchestrator import PredictionOutput

        now = datetime.now(UTC)
        output = PredictionOutput(
            timestamp=now,
            service_name="test-service",
            horizon_minutes=15,
            p10=90.0,
            p50=100.0,
            p90=110.0,
            confidence_score=0.8,
        )

        d = output.to_dict()

        assert d["p50"] == 100.0
        assert d["service_name"] == "test-service"
        assert d["confidence_score"] == 0.8

    def test_orchestrator_get_stats(self):
        """Test orchestrator statistics."""
        from src.prediction.orchestrator import PredictorOrchestrator

        orchestrator = PredictorOrchestrator()
        stats = orchestrator.get_stats()

        assert "initialized" in stats
        assert "models_loaded" in stats
        assert "predictions_generated" in stats
        assert "predictions_failed" in stats


class TestUncertaintyQuantifier:
    """Test uncertainty quantification."""

    def test_uncertainty_quantifier_import(self):
        """Test UncertaintyQuantifier can be imported."""
        from src.prediction.uncertainty import UncertaintyQuantifier

        assert UncertaintyQuantifier is not None

    def test_uncertainty_result_import(self):
        """Test UncertaintyResult can be imported."""
        from src.prediction.uncertainty import UncertaintyResult

        assert UncertaintyResult is not None

    def test_uncertainty_config_import(self):
        """Test UncertaintyConfig can be imported."""
        from src.prediction.uncertainty import UncertaintyConfig

        assert UncertaintyConfig is not None

    def test_uncertainty_config_defaults(self):
        """Test UncertaintyConfig default values."""
        from src.prediction.uncertainty import UncertaintyConfig

        config = UncertaintyConfig()

        assert len(config.horizon_weights) > 0
        assert 0 < config.high_uncertainty_threshold < 1
        assert 0 < config.low_agreement_threshold < 1

    def test_quantifier_initialization(self):
        """Test UncertaintyQuantifier initialization."""
        from src.prediction.uncertainty import UncertaintyConfig, UncertaintyQuantifier

        config = UncertaintyConfig()
        quantifier = UncertaintyQuantifier(config=config)

        assert quantifier.config is not None

    def test_quantify_single_model(self):
        """Test uncertainty quantification with single model."""
        from src.models.base import PredictionResult
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        pred = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([90.0]),
            p50=np.array([100.0]),
            p90=np.array([110.0]),
            model_name="test",
            model_version="1.0",
            horizon_minutes=15,
        )

        result = quantifier.quantify({"transformer": pred}, horizon_minutes=15)

        assert result.combined_p50 == 100.0
        assert result.model_agreement == 1.0  # Single model = perfect agreement
        assert result.n_models == 1

    def test_quantify_multiple_models(self):
        """Test uncertainty quantification with multiple models."""
        from src.models.base import PredictionResult
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        pred1 = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([90.0]),
            p50=np.array([100.0]),
            p90=np.array([110.0]),
            model_name="model1",
            model_version="1.0",
            horizon_minutes=15,
        )

        pred2 = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([85.0]),
            p50=np.array([95.0]),
            p90=np.array([105.0]),
            model_name="model2",
            model_version="1.0",
            horizon_minutes=15,
        )

        result = quantifier.quantify(
            {"transformer": pred1, "gradient_boosting": pred2},
            horizon_minutes=15,
        )

        assert result.n_models == 2
        # Combined p50 should be between the two
        assert 95.0 <= result.combined_p50 <= 100.0
        # Agreement should be less than 1 since models differ
        assert 0 < result.model_agreement < 1

    def test_quantify_empty_predictions(self):
        """Test uncertainty quantification with no predictions."""
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        result = quantifier.quantify({}, horizon_minutes=15)

        assert result.n_models == 0
        assert result.high_uncertainty is True
        assert result.requires_review is True

    def test_model_agreement_high(self):
        """Test high model agreement when models agree."""
        from src.models.base import PredictionResult
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        # Very similar predictions
        pred1 = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([90.0]),
            p50=np.array([100.0]),
            p90=np.array([110.0]),
            model_name="model1",
            model_version="1.0",
            horizon_minutes=15,
        )

        pred2 = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([91.0]),
            p50=np.array([101.0]),
            p90=np.array([111.0]),
            model_name="model2",
            model_version="1.0",
            horizon_minutes=15,
        )

        result = quantifier.quantify(
            {"transformer": pred1, "gradient_boosting": pred2},
            horizon_minutes=15,
        )

        assert result.model_agreement > 0.8  # High agreement

    def test_model_agreement_low(self):
        """Test low model agreement when models disagree."""
        from src.models.base import PredictionResult
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        # Very different predictions
        pred1 = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([50.0]),
            p50=np.array([100.0]),
            p90=np.array([150.0]),
            model_name="model1",
            model_version="1.0",
            horizon_minutes=15,
        )

        pred2 = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([150.0]),
            p50=np.array([200.0]),
            p90=np.array([250.0]),
            model_name="model2",
            model_version="1.0",
            horizon_minutes=15,
        )

        result = quantifier.quantify(
            {"transformer": pred1, "gradient_boosting": pred2},
            horizon_minutes=15,
        )

        assert result.model_agreement < 0.85  # Lower agreement than very similar predictions
        assert result.requires_review is True

    def test_uncertainty_result_to_dict(self):
        """Test UncertaintyResult to_dict method."""
        from src.prediction.uncertainty import UncertaintyResult

        result = UncertaintyResult(
            combined_p10=90.0,
            combined_p50=100.0,
            combined_p90=110.0,
            model_agreement=0.85,
            confidence_score=0.9,
            interval_width=20.0,
            relative_uncertainty=0.2,
            n_models=2,
        )

        d = result.to_dict()

        assert d["combined_p50"] == 100.0
        assert d["model_agreement"] == 0.85
        assert d["n_models"] == 2

    def test_get_model_contribution(self):
        """Test getting model contribution details."""
        from src.models.base import PredictionResult
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        pred = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([90.0]),
            p50=np.array([100.0]),
            p90=np.array([110.0]),
            model_name="test",
            model_version="1.0",
            horizon_minutes=15,
        )

        contributions = quantifier.get_model_contribution(
            {"transformer": pred},
            horizon_minutes=15,
        )

        assert "transformer" in contributions
        assert contributions["transformer"]["p50"] == 100.0
        assert contributions["transformer"]["interval_width"] == 20.0


class TestPredictionCalibrator:
    """Test prediction calibration."""

    def test_calibrator_import(self):
        """Test PredictionCalibrator can be imported."""
        from src.prediction.calibration import PredictionCalibrator

        assert PredictionCalibrator is not None

    def test_calibration_result_import(self):
        """Test CalibrationResult can be imported."""
        from src.prediction.calibration import CalibrationResult

        assert CalibrationResult is not None

    def test_calibration_config_import(self):
        """Test CalibrationConfig can be imported."""
        from src.prediction.calibration import CalibrationConfig

        assert CalibrationConfig is not None

    def test_calibration_config_defaults(self):
        """Test CalibrationConfig default values."""
        from src.prediction.calibration import CalibrationConfig

        config = CalibrationConfig()

        assert config.target_coverage_90 == 0.90
        assert config.target_coverage_80 == 0.80
        assert config.min_samples > 0
        assert config.max_samples > config.min_samples

    def test_calibrator_initialization(self):
        """Test PredictionCalibrator initialization."""
        from src.prediction.calibration import CalibrationConfig, PredictionCalibrator

        config = CalibrationConfig()
        calibrator = PredictionCalibrator(config=config)

        assert calibrator.config is not None
        assert calibrator._total_samples == 0

    def test_calibration_update(self):
        """Test adding calibration samples."""
        from src.prediction.calibration import PredictionCalibrator

        calibrator = PredictionCalibrator()

        calibrator.update(
            horizon_minutes=15,
            predicted_p10=90.0,
            predicted_p50=100.0,
            predicted_p90=110.0,
            actual_value=105.0,
        )

        assert calibrator._total_samples == 1

    def test_calibration_multiple_updates(self):
        """Test multiple calibration updates."""
        from src.prediction.calibration import PredictionCalibrator

        calibrator = PredictionCalibrator()

        for i in range(10):
            calibrator.update(
                horizon_minutes=15,
                predicted_p10=90.0 + i,
                predicted_p50=100.0 + i,
                predicted_p90=110.0 + i,
                actual_value=100.0 + i + np.random.randn() * 5,
            )

        assert calibrator._total_samples == 10

    def test_get_calibration_factor_default(self):
        """Test default calibration factor."""
        from src.prediction.calibration import PredictionCalibrator

        calibrator = PredictionCalibrator()

        factor = calibrator.get_calibration_factor(horizon_minutes=15)

        assert factor == 1.0  # Default when no calibration data

    def test_calibration_sample_in_interval(self):
        """Test CalibrationSample interval checking."""
        from src.prediction.calibration import CalibrationSample

        sample = CalibrationSample(
            timestamp=datetime.now(UTC),
            horizon_minutes=15,
            predicted_p10=90.0,
            predicted_p50=100.0,
            predicted_p90=110.0,
            actual_value=105.0,
        )

        assert sample.in_90_interval is True
        assert sample.prediction_error == 5.0

    def test_calibration_sample_outside_interval(self):
        """Test CalibrationSample when actual is outside interval."""
        from src.prediction.calibration import CalibrationSample

        sample = CalibrationSample(
            timestamp=datetime.now(UTC),
            horizon_minutes=15,
            predicted_p10=90.0,
            predicted_p50=100.0,
            predicted_p90=110.0,
            actual_value=120.0,  # Outside interval
        )

        assert sample.in_90_interval is False
        assert sample.prediction_error == 20.0

    def test_calibration_result_to_dict(self):
        """Test CalibrationResult to_dict method."""
        from src.prediction.calibration import CalibrationResult

        result = CalibrationResult(
            horizon_minutes=15,
            target_coverage=0.90,
            empirical_coverage=0.85,
            calibration_factor=1.1,
            n_samples=100,
            last_updated=datetime.now(UTC),
            is_calibrated=True,
        )

        d = result.to_dict()

        assert d["horizon_minutes"] == 15
        assert d["target_coverage"] == 0.90
        assert d["empirical_coverage"] == 0.85
        assert d["calibration_factor"] == 1.1

    def test_get_stats(self):
        """Test calibrator statistics."""
        from src.prediction.calibration import PredictionCalibrator

        calibrator = PredictionCalibrator()

        # Add some samples
        for i in range(5):
            calibrator.update(
                horizon_minutes=15,
                predicted_p10=90.0,
                predicted_p50=100.0,
                predicted_p90=110.0,
                actual_value=100.0 + i,
            )

        stats = calibrator.get_stats()

        assert stats["total_samples"] == 5
        assert "horizon_groups" in stats

    def test_calibrator_save_load(self, tmp_path):
        """Test saving and loading calibrator state."""
        from src.prediction.calibration import PredictionCalibrator

        calibrator1 = PredictionCalibrator()

        # Add samples
        for i in range(10):
            calibrator1.update(
                horizon_minutes=15,
                predicted_p10=90.0,
                predicted_p50=100.0,
                predicted_p90=110.0,
                actual_value=100.0 + i,
            )

        # Force a calibration factor
        calibrator1._calibration_factors[(0, 30)] = 1.2

        # Save
        save_path = tmp_path / "calibrator"
        calibrator1.save(save_path)

        # Load into new calibrator
        calibrator2 = PredictionCalibrator()
        calibrator2.load(save_path)

        # Verify
        assert calibrator2._calibration_factors.get((0, 30)) == 1.2

    def test_get_accuracy_metrics(self):
        """Test getting accuracy metrics."""
        from src.prediction.calibration import PredictionCalibrator

        calibrator = PredictionCalibrator()

        # Add samples with known values
        for i in range(20):
            calibrator.update(
                horizon_minutes=15,
                predicted_p10=90.0,
                predicted_p50=100.0,
                predicted_p90=110.0,
                actual_value=100.0 + (i % 5),  # Values 100-104
            )

        metrics = calibrator.get_accuracy_metrics(horizon_minutes=15)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "coverage_90" in metrics
        assert metrics["n_samples"] == 20


class TestPredictionModuleExports:
    """Test prediction module exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.prediction import (
            CalibrationResult,
            PredictionCalibrator,
            PredictorOrchestrator,
            UncertaintyQuantifier,
            UncertaintyResult,
        )

        assert PredictorOrchestrator is not None
        assert UncertaintyQuantifier is not None
        assert UncertaintyResult is not None
        assert PredictionCalibrator is not None
        assert CalibrationResult is not None


class TestPredictionIntegration:
    """Test prediction service integration patterns."""

    def test_uncertainty_to_calibrator_flow(self):
        """Test flow from uncertainty to calibrator."""
        from src.models.base import PredictionResult
        from src.prediction.calibration import PredictionCalibrator
        from src.prediction.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        calibrator = PredictionCalibrator()

        # Generate prediction
        pred = PredictionResult(
            timestamps=[datetime.now(UTC)],
            p10=np.array([90.0]),
            p50=np.array([100.0]),
            p90=np.array([110.0]),
            model_name="test",
            model_version="1.0",
            horizon_minutes=15,
        )

        # Quantify uncertainty
        uncertainty = quantifier.quantify({"transformer": pred}, horizon_minutes=15)

        # Get calibration factor
        factor = calibrator.get_calibration_factor(
            horizon_minutes=15,
            confidence_score=uncertainty.confidence_score,
        )

        # Apply calibration
        interval_width = uncertainty.combined_p90 - uncertainty.combined_p10
        calibrated_half_width = (interval_width / 2) * factor

        calibrated_p10 = uncertainty.combined_p50 - calibrated_half_width
        calibrated_p90 = uncertainty.combined_p50 + calibrated_half_width

        assert calibrated_p10 <= uncertainty.combined_p50 <= calibrated_p90

        # Update calibrator with "actual" value
        actual_value = 98.0
        calibrator.update(
            horizon_minutes=15,
            predicted_p10=calibrated_p10,
            predicted_p50=uncertainty.combined_p50,
            predicted_p90=calibrated_p90,
            actual_value=actual_value,
        )

        assert calibrator._total_samples == 1
