"""
Phase 1 Tests: Configuration and Logging

Tests for:
- Configuration loading and validation
- Structured logging
- Settings validation
"""

import os
from unittest.mock import patch

import pytest


class TestConfiguration:
    """Test configuration loading."""

    def test_settings_import(self):
        """Test that settings can be imported."""
        from config.settings import AppSettings

        assert AppSettings is not None

    def test_default_settings(self):
        """Test default settings values."""
        from config.settings import AppSettings

        settings = AppSettings()

        # Check default values exist
        assert settings.env in ["development", "staging", "production"]
        assert settings.debug is True or settings.debug is False
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_database_settings(self):
        """Test database settings."""
        from config.settings import DatabaseSettings

        db_settings = DatabaseSettings()

        assert db_settings.pool_size >= 1
        assert db_settings.pool_size <= 100
        assert db_settings.pool_recycle > 0

    def test_kafka_settings(self):
        """Test Kafka settings."""
        from config.settings import KafkaSettings

        kafka_settings = KafkaSettings()

        assert kafka_settings.bootstrap_servers is not None
        assert kafka_settings.consumer_group is not None

    def test_model_settings(self):
        """Test ML model settings."""
        from config.settings import ModelSettings

        model_settings = ModelSettings()

        assert model_settings.transformer_d_model > 0
        assert model_settings.transformer_nhead > 0
        assert model_settings.short_term_horizon > 0
        assert model_settings.medium_term_horizon > 0
        assert model_settings.long_term_horizon > 0

    def test_scaling_settings(self):
        """Test scaling settings."""
        from config.settings import ScalingSettings

        scaling_settings = ScalingSettings()

        assert scaling_settings.min_instances >= 1
        assert scaling_settings.max_instances >= scaling_settings.min_instances
        assert 0 < scaling_settings.target_utilization <= 1.0
        assert scaling_settings.headroom_factor >= 1.0

    def test_settings_caching(self):
        """Test that get_settings returns cached instance."""
        from config.settings import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance (cached)
        assert settings1 is settings2

    def test_nested_settings(self):
        """Test that AppSettings contains nested settings."""
        from config.settings import AppSettings

        settings = AppSettings()

        # Check nested settings are accessible
        assert settings.database is not None
        assert settings.kafka is not None
        assert settings.model is not None
        assert settings.scaling is not None


class TestLogging:
    """Test structured logging setup."""

    def test_logger_import(self):
        """Test that logger can be imported."""
        from src.utils.logging import get_logger

        assert get_logger is not None

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        from src.utils.logging import get_logger

        logger = get_logger("test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_logger_with_module_name(self):
        """Test logger with __name__ pattern."""
        from src.utils.logging import get_logger

        logger = get_logger(__name__)
        assert logger is not None

    def test_log_context(self):
        """Test log context manager."""
        from src.utils.logging import get_logger, log_context

        logger = get_logger("test")

        # Should not raise
        with log_context(request_id="test-123"):
            logger.info("Test message")


class TestDatabaseModels:
    """Test database model definitions."""

    def test_metric_model_import(self):
        """Test Metric model can be imported."""
        from src.storage.models import Metric

        assert Metric is not None
        assert hasattr(Metric, "__tablename__")
        assert Metric.__tablename__ == "metrics"

    def test_prediction_model_import(self):
        """Test Prediction model can be imported."""
        from src.storage.models import Prediction

        assert Prediction is not None
        assert hasattr(Prediction, "__tablename__")
        assert Prediction.__tablename__ == "predictions"

    def test_scaling_decision_model_import(self):
        """Test ScalingDecision model can be imported."""
        from src.storage.models import ScalingDecision

        assert ScalingDecision is not None
        assert hasattr(ScalingDecision, "__tablename__")
        assert ScalingDecision.__tablename__ == "scaling_decisions"

    def test_business_event_model_import(self):
        """Test BusinessEvent model can be imported."""
        from src.storage.models import BusinessEvent

        assert BusinessEvent is not None
        assert hasattr(BusinessEvent, "__tablename__")
        assert BusinessEvent.__tablename__ == "business_events"

    def test_model_relationships(self):
        """Test that models have expected columns."""
        from src.storage.models import Metric, Prediction

        # Metric should have timestamp, service_name, value
        assert hasattr(Metric, "timestamp")
        assert hasattr(Metric, "service_name")
        assert hasattr(Metric, "value")

        # Prediction should have quantile predictions
        assert hasattr(Prediction, "prediction_p10")
        assert hasattr(Prediction, "prediction_p50")
        assert hasattr(Prediction, "prediction_p90")


class TestRepositoryPattern:
    """Test repository pattern implementation."""

    def test_base_repository_import(self):
        """Test BaseRepository can be imported."""
        from src.storage.repositories import BaseRepository

        assert BaseRepository is not None

    def test_metrics_repository_import(self):
        """Test MetricsRepository can be imported."""
        from src.storage.repositories import MetricsRepository

        assert MetricsRepository is not None

    def test_predictions_repository_import(self):
        """Test PredictionsRepository can be imported."""
        from src.storage.repositories import PredictionsRepository

        assert PredictionsRepository is not None

    def test_metrics_repository_methods_exist(self):
        """Test that MetricsRepository has expected methods."""
        from src.storage.repositories import MetricsRepository

        # MetricsRepository has insert/query methods (not BaseRepository pattern)
        assert hasattr(MetricsRepository, "insert")
        assert hasattr(MetricsRepository, "insert_batch")
        assert hasattr(MetricsRepository, "get_by_time_range")
        assert hasattr(MetricsRepository, "get_latest")
        assert hasattr(MetricsRepository, "to_dataframe")

    def test_predictions_repository_methods_exist(self):
        """Test that PredictionsRepository has expected methods."""
        from src.storage.repositories import PredictionsRepository

        # PredictionsRepository inherits from BaseRepository
        assert hasattr(PredictionsRepository, "get_by_id")
        assert hasattr(PredictionsRepository, "create")
        assert hasattr(PredictionsRepository, "insert")
        assert hasattr(PredictionsRepository, "get_latest_for_horizon")
