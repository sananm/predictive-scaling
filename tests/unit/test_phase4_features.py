"""
Phase 4 Tests: Feature Engineering

Tests for:
- Feature configuration
- Time features
- Lag features
- Rolling features
- Fourier features
- Business features
- Derivative features
- Feature orchestrator
"""


import pandas as pd


class TestFeatureConfig:
    """Test feature configuration."""

    def test_feature_config_import(self):
        """Test FeatureConfig can be imported."""
        from src.features.config import FeatureConfig

        assert FeatureConfig is not None

    def test_feature_config_defaults(self):
        """Test FeatureConfig default values."""
        from src.features.config import FeatureConfig

        config = FeatureConfig()

        assert config.granularity_minutes > 0
        assert len(config.lag_windows) > 0
        assert len(config.rolling_windows) > 0
        assert config.enable_time_features is True

    def test_minimal_config(self):
        """Test minimal configuration preset."""
        from src.features.config import MINIMAL_CONFIG

        assert len(MINIMAL_CONFIG.lag_windows) < 10
        assert len(MINIMAL_CONFIG.rolling_windows) < 10

    def test_full_config(self):
        """Test full configuration preset."""
        from src.features.config import FULL_CONFIG

        assert len(FULL_CONFIG.lag_windows) > 5
        assert len(FULL_CONFIG.rolling_windows) > 5

    def test_config_get_enabled_groups(self):
        """Test get_enabled_groups method."""
        from src.features.config import FeatureConfig

        config = FeatureConfig()
        groups = config.get_enabled_groups()

        assert "time" in groups
        assert "lag" in groups
        assert "rolling" in groups

    def test_config_required_history(self):
        """Test get_required_history_minutes method."""
        from src.features.config import FeatureConfig

        config = FeatureConfig()
        history = config.get_required_history_minutes()

        assert history > 0


class TestTimeFeatures:
    """Test time feature extraction."""

    def test_time_extractor_import(self):
        """Test TimeFeatureExtractor can be imported."""
        from src.features.time_features import TimeFeatureExtractor

        assert TimeFeatureExtractor is not None

    def test_time_features_extraction(self, time_index):
        """Test time feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.time_features import TimeFeatureExtractor

        config = FeatureConfig()
        extractor = TimeFeatureExtractor(config)
        df = pd.DataFrame({"dummy": [0] * len(time_index)}, index=time_index)

        features = extractor.extract(df)

        # Check cyclical features (prefixed with time_)
        assert "time_hour_sin" in features.columns
        assert "time_hour_cos" in features.columns
        assert "time_dow_sin" in features.columns
        assert "time_dow_cos" in features.columns

        # Check binary features
        assert "time_is_weekend" in features.columns
        assert "time_is_business_hours" in features.columns

    def test_cyclical_encoding_range(self, time_index):
        """Test that cyclical encodings are in [-1, 1] range."""
        from src.features.config import FeatureConfig
        from src.features.time_features import TimeFeatureExtractor

        config = FeatureConfig()
        extractor = TimeFeatureExtractor(config)
        df = pd.DataFrame({"dummy": [0] * len(time_index)}, index=time_index)

        features = extractor.extract(df)

        # Sin/cos should be in [-1, 1]
        assert features["time_hour_sin"].min() >= -1
        assert features["time_hour_sin"].max() <= 1
        assert features["time_hour_cos"].min() >= -1
        assert features["time_hour_cos"].max() <= 1

    def test_holiday_features(self, time_index):
        """Test holiday feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.time_features import HolidayFeatureExtractor

        config = FeatureConfig()
        extractor = HolidayFeatureExtractor(config)
        df = pd.DataFrame({"dummy": [0] * len(time_index)}, index=time_index)

        features = extractor.extract(df)

        assert "holiday_is_holiday" in features.columns
        assert "holiday_days_until_holiday" in features.columns


class TestLagFeatures:
    """Test lag feature extraction."""

    def test_lag_extractor_import(self):
        """Test LagFeatureExtractor can be imported."""
        from src.features.lag_features import LagFeatureExtractor

        assert LagFeatureExtractor is not None

    def test_lag_features_extraction(self, sample_metrics_data):
        """Test lag feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.lag_features import LagFeatureExtractor

        config = FeatureConfig(lag_windows=[5, 15, 60])
        extractor = LagFeatureExtractor(config)

        features = extractor.extract(sample_metrics_data, target_column="requests_per_second")

        # Check for lag features (format: lag_lag_Xm)
        assert any("lag_5m" in col for col in features.columns)
        assert any("lag_15m" in col for col in features.columns)
        assert any("lag_60m" in col for col in features.columns)

    def test_diff_features(self, sample_metrics_data):
        """Test difference features are included."""
        from src.features.config import FeatureConfig
        from src.features.lag_features import LagFeatureExtractor

        config = FeatureConfig(lag_windows=[5, 15])
        extractor = LagFeatureExtractor(config)

        features = extractor.extract(sample_metrics_data, target_column="requests_per_second")

        # Diff features should be present
        assert any("diff_" in col for col in features.columns)

    def test_multi_column_lag_extractor(self, sample_metrics_data):
        """Test multi-column lag extractor."""
        from src.features.config import FeatureConfig
        from src.features.lag_features import MultiColumnLagExtractor

        config = FeatureConfig(lag_windows=[5])
        extractor = MultiColumnLagExtractor(config)

        features = extractor.extract(sample_metrics_data, columns=["requests_per_second", "cpu_utilization"])

        assert len(features.columns) > 0


class TestRollingFeatures:
    """Test rolling feature extraction."""

    def test_rolling_extractor_import(self):
        """Test RollingFeatureExtractor can be imported."""
        from src.features.rolling_features import RollingFeatureExtractor

        assert RollingFeatureExtractor is not None

    def test_rolling_features_extraction(self, sample_metrics_data):
        """Test rolling feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.rolling_features import RollingFeatureExtractor

        config = FeatureConfig(rolling_windows=[15, 60])
        extractor = RollingFeatureExtractor(config)

        features = extractor.extract(sample_metrics_data, target_column="requests_per_second")

        # Check for rolling features
        assert len(features.columns) > 0

    def test_ema_features(self, sample_metrics_data):
        """Test EMA feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.rolling_features import ExponentialMovingAverageExtractor

        config = FeatureConfig()
        extractor = ExponentialMovingAverageExtractor(config)

        features = extractor.extract(sample_metrics_data, target_column="requests_per_second")

        assert len(features.columns) > 0


class TestFourierFeatures:
    """Test Fourier feature extraction."""

    def test_fourier_extractor_import(self):
        """Test FourierFeatureExtractor can be imported."""
        from src.features.fourier_features import FourierFeatureExtractor

        assert FourierFeatureExtractor is not None

    def test_fourier_features_extraction(self, time_index):
        """Test Fourier feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.fourier_features import FourierFeatureExtractor

        config = FeatureConfig(fourier_periods=[1440], fourier_harmonics={1440: 2})
        extractor = FourierFeatureExtractor(config)
        df = pd.DataFrame({"dummy": [0] * len(time_index)}, index=time_index)

        features = extractor.extract(df)

        # Should have sin and cos for each harmonic
        assert any("sin" in col for col in features.columns)
        assert any("cos" in col for col in features.columns)

    def test_fourier_range(self, time_index):
        """Test that Fourier sin/cos features are in [-1, 1] range."""
        from src.features.config import FeatureConfig
        from src.features.fourier_features import FourierFeatureExtractor

        config = FeatureConfig(fourier_periods=[1440], fourier_harmonics={1440: 1})
        extractor = FourierFeatureExtractor(config)
        df = pd.DataFrame({"dummy": [0] * len(time_index)}, index=time_index)

        features = extractor.extract(df)

        # Only check sin/cos columns (phase can be unbounded)
        for col in features.columns:
            if "sin" in col or "cos" in col:
                assert features[col].min() >= -1.01  # Small tolerance
                assert features[col].max() <= 1.01


class TestBusinessFeatures:
    """Test business feature extraction."""

    def test_business_extractor_import(self):
        """Test BusinessFeatureExtractor can be imported."""
        from src.features.business_features import BusinessFeatureExtractor

        assert BusinessFeatureExtractor is not None

    def test_business_features_extraction(self, time_index, sample_business_events):
        """Test business feature extraction."""
        from src.features.business_features import BusinessFeatureExtractor
        from src.features.config import FeatureConfig

        config = FeatureConfig()
        extractor = BusinessFeatureExtractor(config)
        df = pd.DataFrame({"dummy": [0] * len(time_index)}, index=time_index)

        features = extractor.extract(df, events=sample_business_events)

        # Should have event-related features
        assert len(features.columns) > 0


class TestDerivativeFeatures:
    """Test derivative feature extraction."""

    def test_derivative_extractor_import(self):
        """Test DerivativeFeatureExtractor can be imported."""
        from src.features.derivative_features import DerivativeFeatureExtractor

        assert DerivativeFeatureExtractor is not None

    def test_derivative_features_extraction(self, sample_metrics_data):
        """Test derivative feature extraction."""
        from src.features.config import FeatureConfig
        from src.features.derivative_features import DerivativeFeatureExtractor

        config = FeatureConfig(derivative_windows=[5])
        extractor = DerivativeFeatureExtractor(config)

        features = extractor.extract(sample_metrics_data, target_column="requests_per_second")

        # Should have derivative features
        assert len(features.columns) > 0


class TestFeatureEngineer:
    """Test feature engineering orchestrator."""

    def test_engineer_import(self):
        """Test FeatureEngineer can be imported."""
        from src.features.engineer import FeatureEngineer

        assert FeatureEngineer is not None

    def test_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        from src.features.config import FeatureConfig
        from src.features.engineer import FeatureEngineer

        config = FeatureConfig()
        engineer = FeatureEngineer(config=config)

        assert engineer.config is not None

    def test_compute_features(self, sample_metrics_data, sample_business_events):
        """Test full feature computation."""
        from src.features.config import MINIMAL_CONFIG
        from src.features.engineer import FeatureEngineer

        engineer = FeatureEngineer(config=MINIMAL_CONFIG, target_column="requests_per_second")

        features = engineer.compute_features(
            sample_metrics_data,
            events=sample_business_events,
        )

        # Should have many features
        assert len(features.columns) > 5
        assert len(features) == len(sample_metrics_data)

    def test_feature_caching(self, sample_metrics_data):
        """Test feature caching behavior."""
        from src.features.config import MINIMAL_CONFIG
        from src.features.engineer import FeatureEngineer

        engineer = FeatureEngineer(config=MINIMAL_CONFIG, target_column="requests_per_second")

        # First computation
        features1 = engineer.compute_features(sample_metrics_data)

        # Second computation (should use cache)
        features2 = engineer.compute_features(sample_metrics_data)

        # Results should be equal
        pd.testing.assert_frame_equal(features1, features2)

    def test_get_metadata(self, sample_metrics_data):
        """Test feature engineering metadata."""
        from src.features.config import MINIMAL_CONFIG
        from src.features.engineer import FeatureEngineer

        engineer = FeatureEngineer(config=MINIMAL_CONFIG, target_column="requests_per_second")
        features = engineer.compute_features(sample_metrics_data)

        metadata = engineer.get_metadata()

        assert "n_features" in metadata
        assert "feature_hash" in metadata
        assert "enabled_groups" in metadata

    def test_feature_names_property(self, sample_metrics_data):
        """Test feature names property."""
        from src.features.config import MINIMAL_CONFIG
        from src.features.engineer import FeatureEngineer

        engineer = FeatureEngineer(config=MINIMAL_CONFIG, target_column="requests_per_second")
        features = engineer.compute_features(sample_metrics_data)

        assert len(engineer.feature_names) == len(features.columns)

    def test_clear_cache(self, sample_metrics_data):
        """Test cache clearing."""
        from src.features.config import MINIMAL_CONFIG
        from src.features.engineer import FeatureEngineer

        engineer = FeatureEngineer(config=MINIMAL_CONFIG, target_column="requests_per_second")

        # Compute to populate cache
        engineer.compute_features(sample_metrics_data)

        # Clear cache
        engineer.clear_cache()

        assert len(engineer._cache) == 0


class TestFeatureFactory:
    """Test feature engineer factory."""

    def test_factory_import(self):
        """Test FeatureEngineerFactory can be imported."""
        from src.features.engineer import FeatureEngineerFactory

        assert FeatureEngineerFactory is not None

    def test_create_minimal(self):
        """Test creating minimal feature engineer."""
        from src.features.engineer import FeatureEngineerFactory

        engineer = FeatureEngineerFactory.create_minimal()
        assert engineer is not None

    def test_create_full(self):
        """Test creating full feature engineer."""
        from src.features.engineer import FeatureEngineerFactory

        engineer = FeatureEngineerFactory.create_full()
        assert engineer is not None

    def test_create_default(self):
        """Test creating default feature engineer."""
        from src.features.engineer import FeatureEngineerFactory

        engineer = FeatureEngineerFactory.create_default()
        assert engineer is not None

    def test_create_custom(self):
        """Test creating custom feature engineer."""
        from src.features.engineer import FeatureEngineerFactory

        engineer = FeatureEngineerFactory.create_custom(
            enable_groups=["time", "lag"],
            lag_windows=[5, 10],
        )
        assert engineer is not None
        assert engineer.config.lag_windows == [5, 10]
