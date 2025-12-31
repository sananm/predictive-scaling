"""
Configuration management using pydantic-settings.

Hierarchical configuration with environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/predictive_scaler",
        description="Database connection URL",
    )
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=3600, ge=60)
    echo: bool = Field(default=False, description="Echo SQL queries")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=10, ge=1)
    socket_timeout: float = Field(default=5.0, ge=0.1)
    retry_on_timeout: bool = Field(default=True)


class KafkaSettings(BaseSettings):
    """Kafka connection settings."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_")

    bootstrap_servers: str = Field(default="localhost:29092")
    metrics_topic: str = Field(default="metrics")
    features_topic: str = Field(default="features")
    predictions_topic: str = Field(default="predictions")
    consumer_group: str = Field(default="predictive-scaler")
    auto_offset_reset: str = Field(default="latest")
    enable_auto_commit: bool = Field(default=True)
    auto_commit_interval_ms: int = Field(default=5000)


class PrometheusSettings(BaseSettings):
    """Prometheus connection settings."""

    model_config = SettingsConfigDict(env_prefix="PROMETHEUS_")

    url: str = Field(default="http://localhost:9090")
    scrape_interval: int = Field(default=15, ge=1, description="Scrape interval in seconds")
    query_timeout: int = Field(default=30, ge=1)


class KubernetesSettings(BaseSettings):
    """Kubernetes connection settings."""

    model_config = SettingsConfigDict(env_prefix="KUBERNETES_")

    namespace: str = Field(default="default")
    deployment_name: str = Field(default="dummy-app")
    hpa_name: str | None = Field(default=None)
    in_cluster: bool = Field(default=False)
    config_path: str | None = Field(default=None)
    verify_ssl: bool = Field(default=True)


class AWSSettings(BaseSettings):
    """AWS connection settings (optional)."""

    model_config = SettingsConfigDict(env_prefix="AWS_")

    region: str = Field(default="us-east-1")
    access_key_id: str | None = Field(default=None)
    secret_access_key: str | None = Field(default=None)
    session_token: str | None = Field(default=None)


class ModelSettings(BaseSettings):
    """ML model settings."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    dir: Path = Field(default=Path("./models"))

    # Prediction horizons in minutes
    short_term_horizon: int = Field(default=15, ge=1, description="Short-term horizon (minutes)")
    medium_term_horizon: int = Field(default=1440, ge=1, description="Medium-term horizon (minutes)")
    long_term_horizon: int = Field(default=10080, ge=1, description="Long-term horizon (minutes)")

    # Context window for sequence models
    context_window: int = Field(default=60, ge=1, description="Context window size (minutes)")

    # Retraining
    retrain_interval: int = Field(default=86400, ge=3600, description="Retrain interval (seconds)")

    # Transformer model config
    transformer_d_model: int = Field(default=128)
    transformer_nhead: int = Field(default=8)
    transformer_num_layers: int = Field(default=4)
    transformer_dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    @field_validator("dir", mode="before")
    @classmethod
    def validate_dir(cls, v: str | Path) -> Path:
        return Path(v)


class ScalingSettings(BaseSettings):
    """Scaling decision settings."""

    model_config = SettingsConfigDict(env_prefix="SCALING_")

    min_instances: int = Field(default=1, ge=1)
    max_instances: int = Field(default=50, ge=1)
    target_utilization: float = Field(default=0.7, ge=0.1, le=1.0)
    headroom_factor: float = Field(default=1.2, ge=1.0, le=2.0)
    cooldown_seconds: int = Field(default=300, ge=0)
    rate_limit_up: int = Field(default=10, ge=1, description="Max instances to add per action")
    rate_limit_down: int = Field(default=5, ge=1, description="Max instances to remove per action")

    # Optimization weights
    weight_cost: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_performance: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_stability: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_risk: float = Field(default=0.1, ge=0.0, le=1.0)


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(default="json")

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    kubernetes: KubernetesSettings = Field(default_factory=KubernetesSettings)
    aws: AWSSettings = Field(default_factory=AWSSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    scaling: ScalingSettings = Field(default_factory=ScalingSettings)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"


@lru_cache
def get_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()


# Convenience function for accessing settings
settings = get_settings()
