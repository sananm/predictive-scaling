"""
Configuration API routes.

Provides endpoints for viewing and updating runtime configuration.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.settings import get_settings

router = APIRouter(prefix="/config", tags=["Configuration"])


class ScalingConfigResponse(BaseModel):
    """Response model for scaling configuration."""

    min_instances: int
    max_instances: int
    target_utilization: float
    headroom_factor: float
    cooldown_seconds: int
    rate_limit_up: int
    rate_limit_down: int
    weight_cost: float
    weight_performance: float
    weight_stability: float
    weight_risk: float


class ScalingConfigUpdate(BaseModel):
    """Request model for updating scaling configuration."""

    min_instances: int | None = Field(default=None, ge=1)
    max_instances: int | None = Field(default=None, ge=1)
    target_utilization: float | None = Field(default=None, ge=0.1, le=1.0)
    headroom_factor: float | None = Field(default=None, ge=1.0, le=2.0)
    cooldown_seconds: int | None = Field(default=None, ge=0)
    rate_limit_up: int | None = Field(default=None, ge=1)
    rate_limit_down: int | None = Field(default=None, ge=1)
    weight_cost: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_performance: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_risk: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelConfigResponse(BaseModel):
    """Response model for model configuration."""

    model_dir: str
    short_term_horizon: int
    medium_term_horizon: int
    long_term_horizon: int
    context_window: int
    retrain_interval: int
    transformer_d_model: int
    transformer_nhead: int
    transformer_num_layers: int
    transformer_dropout: float


class ModelConfigUpdate(BaseModel):
    """Request model for updating model configuration."""

    short_term_horizon: int | None = Field(default=None, ge=1)
    medium_term_horizon: int | None = Field(default=None, ge=1)
    long_term_horizon: int | None = Field(default=None, ge=1)
    context_window: int | None = Field(default=None, ge=1)
    retrain_interval: int | None = Field(default=None, ge=3600)


class FullConfigResponse(BaseModel):
    """Response model for full configuration."""

    env: str
    debug: bool
    log_level: str
    scaling: ScalingConfigResponse
    model: ModelConfigResponse
    kafka_bootstrap_servers: str
    prometheus_url: str
    kubernetes_namespace: str
    kubernetes_deployment: str


class ConfigUpdateResponse(BaseModel):
    """Response for configuration updates."""

    message: str
    updated_fields: list[str]


# Runtime config overrides (in-memory, not persisted)
_runtime_scaling_overrides: dict[str, Any] = {}
_runtime_model_overrides: dict[str, Any] = {}


def _get_effective_scaling_config() -> ScalingConfigResponse:
    """Get scaling config with runtime overrides applied."""
    settings = get_settings()
    return ScalingConfigResponse(
        min_instances=_runtime_scaling_overrides.get(
            "min_instances", settings.scaling.min_instances
        ),
        max_instances=_runtime_scaling_overrides.get(
            "max_instances", settings.scaling.max_instances
        ),
        target_utilization=_runtime_scaling_overrides.get(
            "target_utilization", settings.scaling.target_utilization
        ),
        headroom_factor=_runtime_scaling_overrides.get(
            "headroom_factor", settings.scaling.headroom_factor
        ),
        cooldown_seconds=_runtime_scaling_overrides.get(
            "cooldown_seconds", settings.scaling.cooldown_seconds
        ),
        rate_limit_up=_runtime_scaling_overrides.get(
            "rate_limit_up", settings.scaling.rate_limit_up
        ),
        rate_limit_down=_runtime_scaling_overrides.get(
            "rate_limit_down", settings.scaling.rate_limit_down
        ),
        weight_cost=_runtime_scaling_overrides.get(
            "weight_cost", settings.scaling.weight_cost
        ),
        weight_performance=_runtime_scaling_overrides.get(
            "weight_performance", settings.scaling.weight_performance
        ),
        weight_stability=_runtime_scaling_overrides.get(
            "weight_stability", settings.scaling.weight_stability
        ),
        weight_risk=_runtime_scaling_overrides.get(
            "weight_risk", settings.scaling.weight_risk
        ),
    )


def _get_effective_model_config() -> ModelConfigResponse:
    """Get model config with runtime overrides applied."""
    settings = get_settings()
    return ModelConfigResponse(
        model_dir=str(settings.model.dir),
        short_term_horizon=_runtime_model_overrides.get(
            "short_term_horizon", settings.model.short_term_horizon
        ),
        medium_term_horizon=_runtime_model_overrides.get(
            "medium_term_horizon", settings.model.medium_term_horizon
        ),
        long_term_horizon=_runtime_model_overrides.get(
            "long_term_horizon", settings.model.long_term_horizon
        ),
        context_window=_runtime_model_overrides.get(
            "context_window", settings.model.context_window
        ),
        retrain_interval=_runtime_model_overrides.get(
            "retrain_interval", settings.model.retrain_interval
        ),
        transformer_d_model=settings.model.transformer_d_model,
        transformer_nhead=settings.model.transformer_nhead,
        transformer_num_layers=settings.model.transformer_num_layers,
        transformer_dropout=settings.model.transformer_dropout,
    )


@router.get("", response_model=FullConfigResponse)
async def get_config() -> FullConfigResponse:
    """Get current configuration."""
    settings = get_settings()

    return FullConfigResponse(
        env=settings.env,
        debug=settings.debug,
        log_level=settings.log_level,
        scaling=_get_effective_scaling_config(),
        model=_get_effective_model_config(),
        kafka_bootstrap_servers=settings.kafka.bootstrap_servers,
        prometheus_url=settings.prometheus.url,
        kubernetes_namespace=settings.kubernetes.namespace,
        kubernetes_deployment=settings.kubernetes.deployment_name,
    )


@router.get("/scaling", response_model=ScalingConfigResponse)
async def get_scaling_config() -> ScalingConfigResponse:
    """Get current scaling configuration."""
    return _get_effective_scaling_config()


@router.put("/scaling", response_model=ConfigUpdateResponse)
async def update_scaling_config(update: ScalingConfigUpdate) -> ConfigUpdateResponse:
    """
    Update scaling configuration at runtime.

    Note: These changes are stored in memory and will be lost on restart.
    For persistent changes, update environment variables or .env file.
    """
    updated_fields: list[str] = []

    # Validate min/max relationship if both provided
    new_min = update.min_instances
    new_max = update.max_instances
    current = _get_effective_scaling_config()

    if new_min is not None and new_max is not None:
        if new_min > new_max:
            raise HTTPException(
                status_code=400,
                detail="min_instances cannot be greater than max_instances",
            )
    elif new_min is not None and new_min > current.max_instances:
        raise HTTPException(
            status_code=400,
            detail=f"min_instances ({new_min}) cannot be greater than current max_instances ({current.max_instances})",
        )
    elif new_max is not None and new_max < current.min_instances:
        raise HTTPException(
            status_code=400,
            detail=f"max_instances ({new_max}) cannot be less than current min_instances ({current.min_instances})",
        )

    # Apply updates
    update_dict = update.model_dump(exclude_none=True)
    for field, value in update_dict.items():
        _runtime_scaling_overrides[field] = value
        updated_fields.append(field)

    return ConfigUpdateResponse(
        message="Scaling configuration updated successfully",
        updated_fields=updated_fields,
    )


@router.get("/models", response_model=ModelConfigResponse)
async def get_model_config() -> ModelConfigResponse:
    """Get current model configuration."""
    return _get_effective_model_config()


@router.put("/models", response_model=ConfigUpdateResponse)
async def update_model_config(update: ModelConfigUpdate) -> ConfigUpdateResponse:
    """
    Update model configuration at runtime.

    Note: These changes are stored in memory and will be lost on restart.
    Changes to horizons will take effect on the next prediction cycle.
    """
    updated_fields: list[str] = []

    update_dict = update.model_dump(exclude_none=True)
    for field, value in update_dict.items():
        _runtime_model_overrides[field] = value
        updated_fields.append(field)

    return ConfigUpdateResponse(
        message="Model configuration updated successfully",
        updated_fields=updated_fields,
    )


@router.post("/reset", response_model=ConfigUpdateResponse)
async def reset_config() -> ConfigUpdateResponse:
    """
    Reset all runtime configuration overrides.

    This restores all settings to their original values from
    environment variables and defaults.
    """
    global _runtime_scaling_overrides, _runtime_model_overrides

    all_fields = list(_runtime_scaling_overrides.keys()) + list(
        _runtime_model_overrides.keys()
    )

    _runtime_scaling_overrides = {}
    _runtime_model_overrides = {}

    return ConfigUpdateResponse(
        message="Configuration reset to defaults",
        updated_fields=all_fields,
    )
