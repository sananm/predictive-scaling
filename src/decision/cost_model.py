"""
Cost Model for cloud infrastructure pricing.

Responsibilities:
- Store pricing for different instance types (on-demand and spot)
- Calculate hourly/monthly costs for a given configuration
- Estimate costs for spot instance usage with interruption risk
- Support multiple cloud providers via strategy pattern
- Update pricing periodically from cloud APIs
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class PricingType(str, Enum):
    """Instance pricing types."""

    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"


@dataclass
class InstancePricing:
    """Pricing information for an instance type."""

    instance_type: str
    provider: CloudProvider
    region: str
    vcpu: int
    memory_gb: float
    on_demand_hourly: float
    spot_hourly: float | None = None
    reserved_hourly: float | None = None
    spot_interruption_rate: float = 0.05  # Historical interruption probability
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def spot_discount(self) -> float:
        """Calculate spot discount percentage."""
        if self.spot_hourly and self.on_demand_hourly > 0:
            return 1.0 - (self.spot_hourly / self.on_demand_hourly)
        return 0.0

    def hourly_cost(self, pricing_type: PricingType) -> float:
        """Get hourly cost for a pricing type."""
        if pricing_type == PricingType.ON_DEMAND:
            return self.on_demand_hourly
        elif pricing_type == PricingType.SPOT:
            return self.spot_hourly or self.on_demand_hourly
        elif pricing_type == PricingType.RESERVED:
            return self.reserved_hourly or self.on_demand_hourly
        return self.on_demand_hourly

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_type": self.instance_type,
            "provider": self.provider.value,
            "region": self.region,
            "vcpu": self.vcpu,
            "memory_gb": self.memory_gb,
            "on_demand_hourly": self.on_demand_hourly,
            "spot_hourly": self.spot_hourly,
            "reserved_hourly": self.reserved_hourly,
            "spot_interruption_rate": self.spot_interruption_rate,
            "spot_discount": self.spot_discount,
        }


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure."""

    instance_type: str
    instance_count: int
    spot_percentage: float = 0.0  # 0.0 to 1.0
    provider: CloudProvider = CloudProvider.AWS
    region: str = "us-east-1"

    @property
    def on_demand_count(self) -> int:
        """Number of on-demand instances."""
        return int(self.instance_count * (1 - self.spot_percentage))

    @property
    def spot_count(self) -> int:
        """Number of spot instances."""
        return self.instance_count - self.on_demand_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "spot_percentage": self.spot_percentage,
            "on_demand_count": self.on_demand_count,
            "spot_count": self.spot_count,
            "provider": self.provider.value,
            "region": self.region,
        }


@dataclass
class CostEstimate:
    """Cost estimate for infrastructure configuration."""

    config: InfrastructureConfig
    hourly_cost: float
    daily_cost: float
    monthly_cost: float
    on_demand_cost: float
    spot_cost: float
    spot_risk_adjusted_cost: float  # Cost accounting for interruption risk
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "hourly_cost": self.hourly_cost,
            "daily_cost": self.daily_cost,
            "monthly_cost": self.monthly_cost,
            "on_demand_cost": self.on_demand_cost,
            "spot_cost": self.spot_cost,
            "spot_risk_adjusted_cost": self.spot_risk_adjusted_cost,
            "currency": self.currency,
        }


class CloudPricingStrategy(ABC):
    """Abstract strategy for cloud pricing."""

    @abstractmethod
    def get_pricing(self, instance_type: str, region: str) -> InstancePricing | None:
        """Get pricing for an instance type."""
        pass

    @abstractmethod
    def list_instance_types(self, region: str) -> list[str]:
        """List available instance types."""
        pass

    @abstractmethod
    async def refresh_pricing(self) -> None:
        """Refresh pricing from cloud API."""
        pass


class AWSPricingStrategy(CloudPricingStrategy):
    """AWS pricing strategy with static pricing data."""

    # Default AWS pricing (us-east-1, approximate values)
    DEFAULT_PRICING: dict[str, dict[str, Any]] = {
        "t3.micro": {"vcpu": 2, "memory_gb": 1, "on_demand": 0.0104, "spot": 0.0031},
        "t3.small": {"vcpu": 2, "memory_gb": 2, "on_demand": 0.0208, "spot": 0.0062},
        "t3.medium": {"vcpu": 2, "memory_gb": 4, "on_demand": 0.0416, "spot": 0.0125},
        "t3.large": {"vcpu": 2, "memory_gb": 8, "on_demand": 0.0832, "spot": 0.0250},
        "t3.xlarge": {"vcpu": 4, "memory_gb": 16, "on_demand": 0.1664, "spot": 0.0499},
        "t3.2xlarge": {"vcpu": 8, "memory_gb": 32, "on_demand": 0.3328, "spot": 0.0998},
        "m5.large": {"vcpu": 2, "memory_gb": 8, "on_demand": 0.096, "spot": 0.0384},
        "m5.xlarge": {"vcpu": 4, "memory_gb": 16, "on_demand": 0.192, "spot": 0.0768},
        "m5.2xlarge": {"vcpu": 8, "memory_gb": 32, "on_demand": 0.384, "spot": 0.1536},
        "m5.4xlarge": {"vcpu": 16, "memory_gb": 64, "on_demand": 0.768, "spot": 0.3072},
        "c5.large": {"vcpu": 2, "memory_gb": 4, "on_demand": 0.085, "spot": 0.034},
        "c5.xlarge": {"vcpu": 4, "memory_gb": 8, "on_demand": 0.17, "spot": 0.068},
        "c5.2xlarge": {"vcpu": 8, "memory_gb": 16, "on_demand": 0.34, "spot": 0.136},
        "c5.4xlarge": {"vcpu": 16, "memory_gb": 32, "on_demand": 0.68, "spot": 0.272},
        "r5.large": {"vcpu": 2, "memory_gb": 16, "on_demand": 0.126, "spot": 0.0504},
        "r5.xlarge": {"vcpu": 4, "memory_gb": 32, "on_demand": 0.252, "spot": 0.1008},
        "r5.2xlarge": {"vcpu": 8, "memory_gb": 64, "on_demand": 0.504, "spot": 0.2016},
    }

    # Spot interruption rates by instance type family
    INTERRUPTION_RATES: dict[str, float] = {
        "t3": 0.02,  # Low interruption
        "m5": 0.05,  # Medium interruption
        "c5": 0.08,  # Higher interruption (compute optimized)
        "r5": 0.06,  # Medium-high interruption
    }

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize AWS pricing strategy."""
        self.region = region
        self._pricing_cache: dict[str, InstancePricing] = {}
        self._load_default_pricing()

    def _load_default_pricing(self) -> None:
        """Load default pricing data."""
        for instance_type, specs in self.DEFAULT_PRICING.items():
            family = instance_type.split(".")[0]
            interruption_rate = self.INTERRUPTION_RATES.get(family, 0.05)

            self._pricing_cache[instance_type] = InstancePricing(
                instance_type=instance_type,
                provider=CloudProvider.AWS,
                region=self.region,
                vcpu=specs["vcpu"],
                memory_gb=specs["memory_gb"],
                on_demand_hourly=specs["on_demand"],
                spot_hourly=specs["spot"],
                spot_interruption_rate=interruption_rate,
            )

    def get_pricing(self, instance_type: str, region: str) -> InstancePricing | None:
        """Get pricing for an instance type."""
        return self._pricing_cache.get(instance_type)

    def list_instance_types(self, region: str) -> list[str]:
        """List available instance types."""
        return list(self._pricing_cache.keys())

    async def refresh_pricing(self) -> None:
        """Refresh pricing from AWS API (placeholder for real implementation)."""
        # In production, this would call AWS Pricing API
        logger.info("AWS pricing refresh requested (using static data)")


class GCPPricingStrategy(CloudPricingStrategy):
    """GCP pricing strategy with static pricing data."""

    DEFAULT_PRICING: dict[str, dict[str, Any]] = {
        "e2-micro": {"vcpu": 0.25, "memory_gb": 1, "on_demand": 0.0084, "spot": 0.0025},
        "e2-small": {"vcpu": 0.5, "memory_gb": 2, "on_demand": 0.0168, "spot": 0.0050},
        "e2-medium": {"vcpu": 1, "memory_gb": 4, "on_demand": 0.0336, "spot": 0.0101},
        "e2-standard-2": {"vcpu": 2, "memory_gb": 8, "on_demand": 0.0671, "spot": 0.0201},
        "e2-standard-4": {"vcpu": 4, "memory_gb": 16, "on_demand": 0.1342, "spot": 0.0403},
        "e2-standard-8": {"vcpu": 8, "memory_gb": 32, "on_demand": 0.2684, "spot": 0.0805},
        "n1-standard-1": {"vcpu": 1, "memory_gb": 3.75, "on_demand": 0.0475, "spot": 0.0100},
        "n1-standard-2": {"vcpu": 2, "memory_gb": 7.5, "on_demand": 0.0950, "spot": 0.0200},
        "n1-standard-4": {"vcpu": 4, "memory_gb": 15, "on_demand": 0.1900, "spot": 0.0400},
        "n1-standard-8": {"vcpu": 8, "memory_gb": 30, "on_demand": 0.3800, "spot": 0.0800},
    }

    def __init__(self, region: str = "us-central1") -> None:
        """Initialize GCP pricing strategy."""
        self.region = region
        self._pricing_cache: dict[str, InstancePricing] = {}
        self._load_default_pricing()

    def _load_default_pricing(self) -> None:
        """Load default pricing data."""
        for instance_type, specs in self.DEFAULT_PRICING.items():
            self._pricing_cache[instance_type] = InstancePricing(
                instance_type=instance_type,
                provider=CloudProvider.GCP,
                region=self.region,
                vcpu=int(specs["vcpu"]) if specs["vcpu"] >= 1 else 1,
                memory_gb=specs["memory_gb"],
                on_demand_hourly=specs["on_demand"],
                spot_hourly=specs["spot"],
                spot_interruption_rate=0.05,  # GCP preemptible default
            )

    def get_pricing(self, instance_type: str, region: str) -> InstancePricing | None:
        """Get pricing for an instance type."""
        return self._pricing_cache.get(instance_type)

    def list_instance_types(self, region: str) -> list[str]:
        """List available instance types."""
        return list(self._pricing_cache.keys())

    async def refresh_pricing(self) -> None:
        """Refresh pricing from GCP API (placeholder)."""
        logger.info("GCP pricing refresh requested (using static data)")


class AzurePricingStrategy(CloudPricingStrategy):
    """Azure pricing strategy with static pricing data."""

    DEFAULT_PRICING: dict[str, dict[str, Any]] = {
        "Standard_B1s": {"vcpu": 1, "memory_gb": 1, "on_demand": 0.0104, "spot": 0.0031},
        "Standard_B1ms": {"vcpu": 1, "memory_gb": 2, "on_demand": 0.0207, "spot": 0.0062},
        "Standard_B2s": {"vcpu": 2, "memory_gb": 4, "on_demand": 0.0416, "spot": 0.0125},
        "Standard_B2ms": {"vcpu": 2, "memory_gb": 8, "on_demand": 0.0832, "spot": 0.0250},
        "Standard_D2s_v3": {"vcpu": 2, "memory_gb": 8, "on_demand": 0.096, "spot": 0.0192},
        "Standard_D4s_v3": {"vcpu": 4, "memory_gb": 16, "on_demand": 0.192, "spot": 0.0384},
        "Standard_D8s_v3": {"vcpu": 8, "memory_gb": 32, "on_demand": 0.384, "spot": 0.0768},
        "Standard_D16s_v3": {"vcpu": 16, "memory_gb": 64, "on_demand": 0.768, "spot": 0.1536},
    }

    def __init__(self, region: str = "eastus") -> None:
        """Initialize Azure pricing strategy."""
        self.region = region
        self._pricing_cache: dict[str, InstancePricing] = {}
        self._load_default_pricing()

    def _load_default_pricing(self) -> None:
        """Load default pricing data."""
        for instance_type, specs in self.DEFAULT_PRICING.items():
            self._pricing_cache[instance_type] = InstancePricing(
                instance_type=instance_type,
                provider=CloudProvider.AZURE,
                region=self.region,
                vcpu=specs["vcpu"],
                memory_gb=specs["memory_gb"],
                on_demand_hourly=specs["on_demand"],
                spot_hourly=specs["spot"],
                spot_interruption_rate=0.05,
            )

    def get_pricing(self, instance_type: str, region: str) -> InstancePricing | None:
        """Get pricing for an instance type."""
        return self._pricing_cache.get(instance_type)

    def list_instance_types(self, region: str) -> list[str]:
        """List available instance types."""
        return list(self._pricing_cache.keys())

    async def refresh_pricing(self) -> None:
        """Refresh pricing from Azure API (placeholder)."""
        logger.info("Azure pricing refresh requested (using static data)")


class CostModel:
    """
    Cloud cost model for infrastructure pricing.

    Supports multiple cloud providers and calculates costs
    for different instance configurations with spot pricing.
    """

    def __init__(
        self,
        provider: CloudProvider = CloudProvider.AWS,
        region: str = "us-east-1",
    ) -> None:
        """
        Initialize cost model.

        Args:
            provider: Cloud provider
            region: Cloud region
        """
        self.provider = provider
        self.region = region
        self._strategy = self._create_strategy(provider, region)

    def _create_strategy(
        self, provider: CloudProvider, region: str
    ) -> CloudPricingStrategy:
        """Create pricing strategy for provider."""
        if provider == CloudProvider.AWS:
            return AWSPricingStrategy(region)
        elif provider == CloudProvider.GCP:
            return GCPPricingStrategy(region)
        elif provider == CloudProvider.AZURE:
            return AzurePricingStrategy(region)
        else:
            return AWSPricingStrategy(region)

    def set_provider(self, provider: CloudProvider, region: str) -> None:
        """Change cloud provider."""
        self.provider = provider
        self.region = region
        self._strategy = self._create_strategy(provider, region)

    def get_pricing(self, instance_type: str) -> InstancePricing | None:
        """Get pricing for an instance type."""
        return self._strategy.get_pricing(instance_type, self.region)

    def list_instance_types(self) -> list[str]:
        """List available instance types."""
        return self._strategy.list_instance_types(self.region)

    def calculate_cost(self, config: InfrastructureConfig) -> CostEstimate:
        """
        Calculate cost for an infrastructure configuration.

        Args:
            config: Infrastructure configuration

        Returns:
            Cost estimate with hourly/daily/monthly costs
        """
        pricing = self.get_pricing(config.instance_type)

        if pricing is None:
            logger.warning(
                "Unknown instance type, using default pricing",
                instance_type=config.instance_type,
            )
            # Default to a reasonable fallback
            pricing = InstancePricing(
                instance_type=config.instance_type,
                provider=self.provider,
                region=self.region,
                vcpu=2,
                memory_gb=4,
                on_demand_hourly=0.10,
                spot_hourly=0.03,
            )

        # Calculate on-demand cost
        on_demand_cost = config.on_demand_count * pricing.on_demand_hourly

        # Calculate spot cost
        spot_hourly = pricing.spot_hourly or pricing.on_demand_hourly
        spot_cost = config.spot_count * spot_hourly

        # Total hourly cost
        hourly_cost = on_demand_cost + spot_cost

        # Calculate risk-adjusted spot cost
        # Account for potential fallback to on-demand during interruptions
        spot_risk_cost = spot_cost + (
            config.spot_count
            * pricing.spot_interruption_rate
            * (pricing.on_demand_hourly - spot_hourly)
        )

        return CostEstimate(
            config=config,
            hourly_cost=hourly_cost,
            daily_cost=hourly_cost * 24,
            monthly_cost=hourly_cost * 24 * 30,
            on_demand_cost=on_demand_cost,
            spot_cost=spot_cost,
            spot_risk_adjusted_cost=on_demand_cost + spot_risk_cost,
        )

    def compare_configs(
        self, configs: list[InfrastructureConfig]
    ) -> list[CostEstimate]:
        """
        Compare costs for multiple configurations.

        Args:
            configs: List of configurations to compare

        Returns:
            List of cost estimates, sorted by hourly cost
        """
        estimates = [self.calculate_cost(config) for config in configs]
        return sorted(estimates, key=lambda x: x.hourly_cost)

    def calculate_savings(
        self,
        current_config: InfrastructureConfig,
        proposed_config: InfrastructureConfig,
        hours: float = 1.0,
    ) -> dict[str, float]:
        """
        Calculate savings from changing configuration.

        Args:
            current_config: Current infrastructure
            proposed_config: Proposed infrastructure
            hours: Time period in hours

        Returns:
            Dictionary with savings information
        """
        current_cost = self.calculate_cost(current_config)
        proposed_cost = self.calculate_cost(proposed_config)

        hourly_savings = current_cost.hourly_cost - proposed_cost.hourly_cost
        total_savings = hourly_savings * hours
        savings_percentage = (
            (hourly_savings / current_cost.hourly_cost * 100)
            if current_cost.hourly_cost > 0
            else 0
        )

        return {
            "current_hourly_cost": current_cost.hourly_cost,
            "proposed_hourly_cost": proposed_cost.hourly_cost,
            "hourly_savings": hourly_savings,
            "total_savings": total_savings,
            "savings_percentage": savings_percentage,
            "hours": hours,
        }

    async def refresh_pricing(self) -> None:
        """Refresh pricing data from cloud API."""
        await self._strategy.refresh_pricing()
        logger.info(
            "Pricing refreshed",
            provider=self.provider.value,
            region=self.region,
        )
