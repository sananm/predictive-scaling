"""
Cost Tracker for infrastructure cost monitoring and savings calculation.

Responsibilities:
- Record actual infrastructure costs
- Simulate reactive scaling costs
- Calculate savings from predictive scaling
- Track costs over time
- Generate cost reports
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from collections import defaultdict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CostPeriod(str, Enum):
    """Time periods for cost tracking."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class InstanceCost:
    """Cost information for an instance type."""

    instance_type: str
    hourly_cost: float
    vcpus: int
    memory_gb: float
    capacity_rps: float  # Requests per second capacity
    provider: str = "aws"
    is_spot: bool = False
    spot_discount: float = 0.7  # Spot is typically 70% cheaper


@dataclass
class CostRecord:
    """Record of infrastructure cost at a point in time."""

    timestamp: datetime
    service_name: str
    instance_type: str
    instance_count: int
    hourly_cost: float
    is_spot: bool = False
    spot_count: int = 0
    utilization: float = 0.0
    scaling_type: str = "predictive"  # "predictive" or "reactive"

    @property
    def total_hourly_cost(self) -> float:
        """Calculate total hourly cost."""
        return self.hourly_cost * self.instance_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "service_name": self.service_name,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "hourly_cost": self.hourly_cost,
            "total_hourly_cost": self.total_hourly_cost,
            "is_spot": self.is_spot,
            "spot_count": self.spot_count,
            "utilization": self.utilization,
            "scaling_type": self.scaling_type,
        }


@dataclass
class SavingsRecord:
    """Record of cost savings from predictive scaling."""

    timestamp: datetime
    service_name: str
    period: CostPeriod
    actual_cost: float
    reactive_cost: float
    savings: float
    savings_percent: float
    over_provision_cost: float  # Cost of unused capacity
    sla_violation_cost: float  # Cost of under-provisioning

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "service_name": self.service_name,
            "period": self.period.value,
            "actual_cost": self.actual_cost,
            "reactive_cost": self.reactive_cost,
            "savings": self.savings,
            "savings_percent": self.savings_percent,
            "over_provision_cost": self.over_provision_cost,
            "sla_violation_cost": self.sla_violation_cost,
        }


@dataclass
class CostSummary:
    """Summary of costs for a period."""

    service_name: str
    period_start: datetime
    period_end: datetime

    # Actual costs
    total_cost: float
    avg_hourly_cost: float
    min_hourly_cost: float
    max_hourly_cost: float

    # Instance utilization
    avg_instances: float
    min_instances: int
    max_instances: int
    avg_utilization: float

    # Savings
    total_reactive_cost: float
    total_savings: float
    savings_percent: float

    # Breakdown
    on_demand_cost: float
    spot_cost: float
    over_provision_cost: float
    sla_violation_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost": self.total_cost,
            "avg_hourly_cost": self.avg_hourly_cost,
            "min_hourly_cost": self.min_hourly_cost,
            "max_hourly_cost": self.max_hourly_cost,
            "avg_instances": self.avg_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "avg_utilization": self.avg_utilization,
            "total_reactive_cost": self.total_reactive_cost,
            "total_savings": self.total_savings,
            "savings_percent": self.savings_percent,
            "on_demand_cost": self.on_demand_cost,
            "spot_cost": self.spot_cost,
            "over_provision_cost": self.over_provision_cost,
            "sla_violation_cost": self.sla_violation_cost,
        }


class CostTracker:
    """
    Tracks infrastructure costs and calculates savings.

    Records actual costs, simulates what reactive scaling would have cost,
    and calculates the savings from predictive scaling.
    """

    def __init__(
        self,
        retention_days: int = 90,
        default_instance_type: str = "m5.large",
        target_utilization: float = 0.7,
        sla_violation_cost_multiplier: float = 10.0,
        reactive_scale_delay_minutes: int = 5,
    ) -> None:
        """
        Initialize cost tracker.

        Args:
            retention_days: How long to keep cost records
            default_instance_type: Default instance type for cost calculations
            target_utilization: Target utilization for optimal scaling
            sla_violation_cost_multiplier: Multiplier for SLA violation cost
            reactive_scale_delay_minutes: Assumed delay for reactive scaling
        """
        self._retention_days = retention_days
        self._default_instance_type = default_instance_type
        self._target_utilization = target_utilization
        self._sla_violation_multiplier = sla_violation_cost_multiplier
        self._reactive_delay = reactive_scale_delay_minutes

        # Instance cost catalog
        self._instance_costs: dict[str, InstanceCost] = {}
        self._init_default_instance_costs()

        # Cost records by service
        self._cost_records: dict[str, list[CostRecord]] = defaultdict(list)

        # Savings records by service
        self._savings_records: dict[str, list[SavingsRecord]] = defaultdict(list)

        # Reactive simulation records (for comparison)
        self._reactive_costs: dict[str, list[CostRecord]] = defaultdict(list)

        logger.info(
            "Cost tracker initialized",
            retention_days=retention_days,
            default_instance=default_instance_type,
        )

    def _init_default_instance_costs(self) -> None:
        """Initialize default instance cost catalog."""
        # AWS instance costs (approximate)
        default_instances = [
            InstanceCost("t3.medium", 0.0416, 2, 4, 500, "aws"),
            InstanceCost("t3.large", 0.0832, 2, 8, 800, "aws"),
            InstanceCost("m5.large", 0.096, 2, 8, 1000, "aws"),
            InstanceCost("m5.xlarge", 0.192, 4, 16, 2000, "aws"),
            InstanceCost("m5.2xlarge", 0.384, 8, 32, 4000, "aws"),
            InstanceCost("c5.large", 0.085, 2, 4, 1200, "aws"),
            InstanceCost("c5.xlarge", 0.17, 4, 8, 2400, "aws"),
            InstanceCost("c5.2xlarge", 0.34, 8, 16, 4800, "aws"),
            InstanceCost("r5.large", 0.126, 2, 16, 800, "aws"),
            InstanceCost("r5.xlarge", 0.252, 4, 32, 1600, "aws"),
        ]

        for instance in default_instances:
            self._instance_costs[instance.instance_type] = instance

    def add_instance_type(self, instance: InstanceCost) -> None:
        """Add or update an instance type in the catalog."""
        self._instance_costs[instance.instance_type] = instance

    def get_instance_cost(self, instance_type: str) -> InstanceCost | None:
        """Get cost info for an instance type."""
        return self._instance_costs.get(instance_type)

    def record_cost(
        self,
        service_name: str,
        instance_type: str,
        instance_count: int,
        utilization: float = 0.0,
        spot_count: int = 0,
        timestamp: datetime | None = None,
    ) -> CostRecord:
        """
        Record infrastructure cost.

        Args:
            service_name: Service name
            instance_type: Instance type
            instance_count: Total instance count
            utilization: Current utilization (0-1)
            spot_count: Number of spot instances
            timestamp: Record timestamp (defaults to now)

        Returns:
            The created CostRecord
        """
        ts = timestamp or datetime.now(timezone.utc)
        instance_cost = self._instance_costs.get(instance_type)

        if instance_cost:
            hourly_cost = instance_cost.hourly_cost
            # Calculate effective cost with spot discount
            on_demand_count = instance_count - spot_count
            spot_discount = instance_cost.spot_discount
            effective_hourly = (
                on_demand_count * hourly_cost
                + spot_count * hourly_cost * (1 - spot_discount)
            ) / instance_count if instance_count > 0 else hourly_cost
        else:
            # Use default if instance type not found
            logger.warning(f"Unknown instance type: {instance_type}, using default cost")
            effective_hourly = 0.10  # Default fallback

        record = CostRecord(
            timestamp=ts,
            service_name=service_name,
            instance_type=instance_type,
            instance_count=instance_count,
            hourly_cost=effective_hourly,
            is_spot=spot_count > 0,
            spot_count=spot_count,
            utilization=utilization,
            scaling_type="predictive",
        )

        self._cost_records[service_name].append(record)

        # Also simulate reactive cost
        self._simulate_reactive_cost(record)

        # Cleanup old records
        self._cleanup_old_records(service_name)

        logger.debug(
            "Cost recorded",
            service=service_name,
            instances=instance_count,
            hourly_cost=f"${record.total_hourly_cost:.2f}",
        )

        return record

    def _simulate_reactive_cost(self, actual_record: CostRecord) -> None:
        """Simulate what reactive scaling would have cost."""
        # Reactive scaling is delayed and often over-provisions
        # It scales based on current load, not predicted load
        service = actual_record.service_name

        # Get recent records to simulate reactive behavior
        recent_records = self._cost_records[service][-60:]  # Last hour of records

        if len(recent_records) < 2:
            # Not enough data, assume same as actual
            reactive_count = actual_record.instance_count
        else:
            # Reactive would see load from X minutes ago
            delay_records = int(self._reactive_delay)
            if len(recent_records) > delay_records:
                delayed_record = recent_records[-delay_records - 1]
                # Reactive would scale to handle the load seen then
                # Usually with some headroom
                reactive_count = max(
                    delayed_record.instance_count,
                    int(actual_record.instance_count * 1.2),  # 20% headroom
                )
            else:
                reactive_count = actual_record.instance_count

        reactive_record = CostRecord(
            timestamp=actual_record.timestamp,
            service_name=service,
            instance_type=actual_record.instance_type,
            instance_count=reactive_count,
            hourly_cost=actual_record.hourly_cost,
            is_spot=False,  # Reactive typically doesn't use spot
            spot_count=0,
            utilization=actual_record.utilization,
            scaling_type="reactive",
        )

        self._reactive_costs[service].append(reactive_record)

    def calculate_savings(
        self,
        service_name: str,
        period: CostPeriod = CostPeriod.DAILY,
    ) -> SavingsRecord | None:
        """
        Calculate savings from predictive scaling.

        Args:
            service_name: Service name
            period: Time period for calculation

        Returns:
            SavingsRecord or None if insufficient data
        """
        now = datetime.now(timezone.utc)

        # Determine period duration
        if period == CostPeriod.HOURLY:
            period_start = now - timedelta(hours=1)
        elif period == CostPeriod.DAILY:
            period_start = now - timedelta(days=1)
        elif period == CostPeriod.WEEKLY:
            period_start = now - timedelta(weeks=1)
        else:  # MONTHLY
            period_start = now - timedelta(days=30)

        # Get records for period
        actual_records = [
            r
            for r in self._cost_records.get(service_name, [])
            if r.timestamp >= period_start
        ]
        reactive_records = [
            r
            for r in self._reactive_costs.get(service_name, [])
            if r.timestamp >= period_start
        ]

        if not actual_records:
            return None

        # Calculate costs
        # Approximate hourly cost by averaging and multiplying by hours
        hours = (now - period_start).total_seconds() / 3600

        actual_total = sum(r.total_hourly_cost for r in actual_records)
        actual_avg = actual_total / len(actual_records)
        actual_cost = actual_avg * hours

        if reactive_records:
            reactive_total = sum(r.total_hourly_cost for r in reactive_records)
            reactive_avg = reactive_total / len(reactive_records)
            reactive_cost = reactive_avg * hours
        else:
            reactive_cost = actual_cost * 1.15  # Assume 15% higher without data

        # Calculate over-provisioning cost
        over_provision_cost = 0.0
        for r in actual_records:
            if r.utilization < self._target_utilization:
                wasted_capacity = 1 - (r.utilization / self._target_utilization)
                over_provision_cost += r.total_hourly_cost * wasted_capacity

        # SLA violation cost (simplified - based on under-provisioning events)
        sla_violation_cost = 0.0
        for r in actual_records:
            if r.utilization > 0.9:  # Over 90% utilization indicates potential SLA risk
                # Penalize high utilization periods
                risk_factor = (r.utilization - 0.9) / 0.1
                sla_violation_cost += (
                    r.total_hourly_cost * risk_factor * self._sla_violation_multiplier
                )

        savings = reactive_cost - actual_cost
        savings_percent = (savings / reactive_cost * 100) if reactive_cost > 0 else 0

        record = SavingsRecord(
            timestamp=now,
            service_name=service_name,
            period=period,
            actual_cost=actual_cost,
            reactive_cost=reactive_cost,
            savings=max(0, savings),  # Can't have negative savings
            savings_percent=max(0, savings_percent),
            over_provision_cost=over_provision_cost / len(actual_records) * hours,
            sla_violation_cost=sla_violation_cost / len(actual_records) * hours,
        )

        self._savings_records[service_name].append(record)

        logger.info(
            "Savings calculated",
            service=service_name,
            period=period.value,
            actual=f"${actual_cost:.2f}",
            reactive=f"${reactive_cost:.2f}",
            savings=f"${savings:.2f} ({savings_percent:.1f}%)",
        )

        return record

    def get_cost_summary(
        self,
        service_name: str,
        period_hours: int = 24,
    ) -> CostSummary | None:
        """
        Get cost summary for a service.

        Args:
            service_name: Service name
            period_hours: Period to analyze

        Returns:
            CostSummary or None if no data
        """
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(hours=period_hours)

        records = [
            r
            for r in self._cost_records.get(service_name, [])
            if r.timestamp >= period_start
        ]

        reactive_records = [
            r
            for r in self._reactive_costs.get(service_name, [])
            if r.timestamp >= period_start
        ]

        if not records:
            return None

        # Calculate metrics
        hourly_costs = [r.total_hourly_cost for r in records]
        instance_counts = [r.instance_count for r in records]
        utilizations = [r.utilization for r in records if r.utilization > 0]

        total_cost = sum(hourly_costs) / len(records) * period_hours
        reactive_total = (
            sum(r.total_hourly_cost for r in reactive_records) / len(reactive_records) * period_hours
            if reactive_records
            else total_cost * 1.15
        )

        # On-demand vs spot breakdown
        on_demand_records = [r for r in records if not r.is_spot]
        spot_records = [r for r in records if r.is_spot]

        on_demand_cost = (
            sum(r.total_hourly_cost for r in on_demand_records) / len(records) * period_hours
            if on_demand_records
            else 0
        )
        spot_cost = (
            sum(r.total_hourly_cost for r in spot_records) / len(records) * period_hours
            if spot_records
            else 0
        )

        # Over-provision and SLA violation costs
        over_provision_cost = 0.0
        sla_violation_cost = 0.0
        for r in records:
            if r.utilization < self._target_utilization and r.utilization > 0:
                wasted = 1 - (r.utilization / self._target_utilization)
                over_provision_cost += r.total_hourly_cost * wasted
            if r.utilization > 0.9:
                risk = (r.utilization - 0.9) / 0.1
                sla_violation_cost += r.total_hourly_cost * risk * self._sla_violation_multiplier

        over_provision_cost = over_provision_cost / len(records) * period_hours
        sla_violation_cost = sla_violation_cost / len(records) * period_hours

        savings = reactive_total - total_cost
        savings_percent = (savings / reactive_total * 100) if reactive_total > 0 else 0

        return CostSummary(
            service_name=service_name,
            period_start=period_start,
            period_end=now,
            total_cost=total_cost,
            avg_hourly_cost=sum(hourly_costs) / len(records),
            min_hourly_cost=min(hourly_costs),
            max_hourly_cost=max(hourly_costs),
            avg_instances=sum(instance_counts) / len(records),
            min_instances=min(instance_counts),
            max_instances=max(instance_counts),
            avg_utilization=sum(utilizations) / len(utilizations) if utilizations else 0,
            total_reactive_cost=reactive_total,
            total_savings=max(0, savings),
            savings_percent=max(0, savings_percent),
            on_demand_cost=on_demand_cost,
            spot_cost=spot_cost,
            over_provision_cost=over_provision_cost,
            sla_violation_cost=sla_violation_cost,
        )

    def _cleanup_old_records(self, service_name: str) -> None:
        """Remove records older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)

        self._cost_records[service_name] = [
            r for r in self._cost_records[service_name] if r.timestamp >= cutoff
        ]
        self._reactive_costs[service_name] = [
            r for r in self._reactive_costs[service_name] if r.timestamp >= cutoff
        ]
        self._savings_records[service_name] = [
            r for r in self._savings_records[service_name] if r.timestamp >= cutoff
        ]

    def get_recent_costs(
        self,
        service_name: str,
        hours: int = 24,
    ) -> list[CostRecord]:
        """Get recent cost records."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            r
            for r in self._cost_records.get(service_name, [])
            if r.timestamp >= cutoff
        ]

    def get_recent_savings(
        self,
        service_name: str,
        count: int = 10,
    ) -> list[SavingsRecord]:
        """Get recent savings records."""
        return list(reversed(self._savings_records.get(service_name, [])[-count:]))

    def generate_report(
        self,
        period_hours: int = 24,
        service_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate cost report.

        Args:
            period_hours: Period to analyze
            service_name: Optional service filter

        Returns:
            Report dictionary
        """
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_hours": period_hours,
            "services": [],
            "totals": {
                "total_cost": 0.0,
                "total_reactive_cost": 0.0,
                "total_savings": 0.0,
                "avg_savings_percent": 0.0,
            },
        }

        services = (
            [service_name]
            if service_name
            else list(self._cost_records.keys())
        )

        summaries = []
        for svc in services:
            summary = self.get_cost_summary(svc, period_hours)
            if summary:
                summaries.append(summary)
                report["services"].append(summary.to_dict())

        if summaries:
            report["totals"]["total_cost"] = sum(s.total_cost for s in summaries)
            report["totals"]["total_reactive_cost"] = sum(
                s.total_reactive_cost for s in summaries
            )
            report["totals"]["total_savings"] = sum(s.total_savings for s in summaries)
            report["totals"]["avg_savings_percent"] = (
                sum(s.savings_percent for s in summaries) / len(summaries)
            )

        return report

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        total_records = sum(len(r) for r in self._cost_records.values())
        total_savings_records = sum(len(r) for r in self._savings_records.values())

        return {
            "tracked_services": len(self._cost_records),
            "total_cost_records": total_records,
            "total_savings_records": total_savings_records,
            "instance_types_cataloged": len(self._instance_costs),
            "retention_days": self._retention_days,
        }


# Global instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def init_cost_tracker(**kwargs) -> CostTracker:
    """Initialize the global cost tracker."""
    global _cost_tracker
    _cost_tracker = CostTracker(**kwargs)
    return _cost_tracker
