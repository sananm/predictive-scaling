"""
Health Check System for the predictive scaling system.

Responsibilities:
- Monitor component health status
- Provide liveness and readiness probes
- Track dependency availability
- Report overall system health
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components."""

    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    MODEL = "model"
    COLLECTOR = "collector"
    EXECUTOR = "executor"
    INTERNAL = "internal"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    component_name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "error": self.error,
        }


@dataclass
class HealthCheck:
    """Definition of a health check."""

    name: str
    component_type: ComponentType
    check_fn: Callable[[], bool | dict[str, Any]]
    timeout_seconds: float = 5.0
    interval_seconds: float = 30.0
    critical: bool = True  # If False, failure doesn't affect overall health
    enabled: bool = True
    description: str = ""

    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = datetime.now(UTC)

        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(self.check_fn):
                result = await asyncio.wait_for(
                    self.check_fn(),
                    timeout=self.timeout_seconds,
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self.check_fn),
                    timeout=self.timeout_seconds,
                )

            latency_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000

            # Handle result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                details = {}
            elif isinstance(result, dict):
                status = result.get("status", HealthStatus.HEALTHY)
                if isinstance(status, str):
                    status = HealthStatus(status)
                message = result.get("message", "OK")
                details = result.get("details", {})
            else:
                status = HealthStatus.HEALTHY
                message = "OK"
                details = {}

            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                latency_ms=latency_ms,
                details=details,
            )

        except TimeoutError:
            latency_ms = self.timeout_seconds * 1000
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                latency_ms=latency_ms,
                error=f"Timeout after {self.timeout_seconds}s",
            )

        except Exception as e:
            latency_ms = (
                datetime.now(UTC) - start_time
            ).total_seconds() * 1000
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message="Health check error",
                latency_ms=latency_ms,
                error=str(e),
            )


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: list[HealthCheckResult]
    checked_at: datetime
    version: str = "unknown"
    uptime_seconds: float = 0

    @property
    def healthy_count(self) -> int:
        """Count of healthy components."""
        return sum(1 for c in self.components if c.is_healthy)

    @property
    def unhealthy_count(self) -> int:
        """Count of unhealthy components."""
        return sum(1 for c in self.components if not c.is_healthy)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "total_components": len(self.components),
            "components": [c.to_dict() for c in self.components],
            "checked_at": self.checked_at.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
        }


class HealthChecker:
    """
    System health checker.

    Manages health checks for all system components and
    provides liveness/readiness probe endpoints.
    """

    def __init__(
        self,
        version: str = "unknown",
    ) -> None:
        """
        Initialize health checker.

        Args:
            version: Application version string
        """
        self._version = version
        self._checks: dict[str, HealthCheck] = {}
        self._results: dict[str, HealthCheckResult] = {}
        self._started_at = datetime.now(UTC)
        self._running = False
        self._callbacks: list[Callable] = []

    @property
    def uptime_seconds(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now(UTC) - self._started_at).total_seconds()

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self._checks[check.name] = check
        logger.debug(
            "Health check added",
            name=check.name,
            type=check.component_type.value,
            critical=check.critical,
        )

    def remove_check(self, name: str) -> bool:
        """Remove a health check."""
        if name in self._checks:
            del self._checks[name]
            self._results.pop(name, None)
            return True
        return False

    def add_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """Add callback for health status changes."""
        self._callbacks.append(callback)

    async def check_component(self, name: str) -> HealthCheckResult | None:
        """Run a specific health check."""
        if name not in self._checks:
            return None

        check = self._checks[name]
        if not check.enabled:
            return HealthCheckResult(
                component_name=name,
                component_type=check.component_type,
                status=HealthStatus.UNKNOWN,
                message="Check disabled",
            )

        result = await check.execute()
        old_result = self._results.get(name)
        self._results[name] = result

        # Notify on status change
        if old_result and old_result.status != result.status:
            logger.info(
                "Health status changed",
                component=name,
                old_status=old_result.status.value,
                new_status=result.status.value,
            )
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    logger.error("Health callback error", error=str(e))

        return result

    async def check_all(self) -> SystemHealth:
        """Run all health checks."""
        results = []

        for name, check in self._checks.items():
            if check.enabled:
                result = await self.check_component(name)
                if result:
                    results.append(result)

        # Determine overall status
        critical_unhealthy = any(
            not r.is_healthy
            for r in results
            if self._checks[r.component_name].critical
        )

        any_unhealthy = any(not r.is_healthy for r in results)

        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_unhealthy:
            overall_status = HealthStatus.DEGRADED
        elif not results:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall_status,
            components=results,
            checked_at=datetime.now(UTC),
            version=self._version,
            uptime_seconds=self.uptime_seconds,
        )

    async def is_live(self) -> bool:
        """
        Liveness probe - is the application running?

        Returns True if the application process is alive.
        This should be a simple check that doesn't depend on external services.
        """
        return True

    async def is_ready(self) -> bool:
        """
        Readiness probe - can the application handle requests?

        Returns True if all critical components are healthy.
        """
        health = await self.check_all()
        return health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def get_cached_result(self, name: str) -> HealthCheckResult | None:
        """Get cached health check result."""
        return self._results.get(name)

    def get_cached_health(self) -> SystemHealth | None:
        """Get cached overall health from last check."""
        if not self._results:
            return None

        results = list(self._results.values())

        critical_unhealthy = any(
            not r.is_healthy
            for r in results
            if r.component_name in self._checks
            and self._checks[r.component_name].critical
        )

        any_unhealthy = any(not r.is_healthy for r in results)

        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_unhealthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall_status,
            components=results,
            checked_at=max(r.checked_at for r in results),
            version=self._version,
            uptime_seconds=self.uptime_seconds,
        )

    async def start_background_checks(self) -> None:
        """Start background health check loop."""
        if self._running:
            return

        self._running = True
        logger.info("Health check background loop started")

        while self._running:
            try:
                # Find the minimum interval
                min_interval = min(
                    (c.interval_seconds for c in self._checks.values() if c.enabled),
                    default=30.0,
                )

                # Check components that are due
                for name, check in self._checks.items():
                    if not check.enabled:
                        continue

                    last_result = self._results.get(name)
                    if last_result:
                        elapsed = (
                            datetime.now(UTC) - last_result.checked_at
                        ).total_seconds()
                        if elapsed < check.interval_seconds:
                            continue

                    await self.check_component(name)

                await asyncio.sleep(min_interval)

            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(5)

    async def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False
        logger.info("Health check background loop stopped")

    # ==========================================================================
    # Common health check factories
    # ==========================================================================

    @staticmethod
    def create_database_check(
        name: str,
        connection_fn: Callable[[], bool],
        timeout_seconds: float = 5.0,
    ) -> HealthCheck:
        """Create a database health check."""
        return HealthCheck(
            name=name,
            component_type=ComponentType.DATABASE,
            check_fn=connection_fn,
            timeout_seconds=timeout_seconds,
            critical=True,
            description="Database connectivity check",
        )

    @staticmethod
    def create_cache_check(
        name: str,
        ping_fn: Callable[[], bool],
        timeout_seconds: float = 2.0,
    ) -> HealthCheck:
        """Create a cache health check."""
        return HealthCheck(
            name=name,
            component_type=ComponentType.CACHE,
            check_fn=ping_fn,
            timeout_seconds=timeout_seconds,
            critical=False,  # Cache failures often aren't critical
            description="Cache connectivity check",
        )

    @staticmethod
    def create_kafka_check(
        name: str,
        check_fn: Callable[[], bool],
        timeout_seconds: float = 5.0,
    ) -> HealthCheck:
        """Create a Kafka health check."""
        return HealthCheck(
            name=name,
            component_type=ComponentType.MESSAGE_QUEUE,
            check_fn=check_fn,
            timeout_seconds=timeout_seconds,
            critical=True,
            description="Kafka connectivity check",
        )

    @staticmethod
    def create_api_check(
        name: str,
        url: str,
        expected_status: int = 200,
        timeout_seconds: float = 5.0,
    ) -> HealthCheck:
        """Create an external API health check."""

        async def check_api() -> dict[str, Any]:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout_seconds)
                if response.status_code == expected_status:
                    return {
                        "status": HealthStatus.HEALTHY,
                        "message": f"API returned {response.status_code}",
                        "details": {"status_code": response.status_code},
                    }
                else:
                    return {
                        "status": HealthStatus.UNHEALTHY,
                        "message": f"API returned {response.status_code}",
                        "details": {"status_code": response.status_code},
                    }

        return HealthCheck(
            name=name,
            component_type=ComponentType.EXTERNAL_API,
            check_fn=check_api,
            timeout_seconds=timeout_seconds,
            critical=False,
            description=f"External API check: {url}",
        )

    @staticmethod
    def create_model_check(
        name: str,
        model_ready_fn: Callable[[], bool],
        timeout_seconds: float = 2.0,
    ) -> HealthCheck:
        """Create a model health check."""
        return HealthCheck(
            name=name,
            component_type=ComponentType.MODEL,
            check_fn=model_ready_fn,
            timeout_seconds=timeout_seconds,
            critical=True,
            description="ML model ready check",
        )

    @staticmethod
    def create_memory_check(
        name: str = "memory",
        max_percent: float = 90.0,
    ) -> HealthCheck:
        """Create a memory usage health check."""

        def check_memory() -> dict[str, Any]:
            import psutil

            memory = psutil.virtual_memory()
            status = (
                HealthStatus.HEALTHY
                if memory.percent < max_percent
                else HealthStatus.DEGRADED
            )
            return {
                "status": status,
                "message": f"Memory usage: {memory.percent:.1f}%",
                "details": {
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                },
            }

        return HealthCheck(
            name=name,
            component_type=ComponentType.INTERNAL,
            check_fn=check_memory,
            timeout_seconds=2.0,
            critical=False,
            description="Memory usage check",
        )

    @staticmethod
    def create_disk_check(
        name: str = "disk",
        path: str = "/",
        max_percent: float = 90.0,
    ) -> HealthCheck:
        """Create a disk usage health check."""

        def check_disk() -> dict[str, Any]:
            import psutil

            disk = psutil.disk_usage(path)
            status = (
                HealthStatus.HEALTHY
                if disk.percent < max_percent
                else HealthStatus.DEGRADED
            )
            return {
                "status": status,
                "message": f"Disk usage: {disk.percent:.1f}%",
                "details": {
                    "percent": disk.percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                },
            }

        return HealthCheck(
            name=name,
            component_type=ComponentType.INTERNAL,
            check_fn=check_disk,
            timeout_seconds=2.0,
            critical=False,
            description=f"Disk usage check for {path}",
        )

    def get_stats(self) -> dict[str, Any]:
        """Get health checker statistics."""
        results = list(self._results.values())

        by_status = {}
        for status in HealthStatus:
            by_status[status.value] = len(
                [r for r in results if r.status == status]
            )

        by_type = {}
        for ctype in ComponentType:
            by_type[ctype.value] = len(
                [r for r in results if r.component_type == ctype]
            )

        avg_latency = (
            sum(r.latency_ms for r in results if r.latency_ms)
            / len([r for r in results if r.latency_ms])
            if results
            else 0
        )

        return {
            "total_checks": len(self._checks),
            "enabled_checks": len([c for c in self._checks.values() if c.enabled]),
            "critical_checks": len([c for c in self._checks.values() if c.critical]),
            "by_status": by_status,
            "by_type": by_type,
            "avg_latency_ms": avg_latency,
            "uptime_seconds": self.uptime_seconds,
            "version": self._version,
        }


# Global health checker instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def init_health_checker(version: str = "unknown") -> HealthChecker:
    """Initialize the global health checker."""
    global _health_checker
    _health_checker = HealthChecker(version=version)
    return _health_checker
