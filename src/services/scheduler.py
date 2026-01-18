"""
Scheduler Service for coordinating background tasks.

Responsibilities:
- Coordinate all periodic background tasks
- Manage task scheduling using APScheduler
- Handle task dependencies and ordering
- Provide task status and history
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    JobExecutionEvent,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED = "missed"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskExecution:
    """Record of a task execution."""

    task_id: str
    job_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: TaskStatus = TaskStatus.RUNNING
    duration_seconds: float | None = None
    error: str | None = None
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class ScheduledTask:
    """Definition of a scheduled task."""

    task_id: str
    name: str
    description: str
    func: Callable
    trigger_type: str  # "interval" or "cron"
    trigger_args: dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int = 60
    dependencies: list[str] = field(default_factory=list)

    # Runtime stats
    last_run: datetime | None = None
    last_status: TaskStatus | None = None
    run_count: int = 0
    error_count: int = 0


class SchedulerService:
    """
    Central scheduler service for all background tasks.

    Manages periodic tasks for:
    - Metrics collection
    - Feature computation
    - Predictions
    - Scaling decisions
    - Model maintenance
    """

    def __init__(
        self,
        timezone: str = "UTC",
        max_execution_history: int = 1000,
    ) -> None:
        """
        Initialize scheduler service.

        Args:
            timezone: Timezone for scheduling
            max_execution_history: Maximum execution records to keep
        """
        self._timezone = timezone
        self._max_history = max_execution_history
        self._scheduler: AsyncIOScheduler | None = None
        self._tasks: dict[str, ScheduledTask] = {}
        self._executions: list[TaskExecution] = []
        self._running = False
        self._callbacks: dict[str, list[Callable]] = {
            "on_start": [],
            "on_complete": [],
            "on_error": [],
            "on_missed": [],
        }

        logger.info("Scheduler service initialized", timezone=timezone)

    def _create_scheduler(self) -> AsyncIOScheduler:
        """Create and configure the APScheduler instance."""
        scheduler = AsyncIOScheduler(
            timezone=self._timezone,
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 60,
            },
        )

        # Add event listeners
        scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)

        return scheduler

    def _on_job_executed(self, event: JobExecutionEvent) -> None:
        """Handle successful job execution."""
        task_id = event.job_id
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.last_run = datetime.now(UTC)
            task.last_status = TaskStatus.COMPLETED
            task.run_count += 1

        # Update execution record
        for execution in reversed(self._executions):
            if execution.job_id == event.job_id and execution.status == TaskStatus.RUNNING:
                execution.status = TaskStatus.COMPLETED
                execution.completed_at = datetime.now(UTC)
                execution.duration_seconds = event.scheduled_run_time.timestamp() if event.scheduled_run_time else 0
                execution.result = event.retval
                break

        # Notify callbacks
        for callback in self._callbacks["on_complete"]:
            try:
                callback(task_id, event.retval)
            except Exception as e:
                logger.error("Callback error", callback="on_complete", error=str(e))

        logger.debug("Job executed successfully", job_id=task_id)

    def _on_job_error(self, event: JobExecutionEvent) -> None:
        """Handle job execution error."""
        task_id = event.job_id
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.last_run = datetime.now(UTC)
            task.last_status = TaskStatus.FAILED
            task.run_count += 1
            task.error_count += 1

        # Update execution record
        for execution in reversed(self._executions):
            if execution.job_id == event.job_id and execution.status == TaskStatus.RUNNING:
                execution.status = TaskStatus.FAILED
                execution.completed_at = datetime.now(UTC)
                execution.error = str(event.exception)
                break

        # Notify callbacks
        for callback in self._callbacks["on_error"]:
            try:
                callback(task_id, event.exception)
            except Exception as e:
                logger.error("Callback error", callback="on_error", error=str(e))

        logger.error("Job execution failed", job_id=task_id, error=str(event.exception))

    def _on_job_missed(self, event: JobExecutionEvent) -> None:
        """Handle missed job execution."""
        task_id = event.job_id
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.last_status = TaskStatus.MISSED

        # Create missed execution record
        execution = TaskExecution(
            task_id=task_id,
            job_id=event.job_id,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            status=TaskStatus.MISSED,
        )
        self._executions.append(execution)
        self._trim_history()

        # Notify callbacks
        for callback in self._callbacks["on_missed"]:
            try:
                callback(task_id)
            except Exception as e:
                logger.error("Callback error", callback="on_missed", error=str(e))

        logger.warning("Job execution missed", job_id=task_id)

    def _trim_history(self) -> None:
        """Trim execution history to max size."""
        if len(self._executions) > self._max_history:
            self._executions = self._executions[-self._max_history:]

    def register_task(self, task: ScheduledTask) -> None:
        """
        Register a scheduled task.

        Args:
            task: Task definition to register
        """
        self._tasks[task.task_id] = task
        logger.info(
            "Task registered",
            task_id=task.task_id,
            name=task.name,
            trigger=task.trigger_type,
        )

    def unregister_task(self, task_id: str) -> bool:
        """
        Unregister a scheduled task.

        Args:
            task_id: ID of task to unregister

        Returns:
            True if task was removed
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            if self._scheduler and self._scheduler.get_job(task_id):
                self._scheduler.remove_job(task_id)
            return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            if self._scheduler:
                self._scheduler.resume_job(task_id)
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            if self._scheduler:
                self._scheduler.pause_job(task_id)
            return True
        return False

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add callback for scheduler events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def start(self) -> None:
        """Start the scheduler service."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._scheduler = self._create_scheduler()

        # Add all registered tasks to scheduler
        for task in self._tasks.values():
            if not task.enabled:
                continue

            # Create trigger
            if task.trigger_type == "interval":
                trigger = IntervalTrigger(**task.trigger_args)
            elif task.trigger_type == "cron":
                trigger = CronTrigger(**task.trigger_args)
            else:
                logger.error(f"Unknown trigger type: {task.trigger_type}")
                continue

            # Wrap function to record execution
            async def wrapped_func(task_id: str = task.task_id, func: Callable = task.func):
                execution = TaskExecution(
                    task_id=task_id,
                    job_id=task_id,
                    started_at=datetime.now(UTC),
                )
                self._executions.append(execution)
                self._trim_history()

                # Notify start callbacks
                for callback in self._callbacks["on_start"]:
                    try:
                        callback(task_id)
                    except Exception as e:
                        logger.error("Callback error", callback="on_start", error=str(e))

                # Execute the actual function
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()

            self._scheduler.add_job(
                wrapped_func,
                trigger,
                id=task.task_id,
                name=task.name,
                max_instances=task.max_instances,
                coalesce=task.coalesce,
                misfire_grace_time=task.misfire_grace_time,
            )

        self._scheduler.start()
        self._running = True

        logger.info(
            "Scheduler service started",
            task_count=len(self._tasks),
            enabled_tasks=len([t for t in self._tasks.values() if t.enabled]),
        )

    async def stop(self) -> None:
        """Stop the scheduler service."""
        if not self._running:
            return

        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        self._running = False
        logger.info("Scheduler service stopped")

    async def run_task_now(self, task_id: str) -> Any:
        """
        Run a task immediately (outside of schedule).

        Args:
            task_id: ID of task to run

        Returns:
            Task result
        """
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")

        task = self._tasks[task_id]

        execution = TaskExecution(
            task_id=task_id,
            job_id=f"{task_id}_manual",
            started_at=datetime.now(UTC),
        )
        self._executions.append(execution)

        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func()
            else:
                result = task.func()

            execution.status = TaskStatus.COMPLETED
            execution.completed_at = datetime.now(UTC)
            execution.result = result

            task.last_run = datetime.now(UTC)
            task.last_status = TaskStatus.COMPLETED
            task.run_count += 1

            return result

        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.completed_at = datetime.now(UTC)
            execution.error = str(e)

            task.last_run = datetime.now(UTC)
            task.last_status = TaskStatus.FAILED
            task.run_count += 1
            task.error_count += 1

            raise

    def get_task(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[ScheduledTask]:
        """Get all registered tasks."""
        return list(self._tasks.values())

    def get_task_executions(
        self,
        task_id: str | None = None,
        limit: int = 100,
    ) -> list[TaskExecution]:
        """Get task execution history."""
        executions = self._executions
        if task_id:
            executions = [e for e in executions if e.task_id == task_id]
        return list(reversed(executions[-limit:]))

    def get_next_run_time(self, task_id: str) -> datetime | None:
        """Get next scheduled run time for a task."""
        if self._scheduler:
            job = self._scheduler.get_job(task_id)
            if job:
                return job.next_run_time
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        total_runs = sum(t.run_count for t in self._tasks.values())
        total_errors = sum(t.error_count for t in self._tasks.values())

        return {
            "running": self._running,
            "total_tasks": len(self._tasks),
            "enabled_tasks": len([t for t in self._tasks.values() if t.enabled]),
            "total_executions": total_runs,
            "total_errors": total_errors,
            "error_rate": total_errors / total_runs if total_runs > 0 else 0,
            "execution_history_size": len(self._executions),
        }

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Factory function for creating scheduler with default tasks
def create_scheduler_with_default_tasks(
    collect_metrics_fn: Callable,
    compute_features_fn: Callable,
    run_short_predictions_fn: Callable,
    run_medium_predictions_fn: Callable,
    run_long_predictions_fn: Callable,
    evaluate_scaling_fn: Callable,
    calculate_savings_fn: Callable,
    retrain_models_fn: Callable,
    generate_reports_fn: Callable,
) -> SchedulerService:
    """
    Create a scheduler with all default tasks configured.

    Args:
        collect_metrics_fn: Function to collect metrics
        compute_features_fn: Function to compute features
        run_short_predictions_fn: Function to run short-term predictions
        run_medium_predictions_fn: Function to run medium-term predictions
        run_long_predictions_fn: Function to run long-term predictions
        evaluate_scaling_fn: Function to evaluate scaling decisions
        calculate_savings_fn: Function to calculate cost savings
        retrain_models_fn: Function to retrain models
        generate_reports_fn: Function to generate reports

    Returns:
        Configured SchedulerService
    """
    scheduler = SchedulerService()

    # Every 1 minute: Collect metrics
    scheduler.register_task(
        ScheduledTask(
            task_id="collect_metrics",
            name="Collect Metrics",
            description="Collect metrics from all sources",
            func=collect_metrics_fn,
            trigger_type="interval",
            trigger_args={"minutes": 1},
            priority=TaskPriority.HIGH,
        )
    )

    # Every 1 minute: Compute features (after metrics)
    scheduler.register_task(
        ScheduledTask(
            task_id="compute_features",
            name="Compute Features",
            description="Compute features from collected metrics",
            func=compute_features_fn,
            trigger_type="interval",
            trigger_args={"minutes": 1},
            priority=TaskPriority.HIGH,
            dependencies=["collect_metrics"],
        )
    )

    # Every 5 minutes: Short-term predictions
    scheduler.register_task(
        ScheduledTask(
            task_id="short_term_predictions",
            name="Short-term Predictions",
            description="Run short-term predictions (15-60 min horizon)",
            func=run_short_predictions_fn,
            trigger_type="interval",
            trigger_args={"minutes": 5},
            priority=TaskPriority.HIGH,
        )
    )

    # Every 15 minutes: Medium-term predictions
    scheduler.register_task(
        ScheduledTask(
            task_id="medium_term_predictions",
            name="Medium-term Predictions",
            description="Run medium-term predictions (1-24 hour horizon)",
            func=run_medium_predictions_fn,
            trigger_type="interval",
            trigger_args={"minutes": 15},
            priority=TaskPriority.NORMAL,
        )
    )

    # Every 15 minutes: Evaluate scaling decisions
    scheduler.register_task(
        ScheduledTask(
            task_id="evaluate_scaling",
            name="Evaluate Scaling",
            description="Evaluate if scaling action is needed",
            func=evaluate_scaling_fn,
            trigger_type="interval",
            trigger_args={"minutes": 15},
            priority=TaskPriority.HIGH,
            dependencies=["medium_term_predictions"],
        )
    )

    # Every 1 hour: Long-term predictions
    scheduler.register_task(
        ScheduledTask(
            task_id="long_term_predictions",
            name="Long-term Predictions",
            description="Run long-term predictions (1-7 day horizon)",
            func=run_long_predictions_fn,
            trigger_type="interval",
            trigger_args={"hours": 1},
            priority=TaskPriority.LOW,
        )
    )

    # Every 1 hour: Calculate cost savings
    scheduler.register_task(
        ScheduledTask(
            task_id="calculate_savings",
            name="Calculate Savings",
            description="Calculate cost savings from predictive scaling",
            func=calculate_savings_fn,
            trigger_type="interval",
            trigger_args={"hours": 1},
            priority=TaskPriority.LOW,
        )
    )

    # Daily at 2 AM: Retrain models
    scheduler.register_task(
        ScheduledTask(
            task_id="retrain_models",
            name="Retrain Models",
            description="Retrain prediction models with new data",
            func=retrain_models_fn,
            trigger_type="cron",
            trigger_args={"hour": 2, "minute": 0},
            priority=TaskPriority.NORMAL,
            max_instances=1,
            misfire_grace_time=3600,
        )
    )

    # Daily at 3 AM: Generate reports
    scheduler.register_task(
        ScheduledTask(
            task_id="generate_reports",
            name="Generate Reports",
            description="Generate daily reports and recalibrate predictions",
            func=generate_reports_fn,
            trigger_type="cron",
            trigger_args={"hour": 3, "minute": 0},
            priority=TaskPriority.LOW,
            dependencies=["retrain_models"],
        )
    )

    return scheduler


# Global scheduler instance
_scheduler_service: SchedulerService | None = None


def get_scheduler_service() -> SchedulerService:
    """Get the global scheduler service."""
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service


def init_scheduler_service(**kwargs) -> SchedulerService:
    """Initialize the global scheduler service."""
    global _scheduler_service
    _scheduler_service = SchedulerService(**kwargs)
    return _scheduler_service
