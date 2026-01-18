"""
AWS Executor for direct AWS infrastructure scaling.

Responsibilities:
- Auto Scaling Group updates (desired capacity, min/max)
- Instance type changes via launch template updates
- Spot fleet management
- ECS service scaling
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.execution.base import (
    BaseExecutor,
    ExecutionResult,
    ExecutionStatus,
    ExecutorType,
    InfrastructureState,
    RollbackResult,
    ScalingAction,
    VerificationResult,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Boto3 imports (optional - gracefully handle if not installed)
try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("Boto3 not installed, AWS executor will use mock mode")


class ScalingType(str):
    """Types of AWS scaling resources."""

    ASG = "asg"
    ECS = "ecs"
    SPOT_FLEET = "spot_fleet"


@dataclass
class AWSConfig:
    """Configuration for AWS executor."""

    # AWS credentials (optional, uses default chain if not provided)
    region: str = "us-east-1"
    profile: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None

    # Resource identification
    scaling_type: str = ScalingType.ASG
    asg_name: str | None = None
    ecs_cluster: str | None = None
    ecs_service: str | None = None
    spot_fleet_request_id: str | None = None

    # Scaling settings
    wait_for_instances: bool = True
    instance_warmup_seconds: float = 300.0
    poll_interval_seconds: float = 10.0
    timeout_seconds: float = 600.0

    # Safety settings
    honor_cooldown: bool = True
    min_capacity: int = 1
    max_capacity: int = 100


@dataclass
class ASGStatus:
    """Status of an Auto Scaling Group."""

    name: str
    desired_capacity: int
    min_size: int
    max_size: int
    instances: list[dict[str, Any]]
    healthy_instances: int
    unhealthy_instances: int
    pending_instances: int

    @property
    def is_stable(self) -> bool:
        """Check if ASG is in stable state."""
        return (
            len(self.instances) == self.desired_capacity
            and self.pending_instances == 0
            and self.unhealthy_instances == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "desired_capacity": self.desired_capacity,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "instance_count": len(self.instances),
            "healthy_instances": self.healthy_instances,
            "unhealthy_instances": self.unhealthy_instances,
            "pending_instances": self.pending_instances,
            "is_stable": self.is_stable,
        }


@dataclass
class ECSServiceStatus:
    """Status of an ECS service."""

    cluster: str
    service_name: str
    desired_count: int
    running_count: int
    pending_count: int
    deployments: list[dict[str, Any]]

    @property
    def is_stable(self) -> bool:
        """Check if service is in stable state."""
        return (
            self.running_count == self.desired_count
            and self.pending_count == 0
            and len(self.deployments) == 1
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster": self.cluster,
            "service_name": self.service_name,
            "desired_count": self.desired_count,
            "running_count": self.running_count,
            "pending_count": self.pending_count,
            "deployment_count": len(self.deployments),
            "is_stable": self.is_stable,
        }


class AWSExecutor(BaseExecutor):
    """
    AWS executor for direct infrastructure scaling.

    Supports:
    - Auto Scaling Group (ASG) scaling
    - ECS service scaling
    - Spot Fleet capacity management
    """

    def __init__(self, aws_config: AWSConfig | None = None) -> None:
        """
        Initialize AWS executor.

        Args:
            aws_config: AWS configuration
        """
        super().__init__(ExecutorType.AWS)
        self.aws_config = aws_config or AWSConfig()
        self._autoscaling_client: Any = None
        self._ecs_client: Any = None
        self._ec2_client: Any = None
        self._connected = False

    async def connect(self) -> bool:
        """
        Connect to AWS services.

        Returns:
            True if connected successfully
        """
        if not BOTO3_AVAILABLE:
            logger.warning("Boto3 not available")
            return False

        try:
            session_kwargs = {"region_name": self.aws_config.region}

            if self.aws_config.profile:
                session_kwargs["profile_name"] = self.aws_config.profile
            elif self.aws_config.access_key_id and self.aws_config.secret_access_key:
                session_kwargs["aws_access_key_id"] = self.aws_config.access_key_id
                session_kwargs["aws_secret_access_key"] = self.aws_config.secret_access_key

            session = boto3.Session(**session_kwargs)

            self._autoscaling_client = session.client("autoscaling")
            self._ecs_client = session.client("ecs")
            self._ec2_client = session.client("ec2")
            self._connected = True

            logger.info("Connected to AWS", region=self.aws_config.region)
            return True

        except Exception as e:
            logger.error("Failed to connect to AWS", error=str(e))
            return False

    async def scale(self, action: ScalingAction) -> ExecutionResult:
        """
        Scale AWS resources.

        Args:
            action: Scaling action to execute

        Returns:
            ExecutionResult with status
        """
        started_at = datetime.now(UTC)

        # Store previous state for rollback
        try:
            previous_state = await self.get_current_state()
            self.store_rollback_state(action.action_id, previous_state)
        except Exception as e:
            logger.warning("Failed to get current state", error=str(e))
            previous_state = None

        try:
            # Connect if not connected
            if not self._connected:
                if not await self.connect():
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                        previous_state=previous_state,
                        error_message="Failed to connect to AWS",
                    )

            # Execute scaling based on type
            if self.aws_config.scaling_type == ScalingType.ASG:
                success = await self._scale_asg(action.target_count)
            elif self.aws_config.scaling_type == ScalingType.ECS:
                success = await self._scale_ecs(action.target_count)
            elif self.aws_config.scaling_type == ScalingType.SPOT_FLEET:
                success = await self._scale_spot_fleet(action.target_count)
            else:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                    previous_state=previous_state,
                    error_message=f"Unknown scaling type: {self.aws_config.scaling_type}",
                )

            if not success:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                    previous_state=previous_state,
                    error_message="Scaling operation failed",
                    rollback_available=True,
                )

            # Wait for scaling to complete
            if self.aws_config.wait_for_instances:
                await self._wait_for_scaling(action.target_count, action.timeout_seconds)

            current_state = await self.get_current_state()

            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                previous_state=previous_state,
                current_state=current_state,
            )
            self.record_execution(result)

            logger.info(
                "AWS scale completed",
                scaling_type=self.aws_config.scaling_type,
                previous_count=previous_state.instance_count if previous_state else None,
                current_count=current_state.instance_count,
            )

            return result

        except Exception as e:
            logger.error("AWS scale failed", error=str(e))
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                previous_state=previous_state,
                error_message=str(e),
                rollback_available=True,
            )
            self.record_execution(result)
            return result

    async def _scale_asg(self, target_count: int) -> bool:
        """Scale Auto Scaling Group."""
        if not self._autoscaling_client or not self.aws_config.asg_name:
            return False

        try:
            # Ensure target is within bounds
            target_count = max(self.aws_config.min_capacity, target_count)
            target_count = min(self.aws_config.max_capacity, target_count)

            # Update desired capacity
            self._autoscaling_client.set_desired_capacity(
                AutoScalingGroupName=self.aws_config.asg_name,
                DesiredCapacity=target_count,
                HonorCooldown=self.aws_config.honor_cooldown,
            )

            logger.info(
                "ASG capacity updated",
                asg=self.aws_config.asg_name,
                target=target_count,
            )

            return True

        except ClientError as e:
            logger.error("ASG scale failed", error=str(e))
            return False

    async def _scale_ecs(self, target_count: int) -> bool:
        """Scale ECS service."""
        if not self._ecs_client:
            return False

        if not self.aws_config.ecs_cluster or not self.aws_config.ecs_service:
            return False

        try:
            self._ecs_client.update_service(
                cluster=self.aws_config.ecs_cluster,
                service=self.aws_config.ecs_service,
                desiredCount=target_count,
            )

            logger.info(
                "ECS service updated",
                cluster=self.aws_config.ecs_cluster,
                service=self.aws_config.ecs_service,
                target=target_count,
            )

            return True

        except ClientError as e:
            logger.error("ECS scale failed", error=str(e))
            return False

    async def _scale_spot_fleet(self, target_count: int) -> bool:
        """Scale Spot Fleet."""
        if not self._ec2_client or not self.aws_config.spot_fleet_request_id:
            return False

        try:
            self._ec2_client.modify_spot_fleet_request(
                SpotFleetRequestId=self.aws_config.spot_fleet_request_id,
                TargetCapacity=target_count,
            )

            logger.info(
                "Spot Fleet updated",
                fleet_id=self.aws_config.spot_fleet_request_id,
                target=target_count,
            )

            return True

        except ClientError as e:
            logger.error("Spot Fleet scale failed", error=str(e))
            return False

    async def _wait_for_scaling(
        self, target_count: int, timeout_seconds: float
    ) -> bool:
        """Wait for scaling operation to complete."""
        start_time = datetime.now(UTC)
        timeout = timeout_seconds or self.aws_config.timeout_seconds

        while True:
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning("Scaling wait timeout", elapsed=elapsed)
                return False

            try:
                state = await self.get_current_state()

                if state.instance_count == target_count and state.is_healthy:
                    logger.info("Scaling complete", count=target_count)
                    return True

            except Exception as e:
                logger.warning("Error checking scaling status", error=str(e))

            await asyncio.sleep(self.aws_config.poll_interval_seconds)

    async def _get_asg_status(self) -> ASGStatus | None:
        """Get ASG status."""
        if not self._autoscaling_client or not self.aws_config.asg_name:
            return None

        try:
            response = self._autoscaling_client.describe_auto_scaling_groups(
                AutoScalingGroupNames=[self.aws_config.asg_name]
            )

            if not response.get("AutoScalingGroups"):
                return None

            asg = response["AutoScalingGroups"][0]
            instances = asg.get("Instances", [])

            healthy = sum(
                1 for i in instances
                if i.get("HealthStatus") == "Healthy"
                and i.get("LifecycleState") == "InService"
            )
            unhealthy = sum(
                1 for i in instances
                if i.get("HealthStatus") == "Unhealthy"
            )
            pending = sum(
                1 for i in instances
                if i.get("LifecycleState") in ("Pending", "Pending:Wait", "Pending:Proceed")
            )

            return ASGStatus(
                name=asg["AutoScalingGroupName"],
                desired_capacity=asg["DesiredCapacity"],
                min_size=asg["MinSize"],
                max_size=asg["MaxSize"],
                instances=instances,
                healthy_instances=healthy,
                unhealthy_instances=unhealthy,
                pending_instances=pending,
            )

        except ClientError as e:
            logger.error("Failed to get ASG status", error=str(e))
            return None

    async def _get_ecs_status(self) -> ECSServiceStatus | None:
        """Get ECS service status."""
        if not self._ecs_client:
            return None

        if not self.aws_config.ecs_cluster or not self.aws_config.ecs_service:
            return None

        try:
            response = self._ecs_client.describe_services(
                cluster=self.aws_config.ecs_cluster,
                services=[self.aws_config.ecs_service],
            )

            if not response.get("services"):
                return None

            service = response["services"][0]

            return ECSServiceStatus(
                cluster=self.aws_config.ecs_cluster,
                service_name=service["serviceName"],
                desired_count=service["desiredCount"],
                running_count=service["runningCount"],
                pending_count=service["pendingCount"],
                deployments=service.get("deployments", []),
            )

        except ClientError as e:
            logger.error("Failed to get ECS status", error=str(e))
            return None

    async def rollback(self, action_id: str) -> RollbackResult:
        """Rollback to previous capacity."""
        previous_state = self.get_rollback_state(action_id)

        if previous_state is None:
            return RollbackResult(
                action_id=action_id,
                success=False,
                previous_state=await self.get_current_state(),
                error_message="No rollback state found",
            )

        try:
            # Scale back to previous count
            rollback_action = ScalingAction(
                action_id=f"{action_id}_rollback",
                target_count=previous_state.instance_count,
                current_count=(await self.get_current_state()).instance_count,
            )

            result = await self.scale(rollback_action)

            if result.is_success:
                restored_state = await self.get_current_state()
                self.clear_rollback_state(action_id)

                return RollbackResult(
                    action_id=action_id,
                    success=True,
                    previous_state=previous_state,
                    restored_state=restored_state,
                )
            else:
                return RollbackResult(
                    action_id=action_id,
                    success=False,
                    previous_state=previous_state,
                    error_message=result.error_message,
                )

        except Exception as e:
            return RollbackResult(
                action_id=action_id,
                success=False,
                previous_state=previous_state,
                error_message=str(e),
            )

    async def verify(self, action: ScalingAction) -> VerificationResult:
        """Verify scaling completed successfully."""
        checks_passed = []
        checks_failed = []

        try:
            state = await self.get_current_state()

            # Check instance count
            if state.instance_count == action.target_count:
                checks_passed.append("instance_count")
            else:
                checks_failed.append(
                    f"instance_count (expected {action.target_count}, got {state.instance_count})"
                )

            # Check health
            if state.is_healthy:
                checks_passed.append("health")
            else:
                checks_failed.append(
                    f"health ({state.unhealthy_count} unhealthy)"
                )

            # Check for pending
            if state.pending_count == 0:
                checks_passed.append("no_pending")
            else:
                checks_failed.append(f"pending ({state.pending_count} pending)")

            return VerificationResult(
                action_id=action.action_id,
                verified=len(checks_failed) == 0,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                target_count=action.target_count,
                actual_count=state.instance_count,
                healthy_count=state.healthy_count,
            )

        except Exception as e:
            return VerificationResult(
                action_id=action.action_id,
                verified=False,
                checks_passed=[],
                checks_failed=[f"verification_error: {str(e)}"],
                target_count=action.target_count,
                actual_count=0,
                healthy_count=0,
            )

    async def get_current_state(self) -> InfrastructureState:
        """Get current AWS resource state."""
        if not self._connected:
            if not await self.connect():
                return InfrastructureState(
                    executor_type=self.executor_type,
                    timestamp=datetime.now(UTC),
                    instance_count=0,
                    metadata={"connected": False},
                )

        try:
            if self.aws_config.scaling_type == ScalingType.ASG:
                asg_status = await self._get_asg_status()
                if asg_status:
                    return InfrastructureState(
                        executor_type=self.executor_type,
                        timestamp=datetime.now(UTC),
                        instance_count=asg_status.desired_capacity,
                        healthy_count=asg_status.healthy_instances,
                        unhealthy_count=asg_status.unhealthy_instances,
                        pending_count=asg_status.pending_instances,
                        metadata={"asg_status": asg_status.to_dict()},
                    )

            elif self.aws_config.scaling_type == ScalingType.ECS:
                ecs_status = await self._get_ecs_status()
                if ecs_status:
                    return InfrastructureState(
                        executor_type=self.executor_type,
                        timestamp=datetime.now(UTC),
                        instance_count=ecs_status.desired_count,
                        healthy_count=ecs_status.running_count,
                        pending_count=ecs_status.pending_count,
                        metadata={"ecs_status": ecs_status.to_dict()},
                    )

        except Exception as e:
            logger.error("Failed to get AWS state", error=str(e))

        return InfrastructureState(
            executor_type=self.executor_type,
            timestamp=datetime.now(UTC),
            instance_count=0,
            metadata={"error": "Failed to get state"},
        )

    async def update_asg_limits(
        self, min_size: int | None = None, max_size: int | None = None
    ) -> bool:
        """Update ASG min/max limits."""
        if not self._autoscaling_client or not self.aws_config.asg_name:
            return False

        try:
            update_params = {"AutoScalingGroupName": self.aws_config.asg_name}

            if min_size is not None:
                update_params["MinSize"] = min_size
            if max_size is not None:
                update_params["MaxSize"] = max_size

            self._autoscaling_client.update_auto_scaling_group(**update_params)

            logger.info(
                "ASG limits updated",
                asg=self.aws_config.asg_name,
                min_size=min_size,
                max_size=max_size,
            )
            return True

        except ClientError as e:
            logger.error("Failed to update ASG limits", error=str(e))
            return False
