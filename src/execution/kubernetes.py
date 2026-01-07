"""
Kubernetes Executor for scaling Kubernetes deployments.

Responsibilities:
- Connect to cluster (in-cluster or via kubeconfig)
- Scale deployments by updating replica count
- Adjust HPA min/max replicas
- Suspend HPA for preemptive scaling
- Wait for rollout completion
- Verify pod readiness
- Support for multiple deployments
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

# Kubernetes client imports (optional - gracefully handle if not installed)
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    logger.warning("Kubernetes client not installed, K8s executor will use mock mode")


@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes executor."""

    # Cluster connection
    in_cluster: bool = False
    kubeconfig_path: str | None = None

    # Target resources
    namespace: str = "default"
    deployment_name: str = ""
    hpa_name: str | None = None

    # Scaling settings
    wait_for_rollout: bool = True
    rollout_timeout_seconds: float = 300.0
    poll_interval_seconds: float = 5.0

    # HPA settings
    suspend_hpa_on_scale: bool = False
    restore_hpa_after_seconds: float = 300.0

    # Health check settings
    readiness_timeout_seconds: float = 120.0
    min_ready_seconds: int = 10


@dataclass
class PodStatus:
    """Status of a Kubernetes pod."""

    name: str
    phase: str
    ready: bool
    restart_count: int
    started_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "phase": self.phase,
            "ready": self.ready,
            "restart_count": self.restart_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


@dataclass
class DeploymentStatus:
    """Status of a Kubernetes deployment."""

    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    available_replicas: int
    unavailable_replicas: int
    updated_replicas: int
    pods: list[PodStatus] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Check if deployment is fully ready."""
        return (
            self.ready_replicas == self.replicas
            and self.available_replicas == self.replicas
            and self.unavailable_replicas == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "replicas": self.replicas,
            "ready_replicas": self.ready_replicas,
            "available_replicas": self.available_replicas,
            "unavailable_replicas": self.unavailable_replicas,
            "updated_replicas": self.updated_replicas,
            "is_ready": self.is_ready,
            "pods": [p.to_dict() for p in self.pods],
        }


@dataclass
class HPAStatus:
    """Status of a Horizontal Pod Autoscaler."""

    name: str
    namespace: str
    min_replicas: int
    max_replicas: int
    current_replicas: int
    desired_replicas: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "current_replicas": self.current_replicas,
            "desired_replicas": self.desired_replicas,
        }


class KubernetesExecutor(BaseExecutor):
    """
    Kubernetes executor for scaling deployments.

    Supports:
    - Deployment replica scaling
    - HPA min/max adjustment
    - Rollout waiting and verification
    - Pod health checking
    """

    def __init__(self, k8s_config: KubernetesConfig | None = None) -> None:
        """
        Initialize Kubernetes executor.

        Args:
            k8s_config: Kubernetes configuration
        """
        super().__init__(ExecutorType.KUBERNETES)
        self.k8s_config = k8s_config or KubernetesConfig()
        self._apps_v1: Any = None
        self._core_v1: Any = None
        self._autoscaling_v1: Any = None
        self._connected = False
        self._original_hpa_settings: dict[str, dict[str, int]] = {}

    async def connect(self) -> bool:
        """
        Connect to Kubernetes cluster.

        Returns:
            True if connected successfully
        """
        if not K8S_AVAILABLE:
            logger.warning("Kubernetes client not available")
            return False

        try:
            if self.k8s_config.in_cluster:
                config.load_incluster_config()
            elif self.k8s_config.kubeconfig_path:
                config.load_kube_config(config_file=self.k8s_config.kubeconfig_path)
            else:
                config.load_kube_config()

            self._apps_v1 = client.AppsV1Api()
            self._core_v1 = client.CoreV1Api()
            self._autoscaling_v1 = client.AutoscalingV1Api()
            self._connected = True

            logger.info(
                "Connected to Kubernetes cluster",
                namespace=self.k8s_config.namespace,
            )
            return True

        except Exception as e:
            logger.error("Failed to connect to Kubernetes", error=str(e))
            return False

    async def scale(self, action: ScalingAction) -> ExecutionResult:
        """
        Scale a Kubernetes deployment.

        Args:
            action: Scaling action to execute

        Returns:
            ExecutionResult with status
        """
        started_at = datetime.now(timezone.utc)

        # Store previous state for rollback
        try:
            previous_state = await self.get_current_state()
            self.store_rollback_state(action.action_id, previous_state)
        except Exception as e:
            logger.error("Failed to get current state", error=str(e))
            previous_state = None

        try:
            # Connect if not connected
            if not self._connected:
                if not await self.connect():
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        previous_state=previous_state,
                        error_message="Failed to connect to Kubernetes cluster",
                    )

            # Suspend HPA if configured
            if self.k8s_config.suspend_hpa_on_scale and self.k8s_config.hpa_name:
                await self._suspend_hpa(action.target_count)

            # Scale the deployment
            await self._scale_deployment(action.target_count)

            # Wait for rollout if configured
            if self.k8s_config.wait_for_rollout:
                success = await self._wait_for_rollout(action.timeout_seconds)
                if not success:
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        previous_state=previous_state,
                        error_message="Rollout timeout",
                        rollback_available=True,
                    )

            current_state = await self.get_current_state()

            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                previous_state=previous_state,
                current_state=current_state,
            )
            self.record_execution(result)

            logger.info(
                "Kubernetes scale completed",
                deployment=self.k8s_config.deployment_name,
                previous_replicas=previous_state.instance_count if previous_state else None,
                current_replicas=current_state.instance_count,
            )

            return result

        except Exception as e:
            logger.error("Kubernetes scale failed", error=str(e))
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                previous_state=previous_state,
                error_message=str(e),
                rollback_available=True,
            )
            self.record_execution(result)
            return result

    async def _scale_deployment(self, target_replicas: int) -> None:
        """Scale the deployment to target replicas."""
        if not self._apps_v1:
            raise RuntimeError("Not connected to Kubernetes")

        body = {"spec": {"replicas": target_replicas}}

        self._apps_v1.patch_namespaced_deployment_scale(
            name=self.k8s_config.deployment_name,
            namespace=self.k8s_config.namespace,
            body=body,
        )

        logger.info(
            "Deployment scale patched",
            deployment=self.k8s_config.deployment_name,
            target_replicas=target_replicas,
        )

    async def _wait_for_rollout(self, timeout_seconds: float) -> bool:
        """Wait for deployment rollout to complete."""
        if not self._apps_v1:
            return False

        start_time = datetime.now(timezone.utc)
        timeout = timeout_seconds or self.k8s_config.rollout_timeout_seconds

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning("Rollout timeout", elapsed=elapsed, timeout=timeout)
                return False

            try:
                deployment = self._apps_v1.read_namespaced_deployment(
                    name=self.k8s_config.deployment_name,
                    namespace=self.k8s_config.namespace,
                )

                status = deployment.status
                spec_replicas = deployment.spec.replicas or 0

                if (
                    status.ready_replicas == spec_replicas
                    and status.available_replicas == spec_replicas
                    and (status.unavailable_replicas or 0) == 0
                ):
                    logger.info("Rollout completed", replicas=spec_replicas)
                    return True

            except Exception as e:
                logger.warning("Error checking rollout status", error=str(e))

            await asyncio.sleep(self.k8s_config.poll_interval_seconds)

    async def _suspend_hpa(self, target_replicas: int) -> None:
        """Suspend HPA by setting min=max=target."""
        if not self._autoscaling_v1 or not self.k8s_config.hpa_name:
            return

        try:
            # Get current HPA settings
            hpa = self._autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
                name=self.k8s_config.hpa_name,
                namespace=self.k8s_config.namespace,
            )

            # Store original settings
            self._original_hpa_settings[self.k8s_config.hpa_name] = {
                "min_replicas": hpa.spec.min_replicas,
                "max_replicas": hpa.spec.max_replicas,
            }

            # Set min=max=target to effectively suspend
            body = {
                "spec": {
                    "minReplicas": target_replicas,
                    "maxReplicas": target_replicas,
                }
            }

            self._autoscaling_v1.patch_namespaced_horizontal_pod_autoscaler(
                name=self.k8s_config.hpa_name,
                namespace=self.k8s_config.namespace,
                body=body,
            )

            logger.info(
                "HPA suspended",
                hpa=self.k8s_config.hpa_name,
                target=target_replicas,
            )

        except Exception as e:
            logger.warning("Failed to suspend HPA", error=str(e))

    async def _restore_hpa(self) -> None:
        """Restore HPA to original settings."""
        if not self._autoscaling_v1 or not self.k8s_config.hpa_name:
            return

        original = self._original_hpa_settings.get(self.k8s_config.hpa_name)
        if not original:
            return

        try:
            body = {
                "spec": {
                    "minReplicas": original["min_replicas"],
                    "maxReplicas": original["max_replicas"],
                }
            }

            self._autoscaling_v1.patch_namespaced_horizontal_pod_autoscaler(
                name=self.k8s_config.hpa_name,
                namespace=self.k8s_config.namespace,
                body=body,
            )

            del self._original_hpa_settings[self.k8s_config.hpa_name]

            logger.info(
                "HPA restored",
                hpa=self.k8s_config.hpa_name,
                min_replicas=original["min_replicas"],
                max_replicas=original["max_replicas"],
            )

        except Exception as e:
            logger.warning("Failed to restore HPA", error=str(e))

    async def rollback(self, action_id: str) -> RollbackResult:
        """
        Rollback a previous scaling action.

        Args:
            action_id: ID of the action to rollback

        Returns:
            RollbackResult with status
        """
        previous_state = self.get_rollback_state(action_id)

        if previous_state is None:
            return RollbackResult(
                action_id=action_id,
                success=False,
                previous_state=await self.get_current_state(),
                error_message="No rollback state found for this action",
            )

        try:
            # Scale back to previous count
            await self._scale_deployment(previous_state.instance_count)

            # Wait for rollout
            if self.k8s_config.wait_for_rollout:
                await self._wait_for_rollout(self.k8s_config.rollout_timeout_seconds)

            # Restore HPA if suspended
            await self._restore_hpa()

            restored_state = await self.get_current_state()
            self.clear_rollback_state(action_id)

            logger.info(
                "Kubernetes rollback completed",
                action_id=action_id,
                restored_replicas=restored_state.instance_count,
            )

            return RollbackResult(
                action_id=action_id,
                success=True,
                previous_state=previous_state,
                restored_state=restored_state,
            )

        except Exception as e:
            logger.error("Kubernetes rollback failed", error=str(e))
            return RollbackResult(
                action_id=action_id,
                success=False,
                previous_state=previous_state,
                error_message=str(e),
            )

    async def verify(self, action: ScalingAction) -> VerificationResult:
        """
        Verify that scaling completed successfully.

        Args:
            action: The scaling action to verify

        Returns:
            VerificationResult with verification details
        """
        checks_passed = []
        checks_failed = []

        try:
            deployment_status = await self._get_deployment_status()

            # Check replica count
            if deployment_status.replicas == action.target_count:
                checks_passed.append("replica_count")
            else:
                checks_failed.append(
                    f"replica_count (expected {action.target_count}, got {deployment_status.replicas})"
                )

            # Check ready replicas
            if deployment_status.ready_replicas == action.target_count:
                checks_passed.append("ready_replicas")
            else:
                checks_failed.append(
                    f"ready_replicas (expected {action.target_count}, got {deployment_status.ready_replicas})"
                )

            # Check available replicas
            if deployment_status.available_replicas == action.target_count:
                checks_passed.append("available_replicas")
            else:
                checks_failed.append("available_replicas")

            # Check for unavailable pods
            if deployment_status.unavailable_replicas == 0:
                checks_passed.append("no_unavailable_pods")
            else:
                checks_failed.append(
                    f"unavailable_pods ({deployment_status.unavailable_replicas} unavailable)"
                )

            verified = len(checks_failed) == 0

            return VerificationResult(
                action_id=action.action_id,
                verified=verified,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                target_count=action.target_count,
                actual_count=deployment_status.replicas,
                healthy_count=deployment_status.ready_replicas,
                metadata={"deployment_status": deployment_status.to_dict()},
            )

        except Exception as e:
            logger.error("Verification failed", error=str(e))
            return VerificationResult(
                action_id=action.action_id,
                verified=False,
                checks_passed=checks_passed,
                checks_failed=["verification_error: " + str(e)],
                target_count=action.target_count,
                actual_count=0,
                healthy_count=0,
            )

    async def _get_deployment_status(self) -> DeploymentStatus:
        """Get current deployment status."""
        if not self._apps_v1:
            raise RuntimeError("Not connected to Kubernetes")

        deployment = self._apps_v1.read_namespaced_deployment(
            name=self.k8s_config.deployment_name,
            namespace=self.k8s_config.namespace,
        )

        status = deployment.status

        return DeploymentStatus(
            name=self.k8s_config.deployment_name,
            namespace=self.k8s_config.namespace,
            replicas=deployment.spec.replicas or 0,
            ready_replicas=status.ready_replicas or 0,
            available_replicas=status.available_replicas or 0,
            unavailable_replicas=status.unavailable_replicas or 0,
            updated_replicas=status.updated_replicas or 0,
        )

    async def get_current_state(self) -> InfrastructureState:
        """
        Get current Kubernetes infrastructure state.

        Returns:
            Current InfrastructureState
        """
        if not self._connected:
            if not await self.connect():
                # Return mock state if not connected
                return InfrastructureState(
                    executor_type=self.executor_type,
                    timestamp=datetime.now(timezone.utc),
                    instance_count=0,
                    metadata={"connected": False},
                )

        try:
            deployment_status = await self._get_deployment_status()

            return InfrastructureState(
                executor_type=self.executor_type,
                timestamp=datetime.now(timezone.utc),
                instance_count=deployment_status.replicas,
                healthy_count=deployment_status.ready_replicas,
                unhealthy_count=deployment_status.unavailable_replicas,
                pending_count=deployment_status.replicas - deployment_status.ready_replicas,
                metadata={
                    "deployment": self.k8s_config.deployment_name,
                    "namespace": self.k8s_config.namespace,
                    "deployment_status": deployment_status.to_dict(),
                },
            )

        except Exception as e:
            logger.error("Failed to get Kubernetes state", error=str(e))
            return InfrastructureState(
                executor_type=self.executor_type,
                timestamp=datetime.now(timezone.utc),
                instance_count=0,
                metadata={"error": str(e)},
            )

    async def get_hpa_status(self) -> HPAStatus | None:
        """Get HPA status if configured."""
        if not self._autoscaling_v1 or not self.k8s_config.hpa_name:
            return None

        try:
            hpa = self._autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
                name=self.k8s_config.hpa_name,
                namespace=self.k8s_config.namespace,
            )

            return HPAStatus(
                name=hpa.metadata.name,
                namespace=hpa.metadata.namespace,
                min_replicas=hpa.spec.min_replicas,
                max_replicas=hpa.spec.max_replicas,
                current_replicas=hpa.status.current_replicas or 0,
                desired_replicas=hpa.status.desired_replicas or 0,
            )

        except Exception as e:
            logger.warning("Failed to get HPA status", error=str(e))
            return None
