"""
Kubernetes metrics collector.

Collects pod counts, replica status, resource usage, HPA status,
and deployment events from a Kubernetes cluster.
"""

from datetime import UTC, datetime
from typing import Any

from kubernetes.client.exceptions import ApiException

from config.settings import get_settings
from kubernetes import client, config
from src.utils.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)
settings = get_settings()


class KubernetesCollector(BaseCollector):
    """
    Collector for Kubernetes cluster metrics.

    Collects:
    - Pod count and status
    - Deployment replica status
    - Resource requests and limits
    - HPA current/desired replicas
    - Node capacity (optional)

    Can connect via:
    - In-cluster config (when running inside K8s)
    - Kubeconfig file (when running locally)
    """

    def __init__(
        self,
        service_name: str = "default",
        namespace: str | None = None,
        deployment_name: str | None = None,
        in_cluster: bool | None = None,
        kubeconfig_path: str | None = None,
        collection_interval: float = 30.0,
    ) -> None:
        """
        Initialize Kubernetes collector.

        Args:
            service_name: Name for metrics labeling
            namespace: K8s namespace to monitor (defaults to settings)
            deployment_name: Specific deployment to monitor (defaults to settings)
            in_cluster: Whether running inside K8s (defaults to settings)
            kubeconfig_path: Path to kubeconfig file
            collection_interval: Seconds between collections
        """
        super().__init__(
            name=f"kubernetes-{service_name}",
            collection_interval=collection_interval,
        )

        self.service_name = service_name
        self.namespace = namespace or settings.kubernetes.namespace
        self.deployment_name = deployment_name or settings.kubernetes.deployment_name
        self.in_cluster = in_cluster if in_cluster is not None else settings.kubernetes.in_cluster
        self.kubeconfig_path = kubeconfig_path or settings.kubernetes.config_path

        # K8s API clients (initialized on first use)
        self._core_api: client.CoreV1Api | None = None
        self._apps_api: client.AppsV1Api | None = None
        self._autoscaling_api: client.AutoscalingV1Api | None = None
        self._initialized = False

    def _init_client(self) -> None:
        """Initialize Kubernetes client."""
        if self._initialized:
            return

        try:
            if self.in_cluster:
                # Running inside Kubernetes
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config")
            else:
                # Running locally with kubeconfig
                config.load_kube_config(config_file=self.kubeconfig_path)
                logger.info(
                    "Loaded kubeconfig",
                    path=self.kubeconfig_path or "~/.kube/config",
                )

            self._core_api = client.CoreV1Api()
            self._apps_api = client.AppsV1Api()
            self._autoscaling_api = client.AutoscalingV1Api()
            self._initialized = True

        except Exception as e:
            logger.error("Failed to initialize Kubernetes client", error=str(e))
            raise

    async def collect(self) -> list[dict[str, Any]]:
        """
        Collect all Kubernetes metrics.

        Returns:
            List of metric dictionaries
        """
        self._init_client()

        metrics = []
        timestamp = datetime.now(UTC)

        # Collect pod metrics
        pod_metrics = await self._collect_pod_metrics(timestamp)
        metrics.extend(pod_metrics)

        # Collect deployment metrics
        deployment_metrics = await self._collect_deployment_metrics(timestamp)
        metrics.extend(deployment_metrics)

        # Collect HPA metrics
        hpa_metrics = await self._collect_hpa_metrics(timestamp)
        metrics.extend(hpa_metrics)

        logger.info(
            "Kubernetes collection complete",
            namespace=self.namespace,
            total_metrics=len(metrics),
        )

        return metrics

    async def _collect_pod_metrics(self, timestamp: datetime) -> list[dict[str, Any]]:
        """Collect pod-related metrics."""
        metrics = []

        try:
            # List pods in namespace
            pods = self._core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={self.deployment_name}" if self.deployment_name else None,
            )

            # Count pods by phase
            phase_counts: dict[str, int] = {
                "Running": 0,
                "Pending": 0,
                "Succeeded": 0,
                "Failed": 0,
                "Unknown": 0,
            }

            total_cpu_requests = 0.0
            total_memory_requests = 0.0
            total_cpu_limits = 0.0
            total_memory_limits = 0.0

            for pod in pods.items:
                phase = pod.status.phase or "Unknown"
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

                # Sum resource requests/limits
                if pod.spec.containers:
                    for container in pod.spec.containers:
                        resources = container.resources
                        if resources:
                            if resources.requests:
                                total_cpu_requests += self._parse_cpu(
                                    resources.requests.get("cpu", "0")
                                )
                                total_memory_requests += self._parse_memory(
                                    resources.requests.get("memory", "0")
                                )
                            if resources.limits:
                                total_cpu_limits += self._parse_cpu(
                                    resources.limits.get("cpu", "0")
                                )
                                total_memory_limits += self._parse_memory(
                                    resources.limits.get("memory", "0")
                                )

            # Add pod count metrics
            metrics.append({
                "timestamp": timestamp,
                "service_name": self.service_name,
                "metric_name": "pod_count_total",
                "value": float(len(pods.items)),
                "labels": {"namespace": self.namespace},
            })

            for phase, count in phase_counts.items():
                metrics.append({
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": f"pod_count_{phase.lower()}",
                    "value": float(count),
                    "labels": {"namespace": self.namespace, "phase": phase},
                })

            # Add resource metrics
            metrics.append({
                "timestamp": timestamp,
                "service_name": self.service_name,
                "metric_name": "total_cpu_requests_cores",
                "value": total_cpu_requests,
                "labels": {"namespace": self.namespace},
            })

            metrics.append({
                "timestamp": timestamp,
                "service_name": self.service_name,
                "metric_name": "total_memory_requests_bytes",
                "value": total_memory_requests,
                "labels": {"namespace": self.namespace},
            })

        except ApiException as e:
            logger.error("Failed to collect pod metrics", error=str(e))

        return metrics

    async def _collect_deployment_metrics(self, timestamp: datetime) -> list[dict[str, Any]]:
        """Collect deployment-related metrics."""
        metrics = []

        if not self.deployment_name:
            return metrics

        try:
            deployment = self._apps_api.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
            )

            spec_replicas = deployment.spec.replicas or 0
            status = deployment.status

            metrics.extend([
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "deployment_replicas_desired",
                    "value": float(spec_replicas),
                    "labels": {
                        "namespace": self.namespace,
                        "deployment": self.deployment_name,
                    },
                },
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "deployment_replicas_available",
                    "value": float(status.available_replicas or 0),
                    "labels": {
                        "namespace": self.namespace,
                        "deployment": self.deployment_name,
                    },
                },
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "deployment_replicas_ready",
                    "value": float(status.ready_replicas or 0),
                    "labels": {
                        "namespace": self.namespace,
                        "deployment": self.deployment_name,
                    },
                },
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "deployment_replicas_updated",
                    "value": float(status.updated_replicas or 0),
                    "labels": {
                        "namespace": self.namespace,
                        "deployment": self.deployment_name,
                    },
                },
            ])

        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    "Deployment not found",
                    deployment=self.deployment_name,
                    namespace=self.namespace,
                )
            else:
                logger.error("Failed to collect deployment metrics", error=str(e))

        return metrics

    async def _collect_hpa_metrics(self, timestamp: datetime) -> list[dict[str, Any]]:
        """Collect HPA (Horizontal Pod Autoscaler) metrics."""
        metrics = []

        hpa_name = settings.kubernetes.hpa_name or self.deployment_name
        if not hpa_name:
            return metrics

        try:
            hpa = self._autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name=hpa_name,
                namespace=self.namespace,
            )

            metrics.extend([
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "hpa_min_replicas",
                    "value": float(hpa.spec.min_replicas or 1),
                    "labels": {"namespace": self.namespace, "hpa": hpa_name},
                },
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "hpa_max_replicas",
                    "value": float(hpa.spec.max_replicas),
                    "labels": {"namespace": self.namespace, "hpa": hpa_name},
                },
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "hpa_current_replicas",
                    "value": float(hpa.status.current_replicas or 0),
                    "labels": {"namespace": self.namespace, "hpa": hpa_name},
                },
                {
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "hpa_desired_replicas",
                    "value": float(hpa.status.desired_replicas or 0),
                    "labels": {"namespace": self.namespace, "hpa": hpa_name},
                },
            ])

            # Add current CPU utilization if available
            if hpa.status.current_cpu_utilization_percentage is not None:
                metrics.append({
                    "timestamp": timestamp,
                    "service_name": self.service_name,
                    "metric_name": "hpa_cpu_utilization_percentage",
                    "value": float(hpa.status.current_cpu_utilization_percentage),
                    "labels": {"namespace": self.namespace, "hpa": hpa_name},
                })

        except ApiException as e:
            if e.status == 404:
                logger.debug("HPA not found", hpa=hpa_name, namespace=self.namespace)
            else:
                logger.error("Failed to collect HPA metrics", error=str(e))

        return metrics

    def _parse_cpu(self, cpu_str: str) -> float:
        """
        Parse Kubernetes CPU string to cores.

        Examples:
            "100m" -> 0.1
            "1" -> 1.0
            "2.5" -> 2.5
        """
        if not cpu_str:
            return 0.0

        cpu_str = str(cpu_str)
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)

    def _parse_memory(self, mem_str: str) -> float:
        """
        Parse Kubernetes memory string to bytes.

        Examples:
            "128Mi" -> 134217728
            "1Gi" -> 1073741824
            "1000000" -> 1000000
        """
        if not mem_str:
            return 0.0

        mem_str = str(mem_str)

        units = {
            "Ki": 1024,
            "Mi": 1024**2,
            "Gi": 1024**3,
            "Ti": 1024**4,
            "K": 1000,
            "M": 1000**2,
            "G": 1000**3,
            "T": 1000**4,
        }

        for suffix, multiplier in units.items():
            if mem_str.endswith(suffix):
                return float(mem_str[: -len(suffix)]) * multiplier

        return float(mem_str)

    async def get_current_replicas(self) -> int:
        """Get current replica count for the deployment."""
        self._init_client()

        if not self.deployment_name:
            return 0

        try:
            deployment = self._apps_api.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
            )
            return deployment.status.ready_replicas or 0
        except ApiException:
            return 0

    async def health_check(self) -> bool:
        """Check if Kubernetes API is reachable."""
        try:
            self._init_client()
            self._core_api.get_api_versions()
            return True
        except Exception:
            return False
