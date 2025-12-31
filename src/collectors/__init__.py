"""
Data collectors for the predictive scaling system.

Collectors gather metrics from various sources:
- Prometheus: Application metrics (request rate, latency, etc.)
- Kubernetes: Cluster state (pods, replicas, HPA)
- Business: Business events (campaigns, launches)
- External: External signals (social media, news)
"""

from .base import BaseCollector
from .business import BusinessContextCollector
from .external import ExternalSignalsCollector
from .kubernetes import KubernetesCollector
from .prometheus import PrometheusCollector

__all__ = [
    "BaseCollector",
    "PrometheusCollector",
    "KubernetesCollector",
    "BusinessContextCollector",
    "ExternalSignalsCollector",
]
