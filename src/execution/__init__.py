"""
Execution Layer for predictive scaling system.

This module provides infrastructure executors for applying scaling actions
to various backends (Kubernetes, Terraform, AWS), along with verification
and rollback capabilities.
"""

from src.execution.base import (
    BaseExecutor,
    ExecutionResult,
    ExecutionStatus,
    ExecutorType,
    InfrastructureState,
    MockExecutor,
    RollbackResult,
    ScalingAction,
    VerificationResult,
)
from src.execution.rollback import (
    RollbackManager,
    RollbackPolicy,
    RollbackReason,
    RollbackRecord,
    RollbackRequest,
    RollbackStrategy,
)
from src.execution.verification import (
    VerificationCheck,
    VerificationCheckType,
    VerificationConfig,
    VerificationSession,
    VerificationStatus,
    VerificationSystem,
)

# Optional imports - these may not be available in all environments
try:
    from src.execution.kubernetes import (
        KubernetesConfig,
        KubernetesExecutor,
    )
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False
    KubernetesConfig = None
    KubernetesExecutor = None

try:
    from src.execution.aws import (
        AWSConfig,
        AWSExecutor,
        AWSResourceType,
    )
    HAS_AWS = True
except ImportError:
    HAS_AWS = False
    AWSConfig = None
    AWSExecutor = None
    AWSResourceType = None

try:
    from src.execution.terraform import (
        TerraformConfig,
        TerraformExecutor,
    )
    HAS_TERRAFORM = True
except ImportError:
    HAS_TERRAFORM = False
    TerraformConfig = None
    TerraformExecutor = None


__all__ = [
    # Base
    "BaseExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutorType",
    "InfrastructureState",
    "MockExecutor",
    "RollbackResult",
    "ScalingAction",
    "VerificationResult",
    # Verification
    "VerificationCheck",
    "VerificationCheckType",
    "VerificationConfig",
    "VerificationSession",
    "VerificationStatus",
    "VerificationSystem",
    # Rollback
    "RollbackManager",
    "RollbackPolicy",
    "RollbackReason",
    "RollbackRecord",
    "RollbackRequest",
    "RollbackStrategy",
    # Kubernetes (optional)
    "KubernetesConfig",
    "KubernetesExecutor",
    "HAS_KUBERNETES",
    # AWS (optional)
    "AWSConfig",
    "AWSExecutor",
    "AWSResourceType",
    "HAS_AWS",
    # Terraform (optional)
    "TerraformConfig",
    "TerraformExecutor",
    "HAS_TERRAFORM",
]
