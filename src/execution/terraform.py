"""
Terraform Executor for infrastructure scaling.

Responsibilities:
- Generate tfvars from scaling decisions
- Run terraform plan and parse output
- Safety check for destructive changes
- Run terraform apply with auto-approve
- Parse terraform output for new resource IDs
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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


@dataclass
class TerraformConfig:
    """Configuration for Terraform executor."""

    # Terraform paths
    terraform_dir: Path = Path("terraform")
    terraform_binary: str = "terraform"

    # Workspace settings
    workspace: str = "default"

    # Variable settings
    var_file: str = "terraform.tfvars"

    # Execution settings
    auto_approve: bool = False
    plan_timeout_seconds: float = 300.0
    apply_timeout_seconds: float = 600.0

    # Safety settings
    allow_destroy: bool = False
    max_resources_to_change: int = 10

    # State settings
    backend_config: dict[str, str] = field(default_factory=dict)


@dataclass
class TerraformPlanResult:
    """Result of terraform plan."""

    success: bool
    has_changes: bool
    resources_to_add: int
    resources_to_change: int
    resources_to_destroy: int
    plan_output: str
    plan_file: str | None = None
    error_message: str | None = None

    @property
    def is_safe(self) -> bool:
        """Check if plan is safe to apply."""
        return self.success and not self.has_destroy_only

    @property
    def has_destroy_only(self) -> bool:
        """Check if plan only destroys resources."""
        return (
            self.resources_to_destroy > 0
            and self.resources_to_add == 0
            and self.resources_to_change == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "has_changes": self.has_changes,
            "resources_to_add": self.resources_to_add,
            "resources_to_change": self.resources_to_change,
            "resources_to_destroy": self.resources_to_destroy,
            "is_safe": self.is_safe,
            "error_message": self.error_message,
        }


@dataclass
class TerraformApplyResult:
    """Result of terraform apply."""

    success: bool
    resources_created: list[str]
    resources_updated: list[str]
    resources_destroyed: list[str]
    outputs: dict[str, Any]
    apply_output: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "resources_created": self.resources_created,
            "resources_updated": self.resources_updated,
            "resources_destroyed": self.resources_destroyed,
            "outputs": self.outputs,
            "error_message": self.error_message,
        }


class TerraformExecutor(BaseExecutor):
    """
    Terraform executor for infrastructure scaling.

    Supports:
    - Plan and apply workflows
    - Variable file generation
    - Safety checks for destructive changes
    - Output parsing
    """

    def __init__(self, tf_config: TerraformConfig | None = None) -> None:
        """
        Initialize Terraform executor.

        Args:
            tf_config: Terraform configuration
        """
        super().__init__(ExecutorType.TERRAFORM)
        self.tf_config = tf_config or TerraformConfig()
        self._initialized = False
        self._last_plan: TerraformPlanResult | None = None
        self._current_state_cache: dict[str, Any] | None = None

    async def initialize(self) -> bool:
        """
        Initialize Terraform (run terraform init).

        Returns:
            True if initialization succeeded
        """
        try:
            result = await self._run_terraform(["init", "-no-color"])
            self._initialized = result.returncode == 0

            if self._initialized:
                logger.info("Terraform initialized", dir=str(self.tf_config.terraform_dir))
            else:
                logger.error("Terraform init failed", output=result.stderr)

            return self._initialized

        except Exception as e:
            logger.error("Terraform initialization error", error=str(e))
            return False

    async def scale(self, action: ScalingAction) -> ExecutionResult:
        """
        Scale infrastructure using Terraform.

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
            logger.warning("Failed to get current state", error=str(e))
            previous_state = None

        try:
            # Initialize if needed
            if not self._initialized:
                if not await self.initialize():
                    return ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        previous_state=previous_state,
                        error_message="Terraform initialization failed",
                    )

            # Generate tfvars
            await self._write_tfvars(action)

            # Run terraform plan
            plan_result = await self._plan()

            if not plan_result.success:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    previous_state=previous_state,
                    error_message=f"Terraform plan failed: {plan_result.error_message}",
                )

            # Safety checks
            if not self._is_plan_safe(plan_result):
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    previous_state=previous_state,
                    error_message="Plan failed safety checks",
                    metadata={"plan": plan_result.to_dict()},
                )

            # Apply if auto-approve or no changes
            if not plan_result.has_changes:
                logger.info("No changes to apply")
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    previous_state=previous_state,
                    current_state=previous_state,
                    metadata={"no_changes": True},
                )

            if not self.tf_config.auto_approve:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.PENDING,
                    started_at=started_at,
                    previous_state=previous_state,
                    metadata={"plan": plan_result.to_dict(), "requires_approval": True},
                )

            # Run terraform apply
            apply_result = await self._apply(plan_result.plan_file)

            if not apply_result.success:
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    previous_state=previous_state,
                    error_message=f"Terraform apply failed: {apply_result.error_message}",
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
                metadata={
                    "plan": plan_result.to_dict(),
                    "apply": apply_result.to_dict(),
                },
            )
            self.record_execution(result)

            logger.info(
                "Terraform scale completed",
                created=len(apply_result.resources_created),
                updated=len(apply_result.resources_updated),
                destroyed=len(apply_result.resources_destroyed),
            )

            return result

        except Exception as e:
            logger.error("Terraform scale failed", error=str(e))
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

    async def _write_tfvars(self, action: ScalingAction) -> None:
        """Generate tfvars file from scaling action."""
        tfvars = {
            "instance_count": action.target_count,
        }

        if action.instance_type:
            tfvars["instance_type"] = action.instance_type

        # Add any metadata as variables
        tfvars.update(action.metadata)

        var_file_path = self.tf_config.terraform_dir / self.tf_config.var_file

        with open(var_file_path, "w") as f:
            for key, value in tfvars.items():
                if isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                elif isinstance(value, bool):
                    f.write(f"{key} = {str(value).lower()}\n")
                elif isinstance(value, (list, dict)):
                    f.write(f"{key} = {json.dumps(value)}\n")
                else:
                    f.write(f"{key} = {value}\n")

        logger.debug("Wrote tfvars", path=str(var_file_path), vars=tfvars)

    async def _plan(self) -> TerraformPlanResult:
        """Run terraform plan."""
        with tempfile.NamedTemporaryFile(suffix=".tfplan", delete=False) as plan_file:
            plan_path = plan_file.name

        try:
            result = await self._run_terraform(
                [
                    "plan",
                    "-no-color",
                    "-detailed-exitcode",
                    f"-out={plan_path}",
                    f"-var-file={self.tf_config.var_file}",
                ],
                timeout=self.tf_config.plan_timeout_seconds,
            )

            # Exit codes: 0 = no changes, 1 = error, 2 = changes present
            has_changes = result.returncode == 2
            success = result.returncode in (0, 2)

            # Parse plan output for resource counts
            add_count, change_count, destroy_count = self._parse_plan_output(result.stdout)

            self._last_plan = TerraformPlanResult(
                success=success,
                has_changes=has_changes,
                resources_to_add=add_count,
                resources_to_change=change_count,
                resources_to_destroy=destroy_count,
                plan_output=result.stdout,
                plan_file=plan_path if has_changes else None,
                error_message=result.stderr if not success else None,
            )

            return self._last_plan

        except Exception as e:
            return TerraformPlanResult(
                success=False,
                has_changes=False,
                resources_to_add=0,
                resources_to_change=0,
                resources_to_destroy=0,
                plan_output="",
                error_message=str(e),
            )

    def _parse_plan_output(self, output: str) -> tuple[int, int, int]:
        """Parse plan output to get resource counts."""
        add_count = 0
        change_count = 0
        destroy_count = 0

        for line in output.split("\n"):
            if "to add" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "to" and i > 0:
                            add_count = int(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass

            if "to change" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "to" and i > 0 and "change" in parts[i + 1]:
                            change_count = int(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass

            if "to destroy" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "to" and i > 0:
                            destroy_count = int(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass

        return add_count, change_count, destroy_count

    def _is_plan_safe(self, plan: TerraformPlanResult) -> bool:
        """Check if plan is safe to apply."""
        # Check for destroy-only plans
        if plan.has_destroy_only and not self.tf_config.allow_destroy:
            logger.warning("Plan would only destroy resources")
            return False

        # Check for too many changes
        total_changes = (
            plan.resources_to_add
            + plan.resources_to_change
            + plan.resources_to_destroy
        )
        if total_changes > self.tf_config.max_resources_to_change:
            logger.warning(
                "Plan affects too many resources",
                total=total_changes,
                max=self.tf_config.max_resources_to_change,
            )
            return False

        return True

    async def _apply(self, plan_file: str | None = None) -> TerraformApplyResult:
        """Run terraform apply."""
        args = ["apply", "-no-color", "-auto-approve"]

        if plan_file:
            args.append(plan_file)
        else:
            args.append(f"-var-file={self.tf_config.var_file}")

        try:
            result = await self._run_terraform(
                args,
                timeout=self.tf_config.apply_timeout_seconds,
            )

            success = result.returncode == 0

            # Parse output for resource changes
            created, updated, destroyed = self._parse_apply_output(result.stdout)

            # Get outputs
            outputs = await self._get_outputs()

            return TerraformApplyResult(
                success=success,
                resources_created=created,
                resources_updated=updated,
                resources_destroyed=destroyed,
                outputs=outputs,
                apply_output=result.stdout,
                error_message=result.stderr if not success else None,
            )

        except Exception as e:
            return TerraformApplyResult(
                success=False,
                resources_created=[],
                resources_updated=[],
                resources_destroyed=[],
                outputs={},
                apply_output="",
                error_message=str(e),
            )

    def _parse_apply_output(
        self, output: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Parse apply output for resource changes."""
        created = []
        updated = []
        destroyed = []

        for line in output.split("\n"):
            if ": Creating..." in line or ": Creation complete" in line:
                resource = line.split(":")[0].strip()
                if resource and resource not in created:
                    created.append(resource)
            elif ": Modifying..." in line or ": Modifications complete" in line:
                resource = line.split(":")[0].strip()
                if resource and resource not in updated:
                    updated.append(resource)
            elif ": Destroying..." in line or ": Destruction complete" in line:
                resource = line.split(":")[0].strip()
                if resource and resource not in destroyed:
                    destroyed.append(resource)

        return created, updated, destroyed

    async def _get_outputs(self) -> dict[str, Any]:
        """Get terraform outputs."""
        try:
            result = await self._run_terraform(["output", "-json"])
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.warning("Failed to get outputs", error=str(e))
        return {}

    async def _run_terraform(
        self,
        args: list[str],
        timeout: float | None = None,
    ) -> asyncio.subprocess.Process:
        """Run a terraform command."""
        cmd = [self.tf_config.terraform_binary] + args

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.tf_config.terraform_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout or 300.0,
            )

            # Create a result-like object
            class Result:
                pass

            result = Result()
            result.returncode = process.returncode
            result.stdout = stdout.decode() if stdout else ""
            result.stderr = stderr.decode() if stderr else ""

            return result

        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Terraform command timed out: {' '.join(args)}")

    async def rollback(self, action_id: str) -> RollbackResult:
        """Rollback by applying previous state."""
        previous_state = self.get_rollback_state(action_id)

        if previous_state is None:
            return RollbackResult(
                action_id=action_id,
                success=False,
                previous_state=await self.get_current_state(),
                error_message="No rollback state found",
            )

        try:
            # Create a rollback action with previous count
            rollback_action = ScalingAction(
                action_id=f"{action_id}_rollback",
                target_count=previous_state.instance_count,
                current_count=(await self.get_current_state()).instance_count,
            )

            # Execute the rollback
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
        """Verify terraform state matches expected."""
        checks_passed = []
        checks_failed = []

        try:
            current_state = await self.get_current_state()

            # Check instance count
            if current_state.instance_count == action.target_count:
                checks_passed.append("instance_count")
            else:
                checks_failed.append(
                    f"instance_count (expected {action.target_count}, got {current_state.instance_count})"
                )

            # Check for healthy state
            if current_state.is_healthy:
                checks_passed.append("health")
            else:
                checks_failed.append("health")

            return VerificationResult(
                action_id=action.action_id,
                verified=len(checks_failed) == 0,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                target_count=action.target_count,
                actual_count=current_state.instance_count,
                healthy_count=current_state.healthy_count,
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
        """Get current state from terraform."""
        try:
            # Get terraform state
            result = await self._run_terraform(["show", "-json"])

            if result.returncode == 0:
                state_data = json.loads(result.stdout)

                # Parse instance count from state
                instance_count = self._parse_state_instance_count(state_data)

                return InfrastructureState(
                    executor_type=self.executor_type,
                    timestamp=datetime.now(timezone.utc),
                    instance_count=instance_count,
                    healthy_count=instance_count,
                    metadata={"state_version": state_data.get("format_version")},
                )

        except Exception as e:
            logger.warning("Failed to get terraform state", error=str(e))

        return InfrastructureState(
            executor_type=self.executor_type,
            timestamp=datetime.now(timezone.utc),
            instance_count=0,
            metadata={"error": "Failed to read state"},
        )

    def _parse_state_instance_count(self, state_data: dict) -> int:
        """Parse instance count from terraform state."""
        # This is simplified - real implementation would parse based on resource type
        try:
            values = state_data.get("values", {})
            root_module = values.get("root_module", {})
            resources = root_module.get("resources", [])

            # Look for ASG or similar resources
            for resource in resources:
                if "autoscaling_group" in resource.get("type", "").lower():
                    return resource.get("values", {}).get("desired_capacity", 0)
                if "instance" in resource.get("type", "").lower():
                    return len([r for r in resources if "instance" in r.get("type", "")])

            return 0

        except Exception:
            return 0
