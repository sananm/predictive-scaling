"""
Unit tests for Phase 8: Execution Layer.

Tests cover:
- Base executor and MockExecutor
- Scaling actions and execution results
- Verification system
- Rollback system and manager
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

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
from src.execution.verification import (
    VerificationCheck,
    VerificationCheckType,
    VerificationConfig,
    VerificationSession,
    VerificationStatus,
    VerificationSystem,
)
from src.execution.rollback import (
    RollbackManager,
    RollbackPolicy,
    RollbackReason,
    RollbackRecord,
    RollbackRequest,
    RollbackStrategy,
)


# =============================================================================
# Infrastructure State Tests
# =============================================================================


class TestInfrastructureState:
    """Tests for InfrastructureState dataclass."""

    def test_create_state(self):
        """Test creating infrastructure state."""
        state = InfrastructureState(
            executor_type=ExecutorType.MOCK,
            timestamp=datetime.now(timezone.utc),
            instance_count=5,
            instance_type="t3.medium",
            healthy_count=5,
            unhealthy_count=0,
            pending_count=0,
        )

        assert state.instance_count == 5
        assert state.is_healthy is True

    def test_unhealthy_state(self):
        """Test unhealthy infrastructure state."""
        state = InfrastructureState(
            executor_type=ExecutorType.MOCK,
            timestamp=datetime.now(timezone.utc),
            instance_count=5,
            healthy_count=3,
            unhealthy_count=2,
            pending_count=0,
        )

        assert state.is_healthy is False

    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        state = InfrastructureState(
            executor_type=ExecutorType.KUBERNETES,
            timestamp=datetime.now(timezone.utc),
            instance_count=10,
            healthy_count=10,
        )

        data = state.to_dict()
        assert data["executor_type"] == "kubernetes"
        assert data["instance_count"] == 10
        assert data["is_healthy"] is True


# =============================================================================
# Scaling Action Tests
# =============================================================================


class TestScalingAction:
    """Tests for ScalingAction dataclass."""

    def test_scale_up_action(self):
        """Test scale-up action properties."""
        action = ScalingAction(
            action_id="test-001",
            target_count=10,
            current_count=5,
        )

        assert action.is_scale_up is True
        assert action.is_scale_down is False
        assert action.scale_delta == 5

    def test_scale_down_action(self):
        """Test scale-down action properties."""
        action = ScalingAction(
            action_id="test-002",
            target_count=3,
            current_count=8,
        )

        assert action.is_scale_up is False
        assert action.is_scale_down is True
        assert action.scale_delta == -5

    def test_no_change_action(self):
        """Test action with no change."""
        action = ScalingAction(
            action_id="test-003",
            target_count=5,
            current_count=5,
        )

        assert action.is_scale_up is False
        assert action.is_scale_down is False
        assert action.scale_delta == 0

    def test_action_with_metadata(self):
        """Test action with metadata."""
        action = ScalingAction(
            action_id="test-004",
            target_count=10,
            current_count=5,
            instance_type="t3.large",
            strategy="gradual",
            metadata={"service_name": "api", "reason": "predicted_load"},
        )

        data = action.to_dict()
        assert data["instance_type"] == "t3.large"
        assert data["strategy"] == "gradual"
        assert data["metadata"]["service_name"] == "api"


# =============================================================================
# Execution Result Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            action_id="test-001",
            status=ExecutionStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
        )

        assert result.is_success is True
        assert result.duration_seconds == 300.0

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            action_id="test-002",
            status=ExecutionStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            error_message="Connection timeout",
        )

        assert result.is_success is False
        assert result.error_message == "Connection timeout"

    def test_verified_result_is_success(self):
        """Test that verified status counts as success."""
        result = ExecutionResult(
            action_id="test-003",
            status=ExecutionStatus.VERIFIED,
            started_at=datetime.now(timezone.utc),
        )

        assert result.is_success is True


# =============================================================================
# MockExecutor Tests
# =============================================================================


class TestMockExecutor:
    """Tests for MockExecutor."""

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        return MockExecutor()

    @pytest.mark.asyncio
    async def test_get_current_state(self, executor):
        """Test getting current state."""
        state = await executor.get_current_state()

        assert state.executor_type == ExecutorType.MOCK
        assert state.instance_count == 3  # Default
        assert state.is_healthy is True

    @pytest.mark.asyncio
    async def test_scale_up(self, executor):
        """Test scaling up."""
        action = ScalingAction(
            action_id="scale-up-001",
            target_count=5,
            current_count=3,
        )

        result = await executor.scale(action)

        assert result.is_success is True
        assert result.status == ExecutionStatus.COMPLETED
        assert result.previous_state.instance_count == 3
        assert result.current_state.instance_count == 5

    @pytest.mark.asyncio
    async def test_scale_down(self, executor):
        """Test scaling down."""
        executor.set_current_count(10)

        action = ScalingAction(
            action_id="scale-down-001",
            target_count=5,
            current_count=10,
        )

        result = await executor.scale(action)

        assert result.is_success is True
        assert result.current_state.instance_count == 5

    @pytest.mark.asyncio
    async def test_scale_failure(self, executor):
        """Test scale failure."""
        executor.set_should_fail(True)

        action = ScalingAction(
            action_id="fail-001",
            target_count=5,
            current_count=3,
        )

        result = await executor.scale(action)

        assert result.is_success is False
        assert result.status == ExecutionStatus.FAILED
        assert result.error_message == "Mock failure"

    @pytest.mark.asyncio
    async def test_rollback(self, executor):
        """Test rollback operation."""
        # First scale
        action = ScalingAction(
            action_id="rollback-test-001",
            target_count=10,
            current_count=3,
        )
        await executor.scale(action)

        # Then rollback
        rollback_result = await executor.rollback(action.action_id)

        assert rollback_result.success is True
        assert rollback_result.restored_state.instance_count == 3

    @pytest.mark.asyncio
    async def test_rollback_no_state(self, executor):
        """Test rollback with no stored state."""
        result = await executor.rollback("nonexistent-action")

        assert result.success is False
        assert "No rollback state found" in result.error_message

    @pytest.mark.asyncio
    async def test_verify(self, executor):
        """Test verification."""
        action = ScalingAction(
            action_id="verify-001",
            target_count=3,  # Matches default
            current_count=3,
        )

        result = await executor.verify(action)

        assert result.verified is True
        assert "instance_count" in result.checks_passed
        assert "health" in result.checks_passed

    @pytest.mark.asyncio
    async def test_verify_count_mismatch(self, executor):
        """Test verification with count mismatch."""
        action = ScalingAction(
            action_id="verify-002",
            target_count=10,  # Doesn't match current 3
            current_count=3,
        )

        result = await executor.verify(action)

        assert result.verified is False
        assert "instance_count" in result.checks_failed

    def test_execution_history(self, executor):
        """Test execution history tracking."""
        asyncio.run(self._run_multiple_operations(executor))

        history = executor.get_execution_history(limit=5)
        assert len(history) == 3

    async def _run_multiple_operations(self, executor):
        """Run multiple operations for history test."""
        for i in range(3):
            action = ScalingAction(
                action_id=f"history-{i}",
                target_count=i + 1,
                current_count=i,
            )
            await executor.scale(action)

    def test_executor_stats(self, executor):
        """Test executor statistics."""
        asyncio.run(self._run_success_and_failure(executor))

        stats = executor.get_stats()
        assert stats["total_executions"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1

    async def _run_success_and_failure(self, executor):
        """Run success and failure for stats test."""
        # Success
        action1 = ScalingAction(
            action_id="stats-1",
            target_count=5,
            current_count=3,
        )
        await executor.scale(action1)

        # Failure
        executor.set_should_fail(True)
        action2 = ScalingAction(
            action_id="stats-2",
            target_count=10,
            current_count=5,
        )
        await executor.scale(action2)


# =============================================================================
# Verification System Tests
# =============================================================================


class TestVerificationCheck:
    """Tests for VerificationCheck dataclass."""

    def test_create_check(self):
        """Test creating a verification check."""
        check = VerificationCheck(
            check_type=VerificationCheckType.INSTANCE_COUNT,
            name="instance_count",
            passed=True,
            message="Instance count: 5 (target: 5)",
            value=5,
            threshold=5,
        )

        assert check.passed is True
        assert check.check_type == VerificationCheckType.INSTANCE_COUNT

    def test_check_to_dict(self):
        """Test converting check to dictionary."""
        check = VerificationCheck(
            check_type=VerificationCheckType.LATENCY_CHECK,
            name="latency",
            passed=False,
            message="Latency p99=600ms exceeds threshold",
            value={"p50": 100, "p99": 600},
            threshold={"p50": 200, "p99": 500},
        )

        data = check.to_dict()
        assert data["check_type"] == "latency_check"
        assert data["passed"] is False
        assert data["value"]["p99"] == 600


class TestVerificationConfig:
    """Tests for VerificationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VerificationConfig()

        assert config.timeout_seconds == 300.0
        assert config.poll_interval_seconds == 10.0
        assert config.min_healthy_percentage == 0.95
        assert config.max_error_rate == 0.01
        assert config.auto_rollback_on_failure is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = VerificationConfig(
            timeout_seconds=600.0,
            max_latency_p99_ms=1000.0,
            auto_rollback_on_failure=False,
        )

        assert config.timeout_seconds == 600.0
        assert config.max_latency_p99_ms == 1000.0
        assert config.auto_rollback_on_failure is False


class TestVerificationSession:
    """Tests for VerificationSession."""

    def test_session_passed(self):
        """Test passed verification session."""
        action = ScalingAction(
            action_id="session-001",
            target_count=5,
            current_count=3,
        )

        session = VerificationSession(
            action_id=action.action_id,
            action=action,
            config=VerificationConfig(),
            status=VerificationStatus.PASSED,
            checks=[
                VerificationCheck(
                    check_type=VerificationCheckType.INSTANCE_COUNT,
                    name="instance_count",
                    passed=True,
                    message="OK",
                ),
                VerificationCheck(
                    check_type=VerificationCheckType.HEALTH_CHECK,
                    name="health",
                    passed=True,
                    message="OK",
                ),
            ],
        )

        assert session.is_passed is True
        assert session.all_checks_passed is True

    def test_session_failed_checks(self):
        """Test session with failed checks."""
        action = ScalingAction(
            action_id="session-002",
            target_count=5,
            current_count=3,
        )

        session = VerificationSession(
            action_id=action.action_id,
            action=action,
            config=VerificationConfig(),
            status=VerificationStatus.IN_PROGRESS,
            checks=[
                VerificationCheck(
                    check_type=VerificationCheckType.INSTANCE_COUNT,
                    name="instance_count",
                    passed=True,
                    message="OK",
                ),
                VerificationCheck(
                    check_type=VerificationCheckType.HEALTH_CHECK,
                    name="health",
                    passed=False,
                    message="Unhealthy instances",
                ),
            ],
        )

        assert session.is_passed is False
        assert session.all_checks_passed is False


class TestVerificationSystem:
    """Tests for VerificationSystem."""

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        return MockExecutor()

    @pytest.fixture
    def verification_system(self, executor):
        """Create verification system."""
        config = VerificationConfig(
            timeout_seconds=5.0,  # Short for tests
            poll_interval_seconds=0.1,
            stabilization_seconds=0.1,
        )
        return VerificationSystem(executor, config)

    @pytest.mark.asyncio
    async def test_verify_success(self, verification_system, executor):
        """Test successful verification."""
        action = ScalingAction(
            action_id="verify-success-001",
            target_count=3,  # Matches mock default
            current_count=3,
        )

        session = await verification_system.verify(action)

        assert session.status == VerificationStatus.PASSED
        assert session.is_passed is True
        assert session.all_checks_passed is True

    @pytest.mark.asyncio
    async def test_verify_with_metrics_provider(self, verification_system, executor):
        """Test verification with metrics provider."""

        def metrics_provider(service_name):
            return {
                "latency_p50_ms": 100.0,
                "latency_p99_ms": 200.0,
                "error_rate": 0.005,
            }

        verification_system.set_metrics_provider(metrics_provider)

        action = ScalingAction(
            action_id="verify-metrics-001",
            target_count=3,
            current_count=3,
            metadata={"service_name": "api"},
        )

        session = await verification_system.verify(action)

        assert session.is_passed is True
        # Should have latency and error rate checks
        check_names = [c.name for c in session.checks]
        assert "latency" in check_names
        assert "error_rate" in check_names

    @pytest.mark.asyncio
    async def test_verify_custom_check(self, verification_system, executor):
        """Test verification with custom check."""

        def custom_check(action):
            return VerificationCheck(
                check_type=VerificationCheckType.CUSTOM_CHECK,
                name="custom_validation",
                passed=True,
                message="Custom validation passed",
            )

        verification_system.add_custom_check(custom_check)

        action = ScalingAction(
            action_id="verify-custom-001",
            target_count=3,
            current_count=3,
        )

        session = await verification_system.verify(action)

        check_names = [c.name for c in session.checks]
        assert "custom_validation" in check_names

    def test_get_stats(self, verification_system):
        """Test getting verification statistics."""
        stats = verification_system.get_stats()

        assert "total_verifications" in stats
        assert "success_rate" in stats


# =============================================================================
# Rollback System Tests
# =============================================================================


class TestRollbackPolicy:
    """Tests for RollbackPolicy."""

    def test_default_policy(self):
        """Test default rollback policy."""
        policy = RollbackPolicy()

        assert policy.strategy == RollbackStrategy.IMMEDIATE
        assert policy.delay_seconds == 30.0
        assert policy.max_retries == 3
        assert policy.on_verification_failure is True

    def test_custom_policy(self):
        """Test custom rollback policy."""
        policy = RollbackPolicy(
            strategy=RollbackStrategy.GRADUAL,
            gradual_steps=5,
            gradual_interval_seconds=60.0,
            require_confirmation=True,
        )

        assert policy.strategy == RollbackStrategy.GRADUAL
        assert policy.gradual_steps == 5
        assert policy.require_confirmation is True


class TestRollbackRecord:
    """Tests for RollbackRecord."""

    def test_successful_record(self):
        """Test successful rollback record."""
        state = InfrastructureState(
            executor_type=ExecutorType.MOCK,
            timestamp=datetime.now(timezone.utc),
            instance_count=5,
            healthy_count=5,
        )

        result = RollbackResult(
            action_id="rb-001",
            success=True,
            previous_state=state,
            restored_state=state,
        )

        record = RollbackRecord(
            request_id="req-001",
            action_id="rb-001",
            reason=RollbackReason.VERIFICATION_FAILED,
            strategy=RollbackStrategy.IMMEDIATE,
            result=result,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
        )

        assert record.success is True
        assert record.duration_seconds == 60.0

    def test_failed_record(self):
        """Test failed rollback record."""
        record = RollbackRecord(
            request_id="req-002",
            action_id="rb-002",
            reason=RollbackReason.MANUAL_REQUEST,
            strategy=RollbackStrategy.IMMEDIATE,
            result=None,
            error_message="No rollback state found",
        )

        assert record.success is False


class TestRollbackManager:
    """Tests for RollbackManager."""

    @pytest.fixture
    def executor(self):
        """Create mock executor."""
        return MockExecutor()

    @pytest.fixture
    def manager(self):
        """Create rollback manager."""
        return RollbackManager()

    @pytest.mark.asyncio
    async def test_immediate_rollback(self, manager, executor):
        """Test immediate rollback strategy."""
        # First scale to create rollback state
        action = ScalingAction(
            action_id="immed-rb-001",
            target_count=10,
            current_count=3,
        )
        await executor.scale(action)

        # Request rollback
        request = RollbackRequest(
            request_id="req-immed-001",
            action_id=action.action_id,
            reason=RollbackReason.VERIFICATION_FAILED,
            executor=executor,
            policy=RollbackPolicy(strategy=RollbackStrategy.IMMEDIATE),
        )

        record = await manager.request_rollback(request)

        assert record.success is True
        assert record.strategy == RollbackStrategy.IMMEDIATE

        # Verify state was restored
        state = await executor.get_current_state()
        assert state.instance_count == 3

    @pytest.mark.asyncio
    async def test_delayed_rollback(self, manager, executor):
        """Test delayed rollback strategy."""
        # First scale
        action = ScalingAction(
            action_id="delayed-rb-001",
            target_count=10,
            current_count=3,
        )
        await executor.scale(action)

        # Request delayed rollback
        request = RollbackRequest(
            request_id="req-delayed-001",
            action_id=action.action_id,
            reason=RollbackReason.LATENCY_EXCEEDED,
            executor=executor,
            policy=RollbackPolicy(
                strategy=RollbackStrategy.DELAYED,
                delay_seconds=0.1,  # Short for test
            ),
        )

        record = await manager.request_rollback(request)

        assert record.success is True
        assert record.strategy == RollbackStrategy.DELAYED

    @pytest.mark.asyncio
    async def test_gradual_rollback(self, manager, executor):
        """Test gradual rollback strategy."""
        # Scale up significantly
        executor.set_current_count(3)
        action = ScalingAction(
            action_id="gradual-rb-001",
            target_count=12,
            current_count=3,
        )
        await executor.scale(action)

        # Request gradual rollback
        request = RollbackRequest(
            request_id="req-gradual-001",
            action_id=action.action_id,
            reason=RollbackReason.ERROR_RATE_EXCEEDED,
            executor=executor,
            policy=RollbackPolicy(
                strategy=RollbackStrategy.GRADUAL,
                gradual_steps=3,
                gradual_interval_seconds=0.1,  # Short for test
            ),
        )

        record = await manager.request_rollback(request)

        assert record.success is True
        assert record.strategy == RollbackStrategy.GRADUAL
        assert record.attempts == 3  # 3 steps

        # Verify state was restored
        state = await executor.get_current_state()
        assert state.instance_count == 3

    @pytest.mark.asyncio
    async def test_retry_first_rollback_succeeds(self, manager, executor):
        """Test retry-first strategy where retry succeeds."""
        # Scale
        action = ScalingAction(
            action_id="retry-rb-001",
            target_count=5,
            current_count=3,
        )
        await executor.scale(action)

        # Request with retry-first (retry will succeed since executor works)
        request = RollbackRequest(
            request_id="req-retry-001",
            action_id=action.action_id,
            reason=RollbackReason.VERIFICATION_FAILED,
            executor=executor,
            action=action,
            policy=RollbackPolicy(
                strategy=RollbackStrategy.RETRY_FIRST,
                max_retries=2,
                retry_delay_seconds=0.1,
            ),
        )

        record = await manager.request_rollback(request)

        assert record.success is True
        assert record.metadata.get("retry_succeeded") is True

    @pytest.mark.asyncio
    async def test_rollback_callback(self, manager, executor):
        """Test rollback callbacks."""
        events = []

        async def callback(request_id, event_type, data):
            events.append((request_id, event_type, data))

        manager.add_callback(callback)

        # Scale and rollback
        action = ScalingAction(
            action_id="callback-rb-001",
            target_count=10,
            current_count=3,
        )
        await executor.scale(action)

        request = RollbackRequest(
            request_id="req-callback-001",
            action_id=action.action_id,
            reason=RollbackReason.MANUAL_REQUEST,
            executor=executor,
        )

        await manager.request_rollback(request)

        # Should have received callbacks
        event_types = [e[1] for e in events]
        assert "rollback_requested" in event_types
        assert "rollback_completed" in event_types

    def test_get_stats(self, manager):
        """Test getting rollback statistics."""
        stats = manager.get_stats()

        assert "total_rollbacks" in stats
        assert "success_rate" in stats
        assert "by_reason" in stats
        assert "by_strategy" in stats

    def test_get_records_for_action(self, manager, executor):
        """Test getting records for specific action."""
        asyncio.run(self._create_multiple_rollbacks(manager, executor))

        records = manager.get_records_for_action("multi-rb-001")
        assert len(records) >= 1

    async def _create_multiple_rollbacks(self, manager, executor):
        """Create multiple rollback records."""
        action = ScalingAction(
            action_id="multi-rb-001",
            target_count=10,
            current_count=3,
        )
        await executor.scale(action)

        request = RollbackRequest(
            request_id="req-multi-001",
            action_id=action.action_id,
            reason=RollbackReason.MANUAL_REQUEST,
            executor=executor,
        )
        await manager.request_rollback(request)


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutionIntegration:
    """Integration tests for execution layer."""

    @pytest.mark.asyncio
    async def test_scale_verify_flow(self):
        """Test complete scale and verify flow."""
        executor = MockExecutor()
        verification = VerificationSystem(
            executor,
            VerificationConfig(
                timeout_seconds=5.0,
                poll_interval_seconds=0.1,
                stabilization_seconds=0.1,
            ),
        )

        # Scale
        action = ScalingAction(
            action_id="integration-001",
            target_count=5,
            current_count=3,
        )
        scale_result = await executor.scale(action)
        assert scale_result.is_success is True

        # Verify
        session = await verification.verify(action)
        assert session.is_passed is True

    @pytest.mark.asyncio
    async def test_scale_fail_rollback_flow(self):
        """Test scale failure with automatic rollback."""
        executor = MockExecutor()
        manager = RollbackManager()

        # Scale
        action = ScalingAction(
            action_id="integration-fail-001",
            target_count=10,
            current_count=3,
        )
        await executor.scale(action)

        # Simulate verification failure and trigger rollback
        request = RollbackRequest(
            request_id="req-int-fail-001",
            action_id=action.action_id,
            reason=RollbackReason.VERIFICATION_FAILED,
            executor=executor,
        )

        record = await manager.request_rollback(request)

        assert record.success is True

        # State should be restored
        state = await executor.get_current_state()
        assert state.instance_count == 3

    @pytest.mark.asyncio
    async def test_multi_action_sequence(self):
        """Test multiple scaling actions in sequence."""
        executor = MockExecutor()

        # Sequence of actions
        for i in range(3):
            action = ScalingAction(
                action_id=f"sequence-{i}",
                target_count=(i + 1) * 3,
                current_count=i * 3 if i > 0 else 3,
            )
            result = await executor.scale(action)
            assert result.is_success is True

        # Final state
        state = await executor.get_current_state()
        assert state.instance_count == 9

        # History should have all actions
        history = executor.get_execution_history(limit=10)
        assert len(history) == 3


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_import_base(self):
        """Test importing base module."""
        from src.execution import (
            BaseExecutor,
            MockExecutor,
            ExecutorType,
            ScalingAction,
            ExecutionResult,
        )

        assert BaseExecutor is not None
        assert MockExecutor is not None

    def test_import_verification(self):
        """Test importing verification module."""
        from src.execution import (
            VerificationSystem,
            VerificationConfig,
            VerificationSession,
            VerificationCheck,
        )

        assert VerificationSystem is not None
        assert VerificationConfig is not None

    def test_import_rollback(self):
        """Test importing rollback module."""
        from src.execution import (
            RollbackManager,
            RollbackPolicy,
            RollbackStrategy,
            RollbackReason,
        )

        assert RollbackManager is not None
        assert RollbackPolicy is not None

    def test_optional_imports_flags(self):
        """Test optional import flags."""
        from src.execution import (
            HAS_KUBERNETES,
            HAS_AWS,
            HAS_TERRAFORM,
        )

        # These should be booleans
        assert isinstance(HAS_KUBERNETES, bool)
        assert isinstance(HAS_AWS, bool)
        assert isinstance(HAS_TERRAFORM, bool)
