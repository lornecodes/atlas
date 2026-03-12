"""Integration tests for security in SlotManager and ExecutionPool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from atlas.contract.permissions import PermissionsSpec
from atlas.contract.registry import AgentRegistry
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.pool.slot_manager import SlotManager, SlotState
from atlas.security.policy import SecurityPolicy
from atlas.security.secrets import EnvSecretProvider, SecretError, SecretResolver

from conftest import AGENTS_DIR


@pytest.fixture
def registry():
    r = AgentRegistry(search_paths=[str(AGENTS_DIR)])
    r.discover()
    return r


@pytest.fixture
def queue():
    return JobQueue()


# === SlotManager Container Branch ===

class TestSlotManagerContainerBranch:
    @pytest.mark.asyncio
    async def test_process_isolation_uses_standard_path(self, registry):
        """Default process isolation should use the normal AgentSlot path."""
        sm = SlotManager(registry, warm_pool_size=1)
        perms = PermissionsSpec(isolation="process")
        slot, warmup_ms = await sm.acquire("echo", resolved_permissions=perms)
        assert slot.state == SlotState.BUSY
        assert slot.agent_name == "echo"
        await sm.release(slot)

    @pytest.mark.asyncio
    async def test_none_permissions_uses_standard_path(self, registry):
        """No permissions should use the normal AgentSlot path."""
        sm = SlotManager(registry, warm_pool_size=1)
        slot, warmup_ms = await sm.acquire("echo")
        assert slot.state == SlotState.BUSY
        await sm.release(slot)

    @pytest.mark.asyncio
    async def test_container_isolation_creates_container_slot(self, registry):
        """Container isolation should create a ContainerSlot via _acquire_container."""
        policy = SecurityPolicy(container_network="bridge")
        sm = SlotManager(registry, warm_pool_size=1, security_policy=policy)
        perms = PermissionsSpec(
            isolation="container",
            container_image="test:v1",
            max_memory_mb=256,
        )

        with patch("shutil.which", return_value="/usr/bin/docker"):
            slot, warmup_ms = await sm.acquire(
                "echo",
                resolved_permissions=perms,
                secrets={"KEY": "val"},
            )

        assert slot.state == SlotState.BUSY
        assert slot.agent_name == "echo"
        # The adapter wraps a ContainerSlot
        from atlas.pool.slot_manager import _ContainerAdapter
        assert isinstance(slot.instance, _ContainerAdapter)

    @pytest.mark.asyncio
    async def test_container_slot_docker_not_found(self, registry):
        """Container slot should fail if Docker is not available."""
        sm = SlotManager(registry, warm_pool_size=1)
        perms = PermissionsSpec(isolation="container", container_image="test:v1")

        with patch("shutil.which", return_value=None):
            with pytest.raises(Exception, match="Docker not found"):
                await sm.acquire("echo", resolved_permissions=perms)


# === ExecutionPool with Security ===

class TestExecutionPoolSecurity:
    @pytest.mark.asyncio
    async def test_pool_without_policy_works_normally(self, registry, queue):
        """Pool without security policy should work exactly as before."""
        pool = ExecutionPool(registry, queue, max_concurrent=1, warm_pool_size=1)
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "completed"
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_with_policy_resolves_permissions(self, registry, queue):
        """Pool with policy should resolve permissions and pass to slot manager."""
        policy = SecurityPolicy(max_memory_mb=256, max_cpu_seconds=30)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=1, warm_pool_size=1,
            security_policy=policy,
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "completed"
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_secret_resolution_failure(self, registry, queue):
        """Pool should fail job if secret resolution fails."""
        policy = SecurityPolicy(
            allowed_secrets={"MISSING_SECRET"},
            max_memory_mb=1024,
        )
        # Resolver that will fail (secret not in env)
        resolver = SecretResolver(
            EnvSecretProvider(),
            allowed_secrets={"MISSING_SECRET"},
        )
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=1, warm_pool_size=1,
            security_policy=policy,
            secret_resolver=resolver,
        )

        # We need an agent that declares secrets in its contract
        # Since echo doesn't, we'll mock the registry entry
        from atlas.contract.types import AgentContract
        original_get = registry.get

        def mock_get(name):
            entry = original_get(name)
            if entry and name == "echo":
                # Create a contract with secrets declared
                perms = PermissionsSpec(secrets=["MISSING_SECRET"])
                contract = AgentContract(
                    name=entry.contract.name,
                    version=entry.contract.version,
                    permissions=perms,
                    input_schema=entry.contract.input_schema,
                    output_schema=entry.contract.output_schema,
                )
                from atlas.contract.registry import RegisteredAgent
                return RegisteredAgent(
                    contract=contract,
                    source_path=entry.source_path,
                    module_path=entry.module_path,
                    _agent_class=entry.agent_class,
                )
            return entry

        registry.get = mock_get

        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "failed"
            assert "Secret resolution failed" in result.error
        finally:
            await pool.stop()
            registry.get = original_get


# === SecretResolver Integration ===

class TestSecretResolverIntegration:
    @pytest.mark.asyncio
    async def test_resolve_with_env(self, monkeypatch):
        monkeypatch.setenv("ATLAS_SECRET_MY_KEY", "my-value")
        resolver = SecretResolver(
            EnvSecretProvider(),
            allowed_secrets={"MY_KEY"},
        )
        result = await resolver.resolve(["MY_KEY"])
        assert result == {"MY_KEY": "my-value"}

    @pytest.mark.asyncio
    async def test_resolve_not_allowed(self):
        resolver = SecretResolver(
            EnvSecretProvider(),
            allowed_secrets={"ONLY_THIS"},
        )
        with pytest.raises(SecretError, match="not in the allowed"):
            await resolver.resolve(["OTHER"])

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self):
        resolver = SecretResolver(EnvSecretProvider())
        result = await resolver.resolve([])
        assert result == {}


# === RuntimeContext Security Fields ===

class TestRuntimeContextSecurity:
    def test_context_has_permissions_field(self):
        from atlas.runtime.context import AgentContext
        ctx = AgentContext()
        assert ctx.permissions is None

    def test_context_has_secrets_field(self):
        from atlas.runtime.context import AgentContext
        ctx = AgentContext()
        assert ctx.secrets == {}

    def test_context_secrets_settable(self):
        from atlas.runtime.context import AgentContext
        ctx = AgentContext()
        ctx.secrets = {"KEY": "val"}
        assert ctx.secrets == {"KEY": "val"}

    def test_context_permissions_settable(self):
        from atlas.runtime.context import AgentContext
        perms = PermissionsSpec(spawn=True)
        ctx = AgentContext()
        ctx.permissions = perms
        assert ctx.permissions.spawn is True


# === Edge Case Tests ===

class TestPermissionsValidation:
    def test_invalid_isolation_raises_value_error(self):
        """Invalid isolation mode in from_dict should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid isolation mode"):
            PermissionsSpec.from_dict({"isolation": "kubernetes"})

    def test_unknown_isolation_rejected(self):
        """Arbitrary strings are not valid isolation modes."""
        with pytest.raises(ValueError, match="Invalid isolation mode"):
            PermissionsSpec.from_dict({"isolation": "podman"})


class TestContainerEdgeCases:
    @pytest.mark.asyncio
    async def test_container_empty_image_raises_clear_error(self, registry):
        """Container isolation with empty image should raise a clear RuntimeError."""
        sm = SlotManager(registry, warm_pool_size=1)
        perms = PermissionsSpec(isolation="container", container_image="")

        with pytest.raises(RuntimeError, match="container_image is set"):
            await sm.acquire("echo", resolved_permissions=perms)

    @pytest.mark.asyncio
    async def test_container_timeout_matches_contract(self, registry):
        """Container slot timeout should derive from contract's execution_timeout."""
        from atlas.security.container import ContainerSlot as DockerSlot

        policy = SecurityPolicy(container_network="none")
        sm = SlotManager(registry, warm_pool_size=1, security_policy=policy)
        perms = PermissionsSpec(isolation="container", container_image="test:v1")

        # Patch ContainerSlot to capture the timeout it receives
        captured = {}
        original_init = DockerSlot.__init__

        def spy_init(self, **kwargs):
            captured.update(kwargs)
            original_init(self, **kwargs)

        with patch.object(DockerSlot, "__init__", spy_init), \
             patch("shutil.which", return_value="/usr/bin/docker"):
            slot, _ = await sm.acquire("echo", resolved_permissions=perms)

        # echo agent's contract has a default execution_timeout
        entry = registry.get("echo")
        expected_timeout = entry.contract.execution_timeout + 5.0
        assert captured["timeout"] == expected_timeout
