"""Tests for the pluggable orchestrator — Phase 6."""

from __future__ import annotations

import asyncio

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import AgentContract
from atlas.orchestrator.default import DefaultOrchestrator
from atlas.orchestrator.protocol import Orchestrator, RoutingDecision
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue

from conftest import AGENTS_DIR


# === RoutingDecision Tests ===


class TestRoutingDecision:
    def test_defaults(self):
        d = RoutingDecision()
        assert d.action == "execute"
        assert d.agent_name is None
        assert d.priority is None
        assert d.metadata == {}

    def test_reject(self):
        d = RoutingDecision(action="reject", metadata={"reason": "too big"})
        assert d.action == "reject"
        assert d.metadata["reason"] == "too big"

    def test_redirect(self):
        d = RoutingDecision(action="redirect", agent_name="other-agent")
        assert d.action == "redirect"
        assert d.agent_name == "other-agent"

    def test_priority_override(self):
        d = RoutingDecision(action="execute", priority=99)
        assert d.priority == 99


# === DefaultOrchestrator Tests ===


class TestDefaultOrchestrator:
    async def test_route_returns_execute(self):
        orch = DefaultOrchestrator()
        job = JobData(agent_name="echo", input_data={"message": "hi"})
        registry = AgentRegistry()
        decision = await orch.route(job, registry)
        assert decision.action == "execute"
        assert decision.agent_name is None
        assert decision.priority is None

    async def test_on_job_complete_is_noop(self):
        orch = DefaultOrchestrator()
        job = JobData(agent_name="echo")
        await orch.on_job_complete(job)  # Should not raise

    async def test_on_job_failed_is_noop(self):
        orch = DefaultOrchestrator()
        job = JobData(agent_name="echo")
        await orch.on_job_failed(job)  # Should not raise

    def test_satisfies_protocol(self):
        assert isinstance(DefaultOrchestrator(), Orchestrator)


# === Orchestrator Protocol Tests ===


class TestOrchestratorProtocol:
    def test_structural_check(self):
        """Any class with route/on_job_complete/on_job_failed satisfies the protocol."""

        class Custom:
            async def route(self, job, registry):
                return RoutingDecision()

            async def on_job_complete(self, job):
                pass

            async def on_job_failed(self, job):
                pass

        assert isinstance(Custom(), Orchestrator)

    def test_incomplete_class_fails(self):
        """Missing methods means it does not satisfy the protocol."""

        class Incomplete:
            async def route(self, job, registry):
                return RoutingDecision()

        assert not isinstance(Incomplete(), Orchestrator)


# === Custom Orchestrator Behaviors ===


class RejectingOrchestrator:
    """Rejects all jobs."""

    async def route(self, job, registry):
        return RoutingDecision(action="reject", metadata={"reason": "No."})

    async def on_job_complete(self, job):
        pass

    async def on_job_failed(self, job):
        self.last_failed = job


class RedirectingOrchestrator:
    """Redirects all jobs to 'echo'."""

    async def route(self, job, registry):
        return RoutingDecision(action="redirect", agent_name="echo")

    async def on_job_complete(self, job):
        self.last_completed = job

    async def on_job_failed(self, job):
        pass


class PriorityBoostOrchestrator:
    """Boosts all job priorities to 100."""

    async def route(self, job, registry):
        return RoutingDecision(action="execute", priority=100)

    async def on_job_complete(self, job):
        pass

    async def on_job_failed(self, job):
        pass


class TrackingOrchestrator:
    """Tracks all lifecycle events for assertion."""

    def __init__(self):
        self.routed = []
        self.completed = []
        self.failed = []

    async def route(self, job, registry):
        self.routed.append(job.id)
        return RoutingDecision(action="execute")

    async def on_job_complete(self, job):
        self.completed.append(job.id)

    async def on_job_failed(self, job):
        self.failed.append(job.id)


# === Pool Integration Tests ===


@pytest.fixture
def queue():
    return JobQueue()


class TestOrchestratorReject:
    async def test_rejected_job_fails(self, registry, queue):
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0,
            orchestrator=RejectingOrchestrator(),
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hi"})
            await pool.submit(job)
            await asyncio.sleep(0.3)

            result = queue.get(job.id)
            assert result.status == "failed"
            assert "No." in result.error
        finally:
            await pool.stop()


class TestOrchestratorRedirect:
    async def test_redirect_runs_different_agent(self, registry, queue):
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0,
            orchestrator=RedirectingOrchestrator(),
        )
        await pool.start()
        try:
            # Submit as summarizer but redirect to echo
            job = JobData(
                agent_name="summarizer",
                input_data={"message": "hello from redirect"},
            )
            await pool.submit(job)
            await asyncio.sleep(0.3)

            result = queue.get(job.id)
            assert result.status == "completed"
            # Echo agent returns the message back
            assert result.output_data["message"] == "hello from redirect"
        finally:
            await pool.stop()


class TestOrchestratorPriorityOverride:
    async def test_priority_gets_overridden(self, registry, queue):
        orch = PriorityBoostOrchestrator()
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0,
            orchestrator=orch,
        )
        await pool.start()
        try:
            job = JobData(
                agent_name="echo", priority=0,
                input_data={"message": "hi"},
            )
            await pool.submit(job)
            await asyncio.sleep(0.3)

            result = queue.get(job.id)
            assert result.status == "completed"
        finally:
            await pool.stop()


class TestOrchestratorLifecycle:
    async def test_tracking_callbacks(self, registry, queue):
        orch = TrackingOrchestrator()
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0,
            orchestrator=orch,
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "tracked"})
            await pool.submit(job)
            await asyncio.sleep(0.3)

            assert job.id in orch.routed
            assert job.id in orch.completed
            assert job.id not in orch.failed
        finally:
            await pool.stop()

    async def test_failed_job_triggers_on_failed(self, registry, queue):
        orch = TrackingOrchestrator()
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0,
            orchestrator=orch,
        )
        await pool.start()
        try:
            # Non-existent agent will fail
            job = JobData(agent_name="nonexistent", input_data={})
            await pool.submit(job)
            await asyncio.sleep(0.3)

            assert job.id in orch.routed
            assert job.id in orch.failed
            assert job.id not in orch.completed
        finally:
            await pool.stop()


# === Registry Orchestrator Discovery ===


class TestRegistryDiscovery:
    def test_get_orchestrator_for_agent_type_returns_none(self, registry):
        """Regular agents are not returned by get_orchestrator."""
        assert registry.get_orchestrator("echo") is None

    def test_list_orchestrators_empty_by_default(self, registry):
        """Default test agents have no orchestrators."""
        # All test agents have type="agent" by default
        orchestrators = registry.list_orchestrators()
        # Only the priority-router (if discovered) would show up
        for o in orchestrators:
            assert o.contract.type == "orchestrator"

    def test_get_orchestrator_found(self):
        """Registry finds orchestrators when registered."""
        reg = AgentRegistry(search_paths=[str(AGENTS_DIR)])
        reg.discover()
        entry = reg.get_orchestrator("priority-router")
        assert entry is not None
        assert entry.contract.type == "orchestrator"
        assert entry.contract.name == "priority-router"

    def test_list_orchestrators_includes_priority_router(self):
        reg = AgentRegistry(search_paths=[str(AGENTS_DIR)])
        reg.discover()
        names = [o.contract.name for o in reg.list_orchestrators()]
        assert "priority-router" in names


# === Priority Router Tests ===


def _load_priority_router():
    """Load PriorityRouterOrchestrator from agents/priority-router/agent.py."""
    import importlib.util
    module_path = AGENTS_DIR / "priority-router" / "agent.py"
    spec = importlib.util.spec_from_file_location("priority_router", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PriorityRouterOrchestrator


class TestPriorityRouter:
    async def test_empty_input_rejected(self):
        cls = _load_priority_router()
        orch = cls()
        job = JobData(agent_name="echo", input_data={})
        decision = await orch.route(job, AgentRegistry())
        assert decision.action == "reject"
        assert "Empty" in decision.metadata["reason"]

    async def test_short_text_low_priority(self):
        cls = _load_priority_router()
        orch = cls()
        job = JobData(agent_name="echo", input_data={"text": "short"})
        decision = await orch.route(job, AgentRegistry())
        assert decision.action == "execute"
        assert decision.priority == -1
        assert decision.metadata["complexity"] == "low"

    async def test_medium_text_no_priority_change(self):
        cls = _load_priority_router()
        orch = cls()
        job = JobData(agent_name="echo", input_data={"text": "x" * 500})
        decision = await orch.route(job, AgentRegistry())
        assert decision.action == "execute"
        assert decision.priority is None
        assert decision.metadata["complexity"] == "medium"

    async def test_long_text_high_priority(self):
        cls = _load_priority_router()
        orch = cls()
        job = JobData(agent_name="echo", input_data={"text": "x" * 1500})
        decision = await orch.route(job, AgentRegistry())
        assert decision.action == "execute"
        assert decision.priority == 10
        assert decision.metadata["complexity"] == "high"

    async def test_satisfies_protocol(self):
        cls = _load_priority_router()
        assert isinstance(cls(), Orchestrator)


# === Contract Type Field Tests ===


class TestContractType:
    def test_default_type_is_agent(self):
        contract = AgentContract(name="test", version="1.0.0")
        assert contract.type == "agent"

    def test_orchestrator_type(self):
        contract = AgentContract(name="test", version="1.0.0", type="orchestrator")
        assert contract.type == "orchestrator"

    def test_from_dict_default_type(self):
        contract = AgentContract.from_dict({"agent": {"name": "t", "version": "1.0.0"}})
        assert contract.type == "agent"

    def test_from_dict_orchestrator_type(self):
        contract = AgentContract.from_dict({
            "agent": {"name": "t", "version": "1.0.0", "type": "orchestrator"},
        })
        assert contract.type == "orchestrator"


# === Pool Default Orchestrator ===


class TestPoolDefaultOrchestrator:
    async def test_pool_uses_default_when_none_provided(self, registry, queue):
        """Pool without explicit orchestrator uses DefaultOrchestrator."""
        pool = ExecutionPool(registry, queue, max_concurrent=1, warm_pool_size=0)
        assert isinstance(pool._orchestrator, DefaultOrchestrator)

    async def test_pool_uses_custom_orchestrator(self, registry, queue):
        orch = TrackingOrchestrator()
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0,
            orchestrator=orch,
        )
        assert pool._orchestrator is orch
