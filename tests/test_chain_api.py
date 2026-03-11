"""Tests for Chain HTTP API + ChainExecutor — Phase 9."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from atlas.chains.definition import ChainDefinition, ChainStep
from atlas.chains.executor import ChainExecution, ChainExecutor
from atlas.chains.runner import ChainRunner
from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.mediation.engine import MediationEngine
from atlas.pool.executor import ExecutionPool
from atlas.pool.queue import JobQueue
from atlas.runtime.context import AgentContext
from atlas.serve import create_app

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# === Fixtures ===


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
def chain_executor(registry: AgentRegistry) -> ChainExecutor:
    return ChainExecutor(registry)


@pytest.fixture
async def chain_client(registry: AgentRegistry, chain_executor: ChainExecutor):
    """Test client with chain executor wired in."""
    bus = EventBus()
    queue = JobQueue(event_bus=bus)
    pool = ExecutionPool(registry, queue, max_concurrent=2, warm_pool_size=0)

    app = create_app(registry, queue, pool, event_bus=bus, chain_executor=chain_executor)
    await pool.start()

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    yield client

    await client.close()
    await pool.stop()


# === ChainExecution Tests ===


class TestChainExecution:
    def test_initial_state(self):
        ex = ChainExecution(chain_name="test", total_steps=3)
        assert ex.id.startswith("chain-")
        assert ex.status == "pending"
        assert ex.result is None
        assert ex.current_step == -1
        assert ex.total_steps == 3
        assert ex.created_at > 0

    def test_to_dict_no_result(self):
        ex = ChainExecution(chain_name="test", total_steps=2)
        d = ex.to_dict()
        assert d["chain_name"] == "test"
        assert d["status"] == "pending"
        assert d["result"] is None
        assert d["total_steps"] == 2


# === ChainExecutor Tests ===


class TestChainExecutor:
    async def test_submit_returns_chain_id(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="test", steps=[ChainStep(agent_name="echo")]
        )
        eid = chain_executor.submit(chain, {"message": "hi"})
        assert eid.startswith("chain-")

    async def test_get_returns_execution(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="test", steps=[ChainStep(agent_name="echo")]
        )
        eid = chain_executor.submit(chain, {"message": "hi"})
        ex = chain_executor.get(eid)
        assert ex is not None
        assert ex.chain_name == "test"

    def test_get_unknown_returns_none(self, chain_executor: ChainExecutor):
        assert chain_executor.get("chain-nonexistent") is None

    async def test_execution_completes_success(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="echo-chain", steps=[ChainStep(agent_name="echo")]
        )
        eid = chain_executor.submit(chain, {"message": "hello"})

        # Wait for async completion
        await asyncio.sleep(0.5)

        ex = chain_executor.get(eid)
        assert ex.status == "completed"
        assert ex.result is not None
        assert ex.result.success
        assert ex.result.output == {"message": "hello"}
        assert ex.completed_at > 0

    async def test_execution_records_failure(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="bad-chain", steps=[ChainStep(agent_name="nonexistent")]
        )
        eid = chain_executor.submit(chain, {"message": "hi"})

        await asyncio.sleep(0.5)

        ex = chain_executor.get(eid)
        assert ex.status == "failed"
        assert ex.result is not None
        assert not ex.result.success

    async def test_multi_step_execution(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="echo-echo",
            steps=[ChainStep(agent_name="echo"), ChainStep(agent_name="echo")],
        )
        eid = chain_executor.submit(chain, {"message": "round trip"})

        await asyncio.sleep(0.5)

        ex = chain_executor.get(eid)
        assert ex.status == "completed"
        assert ex.result.success
        assert ex.result.output == {"message": "round trip"}
        assert ex.current_step == 1  # 0-indexed, last step

    async def test_list_executions(self, chain_executor: ChainExecutor):
        for i in range(3):
            chain = ChainDefinition(
                name=f"chain-{i}", steps=[ChainStep(agent_name="echo")]
            )
            chain_executor.submit(chain, {"message": str(i)})

        executions = chain_executor.list()
        assert len(executions) == 3

    async def test_list_filter_by_status(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="done", steps=[ChainStep(agent_name="echo")]
        )
        chain_executor.submit(chain, {"message": "hi"})
        await asyncio.sleep(0.5)

        completed = chain_executor.list(status="completed")
        assert len(completed) >= 1
        assert all(e.status == "completed" for e in completed)

    async def test_to_dict_with_result(self, chain_executor: ChainExecutor):
        chain = ChainDefinition(
            name="dict-test", steps=[ChainStep(agent_name="echo")]
        )
        eid = chain_executor.submit(chain, {"message": "hi"})
        await asyncio.sleep(0.5)

        ex = chain_executor.get(eid)
        d = ex.to_dict()
        assert d["status"] == "completed"
        assert d["result"]["success"] is True
        assert d["result"]["output"] == {"message": "hi"}
        assert len(d["result"]["steps"]) == 1


# === HTTP Integration Tests ===


class TestChainHTTPEndpoints:
    async def test_post_submit_chain(self, chain_client):
        resp = await chain_client.post("/api/chains", json={
            "chain": {
                "name": "echo-api",
                "steps": [{"agent": "echo"}],
            },
            "input": {"message": "hello from api"},
        })
        assert resp.status == 201
        data = await resp.json()
        assert data["id"].startswith("chain-")

    async def test_post_invalid_json(self, chain_client):
        resp = await chain_client.post(
            "/api/chains",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_post_missing_chain(self, chain_client):
        resp = await chain_client.post("/api/chains", json={
            "input": {"message": "no chain field"},
        })
        assert resp.status == 400

    async def test_post_empty_steps(self, chain_client):
        resp = await chain_client.post("/api/chains", json={
            "chain": {"name": "empty", "steps": []},
            "input": {},
        })
        assert resp.status == 400

    async def test_get_chain_execution(self, chain_client):
        # Submit
        resp = await chain_client.post("/api/chains", json={
            "chain": {"name": "get-test", "steps": [{"agent": "echo"}]},
            "input": {"message": "get me"},
        })
        data = await resp.json()
        eid = data["id"]

        await asyncio.sleep(0.5)

        # Get
        resp = await chain_client.get(f"/api/chains/{eid}")
        assert resp.status == 200
        data = await resp.json()
        assert data["id"] == eid
        assert data["status"] == "completed"
        assert data["result"]["success"] is True

    async def test_get_chain_not_found(self, chain_client):
        resp = await chain_client.get("/api/chains/chain-nonexistent")
        assert resp.status == 404

    async def test_list_chains(self, chain_client):
        # Submit two chains
        for i in range(2):
            await chain_client.post("/api/chains", json={
                "chain": {"name": f"list-{i}", "steps": [{"agent": "echo"}]},
                "input": {"message": str(i)},
            })

        await asyncio.sleep(0.5)

        resp = await chain_client.get("/api/chains")
        assert resp.status == 200
        data = await resp.json()
        assert len(data) >= 2

    async def test_list_chains_filter_status(self, chain_client):
        await chain_client.post("/api/chains", json={
            "chain": {"name": "filter-test", "steps": [{"agent": "echo"}]},
            "input": {"message": "hi"},
        })
        await asyncio.sleep(0.5)

        resp = await chain_client.get("/api/chains?status=completed")
        assert resp.status == 200
        data = await resp.json()
        assert all(d["status"] == "completed" for d in data)


# === Context Population Tests ===


class TestChainContextPopulation:
    async def test_context_populated_in_chain(self, registry: AgentRegistry):
        """Verify chain_name and step_index are set on AgentContext during chain execution."""
        # We'll track contexts by patching run_agent
        captured_contexts: list[AgentContext] = []

        from atlas.chains import runner as chain_runner_mod
        original_run = chain_runner_mod.run_agent

        async def tracking_run(reg, name, input_data, *, context=None):
            if context:
                captured_contexts.append(context)
            return await original_run(reg, name, input_data, context=context)

        chain_runner_mod.run_agent = tracking_run
        try:
            engine = MediationEngine()
            runner = ChainRunner(registry, engine)
            chain = ChainDefinition(
                name="ctx-test",
                steps=[
                    ChainStep(agent_name="echo"),
                    ChainStep(agent_name="echo"),
                ],
            )
            result = await runner.execute(chain, {"message": "context check"})
            assert result.success

            assert len(captured_contexts) == 2
            assert captured_contexts[0].chain_name == "ctx-test"
            assert captured_contexts[0].step_index == 0
            assert captured_contexts[1].chain_name == "ctx-test"
            assert captured_contexts[1].step_index == 1
        finally:
            chain_runner_mod.run_agent = original_run

    async def test_chain_data_contains_prior_outputs(self, registry: AgentRegistry):
        """chain_data should contain trigger and prior step outputs."""
        captured_contexts: list[AgentContext] = []

        from atlas.chains import runner as chain_runner_mod
        original_run = chain_runner_mod.run_agent

        async def tracking_run(reg, name, input_data, *, context=None):
            if context:
                captured_contexts.append(context)
            return await original_run(reg, name, input_data, context=context)

        chain_runner_mod.run_agent = tracking_run
        try:
            engine = MediationEngine()
            runner = ChainRunner(registry, engine)
            chain = ChainDefinition(
                name="data-test",
                steps=[
                    ChainStep(agent_name="echo"),
                    ChainStep(agent_name="echo"),
                ],
            )
            result = await runner.execute(chain, {"message": "data check"})
            assert result.success

            # Step 0: no prior outputs
            assert captured_contexts[0].chain_data["trigger"] == {"message": "data check"}
            assert captured_contexts[0].chain_data["step_outputs"] == []

            # Step 1: has step 0's output
            assert captured_contexts[1].chain_data["trigger"] == {"message": "data check"}
            assert captured_contexts[1].chain_data["step_outputs"] == [{"message": "data check"}]
        finally:
            chain_runner_mod.run_agent = original_run
