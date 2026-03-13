"""E2E tests — chain execution with real agents.

Tests ChainExecutor + ChainRunner + MediationEngine wired together
with real agents from the agents/ directory. No mocking.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from atlas.chains.definition import ChainDefinition, ChainStep
from atlas.chains.executor import ChainExecution, ChainExecutor
from atlas.chains.runner import ChainRunner
from atlas.contract.registry import AgentRegistry
from atlas.mediation.engine import MediationEngine

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
def executor(registry):
    return ChainExecutor(registry, max_completed=50)


@pytest.fixture
def mediation():
    return MediationEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(name: str, steps: list[tuple[str, dict | None]]) -> ChainDefinition:
    """Build a ChainDefinition from (agent_name, input_map) tuples."""
    return ChainDefinition(
        name=name,
        steps=[
            ChainStep(agent_name=agent, input_map=imap)
            for agent, imap in steps
        ],
    )


async def _wait_for_execution(
    executor: ChainExecutor, exec_id: str, timeout: float = 10.0
) -> ChainExecution:
    """Poll until execution reaches a terminal state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        execution = executor.get(exec_id)
        if execution and execution.status in ("completed", "failed"):
            return execution
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Execution {exec_id} didn't finish within {timeout}s")


# ---------------------------------------------------------------------------
# Tests: Linear chain execution
# ---------------------------------------------------------------------------


class TestLinearChains:
    """Chain execution with real agents."""

    @pytest.mark.asyncio
    async def test_single_step_chain(self, executor):
        """Chain with one agent works (degenerate case)."""
        chain = _make_chain("single", [("echo", None)])
        exec_id = executor.submit(chain, {"message": "hello"})

        execution = await _wait_for_execution(executor, exec_id)

        assert execution.status == "completed"
        assert execution.result is not None
        assert execution.result.success
        assert execution.result.output.get("message") == "hello"

    @pytest.mark.asyncio
    async def test_chain_passes_output_forward(self, executor):
        """Agent A output becomes Agent B input via mediation."""
        # echo → echo: first echoes message, mediation passes output to second
        chain = _make_chain("double-echo", [("echo", None), ("echo", None)])
        exec_id = executor.submit(chain, {"message": "forward"})

        execution = await _wait_for_execution(executor, exec_id)

        assert execution.status == "completed"
        assert execution.result is not None
        assert execution.result.success
        assert len(execution.result.steps) == 2

    @pytest.mark.asyncio
    async def test_chain_failure_stops_execution(self, executor):
        """Chain stops on agent failure, status = failed."""
        # Use a nonexistent agent as the second step
        chain = _make_chain("fail-chain", [
            ("echo", None),
            ("agent_that_does_not_exist", None),
        ])
        exec_id = executor.submit(chain, {"message": "test"})

        execution = await _wait_for_execution(executor, exec_id)

        assert execution.status == "failed"
        assert execution.result is not None
        assert not execution.result.success
        assert execution.result.error  # Has error message

    @pytest.mark.asyncio
    async def test_chain_execution_tracking(self, executor):
        """get() returns execution with step progress."""
        chain = _make_chain("tracked", [("echo", None)])
        exec_id = executor.submit(chain, {"message": "track"})

        # Initially should be pending or running
        execution = executor.get(exec_id)
        assert execution is not None
        assert execution.chain_name == "tracked"
        assert execution.total_steps == 1

        # Wait for completion
        execution = await _wait_for_execution(executor, exec_id)
        assert execution.completed_at > 0


# ---------------------------------------------------------------------------
# Tests: Listing and filtering
# ---------------------------------------------------------------------------


class TestChainListing:
    """Chain listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_by_status(self, executor):
        """list(status=...) filters correctly."""
        # Submit and complete two chains
        id1 = executor.submit(
            _make_chain("c1", [("echo", None)]),
            {"message": "a"},
        )
        id2 = executor.submit(
            _make_chain("c2", [("echo", None)]),
            {"message": "b"},
        )

        await _wait_for_execution(executor, id1)
        await _wait_for_execution(executor, id2)

        completed = executor.list(status="completed")
        assert len(completed) >= 2

        # No running executions left
        running = executor.list(status="running")
        assert len(running) == 0

    @pytest.mark.asyncio
    async def test_list_limit(self, executor):
        """list(limit=N) caps results."""
        for i in range(5):
            exec_id = executor.submit(
                _make_chain(f"batch-{i}", [("echo", None)]),
                {"message": f"msg-{i}"},
            )
            await _wait_for_execution(executor, exec_id)

        results = executor.list(limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Tests: Eviction and concurrency
# ---------------------------------------------------------------------------


class TestChainEviction:
    """Chain execution eviction and concurrency."""

    @pytest.mark.asyncio
    async def test_max_completed_eviction(self, registry):
        """Old completed executions evicted beyond max_completed."""
        executor = ChainExecutor(registry, max_completed=3)

        exec_ids = []
        for i in range(5):
            exec_id = executor.submit(
                _make_chain(f"evict-{i}", [("echo", None)]),
                {"message": f"msg-{i}"},
            )
            exec_ids.append(exec_id)
            await _wait_for_execution(executor, exec_id)

        # Only max_completed (3) terminal executions kept
        all_executions = executor.list()
        assert len(all_executions) <= 3

        # Oldest should have been evicted
        assert executor.get(exec_ids[0]) is None

    @pytest.mark.asyncio
    async def test_concurrent_chain_executions(self, executor):
        """Multiple chains run concurrently."""
        ids = []
        for i in range(3):
            exec_id = executor.submit(
                _make_chain(f"concurrent-{i}", [("echo", None)]),
                {"message": f"msg-{i}"},
            )
            ids.append(exec_id)

        # Wait for all
        results = await asyncio.gather(
            *[_wait_for_execution(executor, eid) for eid in ids]
        )

        for execution in results:
            assert execution.status == "completed"


# ---------------------------------------------------------------------------
# Tests: Chain with mediation
# ---------------------------------------------------------------------------


class TestChainMediation:
    """Chain mediation between agents."""

    @pytest.mark.asyncio
    async def test_chain_with_compatible_schemas(self, executor):
        """Chain between agents with compatible schemas uses direct mediation."""
        # echo → echo: same schema, direct passthrough
        chain = _make_chain("mediated", [("echo", None), ("echo", None)])
        exec_id = executor.submit(chain, {"message": "mediate-test"})

        execution = await _wait_for_execution(executor, exec_id)

        assert execution.status == "completed"
        assert execution.result is not None
        if execution.result.steps:
            # Check mediation summary exists
            summary = execution.result.mediation_summary
            assert isinstance(summary, list)
