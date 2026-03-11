"""Tests for RetrySubscriber — automatic retry with exponential backoff."""

from __future__ import annotations

import asyncio

import pytest

from atlas.contract.registry import AgentRegistry, RegisteredAgent
from pathlib import Path
from atlas.contract.types import AgentContract, RetrySpec, SchemaSpec
from atlas.events import EventBus
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.retry import RetrySubscriber


def _make_contract(
    name: str = "test-agent",
    max_retries: int = 3,
    backoff_base: float = 0.01,  # Fast for tests
) -> AgentContract:
    return AgentContract(
        name=name,
        version="1.0.0",
        retry=RetrySpec(max_retries=max_retries, backoff_base=backoff_base),
    )


def _make_registry(contract: AgentContract) -> AgentRegistry:
    """Create a registry with a single agent entry."""
    reg = AgentRegistry(search_paths=[])
    entry = RegisteredAgent(
        contract=contract,
        source_path=Path("fake"),
    )
    reg._agents[contract.name] = {contract.version: entry}
    return reg


class TestRetrySpec:
    def test_defaults(self):
        spec = RetrySpec()
        assert spec.max_retries == 0
        assert spec.backoff_base == 2.0

    def test_from_dict(self):
        spec = RetrySpec.from_dict({"max_retries": 5, "backoff_base": 1.5})
        assert spec.max_retries == 5
        assert spec.backoff_base == 1.5

    def test_from_dict_none(self):
        spec = RetrySpec.from_dict(None)
        assert spec.max_retries == 0

    def test_from_dict_empty(self):
        spec = RetrySpec.from_dict({})
        assert spec.max_retries == 0
        assert spec.backoff_base == 2.0

    def test_contract_has_retry(self):
        contract = _make_contract(max_retries=2)
        assert contract.retry.max_retries == 2

    def test_contract_default_no_retry(self):
        contract = AgentContract(name="x", version="1.0.0")
        assert contract.retry.max_retries == 0

    def test_contract_from_dict_with_retry(self):
        data = {
            "agent": {
                "name": "retryable",
                "version": "1.0.0",
                "retry": {"max_retries": 3, "backoff_base": 1.0},
            }
        }
        contract = AgentContract.from_dict(data)
        assert contract.retry.max_retries == 3
        assert contract.retry.backoff_base == 1.0

    def test_contract_from_dict_without_retry(self):
        data = {"agent": {"name": "simple", "version": "1.0.0"}}
        contract = AgentContract.from_dict(data)
        assert contract.retry.max_retries == 0


class TestRetrySubscriber:
    async def test_ignores_non_failure(self):
        """Subscriber does nothing for completed/running/cancelled transitions."""
        contract = _make_contract(max_retries=3)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(agent_name="test-agent", input_data={"x": 1})
        await subscriber(job, "pending", "running")
        await subscriber(job, "running", "completed")

        assert queue.pending_count == 0

    async def test_ignores_unknown_agent(self):
        """No retry if agent not in registry."""
        registry = AgentRegistry(search_paths=[])
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(agent_name="unknown", input_data={"x": 1})
        await subscriber(job, "running", "failed")

        assert queue.pending_count == 0

    async def test_ignores_zero_retries(self):
        """No retry when max_retries=0."""
        contract = _make_contract(max_retries=0)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(agent_name="test-agent", input_data={"x": 1})
        await subscriber(job, "running", "failed")

        # Give the event loop a chance
        await asyncio.sleep(0.05)
        assert queue.pending_count == 0

    async def test_retries_on_failure(self):
        """Failed job schedules a retry after backoff delay."""
        contract = _make_contract(max_retries=3, backoff_base=0.01)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(agent_name="test-agent", input_data={"msg": "hi"})
        await subscriber(job, "running", "failed")

        # Wait for the backoff delay + processing
        await asyncio.sleep(0.1)
        assert queue.pending_count == 1

    async def test_retry_increments_count(self):
        """Retry job has incremented retry_count."""
        contract = _make_contract(max_retries=3, backoff_base=0.01)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(agent_name="test-agent", input_data={"msg": "hi"})
        await subscriber(job, "running", "failed")

        await asyncio.sleep(0.1)
        retry_jobs = queue.list_by_status("pending")
        assert len(retry_jobs) == 1
        assert retry_jobs[0].retry_count == 1

    async def test_retry_links_to_original(self):
        """Retry job has original_job_id set to the first job's ID."""
        contract = _make_contract(max_retries=3, backoff_base=0.01)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(agent_name="test-agent", input_data={"msg": "hi"})
        await subscriber(job, "running", "failed")

        await asyncio.sleep(0.1)
        retry_jobs = queue.list_by_status("pending")
        assert retry_jobs[0].original_job_id == job.id

    async def test_retry_preserves_input_and_priority(self):
        """Retry job has the same input_data and priority."""
        contract = _make_contract(max_retries=3, backoff_base=0.01)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(
            agent_name="test-agent",
            input_data={"msg": "hi", "extra": 42},
            priority=5,
        )
        await subscriber(job, "running", "failed")

        await asyncio.sleep(0.1)
        retry_jobs = queue.list_by_status("pending")
        assert retry_jobs[0].input_data == {"msg": "hi", "extra": 42}
        assert retry_jobs[0].priority == 5

    async def test_exhausted_retries(self):
        """No retry when retry_count >= max_retries."""
        contract = _make_contract(max_retries=2, backoff_base=0.01)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        job = JobData(
            agent_name="test-agent",
            input_data={"msg": "hi"},
            retry_count=2,  # Already at max
        )
        await subscriber(job, "running", "failed")

        await asyncio.sleep(0.1)
        assert queue.pending_count == 0

    async def test_chain_original_id_through_retries(self):
        """When a retry itself fails, the next retry still links to original."""
        contract = _make_contract(max_retries=3, backoff_base=0.01)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        original = JobData(agent_name="test-agent", input_data={"msg": "hi"})
        await subscriber(original, "running", "failed")

        await asyncio.sleep(0.1)
        retry1 = queue.list_by_status("pending")[0]
        assert retry1.original_job_id == original.id

        # Simulate retry1 also failing
        retry1.status = "failed"
        await subscriber(retry1, "running", "failed")

        await asyncio.sleep(0.1)
        pending = queue.list_by_status("pending")
        retry2 = [j for j in pending if j.id != retry1.id][0]
        assert retry2.retry_count == 2
        assert retry2.original_job_id == original.id


class TestRetryWithEventBus:
    async def test_retry_via_event_bus(self):
        """End-to-end: EventBus wired to RetrySubscriber triggers retry."""
        contract = _make_contract(max_retries=2, backoff_base=0.01)
        registry = _make_registry(contract)
        bus = EventBus()
        queue = JobQueue(event_bus=bus)
        subscriber = RetrySubscriber(queue, registry)
        bus.subscribe(subscriber)

        job = JobData(agent_name="test-agent", input_data={"msg": "test"})
        await queue.submit(job)
        await queue.update(job.id, status="running")
        await queue.update(job.id, status="failed", error="boom")

        # Wait for retry to be scheduled and submitted
        await asyncio.sleep(0.1)

        # Original job is failed, retry is pending
        assert job.status == "failed"
        assert queue.pending_count == 1

        retry_jobs = queue.list_by_status("pending")
        assert retry_jobs[0].retry_count == 1
        assert retry_jobs[0].original_job_id == job.id

    async def test_backoff_increases(self):
        """Higher retry_count means longer delay."""
        contract = _make_contract(max_retries=5, backoff_base=0.05)
        registry = _make_registry(contract)
        queue = JobQueue()
        subscriber = RetrySubscriber(queue, registry)

        # retry_count=0 → delay 0.05s
        job0 = JobData(agent_name="test-agent", input_data={"x": 1}, retry_count=0)
        await subscriber(job0, "running", "failed")

        # retry_count=2 → delay 0.2s
        job2 = JobData(agent_name="test-agent", input_data={"x": 1}, retry_count=2)
        await subscriber(job2, "running", "failed")

        # After 0.1s, only the first retry should be submitted (0.05 < 0.1 < 0.2)
        await asyncio.sleep(0.1)
        assert queue.pending_count == 1

        # After 0.3s total, both should be submitted
        await asyncio.sleep(0.2)
        assert queue.pending_count == 2
