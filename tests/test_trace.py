"""Tests for ExecutionTrace, TraceCollector, and cost estimation."""

from __future__ import annotations

import asyncio
import time

import pytest

from atlas.events import EventBus
from atlas.pool.job import JobData
from atlas.trace import (
    ChainTrace,
    EvalResult,
    ExecutionTrace,
    TraceCollector,
    estimate_cost,
)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o-mini", 1000, 500)
        # (1000 * 0.00015 + 500 * 0.0006) / 1000 = 0.00045
        assert abs(cost - 0.00045) < 1e-8

    def test_claude_haiku(self):
        cost = estimate_cost("claude-3-haiku-20240307", 1000, 1000)
        # (1000 * 0.00025 + 1000 * 0.00125) / 1000 = 0.0015
        assert abs(cost - 0.0015) < 1e-8

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("unknown-model-v99", 5000, 5000) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("gpt-4o-mini", 0, 0) == 0.0


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_to_dict(self):
        r = EvalResult(name="has_summary", passed=True, message="OK")
        d = r.to_dict()
        assert d["name"] == "has_summary"
        assert d["passed"] is True
        assert d["message"] == "OK"

    def test_to_dict_with_expected_actual(self):
        r = EvalResult(name="check", passed=False, expected=42, actual=99)
        d = r.to_dict()
        assert d["expected"] == 42
        assert d["actual"] == 99


# ---------------------------------------------------------------------------
# ExecutionTrace
# ---------------------------------------------------------------------------

class TestExecutionTrace:
    def test_basic_construction(self):
        t = ExecutionTrace(
            trace_id="tr-1",
            job_id="job-abc",
            agent_name="echo",
            status="completed",
            input_tokens=100,
            output_tokens=50,
        )
        assert t.total_tokens == 150
        assert t.trace_id == "tr-1"

    def test_to_dict(self):
        t = ExecutionTrace(
            trace_id="tr-1",
            job_id="job-abc",
            agent_name="echo",
            status="completed",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o-mini",
            estimated_cost_usd=0.00045,
            warmup_ms=12.5,
            execution_ms=100.3,
        )
        d = t.to_dict()
        assert d["trace_id"] == "tr-1"
        assert d["total_tokens"] == 150
        assert d["model"] == "gpt-4o-mini"
        assert d["warmup_ms"] == 12.5

    def test_to_dict_with_eval_results(self):
        t = ExecutionTrace(
            trace_id="tr-1",
            job_id="job-abc",
            agent_name="echo",
            status="completed",
            eval_results=[
                EvalResult(name="check1", passed=True),
                EvalResult(name="check2", passed=False, message="bad"),
            ],
        )
        d = t.to_dict()
        assert len(d["eval_results"]) == 2
        assert d["eval_results"][0]["passed"] is True
        assert d["eval_results"][1]["message"] == "bad"

    def test_correlation_fields(self):
        t = ExecutionTrace(
            trace_id="tr-child",
            job_id="job-child",
            agent_name="worker",
            status="completed",
            parent_trace_id="tr-parent",
            chain_name="my-chain",
            step_index=2,
        )
        assert t.parent_trace_id == "tr-parent"
        assert t.chain_name == "my-chain"
        assert t.step_index == 2


# ---------------------------------------------------------------------------
# ChainTrace
# ---------------------------------------------------------------------------

class TestChainTrace:
    def _make_step(self, **kwargs) -> ExecutionTrace:
        defaults = {
            "trace_id": "tr-1",
            "job_id": "job-1",
            "agent_name": "echo",
            "status": "completed",
        }
        defaults.update(kwargs)
        return ExecutionTrace(**defaults)

    def test_empty_chain(self):
        ct = ChainTrace(trace_id="chain-1", chain_name="test", status="completed")
        assert ct.total_tokens == 0
        assert ct.total_cost_usd == 0.0
        assert ct.total_execution_ms == 0.0

    def test_aggregation(self):
        ct = ChainTrace(
            trace_id="chain-1",
            chain_name="write-review",
            status="completed",
            steps=[
                self._make_step(
                    trace_id="s1", input_tokens=100, output_tokens=200,
                    estimated_cost_usd=0.001, execution_ms=50.0,
                ),
                self._make_step(
                    trace_id="s2", input_tokens=300, output_tokens=400,
                    estimated_cost_usd=0.002, execution_ms=75.5,
                ),
            ],
        )
        assert ct.total_input_tokens == 400
        assert ct.total_output_tokens == 600
        assert ct.total_tokens == 1000
        assert abs(ct.total_cost_usd - 0.003) < 1e-8
        assert abs(ct.total_execution_ms - 125.5) < 1e-8

    def test_to_dict(self):
        ct = ChainTrace(
            trace_id="chain-1",
            chain_name="pipeline",
            status="completed",
            steps=[self._make_step(input_tokens=50, output_tokens=50)],
        )
        d = ct.to_dict()
        assert d["chain_name"] == "pipeline"
        assert d["total_tokens"] == 100
        assert len(d["steps"]) == 1


# ---------------------------------------------------------------------------
# TraceCollector
# ---------------------------------------------------------------------------

class TestTraceCollector:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def collector(self, bus):
        c = TraceCollector(bus, max_traces=10)
        yield c
        c.close()

    def _make_job(self, **kwargs) -> JobData:
        defaults = {
            "agent_name": "echo",
            "status": "completed",
            "warmup_ms": 5.0,
            "execution_ms": 50.0,
            "started_at": time.time() - 1,
            "completed_at": time.time(),
            "metadata": {},
        }
        defaults.update(kwargs)
        return JobData(**defaults)

    @pytest.mark.asyncio
    async def test_creates_trace_on_completion(self, bus, collector):
        job = self._make_job()
        await bus.emit(job, "running", "completed")
        trace = collector.get(job.id)
        assert trace is not None
        assert trace.agent_name == "echo"
        assert trace.status == "completed"
        assert trace.warmup_ms == 5.0
        assert trace.execution_ms == 50.0

    @pytest.mark.asyncio
    async def test_creates_trace_on_failure(self, bus, collector):
        job = self._make_job(status="failed")
        await bus.emit(job, "running", "failed")
        trace = collector.get(job.id)
        assert trace is not None
        assert trace.status == "failed"

    @pytest.mark.asyncio
    async def test_ignores_non_terminal(self, bus, collector):
        job = self._make_job(status="running")
        await bus.emit(job, "pending", "running")
        assert collector.get(job.id) is None

    @pytest.mark.asyncio
    async def test_captures_token_metadata(self, bus, collector):
        job = self._make_job(metadata={
            "_trace_input_tokens": 150,
            "_trace_output_tokens": 75,
            "_trace_model": "gpt-4o-mini",
        })
        await bus.emit(job, "running", "completed")
        trace = collector.get(job.id)
        assert trace.input_tokens == 150
        assert trace.output_tokens == 75
        assert trace.model == "gpt-4o-mini"
        assert trace.estimated_cost_usd > 0

    @pytest.mark.asyncio
    async def test_captures_parent_trace_id(self, bus, collector):
        job = self._make_job(metadata={
            "_parent_trace_id": "job-parent-123",
        })
        await bus.emit(job, "running", "completed")
        trace = collector.get(job.id)
        assert trace.parent_trace_id == "job-parent-123"

    @pytest.mark.asyncio
    async def test_filters_internal_metadata(self, bus, collector):
        job = self._make_job(metadata={
            "_trace_model": "gpt-4o",
            "_spawn_depth": 1,
            "user_key": "hello",
        })
        await bus.emit(job, "running", "completed")
        trace = collector.get(job.id)
        # Internal keys (starting with _) stripped from metadata
        assert "_trace_model" not in trace.metadata
        assert "_spawn_depth" not in trace.metadata
        assert trace.metadata.get("user_key") == "hello"

    @pytest.mark.asyncio
    async def test_list_newest_first(self, bus, collector):
        for i in range(3):
            job = self._make_job(agent_name=f"agent-{i}")
            await bus.emit(job, "running", "completed")

        traces = collector.list(limit=10)
        assert len(traces) == 3
        assert traces[0].agent_name == "agent-2"

    @pytest.mark.asyncio
    async def test_list_filter_by_agent(self, bus, collector):
        for name in ["a", "b", "a", "c"]:
            job = self._make_job(agent_name=name)
            await bus.emit(job, "running", "completed")

        traces = collector.list(agent_name="a")
        assert len(traces) == 2
        assert all(t.agent_name == "a" for t in traces)

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, bus, collector):
        for i in range(5):
            job = self._make_job()
            await bus.emit(job, "running", "completed")

        traces = collector.list(limit=2)
        assert len(traces) == 2

    @pytest.mark.asyncio
    async def test_eviction(self, bus, collector):
        # max_traces=10
        for i in range(15):
            job = self._make_job(agent_name=f"agent-{i}")
            await bus.emit(job, "running", "completed")

        all_traces = collector.list(limit=100)
        assert len(all_traces) == 10
        # Oldest should be evicted
        names = {t.agent_name for t in all_traces}
        assert "agent-0" not in names
        assert "agent-14" in names

    def test_record_manual(self, collector):
        trace = ExecutionTrace(
            trace_id="manual-1",
            job_id="job-x",
            agent_name="custom",
            status="completed",
        )
        collector.record(trace)
        assert collector.get("manual-1") is not None

    def test_attach_eval_results(self, collector):
        trace = ExecutionTrace(
            trace_id="tr-eval",
            job_id="job-y",
            agent_name="echo",
            status="completed",
        )
        collector.record(trace)

        results = [
            EvalResult(name="check1", passed=True),
            EvalResult(name="check2", passed=False, message="bad"),
        ]
        collector.attach_eval_results("tr-eval", results)

        updated = collector.get("tr-eval")
        assert len(updated.eval_results) == 2
        assert updated.eval_results[0].passed is True

    def test_attach_eval_results_missing_trace(self, collector):
        # Should not raise
        collector.attach_eval_results("nonexistent", [EvalResult(name="x", passed=True)])

    def test_close_unsubscribes(self, bus):
        collector = TraceCollector(bus)
        assert bus.subscriber_count == 1
        collector.close()
        assert bus.subscriber_count == 0
