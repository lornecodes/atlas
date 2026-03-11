"""Tests for MetricsCollector — Phase 8A."""

from __future__ import annotations

import time

import pytest

from atlas.events import EventBus
from atlas.metrics import AgentMetrics, MetricsCollector, _LATENCY_WINDOW
from atlas.pool.job import JobData


# === AgentMetrics Tests ===


class TestAgentMetrics:
    def test_initial_state(self):
        m = AgentMetrics()
        assert m.jobs_by_status["completed"] == 0
        assert m.warm_hits == 0
        assert m.total_retries == 0
        assert m.latencies == []
        assert m.throughput_per_min == 0.0

    def test_record_transition(self):
        m = AgentMetrics()
        m.record_transition("pending", "running")
        assert m.jobs_by_status["pending"] == 0  # was 0, decremented to 0 (max)
        assert m.jobs_by_status["running"] == 1

    def test_transition_pending_to_running_to_completed(self):
        m = AgentMetrics()
        # Simulate: submit (inc pending), start (pending->running), complete (running->completed)
        m.jobs_by_status["pending"] = 1
        m.record_transition("pending", "running")
        assert m.jobs_by_status["pending"] == 0
        assert m.jobs_by_status["running"] == 1
        m.record_transition("running", "completed")
        assert m.jobs_by_status["running"] == 0
        assert m.jobs_by_status["completed"] == 1

    def test_record_completion_latency(self):
        m = AgentMetrics()
        job = JobData(agent_name="echo", execution_ms=150.0, warmup_ms=10.0)
        m.record_completion(job)
        assert m.latencies == [150.0]
        assert m.percentile(50) == 150.0

    def test_latency_window_capped(self):
        m = AgentMetrics()
        for i in range(_LATENCY_WINDOW + 100):
            job = JobData(agent_name="echo", execution_ms=float(i), warmup_ms=1.0)
            m.record_completion(job)
        assert len(m.latencies) <= _LATENCY_WINDOW

    def test_warm_hit_counting(self):
        m = AgentMetrics()
        job_warm = JobData(agent_name="echo", execution_ms=10.0, warmup_ms=0.0)
        job_cold = JobData(agent_name="echo", execution_ms=10.0, warmup_ms=50.0)
        m.record_completion(job_warm)
        m.record_completion(job_cold)
        assert m.warm_hits == 1

    def test_retry_counting(self):
        m = AgentMetrics()
        job = JobData(agent_name="echo", execution_ms=10.0, retry_count=3)
        m.record_completion(job)
        assert m.total_retries == 3

    def test_throughput_per_min(self):
        m = AgentMetrics()
        # Manually inject recent timestamps
        now = time.time()
        m._completed_timestamps = [now - 10, now - 20, now - 30]
        assert m.throughput_per_min == 3.0

    def test_throughput_excludes_old(self):
        m = AgentMetrics()
        now = time.time()
        m._completed_timestamps = [now - 10, now - 120]  # 120s ago excluded
        assert m.throughput_per_min == 1.0

    def test_percentile_empty(self):
        m = AgentMetrics()
        assert m.percentile(50) == 0.0
        assert m.percentile(99) == 0.0

    def test_percentile_values(self):
        m = AgentMetrics()
        m.latencies = list(range(1, 101))  # 1..100, index 0=1, index 50=51
        assert m.percentile(50) == 51
        assert m.percentile(95) == 96
        assert m.percentile(99) == 100

    def test_to_dict_structure(self):
        m = AgentMetrics()
        d = m.to_dict()
        expected_keys = {
            "jobs_by_status", "total_jobs", "throughput_per_min",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "error_rate", "warm_hit_rate", "total_retries",
        }
        assert set(d.keys()) == expected_keys

    def test_error_rate(self):
        m = AgentMetrics()
        m.jobs_by_status["completed"] = 8
        m.jobs_by_status["failed"] = 2
        d = m.to_dict()
        assert d["error_rate"] == 0.2  # 2/10


# === MetricsCollector Tests ===


class TestMetricsCollector:
    def test_subscribes_on_init(self):
        bus = EventBus()
        collector = MetricsCollector(bus)
        assert bus.subscriber_count == 1
        collector.close()

    async def test_on_event_records_transition(self):
        bus = EventBus()
        collector = MetricsCollector(bus)
        job = JobData(agent_name="echo", status="running")
        await bus.emit(job, "pending", "running")

        data = collector.get_agent_metrics("echo")
        assert data is not None
        assert data["jobs_by_status"]["running"] == 1
        collector.close()

    async def test_on_event_records_completion(self):
        bus = EventBus()
        collector = MetricsCollector(bus)
        job = JobData(agent_name="echo", status="completed", execution_ms=42.0)
        await bus.emit(job, "running", "completed")

        data = collector.get_agent_metrics("echo")
        assert data["latency_p50_ms"] == 42.0
        collector.close()

    async def test_global_aggregation(self):
        bus = EventBus()
        collector = MetricsCollector(bus)

        job1 = JobData(agent_name="echo", execution_ms=10.0)
        await bus.emit(job1, "running", "completed")

        job2 = JobData(agent_name="summarizer", execution_ms=50.0)
        await bus.emit(job2, "running", "completed")

        g = collector.get_global_metrics()
        assert g["total_jobs"] == 2
        collector.close()

    def test_get_agent_metrics_unknown(self):
        bus = EventBus()
        collector = MetricsCollector(bus)
        assert collector.get_agent_metrics("nonexistent") is None
        collector.close()

    async def test_error_rate_calculation(self):
        bus = EventBus()
        collector = MetricsCollector(bus)

        for i in range(8):
            await bus.emit(JobData(agent_name="a"), "running", "completed")
        for i in range(2):
            await bus.emit(JobData(agent_name="a"), "running", "failed")

        g = collector.get_global_metrics()
        assert g["error_rate"] == 0.2
        collector.close()

    async def test_retry_rate(self):
        bus = EventBus()
        collector = MetricsCollector(bus)

        await bus.emit(JobData(agent_name="a", retry_count=2), "running", "completed")
        await bus.emit(JobData(agent_name="a"), "running", "completed")

        g = collector.get_global_metrics()
        assert g["retry_rate"] == 1.0  # 2 retries / 2 total jobs
        collector.close()

    def test_close_unsubscribes(self):
        bus = EventBus()
        collector = MetricsCollector(bus)
        assert bus.subscriber_count == 1
        collector.close()
        assert bus.subscriber_count == 0

    async def test_get_all_metrics(self):
        bus = EventBus()
        collector = MetricsCollector(bus)
        await bus.emit(JobData(agent_name="echo"), "pending", "running")
        data = collector.get_all_metrics()
        assert "global" in data
        assert "agents" in data
        assert "echo" in data["agents"]
        collector.close()
