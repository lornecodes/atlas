"""MetricsCollector — in-memory per-agent and global metrics via EventBus."""

from __future__ import annotations

import bisect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from atlas.events import EventBus, EventCallback
from atlas.pool.job import JobData

_LATENCY_WINDOW = 1000
_THROUGHPUT_WINDOW = 300  # seconds


@dataclass
class AgentMetrics:
    """Per-agent accumulated metrics."""

    jobs_by_status: dict[str, int] = field(default_factory=lambda: {
        "pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0,
    })
    latencies: list[float] = field(default_factory=list)
    _completed_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=_THROUGHPUT_WINDOW * 10)
    )
    warm_hits: int = 0
    total_retries: int = 0

    def record_transition(self, old_status: str, new_status: str) -> None:
        """Track status counters on each transition."""
        if old_status in self.jobs_by_status:
            self.jobs_by_status[old_status] = max(0, self.jobs_by_status[old_status] - 1)
        if new_status in self.jobs_by_status:
            self.jobs_by_status[new_status] = self.jobs_by_status.get(new_status, 0) + 1

    def record_completion(self, job: JobData) -> None:
        """Record latency, warm hits, and retries for a terminal job."""
        self._completed_timestamps.append(time.time())

        if job.execution_ms > 0:
            bisect.insort(self.latencies, job.execution_ms)
            if len(self.latencies) > _LATENCY_WINDOW:
                # Keep only the most recent window (already sorted)
                self.latencies = self.latencies[-_LATENCY_WINDOW:]

        if job.warmup_ms == 0:
            self.warm_hits += 1

        if job.retry_count > 0:
            self.total_retries += job.retry_count

    @property
    def throughput_per_min(self) -> float:
        """Jobs completed in the last 60 seconds."""
        if not self._completed_timestamps:
            return 0.0
        cutoff = time.time() - 60
        return float(sum(1 for t in self._completed_timestamps if t > cutoff))

    def percentile(self, p: float) -> float:
        """Compute a latency percentile from the sorted window."""
        if not self.latencies:
            return 0.0
        idx = int(len(self.latencies) * p / 100)
        idx = min(idx, len(self.latencies) - 1)
        return self.latencies[idx]

    def to_dict(self) -> dict[str, Any]:
        total = sum(self.jobs_by_status.values())
        failed = self.jobs_by_status.get("failed", 0)
        completed = self.jobs_by_status.get("completed", 0)
        return {
            "jobs_by_status": dict(self.jobs_by_status),
            "total_jobs": total,
            "throughput_per_min": round(self.throughput_per_min, 2),
            "latency_p50_ms": round(self.percentile(50), 2),
            "latency_p95_ms": round(self.percentile(95), 2),
            "latency_p99_ms": round(self.percentile(99), 2),
            "error_rate": round(failed / max(total, 1), 4),
            "warm_hit_rate": round(self.warm_hits / max(completed, 1), 4),
            "total_retries": self.total_retries,
        }


class MetricsCollector:
    """EventBus subscriber that accumulates per-agent and global metrics."""

    def __init__(self, bus: EventBus) -> None:
        self._agents: dict[str, AgentMetrics] = {}
        self._bus = bus
        self._callback: EventCallback = self._on_event
        bus.subscribe(self._callback)

    def _get_agent(self, name: str) -> AgentMetrics:
        if name not in self._agents:
            self._agents[name] = AgentMetrics()
        return self._agents[name]

    async def _on_event(self, job: JobData, old_status: str, new_status: str) -> None:
        agent = self._get_agent(job.agent_name)
        agent.record_transition(old_status, new_status)
        if new_status in ("completed", "failed"):
            agent.record_completion(job)

    def get_agent_metrics(self, agent_name: str) -> dict[str, Any] | None:
        agent = self._agents.get(agent_name)
        if not agent:
            return None
        return agent.to_dict()

    def get_global_metrics(self) -> dict[str, Any]:
        """Aggregate across all agents."""
        total_jobs = 0
        total_failed = 0
        total_completed = 0
        all_latencies: list[float] = []
        total_warm_hits = 0
        total_retries = 0
        throughput = 0.0

        for agent in self._agents.values():
            total_jobs += sum(agent.jobs_by_status.values())
            total_failed += agent.jobs_by_status.get("failed", 0)
            total_completed += agent.jobs_by_status.get("completed", 0)
            all_latencies.extend(agent.latencies)
            total_warm_hits += agent.warm_hits
            total_retries += agent.total_retries
            throughput += agent.throughput_per_min

        all_latencies.sort()

        def _pct(lst: list[float], p: float) -> float:
            if not lst:
                return 0.0
            idx = min(int(len(lst) * p / 100), len(lst) - 1)
            return lst[idx]

        return {
            "total_jobs": total_jobs,
            "error_rate": round(total_failed / max(total_jobs, 1), 4),
            "retry_rate": round(total_retries / max(total_jobs, 1), 4),
            "warm_hit_rate": round(total_warm_hits / max(total_completed, 1), 4),
            "throughput_per_min": round(throughput, 2),
            "latency_p50_ms": round(_pct(all_latencies, 50), 2),
            "latency_p95_ms": round(_pct(all_latencies, 95), 2),
            "latency_p99_ms": round(_pct(all_latencies, 99), 2),
        }

    def get_all_metrics(self) -> dict[str, Any]:
        return {
            "global": self.get_global_metrics(),
            "agents": {name: agent.to_dict() for name, agent in self._agents.items()},
        }

    def close(self) -> None:
        self._bus.unsubscribe(self._callback)
