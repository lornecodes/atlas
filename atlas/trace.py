"""Execution tracing — structured trace objects for jobs and chains."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from atlas.events import EventBus, EventCallback
from atlas.logging import get_logger

if TYPE_CHECKING:
    from atlas.pool.job import JobData

logger = get_logger(__name__)


# Simple cost lookup: (input_per_1k, output_per_1k) in USD
_COST_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-opus-4-20250514": (0.015, 0.075),
    # OpenAI
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from model name and token counts.

    Returns 0.0 for unknown models.
    """
    rates = _COST_TABLE.get(model)
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1000


@dataclass
class EvalResult:
    """Result of a single eval check."""

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass
class ExecutionTrace:
    """Structured trace for a single agent execution."""

    trace_id: str
    job_id: str
    agent_name: str
    status: str

    # Timing
    warmup_ms: float = 0.0
    execution_ms: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    # LLM metadata
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0

    # Correlation
    parent_trace_id: str = ""
    chain_name: str = ""
    step_index: int = -1

    # Eval results (populated async)
    eval_results: list[EvalResult] = field(default_factory=list)

    # Raw metadata passthrough
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "warmup_ms": self.warmup_ms,
            "execution_ms": self.execution_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "parent_trace_id": self.parent_trace_id,
            "chain_name": self.chain_name,
            "step_index": self.step_index,
            "eval_results": [e.to_dict() for e in self.eval_results],
            "metadata": self.metadata,
        }


@dataclass
class ChainTrace:
    """Aggregated trace for a chain execution."""

    trace_id: str
    chain_name: str
    status: str
    steps: list[ExecutionTrace] = field(default_factory=list)
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(s.estimated_cost_usd for s in self.steps)

    @property
    def total_execution_ms(self) -> float:
        return sum(s.execution_ms for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "chain_name": self.chain_name,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_execution_ms": round(self.total_execution_ms, 2),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class TraceCollector:
    """EventBus subscriber that builds ExecutionTrace on job completion.

    Follows the same architecture as MetricsCollector — subscribes to
    the EventBus, collects data, provides query methods.
    """

    def __init__(self, bus: EventBus, max_traces: int = 5000) -> None:
        self._traces: OrderedDict[str, ExecutionTrace] = OrderedDict()
        self._bus = bus
        self._max_traces = max_traces
        self._callback: EventCallback = self._on_event
        bus.subscribe(self._callback)

    async def _on_event(
        self, job: JobData, old_status: str, new_status: str
    ) -> None:
        if new_status not in ("completed", "failed"):
            return

        meta = job.metadata
        input_tokens = meta.get("_trace_input_tokens", 0)
        output_tokens = meta.get("_trace_output_tokens", 0)
        model = meta.get("_trace_model", "")

        trace = ExecutionTrace(
            trace_id=job.id,
            job_id=job.id,
            agent_name=job.agent_name,
            status=new_status,
            warmup_ms=job.warmup_ms,
            execution_ms=job.execution_ms,
            started_at=job.started_at,
            completed_at=job.completed_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            estimated_cost_usd=estimate_cost(model, input_tokens, output_tokens),
            parent_trace_id=meta.get("_parent_trace_id", ""),
            chain_name=meta.get("_chain_name", ""),
            step_index=meta.get("_step_index", -1),
            metadata={k: v for k, v in meta.items() if not k.startswith("_")},
        )

        self._traces[trace.trace_id] = trace
        self._evict_old()

    def record(self, trace: ExecutionTrace) -> None:
        """Manually record a trace (e.g. from chain runner)."""
        self._traces[trace.trace_id] = trace
        self._evict_old()

    def attach_eval_results(
        self, trace_id: str, results: list[EvalResult]
    ) -> None:
        """Attach eval results to an existing trace."""
        trace = self._traces.get(trace_id)
        if trace:
            trace.eval_results = results

    def get(self, trace_id: str) -> ExecutionTrace | None:
        return self._traces.get(trace_id)

    def list(
        self,
        *,
        limit: int = 50,
        agent_name: str | None = None,
    ) -> list[ExecutionTrace]:
        """List traces, newest first."""
        traces = list(reversed(self._traces.values()))
        if agent_name:
            traces = [t for t in traces if t.agent_name == agent_name]
        return traces[:limit]

    def _evict_old(self) -> None:
        while len(self._traces) > self._max_traces:
            self._traces.popitem(last=False)

    def close(self) -> None:
        self._bus.unsubscribe(self._callback)
