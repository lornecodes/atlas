"""Orchestrator protocol — the pluggable routing interface.

Users implement this to control how jobs are routed, prioritized,
rejected, or redirected before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.pool.job import JobData


@dataclass
class RoutingDecision:
    """The orchestrator's decision for a job.

    Actions:
        execute  — proceed with execution (default)
        reject   — refuse the job with an error message
        redirect — execute with a different agent
    """

    action: str = "execute"  # execute | reject | redirect
    agent_name: str | None = None  # Override agent (for redirect)
    priority: int | None = None  # Override priority
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Orchestrator(Protocol):
    """Protocol for pluggable job orchestration.

    The orchestrator sits between job submission and execution.
    It can inspect, modify, reject, or redirect jobs before
    they reach the pool's execution pipeline.

    Implement this protocol to build custom routing strategies:
    - Model tier routing based on task complexity
    - Cost-aware scheduling
    - Load balancing across agent variants
    - A/B testing between agent versions
    """

    async def route(
        self, job: JobData, registry: AgentRegistry
    ) -> RoutingDecision:
        """Decide how to handle a job before execution.

        Called after the job is dequeued but before slot acquisition.
        Return a RoutingDecision to control execution.
        """
        ...

    async def on_job_complete(self, job: JobData) -> None:
        """Called when a job completes successfully.

        Use for: updating internal state, metrics, adaptive routing.
        """
        ...

    async def on_job_failed(self, job: JobData) -> None:
        """Called when a job fails.

        Use for: circuit breaking, fallback routing, error tracking.
        """
        ...
