"""DefaultOrchestrator — pass-through, no-op baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.orchestrator.protocol import RoutingDecision

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.pool.job import JobData


class DefaultOrchestrator:
    """Pass-through orchestrator — all jobs execute as-is.

    This is the baseline. It makes no routing decisions,
    applies no transformations, and collects no metrics.
    Swap it out for a custom orchestrator to add intelligence.
    """

    async def route(
        self, job: JobData, registry: AgentRegistry
    ) -> RoutingDecision:
        return RoutingDecision(action="execute")

    async def on_job_complete(self, job: JobData) -> None:
        pass

    async def on_job_failed(self, job: JobData) -> None:
        pass
