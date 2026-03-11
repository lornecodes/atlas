"""Priority Router — routes jobs based on input complexity.

Short text inputs get routed to fast models, longer inputs to powerful ones.
Demonstrates the Orchestrator protocol with a non-trivial routing strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.orchestrator.protocol import RoutingDecision

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.pool.job import JobData


# Thresholds for text length classification
SHORT_THRESHOLD = 200   # characters
LONG_THRESHOLD = 1000   # characters


class PriorityRouterOrchestrator:
    """Routes jobs based on input text length.

    - Short text (< 200 chars): low priority, fast model preferred
    - Medium text (200-1000 chars): normal priority
    - Long text (> 1000 chars): high priority, powerful model preferred

    Also rejects jobs with empty input data.
    """

    def _estimate_complexity(self, input_data: dict) -> str:
        """Estimate input complexity from text length."""
        text = ""
        for value in input_data.values():
            if isinstance(value, str):
                text += value

        length = len(text)
        if length < SHORT_THRESHOLD:
            return "low"
        elif length > LONG_THRESHOLD:
            return "high"
        return "medium"

    async def route(
        self, job: JobData, registry: AgentRegistry
    ) -> RoutingDecision:
        if not job.input_data:
            return RoutingDecision(
                action="reject",
                metadata={"reason": "Empty input data"},
            )

        complexity = self._estimate_complexity(job.input_data)

        if complexity == "low":
            return RoutingDecision(
                action="execute",
                priority=-1,  # Lower priority
                metadata={"complexity": "low", "model_tier": "fast"},
            )
        elif complexity == "high":
            return RoutingDecision(
                action="execute",
                priority=10,  # Higher priority
                metadata={"complexity": "high", "model_tier": "powerful"},
            )

        return RoutingDecision(
            action="execute",
            metadata={"complexity": "medium", "model_tier": "balanced"},
        )

    async def on_job_complete(self, job: JobData) -> None:
        pass

    async def on_job_failed(self, job: JobData) -> None:
        pass
